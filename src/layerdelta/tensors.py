"""Streaming safetensors reader and tensor table.

Only the JSON header is parsed; tensor bytes are read on demand with seek+read, so no model is
built, nothing is unpickled and memory use is independent of file size. Header validation follows
checkpoint.py (unique fields, exact dtype*shape extents, contiguous non-overlapping data) but is
generic: it does not know any architecture.
"""

import contextlib
import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

MAX_FILE_BYTES = 16 << 30
MAX_HEADER_BYTES = 100 << 20
MAX_SHARDS = 4096
INDEX_SUFFIX = ".safetensors.index.json"

# safetensors dtype -> (bytes per element, torch dtype name). Unknown dtypes are rejected.
DTYPES = {
    "BOOL": (1, "bool"), "U8": (1, "uint8"), "I8": (1, "int8"), "I16": (2, "int16"),
    "U16": (2, "uint16"), "I32": (4, "int32"), "U32": (4, "uint32"), "I64": (8, "int64"),
    "U64": (8, "uint64"), "F16": (2, "float16"), "BF16": (2, "bfloat16"), "F32": (4, "float32"),
    "F64": (8, "float64"), "F8_E4M3": (1, "float8_e4m3fn"), "F8_E5M2": (1, "float8_e5m2"),
}
# Only these can be edited linearly; everything else is inspect-only.
EDITABLE_DTYPES = frozenset({"F32", "F16", "BF16"})

_HASH_CHUNK = 1 << 20
_EDGE = 64 << 10  # bytes sampled from each end of a tensor by the duplicate pre-filter


class TensorFileError(ValueError):
    """Rejected input. `code` is a stable machine-readable reason."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    path: Path
    start: int  # absolute byte offsets in `path`
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    @property
    def editable(self) -> bool:
        return self.dtype in EDITABLE_DTYPES


@dataclass
class TensorTable:
    tensors: dict[str, TensorInfo]
    metadata: dict[str, str] = field(default_factory=dict)
    sharded: bool = False
    # (size, mtime_ns, st_ino, st_dev) per file at describe time. This detects replaced or resized
    # files, not an in-place overwrite that preserves size and mtime: content identity ultimately
    # relies on the workspace keeping its own content-addressed copy of the checkpoint (T4b).
    _stamps: dict[Path, tuple[int, int, int, int]] = field(default_factory=dict, repr=False)
    _hashes: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def files(self) -> list[Path]:
        """Every file this table was parsed from (the shards of a sharded set), sorted."""
        return sorted(self._stamps)

    def _check(self, info: TensorInfo, stat: os.stat_result) -> None:
        if _stamp(stat) != self._stamps[info.path]:
            raise TensorFileError("file_changed", f"{info.path.name} changed since it was described")

    def _open(self, info: TensorInfo):
        with _unreadable(info.path):
            stream = info.path.open("rb")
        try:
            self._check(info, os.fstat(stream.fileno()))
            stream.seek(info.start)
            return stream
        except BaseException:
            stream.close()
            raise

    def _read_at(self, info: TensorInfo, offset: int, length: int) -> bytearray:
        data = bytearray(length)  # preallocated: peak memory is the tensor, not twice the tensor
        with _unreadable(info.path), self._open(info) as stream:
            stream.seek(info.start + offset)
            if stream.readinto(data) != length:
                raise TensorFileError("file_changed", f"{info.path.name} is truncated")
        return data

    @property
    def identity(self) -> list:
        """What this table was parsed from: (file name, size, mtime_ns, inode, device) per file."""
        return sorted([path.name, *stamp] for path, stamp in self._stamps.items())

    @property
    def file_bytes(self) -> int:
        return sum(stamp[0] for stamp in self._stamps.values())

    @property
    def file_count(self) -> int:
        return len(self._stamps)

    def read_ranges(self, name: str, ranges):
        """Yield one bytearray per (byte_offset, length) in `ranges`, relative to the tensor start.

        The file is opened once and `ranges` may be a lazy iterable, so reading a strided selection
        costs one open, not one per run. Close the generator (or exhaust it) to release the handle.
        """
        info = self.tensors[name]
        with _unreadable(info.path), self._open(info) as stream:
            for offset, length in ranges:
                if offset < 0 or length < 0 or offset + length > info.nbytes:
                    raise ValueError(f"range {offset}+{length} outside tensor {name} ({info.nbytes} bytes)")
                data = bytearray(length)
                stream.seek(info.start + offset)
                if stream.readinto(data) != length:
                    raise TensorFileError("file_changed", f"{info.path.name} is truncated")
                yield data

    def read_bytes(self, name: str) -> bytearray:
        info = self.tensors[name]
        return self._read_at(info, 0, info.nbytes)

    def read_tensor(self, name: str):
        import torch  # lazy: describing a checkpoint should not pay for torch

        info = self.tensors[name]
        data = self.read_bytes(name)
        if not data:
            return torch.empty(info.shape, dtype=getattr(torch, DTYPES[info.dtype][1]))
        flat = torch.frombuffer(data, dtype=getattr(torch, DTYPES[info.dtype][1]))
        return flat.reshape(info.shape)

    def sha256(self, name: str) -> str:
        """Lazy, cached, streamed in 1 MiB chunks. A cache hit still re-checks the file stamp."""
        info = self.tensors[name]
        if name in self._hashes:
            with _unreadable(info.path):
                self._check(info, os.stat(info.path))
            return self._hashes[name]
        digest, left = hashlib.sha256(), info.nbytes
        with _unreadable(info.path), self._open(info) as stream:
            while left:
                chunk = stream.read(min(_HASH_CHUNK, left))
                if not chunk:
                    raise TensorFileError("file_changed", f"{info.path.name} is truncated")
                digest.update(chunk)
                left -= len(chunk)
        self._hashes[name] = digest.hexdigest()
        return self._hashes[name]

    def _edge_digest(self, name: str) -> str:
        """Cheap fingerprint: first and last 64 KiB. Equal tensors always agree; most others differ."""
        info = self.tensors[name]
        edge = min(_EDGE, info.nbytes)
        head = self._read_at(info, 0, edge)
        tail = self._read_at(info, info.nbytes - edge, edge)
        return hashlib.sha256(head + tail).hexdigest()

    def duplicates(self) -> list[list[str]]:
        """Groups of tensors with identical bytes (possible tied weights). Empty tensors ignored.

        This reads data and is NOT part of open_tensors/describe: call it only on request. Cost is
        staged so a whole-model scan stays cheap: (1) group by byte size, (2) group by a hash of the
        first and last 64 KiB (at most 128 KiB read per candidate), (3) full sha256 only for tensors
        that still collide. Same-size tensors with different edges are never fully read.
        """
        def groups(names, key):
            grouped: dict[str, list[str]] = {}
            for name in names:
                grouped.setdefault(key(name), []).append(name)
            return [g for g in grouped.values() if len(g) > 1]

        by_size = groups((n for n, i in self.tensors.items() if i.nbytes), lambda n: str(self.tensors[n].nbytes))
        found = []
        for same_size in by_size:
            for same_edges in groups(same_size, self._edge_digest):
                found.extend(groups(same_edges, self.sha256))
        return found

    def check_schema(self, expected: dict[str, tuple[str, tuple[int, ...]]]) -> None:
        """Require exactly these names with these (safetensors dtype, shape) pairs."""
        actual = {n: (i.dtype, i.shape) for n, i in self.tensors.items()}
        expected = {n: (d, tuple(s)) for n, (d, s) in expected.items()}
        if actual != expected:
            bad = sorted(n for n in actual.keys() | expected.keys() if actual.get(n) != expected.get(n))
            raise TensorFileError("schema_mismatch", f"tensor table differs from schema at {bad}")


def _stamp(stat: os.stat_result) -> tuple[int, int, int, int]:
    return stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev


@contextlib.contextmanager
def _unreadable(path: Path):
    """Surface OS-level failures (missing, directory, permission, vanished) as TensorFileError."""
    try:
        yield
    except OSError as exc:
        raise TensorFileError("unreadable", f"cannot read {path.name}: {exc.strerror or exc}") from exc


def _unique_fields(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise TensorFileError("duplicate_field", f"duplicate header field {name!r}")
        result[name] = value
    return result


def _parse_json(raw: bytes, code: str, what: str):
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_fields)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise TensorFileError(code, f"invalid {what} JSON") from exc


def _read_file(path: Path, max_file_bytes: int, max_header_bytes: int):
    """Validate one safetensors header. Returns (tensors, metadata, stamp) with stamp = (size, mtime_ns, st_ino, st_dev)."""
    with _unreadable(path), path.open("rb") as stream:
        stat = os.fstat(stream.fileno())
        size = stat.st_size
        if not 8 <= size <= max_file_bytes:
            raise TensorFileError("file_size", f"{path.name}: invalid file size {size}")
        prefix = stream.read(8)
        if len(prefix) != 8:  # shrank after fstat
            raise TensorFileError("file_changed", f"{path.name} changed while being read")
        header_len, = struct.unpack("<Q", prefix)
        if not 2 <= header_len <= min(max_header_bytes, size - 8):
            raise TensorFileError("header_size", f"{path.name}: invalid header size {header_len}")
        raw = stream.read(header_len)
    if len(raw) != header_len:
        raise TensorFileError("file_changed", f"{path.name} changed while being read")
    header = _parse_json(raw, "header_json", "header")
    if not isinstance(header, dict):
        raise TensorFileError("header_json", "header is not an object")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(type(v) is not str for v in metadata.values()):
        raise TensorFileError("metadata", "metadata must map strings to strings")

    base, intervals, tensors = 8 + header_len, [], {}
    for name, record in header.items():
        if not isinstance(record, dict) or set(record) != {"dtype", "shape", "data_offsets"}:
            raise TensorFileError("tensor_record", f"invalid tensor header for {name}")
        dtype = record["dtype"]
        if type(dtype) is not str or dtype not in DTYPES:
            raise TensorFileError("dtype_unknown", f"unsupported dtype {dtype!r} for {name}")
        shape = record["shape"]
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            raise TensorFileError("shape", f"invalid shape for {name}")
        offsets = record["data_offsets"]
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(i) is not int or i < 0 for i in offsets) or offsets[0] > offsets[1]):
            raise TensorFileError("extent", f"invalid byte extent for {name}")
        if offsets[1] - offsets[0] != math.prod(shape) * DTYPES[dtype][0]:
            raise TensorFileError("extent_mismatch", f"byte extent does not match dtype*shape for {name}")
        intervals.append((*offsets, name))
        tensors[name] = TensorInfo(name, dtype, tuple(shape), path, base + offsets[0], base + offsets[1])
    end = 0
    for start, stop, name in sorted(intervals):
        if start != end:
            raise TensorFileError("extent_overlap_or_gap", f"overlapping or missing bytes at {name}")
        end = stop
    if end != size - base:
        raise TensorFileError("data_size", f"{path.name}: data size does not match header")
    return tensors, metadata, _stamp(stat)


_WINDOWS_DEVICES = frozenset({"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                              *(f"LPT{i}" for i in range(1, 10))})


def _shard_path(directory: Path, name) -> Path:
    # Plain sibling file names only: no separators, drive letters or "..". Also nothing Windows
    # would reinterpret: ':' (NTFS alternate data streams), device names (NUL.safetensors) and
    # characters a file name cannot hold.
    if type(name) is str and (name.rstrip(" .") != name or any(c in name for c in '<>:"|?*\0')
                              or any(ord(c) < 32 for c in name)
                              or name.split(".", 1)[0].rstrip(" ").upper() in _WINDOWS_DEVICES):
        raise TensorFileError("index_invalid", f"shard name {name!r} is not a plain file name")
    # A sibling may still be a symlink pointing elsewhere: fine for local import; anything serving
    # untrusted index files (HTTP, T4b) must resolve() and confine the path first.
    if type(name) is not str or not name.endswith(".safetensors") or name != Path(name).name:
        raise TensorFileError("index_invalid", f"invalid shard name {name!r}")
    return directory / name


def open_tensors(path, *, max_file_bytes: int = MAX_FILE_BYTES,
                 max_header_bytes: int = MAX_HEADER_BYTES) -> TensorTable:
    """Describe a .safetensors file or a *.safetensors.index.json set without reading tensor data."""
    path = Path(path)
    if not path.name.endswith(INDEX_SUFFIX):
        tensors, metadata, stamp = _read_file(path, max_file_bytes, max_header_bytes)
        return TensorTable(tensors, metadata, False, {path: stamp})

    with _unreadable(path), path.open("rb") as stream:
        raw = stream.read(max_header_bytes + 1)
    if len(raw) > max_header_bytes:
        raise TensorFileError("index_invalid", "index file too large")
    index = _parse_json(raw, "index_invalid", "index")
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if (not isinstance(weight_map, dict) or not weight_map
            or any(type(k) is not str for k in weight_map)):
        raise TensorFileError("index_invalid", "index needs a non-empty weight_map")
    shards = {}
    for shard in weight_map.values():
        shards[shard] = _shard_path(path.parent, shard)
    if len(shards) > MAX_SHARDS:
        raise TensorFileError("index_invalid", "too many shards")

    tensors, metadata, stamps, listed = {}, {}, {}, set(weight_map)
    for shard, shard_path in shards.items():
        with _unreadable(shard_path):
            present = shard_path.is_file()
        if not present:
            raise TensorFileError("shard_missing", f"shard {shard} is missing")
        found, shard_meta, stamps[shard_path] = _read_file(shard_path, max_file_bytes, max_header_bytes)
        for name, info in found.items():
            if name in tensors:
                raise TensorFileError("duplicate_tensor", f"{name} appears in more than one shard")
            if weight_map.get(name) != shard:
                raise TensorFileError("index_mismatch", f"{name} is in {shard} but not mapped to it")
            tensors[name] = info
        for k, v in shard_meta.items():
            if metadata.setdefault(k, v) != v:
                raise TensorFileError("metadata", f"shards disagree on metadata key {k!r}")
    if set(tensors) != listed:
        raise TensorFileError("index_mismatch", f"indexed tensors missing from shards: {sorted(listed - set(tensors))}")
    return TensorTable(tensors, metadata, True, stamps)
