"""Tensor-local selectors: strict parsing, canonical form, selection_id and row-major enumeration.

A selection is a pure value: per axis a sorted list of merged half-open [start, stop) ranges.
Elements are always enumerated in row-major order over the selected indices. Nothing here builds
a model or reads more of the file than the requested byte runs.
"""

import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass

import numpy as np

from .tensors import DTYPES, TensorInfo, TensorTable

MAX_INDEX_LIST = 100_000
CHUNK_ELEMS = 1 << 20  # fixed, so float accumulation order never depends on the machine


class InspectError(ValueError):
    """Structured request error: `code`, human `message`, dotted `field` path into the request."""

    def __init__(self, code: str, message: str, field: str | None = None):
        super().__init__(f"{code}: {message}" + (f" ({field})" if field else ""))
        self.code, self.message, self.field = code, message, field

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "field": self.field}


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False)


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Selection:
    info: TensorInfo
    axes: tuple[tuple[tuple[int, int], ...], ...]  # per axis: sorted, merged [start, stop) ranges

    @property
    def shape(self) -> tuple[int, ...]:
        return self.info.shape

    @property
    def counts(self) -> tuple[int, ...]:
        return tuple(sum(e - s for s, e in ranges) for ranges in self.axes)

    @property
    def total(self) -> int:
        return math.prod(self.counts)

    @property
    def canonical(self) -> dict:
        return {"tensor": self.info.name, "shape": list(self.shape),
                "index": {str(i): [list(r) for r in ranges] for i, ranges in enumerate(self.axes)}}

    @property
    def resolved(self) -> dict:
        """The selection as a request: feeding this back to parse_selector yields the same selection_id."""
        return {"tensor": self.info.name, "shape": list(self.shape),
                "index": {str(i): [{"start": s, "stop": e} for s, e in ranges] for i, ranges in enumerate(self.axes)}}

    @property
    def selection_id(self) -> str:
        return "sel_" + digest(self.canonical)[:32]

    def indices(self, axis: int) -> list[int]:
        return [i for s, e in self.axes[axis] for i in range(s, e)]

    def narrow(self, picks: dict[int, int]) -> "Selection":
        """The same selection restricted to single real indices on some axes."""
        axes = list(self.axes)
        for axis, index in picks.items():
            axes[axis] = ((index, index + 1),)
        return Selection(self.info, tuple(axes))

    def request_form(self) -> dict:
        """Compact selector JSON: only the axes that are not whole, as {start, stop} ranges."""
        index = {}
        for axis, ranges in enumerate(self.axes):
            if ranges != ((0, self.shape[axis]),):
                index[str(axis)] = [{"start": s, "stop": e} for s, e in ranges]
        return {"tensor": self.info.name, **({"index": index} if index else {})}


def _int(value) -> bool:
    return type(value) is int


def _merge(ranges) -> tuple[tuple[int, int], ...]:
    merged: list[list[int]] = []
    for start, stop in sorted(ranges):
        if merged and merged[-1][1] >= start:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return tuple((s, e) for s, e in merged)


def _axis_ranges(spec, dim: int, field: str) -> tuple[tuple[int, int], ...]:
    if isinstance(spec, list) and spec and all(isinstance(item, dict) for item in spec):
        if len(spec) > MAX_INDEX_LIST:
            raise InspectError("cap_exceeded", f"at most {MAX_INDEX_LIST} ranges per axis", field)
        return _merge(r for i, item in enumerate(spec) for r in _axis_ranges(item, dim, f"{field}[{i}]"))
    if isinstance(spec, list):
        if not spec:
            raise InspectError("invalid_index", "index list must not be empty", field)
        if len(spec) > MAX_INDEX_LIST:
            raise InspectError("cap_exceeded", f"at most {MAX_INDEX_LIST} indices per axis", field)
        for position, i in enumerate(spec):
            where = f"{field}[{position}]"
            if not _int(i):
                raise InspectError("invalid_index", "index must be an integer", where)
            if i < 0:
                raise InspectError("invalid_index", f"negative index {i} is not allowed", where)
            if i >= dim:
                raise InspectError("index_out_of_range", f"index {i} is outside axis of size {dim}", where)
        return _merge((i, i + 1) for i in set(spec))
    if isinstance(spec, dict):
        for key in spec:
            if key not in ("start", "stop"):
                raise InspectError("unknown_field", f"unknown range field {key!r}", f"{field}.{key}")
        for key in ("start", "stop"):
            if key not in spec:
                raise InspectError("invalid_range", f"range needs {key}", f"{field}.{key}")
            if not _int(spec[key]):
                raise InspectError("invalid_range", f"{key} must be an integer", f"{field}.{key}")
            if spec[key] < 0:
                raise InspectError("invalid_index", f"negative {key} {spec[key]} is not allowed", f"{field}.{key}")
        start, stop = spec["start"], spec["stop"]
        if stop > dim:
            raise InspectError("index_out_of_range", f"stop {stop} is outside axis of size {dim}", f"{field}.stop")
        if start >= stop:
            raise InspectError("invalid_range", f"start {start} must be below stop {stop}", f"{field}.start")
        return ((start, stop),)
    raise InspectError("invalid_index", "index must be a list of integers or {start, stop}", field)


def parse_selector(table: TensorTable, selector, *, field: str = "selector",
                   checkpoint_id: str | None = None) -> Selection:
    if not isinstance(selector, dict):
        raise InspectError("invalid_argument", "selector must be an object", field)
    for key in selector:
        if key in ("group", "units"):
            raise InspectError("unsupported_selector", "coupled-group selectors arrive with v0.3", f"{field}.{key}")
        if key not in ("checkpoint", "tensor", "index", "shape"):
            raise InspectError("unknown_field", f"unknown selector field {key!r}", f"{field}.{key}")
    name = selector.get("tensor")
    if not isinstance(name, str):
        raise InspectError("invalid_argument", "selector.tensor must be a tensor name", f"{field}.tensor")
    if name not in table.tensors:
        raise InspectError("unknown_tensor", f"no tensor named {name!r}", f"{field}.tensor")
    if "checkpoint" in selector and checkpoint_id is not None and selector["checkpoint"] != checkpoint_id:
        raise InspectError("invalid_argument", "selector.checkpoint differs from the checkpoint of this call",
                           f"{field}.checkpoint")
    info = table.tensors[name]
    if "shape" in selector and selector["shape"] != list(info.shape):
        raise InspectError("invalid_argument", f"selector.shape {selector['shape']!r} differs from the tensor's "
                           f"shape {list(info.shape)}", f"{field}.shape")
    index = selector.get("index", {})
    if not isinstance(index, dict):
        raise InspectError("invalid_argument", "selector.index must be an object", f"{field}.index")
    given: dict[int, object] = {}
    for key, spec in index.items():
        if not isinstance(key, str) or not re.fullmatch("0|[1-9][0-9]*", key) or int(key) >= len(info.shape):
            raise InspectError("invalid_axis", f"axis {key!r} does not exist (tensor has {len(info.shape)} axes)",
                               f"{field}.index.{key}")
        given[int(key)] = spec
    axes = tuple(_axis_ranges(given[i], dim, f"{field}.index.{i}") if i in given else ((0, dim),)
                 for i, dim in enumerate(info.shape))
    return Selection(info, axes)


def _runs(sel: Selection):
    """Contiguous (element_offset, element_count) runs in row-major selection order."""
    shape, axes = sel.shape, sel.axes
    if sel.total == 0:
        return
    whole = [ranges == ((0, n),) for ranges, n in zip(axes, shape)]
    if all(whole):
        yield 0, sel.total
        return
    k = max(i for i, w in enumerate(whole) if not w)  # trailing whole axes fold into one run
    strides = [math.prod(shape[i + 1:]) for i in range(len(shape))]
    outer = [sel.indices(i) for i in range(k)]
    for prefix in itertools.product(*outer):
        base = sum(i * strides[a] for a, i in enumerate(prefix))
        for start, stop in axes[k]:
            yield base + start * strides[k], (stop - start) * strides[k]


def _split(runs, limit):
    for offset, count in runs:
        while count > limit:
            yield offset, limit
            offset, count = offset + limit, count - limit
        yield offset, count


def iter_raw(table: TensorTable, sel: Selection, chunk_elems: int = CHUNK_ELEMS, *, skip: int = 0,
             take: int | None = None):
    """Yield raw byte chunks (<= chunk_elems elements each) of the selection, row-major.

    `skip`/`take` window the flat selection without reading the skipped elements.
    """
    item = DTYPES[sel.info.dtype][0]
    take = sel.total - skip if take is None else take

    def windowed():
        left, to_skip = take, skip
        for offset, count in _split(_runs(sel), chunk_elems):
            if left <= 0:
                return
            if to_skip >= count:
                to_skip -= count
                continue
            offset, count, to_skip = offset + to_skip, count - to_skip, 0
            count = min(count, left)
            left -= count
            yield offset * item, count * item

    parts, size = [], 0
    for data in table.read_ranges(sel.info.name, windowed()):
        parts.append(data)
        size += len(data) // item
        if size >= chunk_elems:
            yield parts[0] if len(parts) == 1 else bytearray().join(parts)
            parts, size = [], 0
    if parts:
        yield parts[0] if len(parts) == 1 else bytearray().join(parts)


_NUMPY = {"F32": "<f4", "F16": "<f2", "F64": "<f8", "I8": "i1", "U8": "u1", "I16": "<i2", "U16": "<u2",
          "I32": "<i4", "U32": "<u4", "I64": "<i8", "U64": "<u8"}


def decode(dtype: str, raw) -> np.ndarray:
    """Stored bytes -> numpy array holding the exact stored values (BF16/F8 widened losslessly)."""
    if dtype in _NUMPY:
        return np.frombuffer(raw, dtype=_NUMPY[dtype])
    if dtype == "BOOL":
        return np.frombuffer(raw, dtype=np.uint8).astype(bool)
    if dtype == "BF16":
        return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    import torch  # F8: no numpy dtype

    torch_dtype = getattr(torch, DTYPES[dtype][1])
    return torch.frombuffer(bytearray(raw), dtype=torch_dtype).float().numpy()


def raw_bits(dtype: str, raw) -> np.ndarray:
    return np.frombuffer(raw, dtype=f"u{DTYPES[dtype][0]}")


def is_float(dtype: str) -> bool:
    return dtype in ("F16", "BF16", "F32", "F64", "F8_E4M3", "F8_E5M2")
