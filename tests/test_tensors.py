import hashlib
import json
import os
import struct
import threading
import time

import psutil
import pytest
import torch
from safetensors.torch import save, save_file

from layerdelta.model import ReferenceCNN
from layerdelta.tensors import TensorFileError, open_tensors

TORCH_TO_ST = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16"}


def _write(tmp_path, blob, name="m.safetensors"):
    path = tmp_path / name
    path.write_bytes(blob)
    return path


def _schema(state):
    return {n: (TORCH_TO_ST[t.dtype], tuple(t.shape)) for n, t in state.items()}


def test_reference_table_matches_state_dict(tmp_path):
    state = ReferenceCNN().state_dict()
    table = open_tensors(_write(tmp_path, save(state)))
    assert sorted(table.tensors) == sorted(state)
    table.check_schema(_schema(state))
    for name, tensor in state.items():
        assert table.tensors[name].editable
        assert torch.equal(table.read_tensor(name), tensor)
        assert table.sha256(name) == hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


@pytest.mark.parametrize("failure,code", [
    ("keys", "schema_mismatch"), ("shape", "schema_mismatch"), ("dtype", "schema_mismatch"),
    ("dtype_unknown", "dtype_unknown"), ("extent", "extent_mismatch"),
    ("overlap", "extent_overlap_or_gap"), ("gap", "extent_overlap_or_gap"), ("trailing", "data_size"),
    ("header_limit", "header_size"), ("file_limit", "file_size"), ("short", "file_size"),
    ("dup_field", "duplicate_field"), ("bad_json", "header_json"), ("not_object", "header_json"),
    ("metadata", "metadata"), ("extra_key", "tensor_record"), ("neg_shape", "shape"),
    ("bool_shape", "shape"), ("float_offset", "extent"),
])
def test_malformed_inputs_rejected_with_code(tmp_path, failure, code):
    state = ReferenceCNN().state_dict()
    snapshot = save(state)
    length, = struct.unpack_from("<Q", snapshot)
    header = json.loads(snapshot[8:8 + length])
    payload = snapshot[8 + length:]
    record = header["classifier.weight"]
    encoded = None
    if failure == "keys":
        # Drop the last tensor so extents stay valid and only the schema comparison can object.
        last = max(header.items(), key=lambda kv: kv[1]["data_offsets"][0] if kv[0] != "__metadata__" else -1)
        del header[last[0]]
        payload = payload[:last[1]["data_offsets"][0]]
    elif failure == "shape":
        record["shape"].reverse()
    elif failure == "dtype":
        record["dtype"] = "I32"
    elif failure == "dtype_unknown":
        record["dtype"] = "F4"
    elif failure == "extent":
        record["data_offsets"][1] -= 4
    elif failure == "overlap":
        record["data_offsets"] = [v - 4 for v in record["data_offsets"]]
    elif failure == "gap":
        header["classifier.bias"]["data_offsets"] = [v + 4 for v in header["classifier.bias"]["data_offsets"]]
        payload += bytes(4)
    elif failure == "trailing":
        payload += bytes(4)
    elif failure == "metadata":
        header["__metadata__"] = {"a": 1}
    elif failure == "extra_key":
        record["extra"] = 1
    elif failure == "neg_shape":
        record["shape"] = [-1, -10]
    elif failure == "bool_shape":
        record["shape"] = [True, 1]
    elif failure == "float_offset":
        record["data_offsets"] = [0.0, 4.0]
    elif failure == "dup_field":
        encoded = snapshot[8:8 + length].replace(b'{"', b'{"a":1,"a":2,"', 1)
    elif failure == "bad_json":
        encoded = b"{nope"
    elif failure == "not_object":
        encoded = b"[]"
    if encoded is None:
        encoded = json.dumps(header).encode()
    blob = struct.pack("<Q", len(encoded)) + encoded + payload
    kwargs = {}
    if failure == "header_limit":
        kwargs["max_header_bytes"] = len(encoded) - 1
    elif failure == "file_limit":
        kwargs["max_file_bytes"] = len(blob) - 1
    elif failure == "short":
        blob = blob[:7]
    path = _write(tmp_path, blob)
    with pytest.raises(TensorFileError) as info:
        open_tensors(path, **kwargs).check_schema(_schema(state))
    assert info.value.code == code
    assert isinstance(info.value, ValueError)


def test_header_size_exceeding_file(tmp_path):
    with pytest.raises(TensorFileError) as info:
        open_tensors(_write(tmp_path, struct.pack("<Q", 1000) + b"{}"))
    assert info.value.code == "header_size"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_half_precision_roundtrip(tmp_path, dtype):
    values = torch.randn(3, 5).to(dtype)
    path = tmp_path / "h.safetensors"
    save_file({"w": values, "empty": torch.empty(0, 4, dtype=dtype)}, path)
    table = open_tensors(path)
    got = table.read_tensor("w")
    assert got.dtype == dtype and torch.equal(got, values)
    assert table.read_tensor("empty").shape == (0, 4)


def test_dtype_editability_and_reading(tmp_path):
    path = tmp_path / "t.safetensors"
    save_file({"f": torch.zeros(2), "i": torch.arange(4), "b": torch.tensor([True, False]),
               "d": torch.zeros(1, dtype=torch.float64)}, path)
    table = open_tensors(path)
    assert {n: t.editable for n, t in table.tensors.items()} == {"f": True, "i": False, "b": False, "d": False}
    assert torch.equal(table.read_tensor("i"), torch.arange(4))
    assert table.read_tensor("b").tolist() == [True, False]


def test_handles_closed_and_file_change_detected(tmp_path):
    path = tmp_path / "m.safetensors"
    save_file({"a": torch.ones(4)}, path)
    table = open_tensors(path)
    table.read_tensor("a")
    table.sha256("a")
    os.replace(path, tmp_path / "moved.safetensors")  # fails on Windows if a handle leaked
    save_file({"a": torch.ones(8)}, path)
    with pytest.raises(TensorFileError) as info:
        table.read_bytes("a")
    assert info.value.code == "file_changed"


def test_duplicates_report_tied_weights(tmp_path):
    shared = torch.randn(4, 4)
    path = tmp_path / "d.safetensors"
    save_file({"embed": shared, "head": shared.clone(), "other": torch.randn(4, 4),
               "e1": torch.empty(0), "e2": torch.empty(0)}, path)
    assert [sorted(g) for g in open_tensors(path).duplicates()] == [["embed", "head"]]


def _shards(tmp_path, weight_map=None, split=None):
    split = split or {"a.safetensors": {"x": torch.ones(2)}, "b.safetensors": {"y": torch.zeros(3)}}
    for name, tensors in split.items():
        save_file(tensors, tmp_path / name)
    if weight_map is None:
        weight_map = {t: shard for shard, ts in split.items() for t in ts}
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return index


def test_sharded_index(tmp_path):
    table = open_tensors(_shards(tmp_path))
    assert table.sharded and set(table.tensors) == {"x", "y"}
    assert table.read_tensor("y").tolist() == [0, 0, 0]


@pytest.mark.parametrize("case,code", [
    ("missing", "shard_missing"), ("twice", "duplicate_tensor"), ("unmapped", "index_mismatch"),
    ("absent", "index_mismatch"), ("traversal", "index_invalid"), ("empty", "index_invalid"),
])
def test_sharded_index_rejections(tmp_path, case, code):
    if case == "missing":
        index = _shards(tmp_path)
        (tmp_path / "b.safetensors").unlink()
    elif case == "twice":
        index = _shards(tmp_path, split={"a.safetensors": {"x": torch.ones(2)}, "b.safetensors": {"x": torch.ones(2)}},
                        weight_map={"x": "a.safetensors", "y": "b.safetensors"})
    elif case == "unmapped":  # y physically lives in b but the index claims it is in a
        index = _shards(tmp_path, weight_map={"x": "a.safetensors", "y": "a.safetensors"})
    elif case == "absent":  # index lists z, which no shard contains
        index = _shards(tmp_path, weight_map={"x": "a.safetensors", "y": "b.safetensors", "z": "b.safetensors"})
    elif case == "traversal":
        index = _shards(tmp_path, weight_map={"x": "../a.safetensors", "y": "b.safetensors"})
    else:
        index = _shards(tmp_path, weight_map={})
    with pytest.raises(TensorFileError) as info:
        open_tensors(index)
    assert info.value.code == code


def test_duplicates_prefilter_avoids_full_reads(tmp_path):
    n = 1 << 20  # 4 MiB per F32 tensor, well past the 128 KiB edge sample
    base = torch.randn(n)
    differ_first = base.clone()
    differ_first[0] += 1
    differ_middle = base.clone()
    differ_middle[n // 2] += 1
    path = tmp_path / "d.safetensors"
    save_file({"a": base, "b": differ_first, "c": torch.randn(n)}, path)
    table = open_tensors(path)
    assert table.duplicates() == [] and table._hashes == {}  # edges differ: nothing fully read
    path2 = tmp_path / "e.safetensors"
    save_file({"a": base, "mid": differ_middle, "copy": base.clone()}, path2)
    table = open_tensors(path2)
    assert [sorted(g) for g in table.duplicates()] == [["a", "copy"]]
    assert set(table._hashes) == {"a", "mid", "copy"}  # equal edges force the full hash, which separates them


def test_same_size_replacement_with_preserved_mtime_detected(tmp_path):
    path = tmp_path / "m.safetensors"
    save_file({"a": torch.zeros(4)}, path)
    table = open_tensors(path)
    table.sha256("a")
    stat = path.stat()
    other = tmp_path / "other.safetensors"
    save_file({"a": torch.ones(4)}, other)
    os.utime(other, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert other.stat().st_size == stat.st_size
    os.replace(other, path)
    for call in (lambda: table.read_bytes("a"), lambda: table.sha256("a")):  # sha256 is a cache hit
        with pytest.raises(TensorFileError) as info:
            call()
        assert info.value.code == "file_changed"


def test_os_level_failures_are_tensor_file_errors(tmp_path, monkeypatch):
    for target in (tmp_path / "missing.safetensors", tmp_path, tmp_path / "x.safetensors.index.json"):
        with pytest.raises(TensorFileError) as info:
            open_tensors(target)
        assert info.value.code == "unreadable"
    # A tensor file that disappears after being described.
    good = tmp_path / "g.safetensors"
    save_file({"a": torch.ones(2)}, good)
    table = open_tensors(good)
    good.unlink()
    with pytest.raises(TensorFileError) as info:
        table.read_bytes("a")
    assert info.value.code == "unreadable"
    # File shrinks after fstat but before the 8-byte length prefix is read.
    path = _write(tmp_path, b"")

    class Stat:
        st_size, st_mtime_ns, st_ino, st_dev = 100, 0, 0, 0

    monkeypatch.setattr("layerdelta.tensors.os.fstat", lambda _: Stat)
    with pytest.raises(TensorFileError) as info:
        open_tensors(path)
    assert info.value.code == "file_changed"


def test_shard_metadata_conflict_rejected(tmp_path):
    save_file({"x": torch.ones(2)}, tmp_path / "a.safetensors", metadata={"format": "pt", "k": "1"})
    save_file({"y": torch.ones(2)}, tmp_path / "b.safetensors", metadata={"format": "pt", "k": "2"})
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"x": "a.safetensors", "y": "b.safetensors"}}))
    with pytest.raises(TensorFileError) as info:
        open_tensors(index)
    assert info.value.code == "metadata"
    save_file({"y": torch.ones(2)}, tmp_path / "b.safetensors", metadata={"format": "pt", "k": "1"})
    assert open_tensors(index).metadata == {"format": "pt", "k": "1"}


@pytest.mark.parametrize("header,payload,code", [
    ({"a": {"dtype": "F32", "shape": [1], "data_offsets": [8, 4]}}, bytes(4), "extent"),
    # empty tensor whose offset lies inside a non-empty tensor's extent
    ({"a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
      "e": {"dtype": "F32", "shape": [0], "data_offsets": [8, 8]}}, bytes(16), "extent_overlap_or_gap"),
    ({"a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
      "e": {"dtype": "F32", "shape": [0], "data_offsets": [20, 20]}}, bytes(16), "extent_overlap_or_gap"),
    ({"a": {"dtype": "F32", "shape": [2 ** 40, 2 ** 40], "data_offsets": [0, 4]}}, bytes(4), "extent_mismatch"),
])
def test_more_malformed_extents(tmp_path, header, payload, code):
    encoded = json.dumps(header).encode()
    with pytest.raises(TensorFileError) as info:
        open_tensors(_write(tmp_path, struct.pack("<Q", len(encoded)) + encoded + payload))
    assert info.value.code == code


def test_empty_tensors_at_valid_offsets_accepted(tmp_path):
    header = {"a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
              "e0": {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]},
              "e1": {"dtype": "F32", "shape": [0], "data_offsets": [16, 16]}}
    encoded = json.dumps(header).encode()
    table = open_tensors(_write(tmp_path, struct.pack("<Q", len(encoded)) + encoded + bytes(16)))
    assert set(table.tensors) == {"a", "e0", "e1"}


@pytest.mark.slow
def test_two_gib_file_described_with_bounded_memory(tmp_path):
    big, small = 512 << 20, 1024  # F32 elements: 2 GiB tensor + a 4 KiB tensor at the end
    header = {
        "big": {"dtype": "F32", "shape": [big], "data_offsets": [0, big * 4]},
        "small": {"dtype": "F32", "shape": [small], "data_offsets": [big * 4, big * 4 + small * 4]},
    }
    encoded = json.dumps(header).encode()
    path = tmp_path / "huge.safetensors"
    tail = torch.arange(small, dtype=torch.float32)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)) + encoded)
        stream.truncate(8 + len(encoded) + big * 4)  # unwritten region reads as zeros
        stream.seek(0, os.SEEK_END)
        stream.write(tail.numpy().tobytes())
    assert path.stat().st_size > 2 << 30
    process = psutil.Process()
    before, peak, stop = process.memory_info().rss, [0], threading.Event()

    def sample():  # RSS snapshots miss transient peaks, so poll while the work runs
        while not stop.is_set():
            peak[0] = max(peak[0], process.memory_info().rss)
            time.sleep(.005)

    sampler = threading.Thread(target=sample)
    sampler.start()
    try:
        table = open_tensors(path)
        assert table.tensors["big"].shape == (big,)
        assert torch.equal(table.read_tensor("small"), tail)
        digest, zeros = hashlib.sha256(), bytes(1 << 20)
        for _ in range(big * 4 >> 20):
            digest.update(zeros)
        assert table.sha256("big") == digest.hexdigest()  # streams the whole 2 GiB tensor
    finally:
        stop.set()
        sampler.join()
    assert peak[0] - before < 200 << 20


@pytest.mark.parametrize("name", ["a.safetensors:b.safetensors", "NUL.safetensors", "com1.x.safetensors", "Aux.safetensors",
                                  "LPT9.safetensors", "a|b.safetensors", "a?.safetensors", "con .safetensors",
                                  "a.safetensors"])
def test_shard_names_windows_would_reinterpret_are_rejected(tmp_path, name):
    index = tmp_path / "m.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"x": name}}))
    with pytest.raises(TensorFileError) as info:
        open_tensors(index)
    assert info.value.code == "index_invalid"


def test_plain_shard_names_are_accepted(tmp_path):
    for name in ("model-00001-of-00002.safetensors", "console.safetensors", "nul2.safetensors", "my model.safetensors"):
        save_file({"x": torch.ones(2)}, tmp_path / name)
        index = tmp_path / "m.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"x": name}}))
        assert set(open_tensors(index).tensors) == {"x"}
