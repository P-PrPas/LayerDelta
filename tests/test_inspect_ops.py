import json
import math

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from layerdelta.inspect_ops import (RANGE_SCAN_MAX, READ_MAX, describe, diff, query, read_tensor, shortest, stats,
                                    to_json)
from layerdelta.model import ReferenceCNN
from layerdelta.selector import InspectError
from layerdelta.tensors import open_tensors

SHAPES = [(7,), (5, 4), (3, 4, 5), (2, 3, 4, 5), (6, 1, 3), (1, 1), ()]


def table_of(tmp_path, tensors, name="m.safetensors"):
    path = tmp_path / name
    save_file({k: v.contiguous() for k, v in tensors.items()}, path)
    return open_tensors(path)


def fmt(table, name, **kw):
    return read_tensor(table, {"tensor": name, **kw})


def as_tensor(values, dtype):
    return torch.tensor(values, dtype=dtype)


def err(call, **fields):
    with pytest.raises(InspectError) as info:
        call()
    for key, value in fields.items():
        assert getattr(info.value, key) == value, info.value.to_dict()
    return info.value


# ------------------------------------------------------------------ read_tensor vs torch

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
def test_read_equals_tensor_indexing_first_and_last_per_axis(tmp_path, shape, dtype):
    torch.manual_seed(1)
    tensor = (torch.randn(shape) * 3).to(dtype)
    table = table_of(tmp_path, {"t": tensor})
    whole = read_tensor(table, {"tensor": "t"})["result"]
    assert torch.equal(as_tensor(whole["values"], dtype), tensor.flatten())
    for axis, size in enumerate(shape):
        for index in (0, size - 1):
            got = read_tensor(table, {"tensor": "t", "index": {str(axis): [index]}})["result"]
            expected = tensor.select(axis, index)
            assert got["selection_shape"] == [1 if a == axis else n for a, n in enumerate(shape)]
            assert torch.equal(as_tensor(got["values"], dtype), expected.flatten())
    if shape:
        ranged = read_tensor(table, {"tensor": "t", "index": {"0": {"start": 0, "stop": max(1, shape[0] - 1)}}})
        assert torch.equal(as_tensor(ranged["result"]["values"], dtype), tensor[:max(1, shape[0] - 1)].flatten())


def test_multi_axis_selection_and_int_bool_dtypes(tmp_path):
    base = torch.arange(3 * 4 * 5)
    table = table_of(tmp_path, {"i": base.reshape(3, 4, 5), "b": (base % 2 == 0).reshape(3, 4, 5),
                                "d": base.double().reshape(3, 4, 5)})
    index = {"0": [0, 2], "2": {"start": 1, "stop": 4}}
    for name in ("i", "b", "d"):
        got = read_tensor(table, {"tensor": name, "index": index})["result"]
        expected = {"i": base, "b": base % 2 == 0, "d": base.double()}[name].reshape(3, 4, 5)[[0, 2]][:, :, 1:4]
        assert as_tensor(got["values"], expected.dtype).equal(expected.flatten())


# ------------------------------------------------------------------ selector validation

@pytest.mark.parametrize("selector,code,field", [
    ({"tensor": "nope"}, "unknown_tensor", "selector.tensor"),
    ({"tensor": "t", "index": {"0": [4]}}, "index_out_of_range", "selector.index.0[0]"),
    ({"tensor": "t", "index": {"0": [0, -1]}}, "invalid_index", "selector.index.0[1]"),
    ({"tensor": "t", "index": {"1": {"start": 0, "stop": 6}}}, "index_out_of_range", "selector.index.1.stop"),
    ({"tensor": "t", "index": {"1": {"start": -1, "stop": 2}}}, "invalid_index", "selector.index.1.start"),
    ({"tensor": "t", "index": {"1": {"start": 3, "stop": 3}}}, "invalid_range", "selector.index.1.start"),
    ({"tensor": "t", "index": {"1": {"start": 0}}}, "invalid_range", "selector.index.1.stop"),
    ({"tensor": "t", "index": {"1": {"start": 0, "stop": 2, "step": 2}}}, "unknown_field", "selector.index.1.step"),
    ({"tensor": "t", "index": {"2": [0]}}, "invalid_axis", "selector.index.2"),
    ({"tensor": "t", "index": {"-1": [0]}}, "invalid_axis", "selector.index.-1"),
    ({"tensor": "t", "index": {"01": [0]}}, "invalid_axis", "selector.index.01"),
    ({"tensor": "t", "index": {"0": []}}, "invalid_index", "selector.index.0"),
    ({"tensor": "t", "index": {"0": [True]}}, "invalid_index", "selector.index.0[0]"),
    ({"tensor": "t", "index": {"0": [1.0]}}, "invalid_index", "selector.index.0[0]"),
    ({"tensor": "t", "index": {"0": "all"}}, "invalid_index", "selector.index.0"),
    ({"tensor": "t", "extra": 1}, "unknown_field", "selector.extra"),
    ({"tensor": "t", "group": "cg:x", "units": [1]}, "unsupported_selector", "selector.group"),
    ({"tensor": 5}, "invalid_argument", "selector.tensor"),
    ({}, "invalid_argument", "selector.tensor"),
    ({"tensor": "t", "checkpoint": "sha256:other"}, "invalid_argument", "selector.checkpoint"),
    ("t", "invalid_argument", "selector"),
])
def test_selector_errors_have_code_and_field(tmp_path, selector, code, field):
    table = table_of(tmp_path, {"t": torch.zeros(4, 5)})
    for op in (lambda: read_tensor(table, selector, checkpoint_id="sha256:me"),
               lambda: stats(table, selector, checkpoint_id="sha256:me")):
        err(op, code=code, field=field)


def test_index_list_cap(tmp_path):
    table = table_of(tmp_path, {"t": torch.zeros(100_002)})
    err(lambda: read_tensor(table, {"tensor": "t", "index": {"0": list(range(100_001))}}), code="cap_exceeded")


def test_selection_id_is_canonical(tmp_path):
    table = table_of(tmp_path, {"t": torch.zeros(6, 4), "u": torch.zeros(6, 4)})
    ids = {read_tensor(table, {"tensor": "t", "index": {"0": spec}})["fingerprints"]["selection"]
           for spec in ([3, 1, 2, 2], {"start": 1, "stop": 4}, [1, 2, 3])}
    assert len(ids) == 1 and ids.pop().startswith("sel_")
    whole = read_tensor(table, {"tensor": "t"})["fingerprints"]["selection"]
    assert whole == read_tensor(table, {"tensor": "t", "index": {"0": {"start": 0, "stop": 6}}})["fingerprints"]["selection"]
    assert whole != read_tensor(table, {"tensor": "u"})["fingerprints"]["selection"]
    env = read_tensor(table, {"tensor": "t", "index": {"1": [0, 1]}}, checkpoint_id="sha256:abc")
    assert env["resolved"]["selector"]["index"]["1"] == [[0, 2]] and env["fingerprints"]["checkpoint"] == "sha256:abc"
    assert env["side_effects"] == [] and env["schema_version"] == 1


# ------------------------------------------------------------------ pagination and caps

@pytest.mark.parametrize("shape,index", [((5, 4), {}), ((3, 4, 5), {"1": [0, 2, 3]}), ((2, 3, 4, 5), {"0": [1], "3": {"start": 1, "stop": 5}}),
                                         ((7,), {"0": [0, 3, 4, 6]})])
@pytest.mark.parametrize("page", [1, 7, 33, 1000])
def test_read_pagination_concatenates_to_full_read(tmp_path, shape, index, page):
    tensor = torch.randn(shape)
    table = table_of(tmp_path, {"t": tensor})
    selector = {"tensor": "t", "index": index}
    full = read_tensor(table, selector)["result"]["values"]
    values, cursor, pages = [], None, 0
    while True:
        env = read_tensor(table, selector, max_elems=page, cursor=cursor)
        values += env["result"]["values"]
        pages += 1
        assert env["truncated"] == (env["cursor"] is not None)
        assert env["result"]["offset"] == len(values) - env["result"]["returned_elems"]
        cursor = env["cursor"]
        if cursor is None:
            break
    assert values == full and pages == math.ceil(len(full) / page)


def test_caps_enforced(tmp_path):
    table = table_of(tmp_path, {"t": torch.zeros(20_000)})
    err(lambda: read_tensor(table, {"tensor": "t"}, max_elems=READ_MAX + 1), code="cap_exceeded", field="max_elems")
    err(lambda: read_tensor(table, {"tensor": "t"}, max_elems=0), code="invalid_argument")
    err(lambda: read_tensor(table, {"tensor": "t"}, max_elems=True), code="invalid_argument")
    assert read_tensor(table, {"tensor": "t"}, max_elems=READ_MAX)["result"]["returned_elems"] == READ_MAX
    err(lambda: stats(table, {"tensor": "t"}, limit=2049), code="cap_exceeded")
    err(lambda: stats(table, {"tensor": "t"}, bins=257), code="cap_exceeded")
    err(lambda: describe(table, limit=20_001), code="cap_exceeded")
    err(lambda: diff(table, table, limit=1001), code="cap_exceeded")
    err(lambda: query(table, limit=501), code="cap_exceeded")
    err(lambda: query(table, where={"stat": "l2", "axis": 0, "op": "lt", "value": 1}, max_scan_elems=10),
        code="cap_exceeded", field="max_scan_elems")


def test_range_skipped_above_scan_cap(tmp_path, monkeypatch):
    table = table_of(tmp_path, {"t": torch.zeros(64)})
    monkeypatch.setattr("layerdelta.inspect_ops.RANGE_SCAN_MAX", 10)
    result = read_tensor(table, {"tensor": "t"})["result"]
    assert result["range"] is None and result["range_skipped"] == "selection_too_large"
    assert RANGE_SCAN_MAX == 1 << 24


def test_cursor_bound_to_request(tmp_path):
    table = table_of(tmp_path, {"t": torch.randn(50), "u": torch.randn(50)})
    cursor = read_tensor(table, {"tensor": "t"}, max_elems=10)["cursor"]
    err(lambda: read_tensor(table, {"tensor": "u"}, max_elems=10, cursor=cursor), code="invalid_cursor", field="cursor")
    err(lambda: stats(table, {"tensor": "t"}, cursor=cursor), code="invalid_cursor")
    for bad in ("junk", "c1.@@", 5):
        err(lambda: read_tensor(table, {"tensor": "t"}, cursor=bad), code="invalid_cursor")


# ------------------------------------------------------------------ determinism and formatting

def test_repeated_calls_are_byte_identical(tmp_path):
    torch.manual_seed(3)
    a = {"w": torch.randn(6, 5, 3), "b": torch.randn(6), "i": torch.arange(5)}
    b = {k: v.clone() for k, v in a.items()}
    b["w"][2] += 0.5
    ta, tb = table_of(tmp_path, a, "a.safetensors"), table_of(tmp_path, b, "b.safetensors")
    sel = {"tensor": "w", "index": {"0": [1, 2, 4]}}
    calls = [lambda: describe(ta), lambda: read_tensor(ta, sel, max_elems=20), lambda: stats(ta, sel, reduce_over_axes=[1, 2], bins=8),
             lambda: diff(ta, tb), lambda: diff(ta, tb, selector=sel, values=True, unit_axis=None),
             lambda: diff(ta, tb, selector=sel), lambda: query(ta, where={"stat": "l2", "axis": 0, "op": "gt", "value": 0})]
    for call in calls:
        assert to_json(call()) == to_json(call())
        json.loads(to_json(call()))  # allow_nan=False would have raised


@pytest.mark.parametrize("dtype,value,expected", [
    (torch.float16, 0.1, 0.1), (torch.float32, 0.1, 0.1), (torch.bfloat16, 0.1, 0.1),
    (torch.float32, 1 / 3, 0.33333334), (torch.float16, 1 / 3, 0.3333), (torch.bfloat16, 1 / 3, 0.334),
    (torch.float32, 3.0, 3.0), (torch.float32, -0.0, -0.0)])
def test_shortest_repr_in_stored_dtype(tmp_path, dtype, value, expected):
    table = table_of(tmp_path, {"t": torch.tensor([value], dtype=dtype)})
    got = read_tensor(table, {"tensor": "t"})["result"]["values"][0]
    assert got == expected and math.copysign(1, got) == math.copysign(1, expected)
    assert json.dumps(got) == json.dumps(expected)


def test_shortest_roundtrips_for_random_values():
    rng = np.random.default_rng(0)
    for dtype, np_dtype in (("F32", np.float32), ("F16", np.float16)):
        for value in rng.standard_normal(500).astype(np_dtype).tolist():
            assert np_dtype(shortest(value, dtype)) == value
    for value in torch.randn(300).to(torch.bfloat16).float().tolist():
        assert torch.tensor(shortest(value, "BF16"), dtype=torch.bfloat16).float().item() == value


def test_nonfinite_encoding_and_stats_nulls(tmp_path):
    t = torch.tensor([1.0, float("nan"), float("inf"), -float("inf"), 2.0])
    table = table_of(tmp_path, {"t": t, "n": torch.full((3,), float("nan"))})
    result = read_tensor(table, {"tensor": "t"})["result"]
    assert result["values"] == [1.0, "NaN", "Inf", "-Inf", 2.0] and result["nonfinite_count"] == 3
    assert result["range"] == {"min": 1.0, "max": 2.0, "abs_max": 2.0, "abs_p99": 2.0}
    unit = stats(table, {"tensor": "t"})["result"]["units"][0]
    assert (unit["count"], unit["nonfinite"], unit["min"], unit["max"], unit["mean"]) == (5, 3, 1.0, 2.0, 1.5)
    nan_only = stats(table, {"tensor": "n"})["result"]["units"][0]
    assert nan_only["mean"] is None and nan_only["min"] is None and nan_only["constant"] is False
    assert read_tensor(table, {"tensor": "n"})["result"]["range"] is None


def test_range_over_whole_selection_not_page(tmp_path):
    t = torch.randn(1000)
    table = table_of(tmp_path, {"t": t})
    first = read_tensor(table, {"tensor": "t"}, max_elems=10)["result"]
    last = read_tensor(table, {"tensor": "t"}, max_elems=10, cursor=read_tensor(table, {"tensor": "t"}, max_elems=990)["cursor"])["result"]
    assert first["range"] == last["range"]
    a = t.abs().numpy()
    assert first["range"]["min"] == t.min().item() and first["range"]["abs_max"] == a.max()
    assert first["range"]["abs_p99"] == np.sort(a)[math.ceil(0.99 * 1000) - 1]


# ------------------------------------------------------------------ describe

def test_describe_table_roles_and_no_data_read(tmp_path, monkeypatch):
    state = ReferenceCNN().state_dict()
    table = table_of(tmp_path, {**state, "step": torch.tensor([3])})
    monkeypatch.setattr(type(table), "read_ranges", lambda *a, **k: pytest.fail("describe read tensor data"))
    monkeypatch.setattr(type(table), "sha256", lambda *a, **k: pytest.fail("describe hashed"))
    env = describe(table, checkpoint_id="sha256:x")
    r = env["result"]
    assert [t["name"] for t in r["tensors"]] == sorted(list(state) + ["step"])
    by = {t["name"]: t for t in r["tensors"]}
    for name, tensor in state.items():
        assert by[name]["shape"] == list(tensor.shape) and by[name]["dtype"] == "F32" and by[name]["editable"]
        assert by[name]["roles"] == [] and by[name]["read_only_reason"] is None and by[name]["shard"] is None
        assert by[name]["tensor_sha256"] is None and by[name]["identical_group"] is None
    assert by["step"]["editable"] is False and by["step"]["read_only_reason"] == "dtype_not_editable"
    assert r["coverage"] == {"unit": "axes", "derived": 0, "declared": 0, "inferred": 0,
                             "unknown": sum(len(t["shape"]) for t in r["tensors"]), "opaque_ops": []}
    assert r["checkpoint"]["element_count"] == sum(t["numel"] for t in r["tensors"])
    assert r["identical_bytes"] is None and r["matched"] == len(by)
    assert env["truncated"] is False and env["fingerprints"]["checkpoint"] == "sha256:x"


def test_describe_glob_pagination_and_shards(tmp_path):
    save_file({"a.x": torch.zeros(2), "a.y": torch.zeros(2)}, tmp_path / "s1.safetensors")
    save_file({"b.x": torch.zeros(2), "a.z": torch.zeros(3)}, tmp_path / "s2.safetensors")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a.x": "s1.safetensors", "a.y": "s1.safetensors",
                                                "b.x": "s2.safetensors", "a.z": "s2.safetensors"}}))
    table = open_tensors(index)
    names, cursor = [], None
    while True:
        env = describe(table, name_glob="a.*", limit=2, cursor=cursor)
        names += [t["name"] for t in env["result"]["tensors"]]
        cursor = env["cursor"]
        if not cursor:
            break
    assert names == ["a.x", "a.y", "a.z"]
    shards = {t["name"]: t["shard"] for t in describe(table)["result"]["tensors"]}
    assert shards == {"a.x": "s1.safetensors", "a.y": "s1.safetensors", "a.z": "s2.safetensors", "b.x": "s2.safetensors"}
    assert describe(table)["result"]["checkpoint"]["shard_count"] == 2


# ------------------------------------------------------------------ stats

def test_stats_match_numpy_per_unit_and_shared_histogram(tmp_path):
    torch.manual_seed(5)
    t = torch.randn(6, 4, 5)
    t[2] = 0
    t[3] = 7
    t[4, :, 0] = 0
    table = table_of(tmp_path, {"t": t})
    env = stats(table, {"tensor": "t"}, reduce_over_axes=[1, 2], bins=16)
    result = env["result"]
    assert result["unit_axes"] == [0] and result["unit_count"] == 6 and result["accumulate_dtype"] == "float64"
    x = t.double().numpy()
    for u, row in enumerate(result["units"]):
        v = x[u].ravel()
        assert row["unit"] == u and row["index"] == {"0": u} and row["count"] == 20
        assert row["min"] == v.min() and row["max"] == v.max()
        np.testing.assert_allclose([row["mean"], row["std"], row["l1"], row["l2"]],
                                   [v.mean(), v.std(), np.abs(v).sum(), np.sqrt((v * v).sum())], rtol=1e-12, atol=1e-12)
        assert row["zero_frac"] == (v == 0).mean()
        assert row["constant"] == (u in (2, 3))
    hist = result["histogram"]
    assert len(hist["edges"]) == 17 and hist["edges"][0] == x.min() and hist["edges"][-1] == x.max()
    assert [sum(c) for c in hist["counts"]] == [20] * 6
    expected = np.histogram(x[0].ravel(), bins=16, range=(x.min(), x.max()))[0]
    assert abs(np.array(hist["counts"][0]) - expected).sum() <= 2  # boundary rounding only
    whole = stats(table, {"tensor": "t"})["result"]
    assert whole["unit_axes"] == [] and whole["unit_count"] == 1 and whole["units"][0]["unit"] is None
    assert whole["units"][0]["count"] == 120 and whole["units"][0]["min"] == x.min()


@pytest.mark.parametrize("reduce,kept", [([0], [1, 2]), ([1], [0, 2]), ([0, 2], [1]), ([0, 1, 2], [])])
def test_stats_unit_axes_and_pagination(tmp_path, reduce, kept):
    t = torch.randn(3, 4, 5)
    table = table_of(tmp_path, {"t": t})
    sel = {"tensor": "t", "index": {"1": [0, 2, 3]}}
    full = stats(table, sel, reduce_over_axes=reduce)["result"]["units"]
    units, cursor = [], None
    while True:
        env = stats(table, sel, reduce_over_axes=reduce, limit=2, cursor=cursor)
        units += env["result"]["units"]
        cursor = env["cursor"]
        if not cursor:
            break
    assert units == full and env["result"]["unit_axes"] == kept
    sub = t[:, [0, 2, 3]].double()
    for row in full:
        index = {int(k): v for k, v in row["index"].items()}
        picked = sub
        for axis in sorted(index, reverse=True):
            real = index[axis]
            picked = picked.select(axis, [0, 2, 3].index(real) if axis == 1 else real)
        assert row["mean"] == pytest.approx(picked.mean().item(), rel=1e-12, abs=1e-12)


def test_stats_argument_errors(tmp_path):
    table = table_of(tmp_path, {"t": torch.randn(3, 4)})
    err(lambda: stats(table, {"tensor": "t"}, reduce_over_axes=[2]), code="invalid_axis", field="reduce_over_axes[0]")
    err(lambda: stats(table, {"tensor": "t"}, reduce_over_axes=[0, 0]), code="invalid_argument")
    err(lambda: stats(table, {"tensor": "t"}, reduce_over_axes="0"), code="invalid_argument")


# ------------------------------------------------------------------ diff

def test_diff_identical_is_all_zero(tmp_path):
    state = ReferenceCNN().state_dict()
    table = table_of(tmp_path, state)
    env = diff(table, table, include_same=True)
    s = env["result"]["summary"]
    assert (s["tensors_total"], s["changed"], s["same"], s["changed_elems_total"]) == (6, 0, 6, 0)
    assert s["max_abs"] == 0.0 and env["result"]["accumulate_dtype"] == "float64"
    for row in env["result"]["tensors"]:
        assert row["status"] == "same" and row["changed_elems"] == 0 and row["changed_units"]["total"] == 0
        assert row["rel_l2"] == 0.0 and row["cosine"] == 1.0
    assert diff(table, table)["result"]["tensors"] == []
    sel = {"tensor": "classifier.weight"}
    one = diff(table, table, selector=sel)["result"]
    assert one["tensor"]["changed_elems"] == 0 and all(u["changed_elems"] == 0 for u in one["units"])


def test_diff_parent_vs_child_reports_exactly_one_changed_out_unit(tmp_path):
    torch.manual_seed(0)
    parent = ReferenceCNN().state_dict()
    child = {k: v.clone() for k, v in parent.items()}
    child["features.block2.conv.weight"][17] = 0
    ta, tb = table_of(tmp_path, parent, "p.safetensors"), table_of(tmp_path, child, "c.safetensors")
    env = diff(ta, tb, checkpoint_a="sha256:p", checkpoint_b="sha256:c")
    r = env["result"]
    assert r["summary"]["changed"] == 1 and r["summary"]["same"] == 5
    assert len(r["tensors"]) == 1
    row = r["tensors"][0]
    assert row["name"] == "features.block2.conv.weight" and row["status"] == "changed"
    assert row["changed_units"] == {"total": 1, "listed": [17], "truncated": False}
    assert row["changed_elems"] == 8 * 3 * 3 and r["summary"]["changed_elems_total"] == 72
    assert row["cosine"] is not None and 0 < row["rel_l2"] < 1
    assert env["fingerprints"]["checkpoint"] == "sha256:p" and env["fingerprints"]["checkpoint_b"] == "sha256:c"
    units = diff(ta, tb, selector={"tensor": "features.block2.conv.weight"})["result"]["units"]
    assert [u["unit"] for u in units if u["changed_elems"]] == [17]
    zeroed = units[17]
    assert zeroed["rel_l2"] == pytest.approx(1.0) and zeroed["cosine"] is None  # child unit is all zero
    # per-axis changed units along another axis
    by_in = diff(ta, tb, unit_axis=1)["result"]["tensors"][0]["changed_units"]
    assert by_in["total"] == 8


def test_diff_statuses_and_undefined_metrics(tmp_path):
    a = {"same": torch.ones(3), "chg": torch.zeros(2, 3), "only_a": torch.ones(1), "shp": torch.ones(2), "dt": torch.ones(2)}
    b = {"same": torch.ones(3), "chg": torch.tensor([[0.0, 0, 0], [1, 2, 3]]), "only_b": torch.ones(1),
         "shp": torch.ones(3), "dt": torch.ones(2).half()}
    ta, tb = table_of(tmp_path, a, "a.safetensors"), table_of(tmp_path, b, "b.safetensors")
    env = diff(ta, tb, include_same=True)
    s = env["result"]["summary"]
    assert (s["only_a"], s["only_b"], s["shape_mismatch"], s["dtype_mismatch"]) == (["only_a"], ["only_b"], ["shp"], ["dt"])
    assert s["tensors_total"] == 6 and s["changed"] == 1 and s["same"] == 1
    rows = {r["name"]: r for r in env["result"]["tensors"]}
    assert {n: r["status"] for n, r in rows.items()} == {"chg": "changed", "dt": "dtype_mismatch", "only_a": "only_a",
                                                        "only_b": "only_b", "same": "same", "shp": "shape_mismatch"}
    chg = rows["chg"]
    assert chg["rel_l2"] is None and chg["cosine"] is None  # zero parent: undefined, explicit null
    assert chg["changed_units"]["listed"] == [1] and chg["max_abs"] == 3.0
    err(lambda: diff(ta, tb, selector={"tensor": "shp"}), code="incompatible_tensors")
    err(lambda: diff(ta, tb, selector={"tensor": "only_a"}), code="incompatible_tensors")
    err(lambda: diff(ta, tb, values=True), code="invalid_argument")
    err(lambda: diff(ta, tb, unit_axis=-1), code="invalid_axis")
    assert diff(ta, tb, unit_axis=5)["result"]["summary"]["changed"] == 1  # no such axis: no per-unit breakdown
    assert "units" not in diff(ta, tb, selector={"tensor": "chg"}, values=True)["result"]  # values mode: no unit rows


def test_diff_bitwise_changed_mask_and_values_window(tmp_path):
    a = torch.tensor([0.0, 1.0, float("nan"), 2.0, 5.0, 1.0])
    b = torch.tensor([-0.0, 1.0, float("nan"), 2.5, 5.0, float("inf")])
    ta, tb = table_of(tmp_path, {"t": a}, "a.safetensors"), table_of(tmp_path, {"t": b}, "b.safetensors")
    env = diff(ta, tb, selector={"tensor": "t"}, unit_axis=None, values=True)
    r = env["result"]
    assert r["values"] == [0.0, 0.0, "NaN", 0.5, 0.0, "Inf"] and r["delta_dtype"] == "float64"
    assert r["changed"] == [1, 0, 0, 1, 0, 1]  # -0.0 vs 0.0 changed with delta 0; NaN==NaN (same bits)
    assert r["tensor"]["changed_elems"] == 3 and r["nonfinite_count"] == 2
    assert r["range"] == {"min": 0.0, "max": 0.5, "abs_max": 0.5, "abs_p99": 0.5}
    window = diff(ta, tb, selector={"tensor": "t", "index": {"0": {"start": 3, "stop": 5}}}, unit_axis=None, values=True)
    assert window["result"]["values"] == [0.5, 0.0] and window["result"]["changed"] == [1, 0]
    # bf16 rounding-only change is visible through the mask
    x = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    y = torch.tensor([1.0, 2.015625], dtype=torch.bfloat16)
    tx, ty = table_of(tmp_path, {"t": x}, "x.safetensors"), table_of(tmp_path, {"t": y}, "y.safetensors")
    r = diff(tx, ty, selector={"tensor": "t"}, unit_axis=None, values=True)["result"]
    assert r["changed"] == [0, 1] and r["values"] == [0.0, 0.015625]


def test_diff_values_pagination_and_unit_pagination(tmp_path):
    a, b = torch.randn(5, 6), torch.randn(5, 6)
    ta, tb = table_of(tmp_path, {"t": a}, "a.safetensors"), table_of(tmp_path, {"t": b}, "b.safetensors")
    sel = {"tensor": "t", "index": {"1": [0, 2, 5]}}
    full = diff(ta, tb, selector=sel, unit_axis=None, values=True)["result"]
    values, changed, cursor = [], [], None
    while True:
        env = diff(ta, tb, selector=sel, unit_axis=None, values=True, max_elems=4, cursor=cursor)
        values += env["result"]["values"]
        changed += env["result"]["changed"]
        cursor = env["cursor"]
        if not cursor:
            break
    assert values == full["values"] and changed == full["changed"]
    assert full["values"] == [float(v) for v in (b.double() - a.double())[:, [0, 2, 5]].flatten()]
    rows, cursor = [], None
    while True:
        env = diff(ta, tb, selector=sel, unit_axis=0, limit=2, cursor=cursor)
        rows += env["result"]["units"]
        cursor = env["cursor"]
        if not cursor:
            break
    assert [u["unit"] for u in rows] == [0, 1, 2, 3, 4]
    listing = diff(ta, tb, include_same=True, limit=1)
    assert listing["truncated"] is False  # one tensor, one row
    many_a = {f"t{i}": torch.randn(2) for i in range(5)}
    many_b = {k: v + 1 for k, v in many_a.items()}
    xa, xb = table_of(tmp_path, many_a, "ma.safetensors"), table_of(tmp_path, many_b, "mb.safetensors")
    names, cursor = [], None
    while True:
        env = diff(xa, xb, limit=2, cursor=cursor)
        names += [r["name"] for r in env["result"]["tensors"]]
        assert env["result"]["summary"]["changed"] == 5
        cursor = env["cursor"]
        if not cursor:
            break
    assert names == [f"t{i}" for i in range(5)]


# ------------------------------------------------------------------ query

def test_query_filters_ranking_and_selection_ids(tmp_path):
    t = torch.ones(5, 4)
    t[1] = 0.001
    t[3] = 0.002
    table = table_of(tmp_path, {"conv.weight": t, "conv.bias": torch.ones(5), "fc.weight": torch.ones(2, 2),
                                "idx": torch.arange(3)})
    env = query(table, name_glob="*.weight")
    assert [m["name"] for m in env["result"]["matches"]] == ["conv.weight", "fc.weight"]
    assert env["fingerprints"]["selection"] is None
    assert [m["name"] for m in query(table, dtype=["I64"])["result"]["matches"]] == ["idx"]
    assert [m["name"] for m in query(table, min_ndim=2, editable=True)["result"]["matches"]] == ["conv.weight", "fc.weight"]
    assert [m["name"] for m in query(table, max_ndim=1, max_bytes=16)["result"]["matches"]] == []
    assert [m["name"] for m in query(table, min_bytes=20, max_ndim=1)["result"]["matches"]] == ["conv.bias", "idx"]
    where = {"stat": "l2", "axis": 0, "op": "lt", "value": 0.01}
    r = query(table, where=where, dtype=["F32"])["result"]
    assert [m["name"] for m in r["matches"]] == ["conv.weight"]
    match = r["matches"][0]
    assert [u["unit"] for u in match["units"]] == [1, 3]  # ranked by value ascending
    assert match["units"][0]["value"] == pytest.approx(0.002) and match["selector"] == {"tensor": "conv.weight", "index": {"0": [1, 3]}}
    chosen = read_tensor(table, match["selector"])
    assert chosen["fingerprints"]["selection"] == match["selection_id"] and chosen["result"]["total_elems"] == match["total_elems"] == 8
    gt = query(table, where={**where, "op": "gt", "value": 1.3}, dtype=["F32"])["result"]["matches"]
    assert [m["name"] for m in gt] == ["conv.weight", "fc.weight"] and [u["unit"] for u in gt[0]["units"]] == [0, 2, 4]  # descending, ties by unit
    err(lambda: query(table, where={"stat": "nope", "axis": 0, "op": "lt", "value": 1}), code="invalid_argument", field="where.stat")
    err(lambda: query(table, where={"stat": "l2", "axis": 0, "op": "lt"}), code="invalid_argument", field="where.value")
    err(lambda: query(table, where={"stat": "l2", "axis": 0, "op": "lt", "value": 1, "x": 1}), code="unknown_field")
    err(lambda: query(table, dtype=["F99"]), code="invalid_argument", field="dtype")
    names, cursor = [], None
    while True:
        env = query(table, limit=1, cursor=cursor)
        names += [m["name"] for m in env["result"]["matches"]]
        cursor = env["cursor"]
        if not cursor:
            break
    assert names == sorted(["conv.weight", "conv.bias", "fc.weight", "idx"])



@pytest.mark.parametrize("seed", range(8))
def test_random_multi_axis_selections_match_torch(tmp_path, seed):
    rng = np.random.default_rng(seed)
    shape = (4, 5, 6)
    tensor = torch.randn(shape)
    table = table_of(tmp_path, {"t": tensor})
    index, expected = {}, tensor
    for axis, size in enumerate(shape):
        if rng.random() < .3:
            continue
        picks = sorted(set(rng.integers(0, size, size=rng.integers(1, size + 1)).tolist()))
        index[str(axis)] = picks if rng.random() < .5 else {"start": picks[0], "stop": picks[-1] + 1}
        keep = picks if isinstance(index[str(axis)], list) else list(range(picks[0], picks[-1] + 1))
        expected = expected.index_select(axis, torch.tensor(keep))
    got = read_tensor(table, {"tensor": "t", "index": index})["result"]
    assert got["selection_shape"] == list(expected.shape)
    assert torch.equal(as_tensor(got["values"], torch.float32), expected.flatten())
    unit = stats(table, {"tensor": "t", "index": index}, reduce_over_axes=[0, 1, 2])["result"]["units"][0]
    assert unit["count"] == expected.numel() and unit["min"] == expected.min().item()


def test_no_file_handle_left_open_after_ops(tmp_path):
    ta = table_of(tmp_path, {"t": torch.randn(50, 40)}, "a.safetensors")
    tb = table_of(tmp_path, {"t": torch.randn(50, 40)}, "b.safetensors")
    sel = {"tensor": "t", "index": {"0": [1, 5, 9]}}
    read_tensor(ta, sel)
    stats(ta, sel, reduce_over_axes=[1], bins=4)
    diff(ta, tb)
    diff(ta, tb, selector=sel)
    diff(ta, tb, selector=sel, values=True)
    query(ta, where={"stat": "l1", "axis": 0, "op": "gt", "value": 0})
    for name in ("a", "b"):  # replacing or deleting fails on Windows if any handle leaked
        (tmp_path / f"{name}.safetensors").replace(tmp_path / f"{name}2.safetensors")
