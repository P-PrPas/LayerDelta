"""inspect-ops rev7 at library level: describe with a graph, group selectors in read/stats/diff/query, the graph view (v0.3-G4)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from layerdelta import inspect_ops
from layerdelta.graphview import GraphView
from layerdelta.selector import InspectError, canonical_json
from layerdelta.tensors import open_tensors
from test_corpus import build, save
from test_graph_ir import graph_of, register_corpus

register_corpus()


def view_of(tmp_path, family="resnet", name=None, state=None):
    model, st = build(family, torch.float32)
    path = save(tmp_path, state or st, name or f"{family}.safetensors")
    table = open_tensors(path)
    record = {"provider": "builtin", "created_at": "2026-11-24T00:00:00.000Z", "ir_sha256": graph_of(family)["ir_sha256"]}
    return GraphView.build(graph_of(family), table, record), table, st


def group_of(view, tensor, axis, size=None, kind=None):
    found = [g for g in view.derived.groups if any(m["tensor"] == tensor and m["axis"] == axis for m in g["members"])
             and (size is None or g["size"] == size) and (kind is None or g["kind"] == kind)]
    assert len(found) == 1, [g["group_id"] for g in found]
    return found[0]


def err(call, code, field=None):
    with pytest.raises(InspectError) as info:
        call()
    assert info.value.code == code and (field is None or info.value.field == field), info.value.to_dict()


def member_indices(m, u):
    return sorted(m["offset"] + u * m["block"] + k * m["period"] + j for k in range(m["repeat"]) for j in range(m["block"]))


# ------------------------------------------------------------------ describe

def test_describe_without_a_graph_is_the_rev6_response(tmp_path):
    _, table, _ = view_of(tmp_path)
    result = inspect_ops.describe(table)["result"]
    assert "graph" not in result and all(row["roles"] == [] and "binding" not in row and "groups" not in row
                                         for row in result["tensors"])
    assert result["coverage"]["derived"] == 0 and result["coverage"]["opaque_ops"] == []


def test_describe_with_a_graph_adds_roles_binding_groups_and_coverage(tmp_path):
    view, table, _ = view_of(tmp_path, "vit")
    result = inspect_ops.describe(table, graph=view)["result"]
    rows = {r["name"]: r for r in result["tensors"]}
    assert result["graph"] == {"ir_sha256": view.ir["ir_sha256"], "provider": "builtin",
                               "table_signature": view.meta()["table_signature"], "attached_at": "2026-11-24T00:00:00.000Z"}
    qkv = rows["blocks.0.attn.qkv.weight"]
    assert qkv["binding"] == "bound" and [r["role"] for r in qkv["roles"]] == ["packed", "in_unit"]
    assert qkv["roles"][0]["k"] == 3 and qkv["roles"][0]["provenance"] == "derived" and qkv["roles"][0]["basis"].startswith("linear.kernel@")
    assert {g["axis"] for g in qkv["groups"]} == {0, 1} and len(qkv["groups"]) == 4  # heads, q/k dims, v dims on axis 0; stream on 1
    assert rows["logit_scale"]["binding"] == "unbound" and rows["tok_embed.weight"]["binding"] == "unbound"
    cov = result["coverage"]
    assert cov["derived"] + cov["unknown"] == sum(len(r["roles"]) for r in rows.values()) and cov["tensors"]["unbound"] == 2
    assert cov["groups"]["closed"] + cov["groups"]["open"] == len(view.derived.groups)
    assert inspect_ops.to_json(inspect_ops.describe(table, graph=view)) == inspect_ops.to_json(inspect_ops.describe(table, graph=view))


# ------------------------------------------------------------------ group selectors

def test_read_group_expands_to_member_slices(tmp_path):
    view, table, state = view_of(tmp_path)
    g = group_of(view, "layer3.0.bn2.weight", 0)
    out = inspect_ops.read_tensor(table, {"group": g["group_id"], "units": [3, 5]}, max_elems=10_000, graph=view)
    result = out["result"]
    assert result["group"]["unit_count"] == 2 and result["total_elems"] == sum(m["result"]["total_elems"] for m in result["members"])
    by = {(m["tensor"], m["axis"]): m for m in result["members"]}
    for m in g["members"]:
        want = state[m["tensor"]].index_select(m["axis"], torch.tensor(sorted({i for u in (3, 5) for i in member_indices(m, u)})))
        got = by[(m["tensor"], m["axis"])]["result"]["values"]
        assert np.allclose(np.array(got, dtype=np.float64), want.numpy().astype(np.float64).ravel())
    assert by[("layer3.0.conv2.weight", 0)]["role"] == "out_unit" and by[("fc.weight", 1)]["role"] == "in_unit"


def test_resolved_selector_is_resubmittable_and_expansion_replays_without_a_graph(tmp_path):
    view, table, _ = view_of(tmp_path)
    g = group_of(view, "layer2.0.conv1.weight", 0)
    first = inspect_ops.read_tensor(table, {"group": g["group_id"], "units": {"start": 2, "stop": 5}}, max_elems=10_000, graph=view)
    resolved = first["resolved"]["selector"]
    assert resolved["units"] == [{"start": 2, "stop": 5}] and resolved["group"] == g["group_id"]
    again = inspect_ops.read_tensor(table, {k: resolved[k] for k in ("group", "units")}, max_elems=10_000, graph=view)
    assert again["fingerprints"]["selection"] == first["fingerprints"]["selection"] and inspect_ops.to_json(again) == inspect_ops.to_json(first)
    for member, expanded in zip(first["result"]["members"], resolved["expanded"]):  # no graph needed to replay it
        plain = inspect_ops.read_tensor(table, expanded, max_elems=10_000)
        assert plain["result"]["values"] == member["result"]["values"]
    other = inspect_ops.read_tensor(table, {"group": g["group_id"], "units": [2]}, graph=view)
    assert other["fingerprints"]["selection"] != first["fingerprints"]["selection"]  # a pure value of {group, graph, units}


def test_group_read_pages_across_members(tmp_path):
    view, table, _ = view_of(tmp_path)
    g = group_of(view, "layer1.0.conv1.weight", 0)
    whole = inspect_ops.read_tensor(table, {"group": g["group_id"], "units": [0, 1]}, max_elems=10_000, graph=view)
    flat = {(m["tensor"], m["axis"]): [] for m in whole["result"]["members"]}
    cursor, pages = None, 0
    while True:
        page = inspect_ops.read_tensor(table, {"group": g["group_id"], "units": [0, 1]}, max_elems=37, cursor=cursor, graph=view)
        for m in page["result"]["members"]:
            flat[(m["tensor"], m["axis"])] += m["result"]["values"]
        pages += 1
        cursor = page["cursor"]
        if not cursor:
            break
    assert pages > 3
    for m in whole["result"]["members"]:
        assert flat[(m["tensor"], m["axis"])] == m["result"]["values"]
    err(lambda: inspect_ops.read_tensor(table, {"group": g["group_id"], "units": [0]}, cursor=cursor or "c1.bad", graph=view),
        "invalid_cursor")


def test_members_filter(tmp_path):
    view, table, _ = view_of(tmp_path)
    g = group_of(view, "layer1.0.conv1.weight", 0)
    names = lambda sel: {(m["tensor"], m["axis"]) for m in inspect_ops.read_tensor(table, sel, graph=view)["result"]["members"]}
    everything = names({"group": g["group_id"]})
    outs = names({"group": g["group_id"], "members": {"roles": ["out_unit"]}})
    assert outs == {("layer1.0.conv1.weight", 0)} < everything
    no_stats = names({"group": g["group_id"], "members": {"exclude": ["norm_stats"]}})
    assert ("layer1.0.bn1.weight", 0) in no_stats and ("layer1.0.bn1.running_mean", 0) not in no_stats
    assert ("layer1.0.bn1.running_mean", 0) in everything
    err(lambda: inspect_ops.read_tensor(table, {"group": g["group_id"], "members": {"roles": ["nope"]}}, graph=view), "no_members",
        "selector.members")
    err(lambda: inspect_ops.read_tensor(table, {"group": g["group_id"], "members": {"exclude": ["x"]}}, graph=view), "invalid_argument")
    err(lambda: inspect_ops.read_tensor(table, {"group": g["group_id"], "members": {"what": 1}}, graph=view), "unknown_field")


def test_group_errors(tmp_path):
    view, table, _ = view_of(tmp_path)
    g = group_of(view, "layer1.0.conv1.weight", 0)
    err(lambda: inspect_ops.read_tensor(table, {"group": g["group_id"]}), "no_graph", "selector.group")
    err(lambda: inspect_ops.stats(table, {"group": g["group_id"]}), "no_graph", "selector.group")
    err(lambda: inspect_ops.read_tensor(table, {"group": "g_" + "0" * 24}, graph=view), "unknown_group", "selector.group")
    err(lambda: inspect_ops.read_tensor(table, {"group": g["group_id"], "units": [g["size"]]}, graph=view), "index_out_of_range")
    err(lambda: inspect_ops.read_tensor(table, {"group": g["group_id"], "tensor": "x"}, graph=view), "unknown_field")
    err(lambda: inspect_ops.read_tensor(table, {"group": 5}, graph=view), "invalid_argument", "selector.group")
    err(lambda: inspect_ops.stats(table, {"group": g["group_id"]}, reduce_over_axes=[0], graph=view), "invalid_argument")
    err(lambda: inspect_ops.stats(table, {"group": g["group_id"]}, bins=4, graph=view), "invalid_argument", "bins")


def test_group_stats_are_per_unit_per_member(tmp_path):
    view, table, state = view_of(tmp_path, "vit")
    g = group_of(view, "blocks.0.attn.qkv.weight", 0, size=2, kind="head")  # heads: each unit is 3 blocks of 8 rows
    out = inspect_ops.stats(table, {"group": g["group_id"]}, graph=view)["result"]
    assert out["unit_count"] == 2 and out["accumulate_dtype"] == "float64"
    qkv = next(m for m in out["members"] if m["tensor"] == "blocks.0.attn.qkv.weight")
    spec = next(m for m in g["members"] if m["tensor"] == "blocks.0.attn.qkv.weight")
    for row in qkv["units"]:
        rows = state["blocks.0.attn.qkv.weight"].index_select(0, torch.tensor(member_indices(spec, row["unit"]))).double()
        assert row["l1"] == pytest.approx(rows.abs().sum().item(), rel=1e-12) and row["max"] == rows.max().item()
    paged = inspect_ops.stats(table, {"group": g["group_id"]}, limit=1, graph=view)
    assert paged["truncated"] and len(paged["result"]["members"]) == 1


def test_group_diff_reports_the_changed_unit(tmp_path):
    view, table, state = view_of(tmp_path)
    g = group_of(view, "layer1.0.conv1.weight", 0)
    child = {k: v.clone() for k, v in state.items()}
    for m in g["members"]:
        if m["tensor"] == "layer1.0.conv1.weight" and m["axis"] == 0:
            child[m["tensor"]][4] *= 2
    other = open_tensors(save(tmp_path, child, "child.safetensors"))
    out = inspect_ops.diff(table, other, selector={"group": g["group_id"]}, graph=view, checkpoint_a="a" * 64, checkpoint_b="b" * 64)
    conv = next(m for m in out["result"]["members"] if m["tensor"] == "layer1.0.conv1.weight")
    changed = {u["unit"] for u in conv["units"] if u["changed_elems"]}
    assert changed == {4} and out["fingerprints"]["checkpoint_b"] == "b" * 64
    err(lambda: inspect_ops.diff(table, other, selector={"group": g["group_id"]}, values=True, graph=view), "invalid_argument", "values")


def test_query_where_group_ranks_the_units_by_the_first_out_unit_member(tmp_path):
    view, table, state = view_of(tmp_path)
    g = group_of(view, "layer1.0.conv1.weight", 0)
    where = {"group": g["group_id"], "stat": "l2", "op": "ge", "value": 0.0}
    out = inspect_ops.query(table, where=where, graph=view)["result"]
    match = out["matches"][0]
    assert match["member_used"]["role"] == "out_unit" and match["member_used"]["reason"] == "out_unit member with the most elements per unit"
    weight = state[match["member_used"]["tensor"]].double()
    expected = sorted(range(g["size"]), key=lambda u: (-weight[u].pow(2).sum().sqrt().item(), u))
    assert [u["unit"] for u in match["units"]] == expected
    assert inspect_ops.read_tensor(table, match["selector"], graph=view)["result"]["group"]["unit_count"] == g["size"]
    none = inspect_ops.query(table, where={**where, "op": "lt", "value": -1.0}, graph=view)["result"]
    assert none["matched"] == 0
    fallback = inspect_ops.query(table, where={"group": g["group_id"], "stat": "l2", "op": "ge", "value": 0.0,
                                               "members": {"roles": ["in_unit"]}}, graph=view)["result"]["matches"][0]
    assert fallback["member_used"]["reason"].startswith("no out_unit member")
    err(lambda: inspect_ops.query(table, where={"group": g["group_id"], "stat": "nope", "op": "ge", "value": 0}, graph=view),
        "invalid_argument", "where.stat")
    err(lambda: inspect_ops.query(table, where={"group": g["group_id"], "stat": "l2", "op": "ge", "value": 0}), "no_graph")


# ------------------------------------------------------------------ the graph view

def test_group_records_carry_boundary_reasons_labels_kinds_notes_and_spans(tmp_path):
    view, table, _ = view_of(tmp_path, "vit")
    records = {g["group_id"]: view.group_record(g) for g in view.derived.groups}
    head = records[group_of(view, "head.weight", 0)["group_id"]]
    assert not head["closed"] and head["boundary"][0]["reason"] == "graph_output" and head["boundary"][0]["output"] == 0
    assert any("model output" in n for n in head["notes"]) and head["label"] == "head channels"
    qkv_groups = {r["kind"]: r for r in records.values() if any(m["tensor"] == "blocks.0.attn.qkv.weight" and m["axis"] == 0
                                                                 for m in r["members"])}
    assert set(qkv_groups) == {"head", "qk_dim", "v_dim"} and qkv_groups["head"]["label"] == "blocks.0.attn.qkv heads"
    some = records[group_of(view, "blocks.0.mlp.0.weight", 0)["group_id"]]
    assert some["spans"] and {"value", "dim", "offset", "block", "period", "repeat"} == set(some["spans"][0])
    assert some["spans_total"] >= len(some["spans"]) and some["closed"] and some["notes"] == []
    pos = next(r for r in records.values() if any(m["tensor"] == "pos_embed" for m in r["members"]) and r["size"] == 4)
    assert pos["kind"] == "position" and not pos["closed"]


def test_span_formula_matches_the_activation_shape(tmp_path):
    view, table, _ = view_of(tmp_path, "vit")
    shapes = {o["value"]: o["shape"] for n in view.ir["nodes"] for o in n["outputs"] if o["kind"] == "tensor"}
    for g in view.derived.groups:
        for s in g["spans"]:
            last = s["offset"] + (g["size"] - 1) * s["block"] + (s["repeat"] - 1) * s["period"] + s["block"] - 1
            assert last < shapes.get(s["value"], [10 ** 9] * 9)[s["dim"]] or s["value"] not in shapes


def test_graph_listing_filters(tmp_path):
    view, _, _ = view_of(tmp_path, "unet")
    assert len(view.list_groups(closed=False)) + len(view.list_groups(closed=True)) == len(view.derived.groups)
    touching = view.list_groups(tensor="dec2.0.weight")
    assert touching and all(any(m["tensor"] == "dec2.0.weight" for m in g["members"]) for g in touching)
    assert {g["group_id"] for g in view.list_groups(module_prefix="dec2")} >= {g["group_id"] for g in touching}
    node = next(n for n in view.ir["nodes"] if n["op"].endswith("conv2d.default") and n["module_path"] == "dec2.0")
    assert {g["group_id"] for g in view.list_groups(node=node["id"])} >= {g["group_id"] for g in touching}
    err(lambda: view.list_groups(node="n99999"), "unknown_node", "node")
    detail = view.node_detail(node["id"])
    assert detail["rule"] == node["op"] and detail["opaque_reason"] is None and "dec2.0.weight" in detail["parameters"]
    modules = {m["path"]: m for m in view.modules()}
    assert modules["dec2"]["node_count"] >= modules["dec2.0"]["node_count"] > 0 and modules[""]["node_count"] == len(view.ir["nodes"])
    assert modules["dec2.0"]["type"] == "Conv2d" and modules["dec2.0"]["axes"]["unknown"] == 0
    assert [m["path"] for m in view.modules("dec2")][0] == "dec2"
    assert all(m["path"] == "dec2" or m["path"].startswith("dec2.") for m in view.modules("dec2"))


def test_opaque_nodes_are_reported_with_their_reason(tmp_path):
    from test_coupling import Cumsum
    from layerdelta.providers import build_graph
    ir = build_graph({"provider": "builtin", "model": "test.cumsum"})
    state = {k: v.detach().clone() for k, v in Cumsum().state_dict().items()}
    table = open_tensors(save(tmp_path, state, "c.safetensors"))
    view = GraphView.build(ir, table, {"provider": "builtin", "created_at": "t"})
    cum = next(n for n in ir["nodes"] if n["op"] == "aten.cumsum.default")
    assert view.node_detail(cum["id"])["opaque_reason"] == "no_rule" and view.node_detail(cum["id"])["rule"] is None
    assert view.modules()[0]["opaque_count"] == 1
    assert any(not g["closed"] and any(b["op"] == "aten.cumsum.default" for b in g["boundary"]) for g in view.derived.groups)


def test_a_group_with_a_non_affine_member_cannot_be_expanded(tmp_path):
    view, table, _ = view_of(tmp_path)
    g = group_of(view, "layer1.0.conv1.weight", 0)
    original = g["members"][0]["offset"]
    g["members"][0]["offset"] = None  # what a group with a non_affine_member looks like
    try:
        for call in (lambda: inspect_ops.read_tensor(table, {"group": g["group_id"]}, graph=view),
                     lambda: inspect_ops.stats(table, {"group": g["group_id"]}, graph=view),
                     lambda: inspect_ops.query(table, where={"group": g["group_id"], "stat": "l2", "op": "ge", "value": 0}, graph=view)):
            err(call, "non_affine_group")
    finally:
        g["members"][0]["offset"] = original


def test_group_query_is_capped_and_refuses_tensor_filters(tmp_path):
    view, table, _ = view_of(tmp_path)
    g = group_of(view, "layer1.0.conv1.weight", 0)
    where = {"group": g["group_id"], "stat": "l2", "op": "ge", "value": 0}
    err(lambda: inspect_ops.query(table, where=where, max_scan_elems=3, graph=view), "cap_exceeded", "where")
    assert inspect_ops.query(table, where=where, max_scan_elems=10 ** 6, graph=view)["resolved"]["max_scan_elems"] == 10 ** 6
    for field, value in (("name_glob", "*"), ("dtype", ["F32"]), ("min_ndim", 1), ("max_ndim", 4), ("editable", True),
                         ("min_bytes", 1), ("max_bytes", 10)):
        err(lambda: inspect_ops.query(table, where=where, graph=view, **{field: value}), "invalid_argument", field)



def test_a_group_query_ranks_by_the_out_unit_member_with_the_most_elements_per_unit():
    from types import SimpleNamespace
    from layerdelta import group_ops

    def member(name, role, per_unit):
        return SimpleNamespace(tensor=name, role=role, unit_selection=lambda unit, n=per_unit: SimpleNamespace(total=n))

    bias, weight, tie = member("rnn.bias_ih", "out_unit", 4), member("rnn.weight_ih", "out_unit", 64), member("rnn.weight_hh", "out_unit", 64)
    reader = member("fc.weight", "in_unit", 100)
    assert group_ops.ranking_member([bias, weight, tie, reader], [0, 1]).tensor == "rnn.weight_ih"  # the bias comes first but is smaller; ties keep order
    assert group_ops.ranking_member([reader, member("norm.bias", "channel", 1)], [0]).tensor == "fc.weight"  # no out_unit member of its own: the first member
    assert group_ops.ranking_member([bias, weight], []).tensor == "rnn.bias_ih"  # no units to size: the first out_unit member
