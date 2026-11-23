"""Coupled-group edits (v0.3-G5): group selectors in programs, permute_units, engine revisions, recipe provenance."""

import copy
import hashlib

import numpy as np
import pytest
import torch
from safetensors.torch import load_file
from torch import nn

from layerdelta import edit
from layerdelta.edit import apply, freeze_recipe, plan, verify_recipe
from layerdelta.graphview import GraphView
from layerdelta.providers import build_graph, register_builtin
from layerdelta.selector import InspectError
from layerdelta.tensors import open_tensors
from test_corpus import FAMILIES, build, save
from test_coupling import Cumsum
from test_graph_ops import group_of, view_of
from test_graph_ir import graph_of, register_corpus

register_corpus()
BASE = "b" * 64
FAKE_VIA = {"group": "g_" + "1" * 24, "ir_sha256": "2" * 64, "units": [{"start": 0, "stop": 3}], "members": None,
            "member": {"tensor": "fc.bias", "axis": 0}}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def err(call, code, field=None):
    with pytest.raises(InspectError) as info:
        call()
    assert info.value.code == code and (field is None or info.value.field == field), info.value.to_dict()


def freeze(table, view, program, hypothesis="h"):
    return freeze_recipe(table, BASE, program, hypothesis, graph=view)


def child_of(table, recipe, tmp_path, name="child.safetensors"):
    out = tmp_path / name
    apply(table, recipe, out, base_id=BASE)  # no graph: a recipe replays without one
    return out


def outputs(model, state, x):
    model.load_state_dict(state)
    with torch.no_grad():
        y = model(x)
    return torch.cat([t.flatten() for t in (y if isinstance(y, (list, tuple)) else [y])])


# ------------------------------------------------------------------ A3: permuting a closed group keeps the function

@pytest.mark.parametrize("family", list(FAMILIES))
def test_permute_units_on_every_closed_group_keeps_the_forward_output(tmp_path, family):
    view, table, state = view_of(tmp_path, family)
    model, _ = build(family, torch.float32)
    x = torch.randn(*FAMILIES[family][1])
    base = outputs(model, state, x)
    gen = torch.Generator().manual_seed(11)
    done = 0
    for g in view.derived.groups:
        if not g["closed"] or g["size"] < 2:
            continue
        perm = torch.arange(g["size"])
        while perm.tolist() == list(range(g["size"])):
            perm = torch.randperm(g["size"], generator=gen)
        recipe = freeze(table, view, [{"op": "permute_units", "args": {"perm": perm.tolist()}, "select": {"group": g["group_id"]}}])
        assert recipe["engine"]["layerdelta_ops_revision"] == 2
        assert all(e["via"]["group"] == g["group_id"] and e["via"]["perm"] == perm.tolist() for e in recipe["program"])
        child = child_of(table, recipe, tmp_path, f"c{done}.safetensors")
        got = outputs(model, load_file(child), x)
        assert (got - base).abs().max() <= 1e-5 * max(1.0, base.abs().max().item()), (family, g["group_id"])
        assert any(not torch.equal(state[k], v) for k, v in load_file(child).items())  # it really moved something
        done += 1
    assert done >= 6


def test_a_permutation_is_a_bit_exact_gather(tmp_path):
    view, table, state = view_of(tmp_path)
    g = group_of(view, "layer1.0.conv1.weight", 0)
    perm = list(reversed(range(g["size"])))
    recipe = freeze(table, view, [{"op": "permute_units", "args": {"perm": perm}, "select": {"group": g["group_id"]}}])
    child = load_file(child_of(table, recipe, tmp_path))
    assert torch.equal(child["layer1.0.conv1.weight"], state["layer1.0.conv1.weight"][perm])  # no rounding, no arithmetic
    assert torch.equal(child["layer1.0.bn1.running_var"], state["layer1.0.bn1.running_var"][perm])  # statistics move with it
    assert torch.equal(child["layer1.0.conv2.weight"], state["layer1.0.conv2.weight"][:, perm])
    assert torch.equal(child["layer2.0.conv1.weight"], state["layer2.0.conv1.weight"])  # an unrelated tensor is untouched


def test_the_inverse_permutation_restores_the_checkpoint_bytes(tmp_path):
    view, table, state = view_of(tmp_path, "vit")
    g = group_of(view, "blocks.0.mlp.0.weight", 0)
    perm = torch.randperm(g["size"], generator=torch.Generator().manual_seed(2)).tolist()
    inverse = [perm.index(i) for i in range(len(perm))]
    first = child_of(table, freeze(table, view, [{"op": "permute_units", "args": {"perm": perm}, "select": {"group": g["group_id"]}}]),
                     tmp_path, "one.safetensors")
    middle = open_tensors(first)
    view2 = GraphView.build(view.ir, middle, view.record)
    second = child_of(middle, freeze(middle, view2, [{"op": "permute_units", "args": {"perm": inverse}, "select": {"group": g["group_id"]}}]),
                      tmp_path, "two.safetensors")
    assert sha(second) == sha(tmp_path / "vit.safetensors")


def test_packed_attention_heads_permute_with_their_q_k_and_v_rows(tmp_path):
    view, table, state = view_of(tmp_path, "vit")
    heads = group_of(view, "blocks.0.attn.qkv.weight", 0, size=2, kind="head")
    recipe = freeze(table, view, [{"op": "permute_units", "args": {"perm": [1, 0]}, "select": {"group": heads["group_id"]}}])
    qkv = next(e for e in recipe["program"] if e["via"]["member"] == {"tensor": "blocks.0.attn.qkv.weight", "axis": 0})
    assert qkv["args"]["axis"] == 0 and qkv["args"]["perm"][:8] == list(range(8, 16)) and len(qkv["args"]["perm"]) == 48
    child = load_file(child_of(table, recipe, tmp_path))
    w, c = state["blocks.0.attn.qkv.weight"], child["blocks.0.attn.qkv.weight"]
    assert torch.equal(c[0:8], w[8:16]) and torch.equal(c[8:16], w[0:8]) and torch.equal(c[16:24], w[24:32])  # q, k and v rows


# ------------------------------------------------------------------ what may and may not be permuted

def test_open_groups_refuse_permute_units_but_allow_other_edits(tmp_path):
    view, table, state = view_of(tmp_path)
    head = group_of(view, "fc.weight", 0)
    assert not head["closed"]
    err(lambda: freeze(table, view, [{"op": "permute_units", "args": {"perm": list(range(10))[::-1]}, "select": {"group": head["group_id"]}}]),
        "group_not_closed", "program[0].select.group")
    zero = freeze(table, view, [{"op": "zero", "args": {}, "select": {"group": head["group_id"], "units": [3]}}])
    assert zero["engine"]["layerdelta_ops_revision"] == 2 and {e["via"]["group"] for e in zero["program"]} == {head["group_id"]}
    planned = plan(table, zero, base_id=BASE, graph=view)["result"]
    assert planned["warnings"][0]["code"] == "group_open" and planned["warnings"][0]["boundary"][0]["reason"] == "graph_output"
    assert planned["groups"][0]["closed"] is False and planned["groups"][0]["notes"]
    child = load_file(child_of(table, zero, tmp_path))
    assert child["fc.weight"][3].abs().sum() == 0 and child["fc.bias"][3] == 0 and torch.equal(child["fc.weight"][4], state["fc.weight"][4])


def test_an_opaque_boundary_refuses_permute_units_and_tensor_edits_still_work(tmp_path):
    ir = build_graph({"provider": "builtin", "model": "test.cumsum"})
    st = {k: v.detach().clone() for k, v in Cumsum().state_dict().items()}
    table = open_tensors(save(tmp_path, st, "c.safetensors"))
    view = GraphView.build(ir, table, {"provider": "builtin", "created_at": "t"})
    g = group_of(view, "a.weight", 0)
    err(lambda: freeze(table, view, [{"op": "permute_units", "args": {"perm": list(range(6))[::-1]}, "select": {"group": g["group_id"]}}]),
        "group_not_closed")
    scaled = freeze(table, view, [{"op": "scale", "args": {"k": 2.0}, "select": {"tensor": "a.weight"}}])
    assert scaled["engine"]["layerdelta_ops_revision"] == 1 and "via" not in scaled["program"][0]
    assert torch.equal(load_file(child_of(table, scaled, tmp_path))["a.weight"], st["a.weight"] * 2)


def test_permutation_argument_errors(tmp_path):
    view, table, _ = view_of(tmp_path)
    g = group_of(view, "layer1.0.conv1.weight", 0)
    sel = {"group": g["group_id"]}
    n = g["size"]
    for perm in (list(range(n - 1)), [0] * n, list(range(1, n + 1)), "x", [], [0.5] * n):
        err(lambda: freeze(table, view, [{"op": "permute_units", "args": {"perm": perm}, "select": sel}]), "invalid_permutation")
    ok = list(range(n))[::-1]
    err(lambda: freeze(table, view, [{"op": "permute_units", "args": {"perm": ok}, "select": {**sel, "units": [0, 1]}}]), "invalid_argument")
    err(lambda: freeze(table, view, [{"op": "permute_units", "args": {"perm": ok}, "select": {**sel, "members": {"roles": ["out_unit"]}}}]),
        "invalid_argument")
    err(lambda: freeze(table, view, [{"op": "permute_units", "args": {"perm": ok, "x": 1}, "select": sel}]), "invalid_argument")
    err(lambda: freeze(table, view, [{"op": "permute_units", "args": {"perm": ok}, "select": {"tensor": "fc.bias"}}]), "invalid_argument")
    err(lambda: freeze(table, None, [{"op": "zero", "args": {}, "select": sel}]), "no_graph", "program[0].select.group")
    err(lambda: freeze(table, view, [{"op": "zero", "args": {}, "select": {"group": "g_" + "0" * 24}}]), "unknown_group")
    err(lambda: freeze(table, view, [{"op": "set", "args": {"blob": "sha256:" + "a" * 64}, "select": sel}]), "invalid_argument",
        "program[0].args.blob")


class TwoNorms(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.g, self.b = nn.Conv1d(3, 8, 1), nn.GroupNorm(2, 8), nn.Conv1d(8, 3, 1)

    def forward(self, x):
        return self.b(self.g(self.a(x)))


register_builtin("test.edit_gn", TwoNorms, [((2, 3, 4), "float32")], replace=True)


def test_group_normalisation_blocks_constrain_the_permutation(tmp_path):
    ir = build_graph({"provider": "builtin", "model": "test.edit_gn"})
    torch.manual_seed(5)
    model = TwoNorms()
    model.g.weight.data.normal_(1, .3)
    st = {k: v.detach().clone() for k, v in model.state_dict().items()}
    table = open_tensors(save(tmp_path, st, "g.safetensors"))
    view = GraphView.build(ir, table, {"provider": "builtin", "created_at": "t"})
    g = group_of(view, "g.weight", 0)
    assert g["closed"] and g["constraints"] == {"kind": "within_blocks", "n": 4, "of": [4]}
    cross = [4, 5, 6, 7, 0, 1, 2, 3]
    err(lambda: freeze(table, view, [{"op": "permute_units", "args": {"perm": cross}, "select": {"group": g["group_id"]}}]),
        "constraint_violation", "program[0].args.perm")
    inside = [3, 2, 1, 0, 4, 5, 6, 7]
    recipe = freeze(table, view, [{"op": "permute_units", "args": {"perm": inside}, "select": {"group": g["group_id"]}}])
    x = torch.randn(2, 3, 4)
    child = load_file(child_of(table, recipe, tmp_path))
    assert torch.allclose(outputs(model, st, x), outputs(model, child, x), atol=1e-5)


# ------------------------------------------------------------------ A4: a group edit is the manual edit

def test_zero_through_a_group_equals_zeroing_the_slices_by_hand(tmp_path):
    view, table, state = view_of(tmp_path)
    g = group_of(view, "layer1.0.conv1.weight", 0)
    recipe = freeze(table, view, [{"op": "zero", "args": {}, "select": {"group": g["group_id"], "units": [2, 5]}}])
    child = child_of(table, recipe, tmp_path)
    manual = {k: v.clone() for k, v in state.items()}
    for m in g["members"]:
        for u in (2, 5):
            idx = [m["offset"] + u * m["block"] + k * m["period"] + j for k in range(m["repeat"]) for j in range(m["block"])]
            manual[m["tensor"]].index_fill_(m["axis"], torch.tensor(idx), 0)
    by_hand = save(tmp_path, manual, "manual.safetensors")
    assert sha(child) == sha(by_hand)
    assert {e["via"]["member"]["tensor"] for e in recipe["program"]} == {m["tensor"] for m in g["members"]}


def test_the_members_filter_limits_a_group_edit(tmp_path):
    view, table, state = view_of(tmp_path)
    g = group_of(view, "layer1.0.conv1.weight", 0)
    recipe = freeze(table, view, [{"op": "scale", "args": {"k": 3.0},
                                   "select": {"group": g["group_id"], "units": [1], "members": {"roles": ["out_unit"]}}}])
    assert [e["via"]["member"] for e in recipe["program"]] == [{"tensor": "layer1.0.conv1.weight", "axis": 0}]
    child = load_file(child_of(table, recipe, tmp_path))
    assert torch.equal(child["layer1.0.conv1.weight"][1], state["layer1.0.conv1.weight"][1] * 3)
    assert torch.equal(child["layer1.0.bn1.weight"], state["layer1.0.bn1.weight"])  # no norm statistic was touched
    both = freeze(table, view, [{"op": "scale", "args": {"k": 3.0}, "select": {"group": g["group_id"], "units": [1]}}])
    assert len(both["program"]) > 1


# ------------------------------------------------------------------ engines, provenance, replay

def test_revision_1_recipes_keep_their_engine_and_hash(tmp_path):
    view, table, _ = view_of(tmp_path)
    plain = freeze(table, None, [{"op": "scale", "args": {"k": 2.0}, "select": {"tensor": "fc.bias"}}])
    assert plain["engine"] == edit.ENGINE and plain["engine"]["layerdelta_ops_revision"] == 1
    assert verify_recipe(plain) is plain
    same = freeze(table, view, [{"op": "scale", "args": {"k": 2.0}, "select": {"tensor": "fc.bias"}}])  # a graph changes nothing for it
    assert same["recipe_hash"] == plain["recipe_hash"]
    pretend = copy.deepcopy(plain)
    pretend["program"][0]["op"] = "permute_units"
    pretend["program"][0]["args"] = {"axis": 0, "perm": list(range(10))}
    err(lambda: verify_recipe(pretend), "unsupported_engine")
    with_via = copy.deepcopy(plain)
    with_via["program"][0]["via"] = {}
    err(lambda: verify_recipe(with_via), "unsupported_engine", "engine")


def test_a_group_recipe_verifies_and_replays_without_the_graph(tmp_path):
    view, table, state = view_of(tmp_path)
    g = group_of(view, "layer1.0.conv1.weight", 0)
    recipe = freeze(table, view, [{"op": "permute_units", "args": {"perm": list(range(g["size"]))[::-1]}, "select": {"group": g["group_id"]}}])
    assert verify_recipe(recipe) is recipe and recipe["engine"] == edit.ENGINE_V2
    again = copy.deepcopy(recipe)
    assert plan(table, again, base_id=BASE)["result"]["changed_elems_total"] > 0  # no graph passed
    first = child_of(table, recipe, tmp_path, "a.safetensors")
    second = child_of(table, again, tmp_path, "b.safetensors")
    assert sha(first) == sha(second)
    report = plan(table, recipe, base_id=BASE, graph=view)["result"]
    assert report["groups"][0]["group"] == g["group_id"] and report["groups"][0]["closed"] is True
    assert report["moved_units"][0]["count"] == g["size"] - (g["size"] % 2) and report["moved_units"][0]["total"] == g["size"]
    assert {"from", "to"} == set(report["moved_units"][0]["units"][0]) and report["warnings"] == []
    assert plan(table, recipe, base_id=BASE)["result"]["groups"][0]["closed"] is None  # without a graph only the provenance
    assert plan(table, recipe, base_id=BASE)["fingerprints"]["op_revision"] == 2
    tampered = copy.deepcopy(recipe)
    tampered["program"][0]["via"]["group"] = "nope"
    err(lambda: verify_recipe(tampered), "noncanonical_recipe")
    tampered = copy.deepcopy(recipe)
    tampered["program"][0]["args"]["perm"][0] = tampered["program"][0]["args"]["perm"][1]
    err(lambda: verify_recipe(tampered), "invalid_permutation")
    tampered = copy.deepcopy(recipe)
    tampered["program"][0]["via"]["perm"] = [0]
    tampered["program"][0]["via"]["units"] = [{"start": 3, "stop": 3}]
    err(lambda: verify_recipe(tampered), "noncanonical_recipe")
    wrong_axis = copy.deepcopy(recipe)
    wrong_axis["program"][0]["args"]["axis"] = 9
    wrong_axis["recipe_hash"] = edit._finish(BASE, wrong_axis["program"], "h", None, edit.ENGINE_V2)["recipe_hash"]
    err(lambda: plan(table, wrong_axis, base_id=BASE), "invalid_axis")


def test_a_permutation_of_the_wrong_length_for_the_selection_is_refused_at_plan(tmp_path):
    view, table, _ = view_of(tmp_path)
    recipe = freeze(table, None, [{"op": "scale", "args": {"k": 2.0}, "select": {"tensor": "fc.bias"}}])
    bad = copy.deepcopy(recipe)
    bad["engine"] = dict(edit.ENGINE_V2)
    bad["program"][0] = {"op": "permute_units", "args": {"axis": 0, "perm": [0, 1, 2]}, "select": bad["program"][0]["select"],
                         "via": dict(FAKE_VIA, perm=[0, 1, 2])}
    bad["recipe_hash"] = edit._finish(BASE, bad["program"], "h", None, edit.ENGINE_V2)["recipe_hash"]
    err(lambda: plan(table, bad, base_id=BASE), "invalid_permutation", "program[0].args.perm")


def test_a_noop_permutation_is_refused_like_any_noop(tmp_path):
    view, table, _ = view_of(tmp_path)
    g = group_of(view, "layer1.0.conv1.weight", 0)
    recipe = freeze(table, view, [{"op": "permute_units", "args": {"perm": list(range(g["size"]))}, "select": {"group": g["group_id"]}}])
    err(lambda: apply(table, recipe, tmp_path / "n.safetensors", base_id=BASE), "no_op")


# ------------------------------------------------------------------ review round 1

class BlockReshape(nn.Module):
    """GroupNorm(2, 8) (blocks of 4 channels), then the channels cut into 4 x 2 and joined again."""

    def __init__(self):
        super().__init__()
        self.a, self.g, self.b = nn.Conv1d(3, 8, 1), nn.GroupNorm(2, 8), nn.Conv1d(8, 3, 1)

    def forward(self, x):
        h = self.g(self.a(x))
        h = torch.relu(h.reshape(h.shape[0], 4, 2, h.shape[2]))
        return self.b(h.reshape(h.shape[0], 8, h.shape[3]))


register_builtin("test.edit_block_reshape", BlockReshape, [((2, 3, 4), "float32")], replace=True)


def test_a_block_constraint_is_translated_to_the_factors_of_a_reshape(tmp_path):
    ir = build_graph({"provider": "builtin", "model": "test.edit_block_reshape"})
    torch.manual_seed(6)
    model = BlockReshape()
    model.g.weight.data.normal_(1, .3)
    model.g.bias.data.normal_(0, .3)
    st = {k: v.detach().clone() for k, v in model.state_dict().items()}
    table = open_tensors(save(tmp_path, st, "r.safetensors"))
    view = GraphView.build(ir, table, {"provider": "builtin", "created_at": "t"})
    x = torch.randn(2, 3, 4)
    base = outputs(model, st, x)
    major = next(g for g in view.derived.groups if g["size"] == 4 and any(m["tensor"] == "g.weight" for m in g["members"]))
    minor = next(g for g in view.derived.groups if g["size"] == 2 and any(m["tensor"] == "g.weight" for m in g["members"]))
    assert major["constraints"]["n"] == 2 and minor["constraints"]["kind"] == "none"  # 4-channel blocks = 2 units of the 4-factor
    err(lambda: freeze(table, view, [{"op": "permute_units", "args": {"perm": [2, 1, 0, 3]}, "select": {"group": major["group_id"]}}]),
        "constraint_violation")  # it would take channels {0,1} and {4,5} into each other's GroupNorm group
    for group, perm in ((major, [1, 0, 3, 2]), (minor, [1, 0])):  # inside a block: function-preserving
        recipe = freeze(table, view, [{"op": "permute_units", "args": {"perm": perm}, "select": {"group": group["group_id"]}}])
        got = outputs(model, load_file(child_of(table, recipe, tmp_path, f"r{group['size']}.safetensors")), x)
        assert torch.allclose(got, base, atol=1e-5)
    err(lambda: freeze(table, view, [{"op": "permute_units", "args": {"perm": [2, 3, 0, 1]}, "select": {"group": major["group_id"]}}]),
        "constraint_violation")  # swapping whole blocks would be fine too; the constraint is the conservative "inside a block"


def test_the_block_translation_rules():
    from layerdelta.coupling import _factor_blocks, _part_blocks
    keep = lambda n: {("within_blocks", n): {"kind": "within_blocks", "n": n}}
    assert _factor_blocks(keep(4), 1, 2) == ({}, False)            # 2 units of stride 1 fit inside a block of 4
    assert _factor_blocks(keep(4), 2, 4) == (keep(2), False)       # the major factor of 4 x 2: blocks of 2 units
    assert _factor_blocks(keep(4), 1, 8) == (keep(4), False)       # the whole space cut as 8 x 1
    assert _factor_blocks(keep(6), 2, 4) == ({}, True)             # 6 does not divide into a 2-strided factor of 4
    assert _factor_blocks(keep(2), 4, 2) == ({}, True)             # a block of 2 is smaller than the factor stride
    assert _part_blocks(keep(4), 0, 8) == (keep(4), False) and _part_blocks(keep(4), 4, 4) == (keep(4), False)
    assert _part_blocks(keep(4), 1, 2) == ({}, False)              # inside one block
    assert _part_blocks(keep(4), 2, 4) == ({}, True)               # straddles two blocks


def test_a_permute_units_op_must_carry_its_via(tmp_path):
    view, table, _ = view_of(tmp_path)
    recipe = freeze(table, None, [{"op": "scale", "args": {"k": 2.0}, "select": {"tensor": "fc.bias"}}])
    bare = copy.deepcopy(recipe)
    bare["engine"] = dict(edit.ENGINE_V2)
    bare["program"][0] = {"op": "permute_units", "args": {"axis": 0, "perm": list(range(10))[::-1]}, "select": bare["program"][0]["select"]}
    bare["recipe_hash"] = edit._finish(BASE, bare["program"], "h", None, edit.ENGINE_V2)["recipe_hash"]
    err(lambda: verify_recipe(bare), "noncanonical_recipe", "program[0].via")
    no_perm = copy.deepcopy(bare)
    no_perm["program"][0]["via"] = dict(FAKE_VIA)
    no_perm["recipe_hash"] = edit._finish(BASE, no_perm["program"], "h", None, edit.ENGINE_V2)["recipe_hash"]
    err(lambda: verify_recipe(no_perm), "noncanonical_recipe", "program[0].via")


def test_a_submitted_via_is_checked_against_the_graph(tmp_path):
    view, table, _ = view_of(tmp_path)
    g = group_of(view, "layer1.0.conv1.weight", 0)
    real = freeze(table, view, [{"op": "permute_units", "args": {"perm": list(range(g["size"]))[::-1]}, "select": {"group": g["group_id"]}}])
    verified = plan(table, real, base_id=BASE, graph=view)["result"]
    assert [x["via_verified"] for x in verified["groups"]] == [True] and verified["moved_units"][0]["count"] > 0
    assert [x["via_verified"] for x in plan(table, real, base_id=BASE)["result"]["groups"]] == [False]  # no graph: provenance only
    other = GraphView.build(graph_of("unet"), open_tensors(save(tmp_path, build("unet", torch.float32)[1], "u.safetensors")),
                            {"provider": "builtin", "created_at": "t"})
    assert plan(table, real, base_id=BASE, graph=other)["result"]["groups"][0]["via_verified"] is False  # a graph for another IR
    scale = freeze(table, view, [{"op": "scale", "args": {"k": 2.0}, "select": {"group": g["group_id"], "units": [1]}}])
    assert plan(table, scale, base_id=BASE, graph=view)["result"]["groups"][0]["via_verified"] is True

    def forge(change):
        fake = copy.deepcopy(real)
        change(fake)
        fake["recipe_hash"] = edit._finish(BASE, fake["program"], "h", None, edit.ENGINE_V2)["recipe_hash"]
        return fake
    fakes = {
        "another perm": forge(lambda r: [e["via"].update(perm=[1, 0] + list(range(2, g["size"]))) for e in r["program"]]),
        "an index swapped": forge(lambda r: r["program"][0]["args"]["perm"].reverse()),
        "a member dropped": forge(lambda r: r["program"].pop()),
        "unknown group": forge(lambda r: [e["via"].update(group="g_" + "0" * 24) for e in r["program"]]),
        "an open group": forge(lambda r: [e["via"].update(group=group_of(view, "fc.weight", 0)["group_id"]) for e in r["program"]]),
    }
    for name, fake in fakes.items():
        err(lambda: plan(table, fake, base_id=BASE, graph=view), "noncanonical_recipe", "program[0].via")
        err(lambda: apply(table, fake, tmp_path / "x.safetensors", base_id=BASE, graph=view), "noncanonical_recipe")
    for name, fake in fakes.items():  # without the graph the same recipes are still graph-free valid recipes, flagged unverified
        if name in ("a member dropped", "an index swapped"):
            assert plan(table, fake, base_id=BASE)["result"]["groups"][0]["via_verified"] is False
