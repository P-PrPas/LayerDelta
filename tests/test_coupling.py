"""Roles, coupling groups, opaque boundaries and coverage (v0.3-G3), on the T6 corpus and small purpose-built models."""

import copy
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from torch import nn

from layerdelta import coupling
from layerdelta.coupling import derive
from layerdelta.graph import finalize_ir
from layerdelta.providers import build_graph, register_builtin
from layerdelta.tensors import open_tensors
from test_corpus import FAMILIES, build, save
from test_graph_ir import graph_of, register_corpus

register_corpus()
TESTS = Path(__file__).resolve().parent
SRC = TESTS.parent / "src"


@lru_cache(maxsize=None)
def _derived(family):
    import tempfile
    _, state = build(family, torch.float32)
    table = open_tensors(save(Path(tempfile.mkdtemp()), state, f"{family}.safetensors"))
    return derive(graph_of(family), table), table


def indices(member, u):
    return {member["offset"] + u * member["block"] + k * member["period"] + j
            for k in range(member["repeat"]) for j in range(member["block"])}


def members_of(group):
    return {(m["tensor"], m["axis"]) for m in group["members"]}


def one(result, tensor, axis, index=0, size=None):
    """The single group whose member (tensor, axis) covers `index`."""
    found = [g for g in result.groups if (size is None or g["size"] == size) and
             any(m["tensor"] == tensor and m["axis"] == axis and index in set().union(*(indices(m, u) for u in range(g["size"])))
                 and any(index in indices(m, u) for u in range(g["size"])) for m in g["members"])]
    assert len(found) == 1, (tensor, axis, index, [g["group_id"] for g in found])
    return found[0]


# ------------------------------------------------------------------ A5: coupling content

def test_resnet_residual_chain_is_one_closed_group():
    result, _ = _derived("resnet")
    stage = one(result, "layer3.0.bn2.weight", 0)
    assert stage["closed"] and stage["size"] == 32
    assert {("layer3.0.bn2.weight", 0), ("layer3.0.conv2.weight", 0), ("layer3.0.downsample.0.weight", 0),
            ("layer3.0.downsample.1.weight", 0), ("fc.weight", 1)} <= members_of(stage)
    assert members_of(one(result, "layer1.0.conv1.weight", 0)) >= {("layer1.0.bn1.weight", 0), ("layer1.0.conv2.weight", 1)}


def test_unet_concat_offsets_and_transposed_conv_axes():
    result, _ = _derived("unet")
    dec2_in = [m for g in result.groups for m in g["members"] if m["tensor"] == "dec2.0.weight" and m["axis"] == 1]
    assert sorted((m["offset"], m["block"]) for m in dec2_in) == [(0, 1), (16, 1)]  # enc2 skip first, then up2
    up2_out = [g for g in result.groups if ("up2.weight", 1) in members_of(g)]  # ConvT: out channels are weight axis 1
    assert len(up2_out) == 1 and ("up2.bias", 0) in members_of(up2_out[0]) and ("dec2.0.weight", 1) in members_of(up2_out[0])
    assert ("up2.weight", 0) not in members_of(up2_out[0])


def test_yolo_depthwise_conv_shares_one_unit_space():
    result, _ = _derived("yolo")
    g = one(result, "dw3.weight", 0)
    assert ("dw3.weight", 1) not in members_of(g) and ("fuse3.bn.weight", 0) in members_of(g)  # one space for in and out channels


def test_vit_attention_is_split_into_heads_qk_and_v():
    result, _ = _derived("vit")
    qkv_rows = [(g, m) for g in result.groups for m in g["members"] if m["tensor"] == "blocks.0.attn.qkv.weight" and m["axis"] == 0]
    sizes = sorted(g["size"] for g, _ in qkv_rows)
    assert sizes == [2, 8, 8]  # heads, the q/k head dimension, the v head dimension
    heads = next(g for g, _ in qkv_rows if g["size"] == 2)
    assert ("blocks.0.attn.proj.weight", 1) in members_of(heads) and heads["closed"]
    heads_member = next(m for m in heads["members"] if m["tensor"] == "blocks.0.attn.qkv.weight")
    assert (heads_member["block"], heads_member["period"], heads_member["repeat"]) == (8, 16, 3)  # q, k and v rows of one head
    eights = [g for g, _ in qkv_rows if g["size"] == 8]
    qk = next(g for g in eights if ("blocks.0.attn.proj.weight", 1) not in members_of(g))
    v = next(g for g in eights if ("blocks.0.attn.proj.weight", 1) in members_of(g))
    qk_member = next(m for m in qk["members"] if m["tensor"] == "blocks.0.attn.qkv.weight")
    v_member = next(m for m in v["members"] if m["tensor"] == "blocks.0.attn.qkv.weight")
    assert indices(qk_member, 0) == {0, 8, 16, 24} and indices(v_member, 0) == {32, 40}  # q,k rows apart from v rows
    assert all(g["closed"] for g in eights)
    packed = [r for r in result.roles["blocks.0.attn.qkv.weight"] if r["axis"] == 0][0]
    assert packed["role"] == "packed" and packed["k"] == 3 and [f["size"] for f in packed["factors"]] == [3, 2, 8]


def test_crnn_lstm_groups():
    result, _ = _derived("crnn")
    layer0 = one(result, "rnn.weight_hh_l0", 1)
    assert layer0["size"] == 24 and layer0["closed"] and layer0["kinds"] == ["hidden"]
    rows = next(m for m in layer0["members"] if m["tensor"] == "rnn.weight_ih_l0" and m["axis"] == 0)
    assert (rows["repeat"], rows["period"], rows["block"]) == (4, 24, 1)  # four gates of hidden size 24
    assert {("rnn.bias_hh_l0", 0), ("rnn.bias_ih_l0", 0), ("rnn.weight_hh_l0", 0)} <= members_of(one(result, "rnn.weight_hh_l0", 0))
    # layer 1 reads layer 0's forward then reverse hidden units as two offset blocks of its input columns
    cols = [(m["offset"], g["size"]) for g in result.groups for m in g["members"] if m["tensor"] == "rnn.weight_ih_l1" and m["axis"] == 1]
    assert sorted(cols) == [(0, 24), (24, 24)]
    flat = [g for g in result.groups if ("rnn.weight_ih_l0", 1) in members_of(g) and g["size"] == 2]
    assert flat and not flat[0]["closed"] and flat[0]["boundary"][0]["reason"] == "data_dimension"  # the frequency axis is data


# ------------------------------------------------------------------ boundaries and the output rule

@pytest.mark.parametrize("family", list(FAMILIES))
def test_groups_reaching_the_graph_boundary_are_open(family):
    result, _ = _derived(family)
    reasons = {b["reason"] for g in result.groups if not g["closed"] for b in g["boundary"]}
    assert {"graph_output"} | ({"graph_input"} if family != "crnn" else set()) <= reasons  # crnn takes one-channel images
    for g in result.groups:
        assert g["closed"] == (not g["boundary"])
        for b in g["boundary"]:
            assert b["reason"] in {"graph_output", "graph_input", "opaque_op", "data_dimension", "indexed_unit", "non_affine_member"}
    outputs = [g for g in result.groups if any(b["reason"] == "graph_output" for b in g["boundary"])]
    assert outputs and all(g["closed"] is False for g in outputs)


def test_the_output_layer_is_open_and_the_hidden_stream_is_closed():
    result, _ = _derived("resnet")
    head = one(result, "fc.weight", 0)
    assert not head["closed"] and {"graph_output"} == {b["reason"] for b in head["boundary"]}
    assert one(result, "layer1.0.bn1.weight", 0)["closed"]


# ------------------------------------------------------------------ A3 (permutation), without G5: apply the member maps by hand

def _permute(state, group, perm):
    out = {k: v.clone() for k, v in state.items()}
    for m in group["members"]:
        t = out[m["tensor"]]
        ref = t.clone()  # a tensor can be a member on two axes (w_hh rows and columns): apply them one after the other
        for u in range(group["size"]):
            src = sorted(indices(m, u))
            dst = sorted(indices(m, int(perm[u])))
            t.index_copy_(m["axis"], torch.tensor(dst), ref.index_select(m["axis"], torch.tensor(src)))
    return out


@pytest.mark.parametrize("family", list(FAMILIES))
def test_permuting_a_closed_group_preserves_the_function(family):
    result, _ = _derived(family)
    model, state = build(family, torch.float32)
    x = torch.randn(*FAMILIES[family][1])
    with torch.no_grad():
        base = model(x)
    flat = lambda y: torch.cat([t.flatten() for t in (y if isinstance(y, (list, tuple)) else [y])])
    gen = torch.Generator().manual_seed(1)
    checked = 0
    for g in result.groups:
        if not g["closed"]:
            continue
        perm = torch.arange(g["size"])
        while perm.tolist() == list(range(g["size"])):
            perm = torch.randperm(g["size"], generator=gen)
        model.load_state_dict(_permute(state, g, perm))
        with torch.no_grad():
            got = model(x)
        assert torch.allclose(flat(base), flat(got), atol=1e-4), (g["group_id"], sorted(members_of(g))[:3])
        checked += 1
    model.load_state_dict(state)
    assert checked >= 6


def test_permuting_an_output_group_changes_the_function():
    result, _ = _derived("resnet")
    model, state = build("resnet", torch.float32)
    x = torch.randn(*FAMILIES["resnet"][1])
    head = one(result, "fc.weight", 0)
    perm = torch.arange(head["size"]).flip(0)
    with torch.no_grad():
        base = model(x)
        model.load_state_dict(_permute(state, head, perm))
        assert not torch.allclose(base, model(x), atol=1e-4)  # this is why graph outputs are boundaries


# ------------------------------------------------------------------ A6: an op without a rule

class Cumsum(nn.Module):
    def __init__(self):
        super().__init__()
        self.a, self.b, self.c = nn.Linear(4, 6), nn.Linear(6, 5), nn.Linear(5, 3)

    def forward(self, x):
        h = self.a(x)
        h = torch.cumsum(h, dim=-1)  # no rule: the order of channels matters
        return self.c(torch.relu(self.b(h)))


register_builtin("test.cumsum", Cumsum, [((2, 4), "float32")], replace=True)


def _derive_model(tmp_path, name, model):
    ir = build_graph({"provider": "builtin", "model": name})
    state = {k: v.detach().clone() for k, v in model().state_dict().items()}
    return derive(ir, open_tensors(save(tmp_path, state, "m.safetensors"))), model, state


def test_an_op_without_a_rule_is_listed_and_opens_its_groups(tmp_path):
    result, model, state = _derive_model(tmp_path, "test.cumsum", Cumsum)
    assert result.opaque_ops == [{"op": "aten.cumsum.default", "count": 1}]
    assert result.coverage["opaque_ops"] == result.opaque_ops
    touched = one(result, "a.weight", 0)
    assert not touched["closed"] and "opaque_op" in {b["reason"] for b in touched["boundary"]}
    assert any(b["op"] == "aten.cumsum.default" for b in touched["boundary"])
    after = one(result, "b.weight", 1)  # the op's output is a different unit space, also behind the boundary
    assert after["group_id"] != touched["group_id"] and not after["closed"]
    assert one(result, "b.weight", 0)["closed"]  # downstream of the op, behind a relu: still understood
    m = model()
    m.load_state_dict(state)
    x = torch.randn(2, 4)
    perm = torch.arange(6).flip(0)
    with torch.no_grad():
        base = m(x)
        m.load_state_dict(_permute(state, touched, perm))
        assert not torch.allclose(base, m(x), atol=1e-4)  # permuting it really would change the function


# ------------------------------------------------------------------ recurrent variants: proj_size and bias=False are read from the node

class ProjLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(6, 8, num_layers=2, bidirectional=True, proj_size=4, batch_first=True)
        self.fc = nn.Linear(8, 3)

    def forward(self, x):
        return self.fc(self.rnn(x)[0])


class NoBiasLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(6, 8, num_layers=2, bias=False)
        self.gru = nn.GRU(8, 5, bias=False)
        self.fc = nn.Linear(5, 3)

    def forward(self, x):
        return self.fc(self.gru(self.rnn(x)[0])[0])


for _name, _cls in (("test.proj_lstm", ProjLSTM), ("test.nobias_rnn", NoBiasLSTM)):
    register_builtin(_name, _cls, [((2, 5, 6), "float32")], replace=True)


def _permutation_check(tmp_path, name, cls):
    result, model, state = _derive_model(tmp_path, name, cls)
    m = cls().eval()
    x = torch.randn(2, 5, 6)
    gen = torch.Generator().manual_seed(3)
    checked = 0
    for g in result.groups:
        if not g["closed"]:
            continue
        perm = torch.arange(g["size"])
        while perm.tolist() == list(range(g["size"])):
            perm = torch.randperm(g["size"], generator=gen)
        m.load_state_dict(state)
        with torch.no_grad():
            base = m(x)
            m.load_state_dict(_permute(state, g, perm))
            assert torch.allclose(base, m(x), atol=1e-4), (name, g["group_id"], sorted(members_of(g))[:4])
        checked += 1
    return result, checked


def test_lstm_projection_weights_are_read_from_the_node(tmp_path):
    result, checked = _permutation_check(tmp_path, "test.proj_lstm", ProjLSTM)
    assert checked >= 5 and result.coverage["tensors"]["unbound"] == 0
    assert [r["role"] for r in result.roles["rnn.weight_hr_l0"]] == ["out_unit", "in_unit"]
    assert result.roles["rnn.weight_hr_l0"][1]["basis"].startswith("rnn.projection_kernel@")
    hidden = one(result, "rnn.weight_hr_l0", 1)  # the cell state: 8 units, gates x 8 rows, projected back to 4
    assert hidden["size"] == 8 and {("rnn.weight_ih_l0", 0), ("rnn.bias_ih_l0", 0), ("rnn.weight_hh_l0", 0)} <= members_of(hidden)
    projected = one(result, "rnn.weight_hr_l0", 0)  # the projected state that the next step and the next layer read
    assert projected["size"] == 4 and {("rnn.weight_hh_l0", 1), ("rnn.weight_ih_l1", 1)} <= members_of(projected)
    assert ("rnn.weight_hh_l0", 1) not in members_of(hidden)
    assert all(r["role"] != "unknown" for rs in result.roles.values() for r in rs)


def test_lstm_and_gru_without_biases_are_decoded(tmp_path):
    result, checked = _permutation_check(tmp_path, "test.nobias_rnn", NoBiasLSTM)
    assert checked == 3 and not any("bias" in name for name in result.roles if name.startswith(("rnn.", "gru.")))
    rows = next(m for g in result.groups for m in g["members"] if m["tensor"] == "rnn.weight_ih_l0" and m["axis"] == 0)
    assert (rows["repeat"], rows["period"]) == (4, 8)
    gru = next(m for g in result.groups for m in g["members"] if m["tensor"] == "gru.weight_ih_l0" and m["axis"] == 0)
    assert (gru["repeat"], gru["period"]) == (3, 5)  # a GRU has three gates
    assert all(r["role"] != "unknown" for rs in result.roles.values() for r in rs)


# ------------------------------------------------------------------ A7: coverage, provenance, basis

@pytest.mark.parametrize("family", list(FAMILIES))
def test_coverage_accounts_for_every_axis(family):
    result, table = _derived(family)
    cov = result.coverage
    assert cov["unit"] == "axes"
    assert cov["derived"] + cov["declared"] + cov["inferred"] + cov["unknown"] == sum(len(r) for r in result.roles.values())
    assert sum(1 for rs in result.roles.values() for r in rs if r["axis"] is not None) == sum(len(i.shape) for i in table.tensors.values())
    assert sum(cov["tensors"].values()) == len(table.tensors)
    assert cov["groups"]["closed"] + cov["groups"]["open"] == len(result.groups)
    unknown = {name for name, rs in result.roles.items() for r in rs if r["role"] == "unknown"}
    expected = {"resnet": set(), "unet": set(), "crnn": set(), "vit": {"logit_scale", "tok_embed.weight"}, "yolo": {"anchors_unused"}}[family]
    assert unknown == expected  # exactly the tensors G2 found unbound


@pytest.mark.parametrize("family", list(FAMILIES))
def test_every_role_has_provenance_and_a_rule_at_node_basis(family):
    result, table = _derived(family)
    ids = {n["id"] for n in graph_of(family)["nodes"]}
    for name, axes in result.roles.items():
        assert axes
        for r in axes:
            assert r["provenance"] in {"derived", "unknown"}
            if r["provenance"] == "derived":
                m = re.fullmatch(r"([A-Za-z_.]+)@(n\d+|graph)", r["basis"])
                assert m and m.group(2) in ids | {"graph"}, (name, r)
    conv = result.roles["layer1.0.conv1.weight"] if "layer1.0.conv1.weight" in result.roles else None
    if family == "resnet":
        assert [r["role"] for r in conv] == ["out_unit", "in_unit", "spatial", "spatial"]
        assert conv[0]["basis"].startswith("conv.kernel@")
        assert [r["role"] for r in result.roles["fc.bias"]] == ["channel"]
    if family == "vit":
        assert result.roles["tok_embed.weight"][0]["basis"] == "unbound"
        assert [r["role"] for r in result.roles["pos_embed"]] == ["position", "position", "channel"]


def test_conv_linear_norm_lstm_embedding_axes_are_all_derived():
    for family in FAMILIES:
        result, _ = _derived(family)
        for name, axes in result.roles.items():
            if re.search(r"(conv|\.weight_ih|\.weight_hh|fc\.|head|qkv|proj|mlp|bn|norm)", name):
                assert all(r["role"] != "unknown" for r in axes), (family, name)


# ------------------------------------------------------------------ group ids, determinism, the version guard

def test_group_ids_are_deterministic_and_value_independent(tmp_path):
    first, _ = _derived("vit")
    _, state = build("vit", torch.float32, seed=7)  # different weights, same graph and layout
    again = derive(graph_of("vit"), open_tensors(save(tmp_path, state, "other.safetensors")))
    assert first.sha256() == again.sha256()
    assert [g["group_id"] for g in first.groups] == [g["group_id"] for g in again.groups]
    assert len({g["group_id"] for g in first.groups}) == len(first.groups)
    code = ("import sys,tempfile;sys.path[:0]=[%r,%r];from pathlib import Path;import torch\n"
            "from test_corpus import build,save;from test_graph_ir import graph_of,register_corpus\n"
            "from layerdelta.coupling import derive;from layerdelta.tensors import open_tensors\n"
            "_,s=build('vit',torch.float32);t=open_tensors(save(Path(tempfile.mkdtemp()),s))\n"
            "print(derive(graph_of('vit'),t).sha256())") % (str(SRC), str(TESTS))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == first.sha256()


def test_group_id_changes_with_the_graph(tmp_path):
    a, _ = _derived("resnet")
    ir = graph_of("resnet")
    body = {k: v for k, v in ir.items() if k != "ir_sha256"}
    body["provider"] = {**body["provider"], "spec_sha256": "0" * 64}  # same layout, different sealed graph
    other = derive(finalize_ir(body), _derived("resnet")[1])
    assert {g["group_id"] for g in a.groups}.isdisjoint({g["group_id"] for g in other.groups})
    assert [g["members"] for g in a.groups] == [g["members"] for g in other.groups]


def test_an_unknown_torch_release_makes_every_op_opaque():
    ir = graph_of("resnet")
    body = copy.deepcopy({k: v for k, v in ir.items() if k != "ir_sha256"})
    body["provider"]["version"] = "torch-9.9.0"
    result = derive(finalize_ir(body), _derived("resnet")[1])
    assert result.warnings and "torch 9.9.0" in result.warnings[0]
    assert all(r["role"] == "unknown" for rs in result.roles.values() for r in rs if r["axis"] is not None)
    assert not any(g["closed"] for g in result.groups)
    assert result.rules["torch"] == ["2.6"]


def test_the_rule_engine_is_generic():
    text = Path(coupling.__file__).read_text(encoding="utf-8")
    for token in ("layer1", "resnet", "qkv", "tok_embed", "corpus", "ViT", "UNet"):
        assert token not in text.replace('"qkv"', ""), token
