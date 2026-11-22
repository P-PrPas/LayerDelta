"""GraphIR, the builtin provider, binding and the graph store (v0.3-G2), on the T6 corpus and the MNIST reference CNN."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from layerdelta import edit, providers
from layerdelta.checkpoints import Catalog
from layerdelta.graph import (GraphMismatch, GraphStore, bind, finalize_ir, ir_json, table_signature, verify_ir)
from layerdelta.model import ReferenceCNN
from layerdelta.providers import GraphBuildError, build_graph, register_builtin
from layerdelta.selector import InspectError
from layerdelta.tensors import open_tensors
from test_corpus import FAMILIES, build, file_sha, save

TESTS = Path(__file__).resolve().parent
SRC = TESTS.parent / "src"


def register_corpus():
    for family, (cls, shape) in FAMILIES.items():
        register_builtin(f"corpus.{family}", cls, [(shape, "float32")], replace=True)


register_corpus()
EXPECTED_UNBOUND = {"resnet": set(), "unet": set(), "vit": {"logit_scale", "tok_embed.weight"}, "crnn": set(),
                    "yolo": {"anchors_unused"}}


def graph_of(family):
    return build_graph({"provider": "builtin", "model": f"corpus.{family}"})


def table_of(tmp_path, family, dtype=torch.float32, name="parent.safetensors"):
    _, state = build(family, dtype)
    return open_tensors(save(tmp_path, state, name)), state


class Tied(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 4)
        self.head = nn.Linear(4, 10, bias=False)
        self.head.weight = self.embed.weight  # weight tying

    def forward(self, ids):
        return self.head(self.embed(ids))


register_builtin("test.tied", Tied, [((2, 3), "int64")], replace=True)


# ------------------------------------------------------------------ the registry is closed

def test_the_builtin_provider_only_builds_registered_names():
    assert {"reference_cnn", "test.tied", *(f"corpus.{f}" for f in FAMILIES)} <= set(providers.registered_models())
    for model, code in (("nope", "unknown_model"), ("os:system", "invalid_spec"), ("pkg.mod:factory", "invalid_spec"),
                        ("../model", "invalid_spec"), ("Corpus.vit", "invalid_spec"), (5, "invalid_spec")):
        with pytest.raises(GraphBuildError) as info:
            build_graph({"provider": "builtin", "model": model})
        assert info.value.code == code
    for spec in ({"provider": "builtin"}, {"provider": "builtin", "model": "reference_cnn", "entry": "x:y"}, {"model": "reference_cnn"},
                 {"provider": "torch_export", "model": "reference_cnn"}, "reference_cnn", None):
        with pytest.raises(GraphBuildError):
            build_graph(spec)
    with pytest.raises(ValueError):
        register_builtin("Bad Name", nn.Identity, [])
    with pytest.raises(ValueError):
        register_builtin("reference_cnn", nn.Identity, [])  # no silent replacement
    assert set(providers.PROVIDERS) == {"builtin"}


# ------------------------------------------------------------------ the IR

@pytest.mark.parametrize("name", ["reference_cnn", *(f"corpus.{f}" for f in FAMILIES), "test.tied"])
def test_ir_is_valid_canonical_and_deterministic(name):
    first = build_graph({"provider": "builtin", "model": name})
    second = build_graph({"provider": "builtin", "model": name})  # a fresh module instance, a fresh export
    assert first["ir_sha256"] == second["ir_sha256"] and ir_json(first) == ir_json(second)
    assert verify_ir(first) is first and first["ir_version"] == 1 and first["provider"]["name"] == "builtin"
    assert first["provider"]["version"].startswith("torch-") and "+" not in first["provider"]["version"]
    assert json.loads(ir_json(first)) == first and verify_ir(json.loads(ir_json(first)))  # canonical JSON round-trips
    assert first["nodes"] and first["outputs"] and first["inputs"] and first["params"]
    assert all(n["module_type"] for n in first["nodes"] if n["module_path"])  # module path implies a module type


def test_ir_is_the_same_in_another_process():
    code = ("import sys; sys.path[:0] = [%r, %r]\n"
            "import json, test_corpus\n"
            "from layerdelta import providers\n"
            "for f, (cls, shape) in test_corpus.FAMILIES.items():\n"
            "    providers.register_builtin('corpus.' + f, cls, [(shape, 'float32')], replace=True)\n"
            "print(json.dumps({m: providers.build_graph({'provider': 'builtin', 'model': m})['ir_sha256'] "
            "for m in ('reference_cnn', 'corpus.vit', 'corpus.crnn')}))") % (str(SRC), str(TESTS))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=300).stdout
    other = json.loads(out.strip().splitlines()[-1])
    here = {m: build_graph({"provider": "builtin", "model": m})["ir_sha256"] for m in other}
    assert other == here


def test_ir_schema_is_strict():
    ir = graph_of("unet")
    body = {k: v for k, v in ir.items() if k != "ir_sha256"}

    def broken(change):
        copy_ = copy.deepcopy(body)
        change(copy_)
        return finalize_ir(copy_)

    cases = {
        "forward reference": lambda b: b["nodes"][0]["args"].append("v99999"),
        "unknown wrapper": lambda b: b["nodes"][0]["args"].append({"bogus": 1}),
        "duplicate param": lambda b: b["params"].append(dict(b["params"][0], value="p9999")),
        "bad kind": lambda b: b["params"][0].update(kind="tensor"),
        "bad dtype": lambda b: b["inputs"][0].update(dtype="F4"),
        "negative dim": lambda b: b["inputs"][0].update(shape=[-1, 3]),
        "bad output": lambda b: b["outputs"].append("v99999"),
        "extra field": lambda b: b.update(extra=1),
        "unsorted ties": lambda b: b.update(ties=[["b", "a"]]),
        "NaN literal": lambda b: b["nodes"][0]["args"].append(float("nan")),
    }
    for name, change in cases.items():
        with pytest.raises(InspectError) as info:
            broken(change)
        assert info.value.code == "invalid_graph", name
    tampered = copy.deepcopy(ir)
    tampered["nodes"][0]["op"] = "aten.relu.default"
    with pytest.raises(InspectError) as info:
        verify_ir(tampered)
    assert info.value.code == "ir_hash_mismatch"
    with pytest.raises(InspectError) as info:
        verify_ir({**ir, "ir_version": 2})
    assert info.value.code == "unsupported_ir_version"


def test_nodes_carry_module_path_and_type():
    ir = graph_of("vit")
    qkv = [n for n in ir["nodes"] if n["module_path"] == "blocks.0.attn.qkv"]
    assert qkv and qkv[0]["op"] == "aten.linear.default" and qkv[0]["module_type"] == "Linear"
    ref = build_graph({"provider": "builtin", "model": "reference_cnn"})
    assert {(n["module_path"], n["module_type"]) for n in ref["nodes"]} >= {("features.block1.conv", "Conv2d"), ("classifier", "Linear")}


# ------------------------------------------------------------------ A1: binding

@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_binding_classifies_every_tensor(tmp_path, family, dtype):
    table, state = table_of(tmp_path, family, dtype)
    bound = bind(graph_of(family), table)  # the F32 graph binds to a BF16 checkpoint: dtype is not part of the meaning
    assert set(bound.tensors) == set(state)
    assert set(bound.bound) | set(bound.unused) | set(bound.unbound) == set(state)
    assert set(bound.unbound) == EXPECTED_UNBOUND[family]
    assert set(bound.unused) == {n for n in state if n.endswith("num_batches_tracked")}  # placeholders with no users in eval
    for name in bound.bound:
        assert bound.tensors[name].uses and all(u["node"].startswith("n") for u in bound.tensors[name].uses)
    for name in bound.unused + bound.unbound:
        assert bound.tensors[name].uses == []


def test_the_lstm_fused_node_is_decoded_by_position(tmp_path):
    ir = graph_of("crnn")
    table, _ = table_of(tmp_path, "crnn")
    bound = bind(ir, table)
    lstm = next(n for n in ir["nodes"] if n["op"] == "aten.lstm.input")
    assert (lstm["args"][3], lstm["args"][4], lstm["args"][7]) == (True, 2, True)  # has_biases, num_layers, bidirectional
    by_position = {}
    for name, tb in bound.tensors.items():
        for use in tb.uses:
            if use["node"] == lstm["id"]:
                assert use["arg"] == 2 and len(use["path"]) == 1 and tb.uses == [use]
                by_position[use["path"][0]] = name
    layers, directions = lstm["args"][4], 2 if lstm["args"][7] else 1
    expected = [f"rnn.{kind}_l{layer}{'_reverse' if reverse else ''}" for layer in range(layers) for reverse in range(directions)
                for kind in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")]
    assert len(lstm["args"][2]) == layers * directions * 4 and [by_position[i] for i in sorted(by_position)] == expected


def test_a_graph_that_does_not_fit_is_refused_with_the_names(tmp_path):
    ir = graph_of("resnet")
    _, state = build("resnet", torch.float32)
    short = {k: v for k, v in state.items() if k not in ("fc.weight", "conv1.weight")}
    with pytest.raises(GraphMismatch) as info:
        bind(ir, open_tensors(save(tmp_path, short, "short.safetensors")))
    assert info.value.code == "graph_mismatch" and info.value.names == ["conv1.weight", "fc.weight"]
    reshaped = dict(state, **{"fc.weight": torch.zeros(10, 33)})
    with pytest.raises(GraphMismatch) as info:
        bind(ir, open_tensors(save(tmp_path, reshaped, "reshaped.safetensors")))
    assert info.value.names == ["fc.weight"] and "shape" in info.value.problems[0][1]
    retyped = dict(state, **{"fc.bias": torch.zeros(10, dtype=torch.int64)})
    with pytest.raises(GraphMismatch) as info:
        bind(ir, open_tensors(save(tmp_path, retyped, "retyped.safetensors")))
    assert info.value.names == ["fc.bias"] and "dtype" in info.value.problems[0][1]
    extra = bind(ir, open_tensors(save(tmp_path, dict(state, extra=torch.zeros(3)), "extra.safetensors")))
    assert extra.unbound == ["extra"]  # a tensor the graph never heard of is reported, not an error


def test_declared_ties_bind_both_names(tmp_path):
    ir = build_graph({"provider": "builtin", "model": "test.tied"})
    assert ir["ties"] == [["embed.weight", "head.weight"]] and [p["name"] for p in ir["params"]] == ["head.weight"]
    both = open_tensors(save(tmp_path, {"embed.weight": torch.randn(10, 4), "head.weight": torch.randn(10, 4)}, "both.safetensors"))
    bound = bind(ir, both)
    assert bound.bound == ["embed.weight", "head.weight"] and bound.tensors["embed.weight"].tied_with == ["head.weight"]
    assert bound.tensors["embed.weight"].uses == bound.tensors["head.weight"].uses and len(bound.tensors["head.weight"].uses) == 2
    once = open_tensors(save(tmp_path, {"embed.weight": torch.randn(10, 4)}, "once.safetensors"))  # tied weights stored once
    assert bind(ir, once).bound == ["embed.weight"] and bind(ir, once).tensors["embed.weight"].tied_with == ["head.weight"]
    with pytest.raises(GraphMismatch):
        bind(ir, open_tensors(save(tmp_path, {"other": torch.zeros(1)}, "none.safetensors")))


def test_the_reference_cnn_binds_completely(tmp_path):
    table = open_tensors(save(tmp_path, ReferenceCNN().state_dict()))
    bound = bind(build_graph({"provider": "builtin", "model": "reference_cnn"}), table)
    assert bound.unbound == [] and bound.unused == [] and len(bound.bound) == 6


# ------------------------------------------------------------------ the store and A8 inheritance

@pytest.fixture
def store(tmp_path):
    catalog = Catalog(tmp_path / "ws")
    catalog.start()
    return GraphStore(catalog)


def test_the_store_keys_graphs_by_table_signature(tmp_path, store):
    table, _ = table_of(tmp_path, "resnet")
    ir = graph_of("resnet")
    record = store.put(ir, table)
    assert record["ir_sha256"] == ir["ir_sha256"] and record["table_signature"] == table_signature(table)
    assert record["provider"] == "builtin" and record["spec_sha256"] == ir["provider"]["spec_sha256"]
    assert store.put(ir, table) == record and len(store.for_table(table)) == 1  # idempotent
    assert store.graph_for(table) == ir and store.load(ir["ir_sha256"]) == ir
    other = finalize_ir({**{k: v for k, v in ir.items() if k != "ir_sha256"},
                         "provider": {**ir["provider"], "name": "onnx", "spec_sha256": "f" * 64}})
    store.put(other, table)
    assert [r["provider"] for r in store.for_table(table)] == ["builtin", "onnx"]
    assert store.graph_for(table, "builtin") == ir and store.graph_for(table) == other  # newest, or per provider
    assert store.graph_for(table, "torch_export") is None
    unrelated, _ = table_of(tmp_path, "vit", name="vit.safetensors")
    assert store.for_table(unrelated) == [] and store.graph_for(unrelated) is None
    with pytest.raises(GraphMismatch):
        store.put(ir, unrelated)
    assert len(store.for_table(unrelated)) == 0
    with pytest.raises(InspectError) as info:
        store.load("0" * 64)
    assert info.value.code == "not_found"
    with store.catalog._db(True) as db:  # a row altered on disk is caught on load
        db.execute("UPDATE graphs SET ir_json = replace(ir_json, 'aten.add', 'aten.mul') WHERE ir_sha256=?", (ir["ir_sha256"],))
    with pytest.raises(InspectError) as info:
        store.load(ir["ir_sha256"])
    assert info.value.code == "ir_hash_mismatch"


def test_a_child_checkpoint_inherits_its_parents_graph(tmp_path, store):
    table, state = table_of(tmp_path, "resnet")
    parent_path = tmp_path / "parent.safetensors"
    ir = graph_of("resnet")
    store.put(ir, table)
    base_id = file_sha(parent_path)
    conv = next(n for n, t in state.items() if t.ndim == 4)
    bias = next(n for n, t in state.items() if t.ndim == 1 and n.endswith("bias"))
    program = [{"op": "scale", "args": {"k": 0.5}, "select": {"tensor": conv, "index": {"0": [0]}}},
               {"op": "add", "args": {"value": 0.25}, "select": {"tensor": bias}}]
    child_path = tmp_path / "child.safetensors"
    edit.apply(table, edit.freeze_recipe(table, base_id, program, "inherit"), child_path, base_id=base_id)
    child = open_tensors(child_path)
    assert file_sha(child_path) != base_id and table_signature(child) == table_signature(table)
    inherited = store.for_table(child)
    assert [r["ir_sha256"] for r in inherited] == [ir["ir_sha256"]] and store.graph_for(child) == ir
    parent_bound, child_bound = bind(ir, table), bind(store.graph_for(child), child)
    assert {n: (b.status, b.uses) for n, b in child_bound.tensors.items()} == {n: (b.status, b.uses) for n, b in parent_bound.tensors.items()}
    assert child_bound.unbound == parent_bound.unbound == [] and child_bound.unused == parent_bound.unused
    # a different layout (here another dtype) is a different signature: no inheritance, no guessing
    half, _ = table_of(tmp_path, "resnet", torch.bfloat16, "half.safetensors")
    assert table_signature(half) != table_signature(table) and store.for_table(half) == []
    assert bind(ir, half).unbound == []  # ...but the graph can still be attached to it explicitly
    assert [r["ir_sha256"] for r in [store.put(ir, half)]] == [ir["ir_sha256"]] and len(store.for_table(half)) == 1


def test_the_table_signature_ignores_values_and_file_layout(tmp_path):
    a, state = table_of(tmp_path, "unet", name="a.safetensors")
    reordered = dict(reversed(list(state.items())))
    b = open_tensors(save(tmp_path, {k: v * 2 if v.is_floating_point() else v for k, v in reordered.items()}, "b.safetensors"))
    assert table_signature(a) == table_signature(b)
    renamed = open_tensors(save(tmp_path, {("x." + k): v for k, v in state.items()}, "c.safetensors"))
    assert table_signature(renamed) != table_signature(a)
