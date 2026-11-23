"""A10: Torch-Pruning's DependencyGraph as a dev-only oracle for the coupling groups (ADR 0001).

Optional: needs `pip install -e ".[oracle]"` (torch-pruning==1.6.1) and runs with `pytest -m oracle`. Without the extra the
module is skipped. Nothing in src/ imports it. Only groups inside the declared scope are compared (conv, conv-transpose, linear,
batch/layer norm, add, concat, flatten, depthwise); everything else is listed in the output, never silently dropped.
"""

import copy
import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

tp = pytest.importorskip("torch_pruning", minversion="1.6")  # the 1.6.1 wheel reports __version__ 1.6.0

from layerdelta.coupling import derive
from layerdelta.tensors import open_tensors
from test_corpus import FAMILIES, build, save
from test_graph_ir import graph_of, register_corpus

pytestmark = pytest.mark.oracle
register_corpus()

# the exact number of DepGraph groups per family that are in scope, passed their own prune check and do not reach an output
# (ViT: the residual-stream group holds DepGraph "unwrapped parameters" and the attention groups are wrong in DepGraph, so only
# the two MLP groups are in scope; CRNN: the second CNN group runs into the LSTM, so only the first is.)
EXPECTED_IN_SCOPE = {"resnet": 8, "unet": 12, "yolo": 8, "vit": 2, "crnn": 1}
SCOPE = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear,
         nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)


def indices(m, u):
    return {m["offset"] + u * m["block"] + k * m["period"] + j for k in range(m["repeat"]) for j in range(m["block"])}


def mapped(module, name, handler, idxs):
    """DepGraph (module, handler, idxs) -> {(tensor, axis, frozenset(idxs))}, or None when the layer is outside the scope."""
    if not isinstance(module, SCOPE):
        return None
    out_side = handler == "prune_out_channels"
    idxs = frozenset(int(i) for i in idxs)
    if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        members = [("weight", 1 if out_side else 0)] + ([("bias", 0)] if out_side and module.bias is not None else [])
    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        depthwise = module.groups > 1 and module.groups == module.in_channels
        members = [("weight", 0 if out_side or depthwise else 1)] + ([("bias", 0)] if out_side and module.bias is not None else [])
    elif isinstance(module, nn.Linear):
        members = [("weight", 0 if out_side else 1)] + ([("bias", 0)] if out_side and module.bias is not None else [])
    elif isinstance(module, nn.LayerNorm):
        members = [(p, module.weight.ndim - 1) for p in ("weight", "bias") if getattr(module, p) is not None]
    else:
        members = [(p, 0) for p in ("weight", "bias", "running_mean", "running_var") if getattr(module, p, None) is not None]
    return {(f"{name}.{p}", axis, idxs) for p, axis in members}


def depgraph_groups(family):
    """[(size, {unit: member set}, in_scope)] for every DepGraph group; the prune check runs on a copy."""
    model, _ = build(family, torch.float32)
    x = torch.randn(*FAMILIES[family][1])
    names = {id(m): n for n, m in model.named_modules()}
    dg = tp.DependencyGraph().build_dependency(model, example_inputs=x)
    found = []
    for group in dg.get_all_groups():
        root_dep = group[0][0]
        module, handler = root_dep.target.module, root_dep.handler.__name__
        size = len(group[0][1])
        units, scope = {}, True
        for u in sorted({0, size - 1}):
            members = set()
            for dep, idxs in dg.get_pruning_group(module, root_dep.handler, idxs=[u]):
                layer = dep.target.module
                if type(layer).__name__.startswith("_") and type(layer).__module__.startswith("torch_pruning"):
                    continue  # DepGraph's own bookkeeping nodes (add, concat, split, reshape): no parameters
                one = mapped(layer, names.get(id(layer), "?"), dep.handler.__name__, idxs)
                if one is None:
                    scope = False
                    continue
                members |= one
            units[u] = members
        found.append((size, units, scope and passes_own_check(family, names[id(module)], handler)))
    return found


def passes_own_check(family, module_name, handler):
    model, _ = build(family, torch.float32)
    x = torch.randn(*FAMILIES[family][1])
    try:
        dg = tp.DependencyGraph().build_dependency(model, example_inputs=x)
        module = dict(model.named_modules())[module_name]
        dg.get_pruning_group(module, getattr(dg.get_pruner_of_module(module), handler), idxs=[0]).prune()
        with torch.no_grad():
            model(x)
        return True
    except Exception:
        return False


@pytest.mark.parametrize("family", list(FAMILIES))
def test_our_closed_groups_equal_depgraphs_in_scope_groups(family, capsys):
    _, state = build(family, torch.float32)
    table = open_tensors(save(Path(tempfile.mkdtemp()), state, f"{family}.safetensors"))
    ours = derive(graph_of(family), table).groups
    compared, excluded = 0, []
    for size, units, in_scope in depgraph_groups(family):
        if not in_scope:
            excluded.append(("out of scope", size, sorted(units[0])[:2]))
            continue
        match = [g for g in ours if g["size"] == size and all(
            {(m["tensor"], m["axis"], frozenset(indices(m, u))) for m in g["members"]} >= units[u] and
            {(m["tensor"], m["axis"], frozenset(indices(m, u))) for m in g["members"]
             if any(m["tensor"] == t for t, _, _ in units[u])} == units[u] for u in units)]
        assert len(match) == 1, (family, size, sorted(units[0]), len(match))
        if any(b["reason"] == "graph_output" for b in match[0]["boundary"]):
            excluded.append(("reaches a graph output", size, sorted(units[0])[:2]))
            assert not match[0]["closed"]  # the oracle must never justify an output-reaching group as permutable
            continue
        assert match[0]["closed"], (family, size, sorted(units[0]), match[0]["boundary"])
        compared += 1
    with capsys.disabled():
        print(f"[oracle] {family}: compared {compared}, excluded {excluded}")
    assert compared == EXPECTED_IN_SCOPE[family], (compared, excluded)
