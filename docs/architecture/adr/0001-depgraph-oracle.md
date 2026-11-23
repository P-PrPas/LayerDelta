# ADR 0001: Torch-Pruning DepGraph as a dev-only test oracle, scoped

Status: **accepted (proposed by the Senior AI Scientist, v0.3-G1)** · Date: 2026-11-23 · Context: [v0.3 design §2, acceptance A10](../v0.3-graph-semantics.md)

## Decision

Use Torch-Pruning's `DependencyGraph` **only as an optional, dev-only test oracle** for the coupling groups that v0.3-G3 derives. It must **not** be a runtime dependency and it is not imported by `src/`. Its verdicts count only inside a **declared scope**: conv, conv-transpose, linear, batch/layer norm, residual add, concat, flatten into a following layer, and depthwise conv. Outside that scope (packed qkv attention written by hand, LSTM/GRU hidden units, free `nn.Parameter`s, graph outputs) it is wrong or silent, and our own acceptance test A3 (permute every closed group, forward max diff ≤ 1e-5) is the oracle there.

## Evidence (spike, 2026-11-23)

Environment: a throwaway venv in `%TEMP%\ld-depgraph-venv` (outside the repo). `pip install --no-deps torch-pruning` was used, with a `.pth` pointing at the project venv's torch 2.6.0+cu124 so nothing else was downloaded. Windows 11, Python 3.12, CPU. Nothing was added to `pyproject.toml`. The spike scripts lived in a scratchpad. They built the graph for each T6 corpus family, listed every group, and pruned unit 0 of each group in a deep copy to check that a forward pass still runs. That is DepGraph's own consistency check.

### Package facts (verified, not from memory)

| Fact | Value |
|---|---|
| Distribution | `torch-pruning` **1.6.1** (wheel metadata). The module's `__version__` string says `1.6.0`, a packaging inconsistency to keep in mind when pinning and with `importorskip(minversion=...)` |
| License | **MIT** (`License-File: LICENSE`, classifier `License :: OSI Approved :: MIT License`, "Copyright (c) 2019 Gongfan Fang"; AUTHORS: Gongfan Fang, Xinyin Ma). Compatible with any dev-only use |
| Requirements | `torch>=2.0`, `numpy`, Python `>=3.7`; pure Python |
| torch 2.6 | imports and builds graphs with no error or warning that matters, on Windows CPU |
| Mechanism | `DependencyGraph().build_dependency(model, example_inputs=x)` runs a forward on the **live module** and walks autograd. Layer pruners (`ConvPruner`, `LinearPruner`, `BatchnormPruner`, `LayernormPruner`, `GroupNormPruner`, `EmbeddingPruner`, `LSTMPruner`, `MultiheadAttentionPruner`, ...) mutate modules |

### Results on the five corpus families

| Family | build | groups | params in some group | prune-one-unit forward ok | Verdict |
|---|---|---|---|---|---|
| ResNet-like | 0.07 s | 9 | 41/41 | **9/9** | correct: each residual chain is one group (conv2, bn2, downsample conv+bn of a stage, the next stage's conv1/downsample inputs, `fc` input) |
| U-Net-like | 0.02 s | 13 | 36/36 | **13/13** | correct: `up2 out ↔ dec2.0 in` and `up1 out ↔ dec1.0 in` (concat offsets handled), ConvT axis handled (`ConvPruner` uses weight axis 1 for `out` when `transposed`) |
| YOLO-like | 0.03 s | 11 | 31/31 | **11/11** | correct, including depthwise `dw3` tied to `fuse3` and `head3` input |
| ViT-tiny | 0.03 s | 6 | 30/34 | **3/6** | residual stream (17 members incl. LayerNorms, `patch_embed`, `cls_token`/`pos_embed` as "UnwrappedParameter") and both MLP groups are correct. **Both attention groups are wrong**: `qkv out (48) ↔ proj in (16)` ignores the `(3, heads, d_head)` packing, and pruning them breaks the forward pass (`shape '[2, 17, 3, 2, 8]' is invalid`) |
| CRNN-like | 0.10 s | 3 | 26/26 | 3/3 | the CNN groups and `cnn.4 out (16) ↔ rnn in (32)` through flatten are correct, but there are **no groups for the LSTM hidden units** at all (2 layers, bidirectional). `LSTMPruner.prune_out_channels` asserts `num_layers == 1`, and its row indexing for `weight_hh` does not expand over the four gates, which is wrong for our purpose |

Two of the four unbound/unused tensors (`logit_scale`, `tok_embed.weight`) are absent from DepGraph's groups too, which agrees with the torch.export finding in the design (§1). Group listing is per live module, so names of free parameters must be mapped by object identity (`named_parameters`), not by the group's display name.

### Finding that feeds G3

DepGraph makes the output layer's units a group (`fc out`, `head out`). Permuting those units is shape-valid but changes the model's output order. **v0.3 must treat any dimension that reaches a graph output as a boundary**: such groups are `closed: false`, and `permute_units` is refused on them. Add this to the G3 rule set and to A3 (the oracle must never be used to justify output-reaching groups as permutable).

## Options considered

1. **Runtime dependency**, using DepGraph instead of writing G3. Rejected. It needs the live module plus autograd, so user-code execution would leak beyond the one provider seam. It is keyed by module objects, not by tensor names on a content-addressed checkpoint. It mutates for pruning. It has no notion of an opaque boundary or provenance. It is wrong on hand-packed attention and silent on multi-layer LSTMs, exactly the cases our rule table must get right.
2. **Drop it.** Rejected. Inside the declared scope it agrees with the expected structure on every group of three families and is self-consistent (prune-forward checks pass). That makes it a cheap, independent second opinion on the most common coupling patterns (residual chains, concat offsets, ConvT axes, depthwise), where a G3 bug would otherwise hide.
3. **Scoped dev-only oracle** (chosen).

## A10 oracle harness (how tests call it)

- **Dependency**: an optional test extra, added by G3 when the harness lands (not in this PR): `[project.optional-dependencies] oracle = ["torch-pruning==1.6.1"]`. Never in the default `test` extra.
- **Test file** `tests/test_depgraph_oracle.py`, marker `oracle` (registered in pytest config), and at the top:
  ```python
  tp = pytest.importorskip("torch_pruning", minversion="1.6")  # module __version__ says 1.6.0 for the 1.6.1 wheel
  ```
  so the suite skips cleanly when the extra is absent. An optional CI job installs the extra and runs `pytest -m oracle`. It is not required for merge until G3 is stable.
- **Normalization** (both sides become a set of members per unit):
  - DepGraph side: for each group from `dg.get_all_groups()`, take its root `(module, handler)` and, for unit `u ∈ {0, size-1}`, call `dg.get_pruning_group(root, handler, idxs=[u])`. Each `(dep, idxs)` maps to tensor axes by layer type and handler:

    | layer | `out` | `in` |
    |---|---|---|
    | Conv | `weight` axis 0, `bias` axis 0 | `weight` axis 1 (per group) |
    | ConvTranspose | `weight` axis 1, `bias` axis 0 | `weight` axis 0 |
    | Linear | `weight` axis 0, `bias` axis 0 | `weight` axis 1 |
    | BatchNorm | `weight`, `bias`, `running_mean`, `running_var` axis 0 | — |
    | LayerNorm | `weight`, `bias` last axis | — |
    | unwrapped parameter | the param, the axis given by DepGraph, mapped to its state-dict name by object identity | — |

    The result is `{(tensor_name, axis, frozenset(indices))}`.
  - LayerDelta side: the G3 group containing the same unit, expanded with `indices(u)` per member (design §3.4), giving the same set shape.
- **Scope filter** (applied before comparing):
  - every DepGraph member is one of the layer types in the table above;
  - the group passes DepGraph's own check (prune unit 0 in a copy, forward runs);
  - no member is an LSTM/GRU/MultiheadAttention module;
  - the group does not reach a graph output.

  Groups filtered out are listed in the test output, not silently dropped.
- **Assertion**: for every in-scope DepGraph group, LayerDelta has exactly one closed group whose member sets for `u = 0` and `u = size-1` are equal. Expected on the corpus: ResNet 8/9 (the `fc out` group is excluded as output-reaching), U-Net all except `head` out, YOLO all except the three detection heads' out, ViT the residual-stream and two MLP groups (attention excluded), CRNN one CNN group (the second one runs into the LSTM, so the "no LSTM member" filter excludes it; corrected in G3, the first estimate said two). G3 records the exact expected counts when the harness lands.
- **Out-of-scope cases** (packed qkv heads, LSTM gates and hidden units, output boundaries) are covered by A3 (permutation then forward equivalence) and A5 (explicit expected members), not by the oracle.

## Consequences

- G3 has an independent check on the commonest coupling patterns, at the cost of one optional pinned dev dependency.
- If a future torch breaks Torch-Pruning, the oracle job skips or fails without affecting product code; A3/A5 remain the authoritative tests.
- The output-boundary rule becomes part of the G3 contract.
- Reproduce the spike: create the venv as above, then run the two scratch scripts (build plus per-group prune check, and group listing) against `tests/test_corpus.py` with the repo's `src` on `sys.path`. G3 turns them into the harness, so they are not committed here.
