# Roles, coupling groups, opaque boundaries and coverage (v0.3-G3)

Library level, no HTTP, no routes (G4), no edits (G5). Design: [v0.3-graph-semantics.md](v0.3-graph-semantics.md) 3.3-3.5, 4, 6 and 7.1; oracle: [ADR 0001](adr/0001-depgraph-oracle.md). Code: `coupling.py` (torch-free, generic, no tensor name and no model family: the test guard covers it).

`derive(ir, table) -> Derived` takes a GraphIR (G2) and the checkpoint's tensor table and returns `roles`, `groups`, `opaque_ops`, `coverage`, the `rules` version and `warnings`; `Derived.sha256()` is a digest of all of it. It binds first (`graph_mismatch` if the graph does not fit the table).

## How groups are derived

Every activation dimension and every parameter axis is a `Dim`: an ordered list of *pieces* (what `cat` makes), each piece a list of *factors* (what `reshape`/`flatten` make, major to minor), each factor a unit space. Rules tie spaces together (conv in-channels with the producer's out-channels, residual `add`, the bias with its weight rows, `matmul` contraction, ...), `reshape` splits a space into mixed-radix factors, `cat`/`slice`/`split` split it into pieces, and `select`/`unbind` of a packed factor restricts its sibling factors to the chosen slice. That last step is what makes the q and k head dimension a different group from the v head dimension in packed attention, while all three share the heads group.

Each space keeps, for every parameter axis that depends on it, the explicit index set of each of its units; a **group** is a live space with at least two units and at least one parameter member, and the index sets are compressed to the member map of the design (3.4):

```
member = {tensor, axis, offset, block, period, repeat}
indices(u) = { offset + u*block + k*period + j  |  k < repeat, j < block }
```

A group whose index sets are not of that shape is kept but marked open (`non_affine_member`). A packed factor itself (size 3 for qkv) is a role (`packed`, `k`, `factors` on the axis) and never a group.

`group_id = "g_" + sha256(ir_sha256, size, canonical member list)[:24]`: deterministic, independent of tensor values, and different for a different sealed graph.

## Boundaries: when a group is closed

`closed` is `true` only when nothing reaches it that could change the function if its units were reordered. Otherwise `boundary` lists `{node, op, reason}`:

| reason | meaning |
|---|---|
| `graph_output` | the unit dimension reaches a graph output (the output layer, detection heads): permuting it reorders the model's answer. Always open (ADR 0001) |
| `graph_input` | the unit dimension is an input's: permuting input channels changes the function |
| `opaque_op` | an op without a rule, a grouped (non-depthwise) convolution, a reshape that cannot be tracked |
| `data_dimension` | a unit space is tied to a data dimension (sequence positions, a frequency axis in front of an RNN) |
| `indexed_unit` | one unit was picked by `select`/`index` |
| `recurrent_state_used` | a recurrent layer's final hidden/cell state is consumed |
| `recurrent_initial_state` | the initial state is not a zeros constant |
| `non_affine_member` | a member's index set has no `{offset, block, period, repeat}` form |
| `overlapping_group` | backstop: two groups cut the same tensor axis into the same units; both are opened |

Group-norm adds a constraint to the group it normalises: `{kind: within_blocks, n, of}` where `of` lists every block size seen (for two GroupNorms on one stream, 4 and 2) and `n` is their gcd; a block that does not lie inside one unit space (after a `cat` or a reshape) opens the group instead. A `slice` with a step other than 1 is opaque. Selecting the same packed slice twice reuses one restricted space (the head dimension of q, say), and a mark or tie on the whole packed tensor afterwards reaches every restricted copy. An `attn_mask` operand is tied to the head/batch/position units it is indexed by, so a per-head mask buffer moves with its heads and a mask computed from the input opens them. Factor kinds (`hidden`, `heads`, `d_head`, `gate`) are labels for display only; they never drive a decision.

## Rule table v1 and the version guard

Rules are keyed by torch.export aten op names (`RULES_FOR["2.6"]`): conv/conv-transpose (1-3d, depthwise), linear, batch/layer/group norm, embedding, `lstm`/`gru`/`rnn` (layers, directions, `has_biases` and `proj_size` read from the node: with projection, `w_hr` per direction), `cat`, elementwise and broadcasting ops, reshape/view/flatten/unflatten/squeeze/unsqueeze, permute/transpose/t, expand, select/unbind/slice/split/chunk, mean/sum/amax reductions, pooling and upsampling, `matmul`/`bmm`, `scaled_dot_product_attention`, tensor creation. A graph exported with another torch release has no table: every op is opaque, the result carries a warning, and no group is closed.

## Roles, provenance, coverage

Every table tensor gets one entry per axis (`axis: null` for a 0-d tensor): `{axis, role, provenance, basis}` with `provenance` `derived` or `unknown` and `basis` `rule@node` (for example `conv.kernel@n12`, `norm.param@n7`; `counter.buffer@graph` for an integer scalar buffer no node reads; `unbound` for a tensor the graph never receives). Roles: `out_unit`, `in_unit`, `spatial`, `channel`, `vocab`, `position`, `packed`, `scalar`, `counter`, `unknown`. A tensor two rules describe differently becomes `unknown` with both bases.

`coverage = {unit: "axes", derived, declared, inferred, unknown, opaque_ops, tensors: {bound, unused, unbound}, groups: {closed, open}}` and `derived + declared + inferred + unknown` equals the number of role entries (declared and inferred arrive with G5). On the five corpus families every axis of every conv, linear, norm, recurrent, embedding and attention parameter is `derived`; the only `unknown` entries are the tensors G2 reports as unbound.

## Tests

`tests/test_coupling.py`: A5 spot groups (ResNet residual chain, U-Net concat offsets and transposed-conv axes, depthwise conv, ViT heads / q-k / v split with the packed role, CRNN LSTM gates and layer-1 offset columns), the output and input rules on every family, **A3 without G5**: for every closed group of every family (and of an LSTM with `proj_size` and a bias-free LSTM+GRU) the member maps are applied by hand to a random permutation of the units and the forward output must not change; an output group permuted does change it. A6 (`cumsum` on a channel dimension is listed in `opaque_ops`, opens its groups, and permuting them really changes the output), A7 coverage and provenance on every role, group-id determinism (also in a subprocess), the version guard.

`tests/test_depgraph_oracle.py` (A10, marker `oracle`, optional extra `pip install -e ".[oracle]"` = `torch-pruning==1.6.1`; skipped when absent; optional CI job `.github/workflows/oracle.yml`, not required for merge): every in-scope DepGraph group (conv, conv-transpose, linear, norms, add, concat, flatten, depthwise, unwrapped parameters mapped by object identity; passes DepGraph's own prune-and-forward check; no LSTM/attention member) must have exactly one closed group of ours that holds the same (tensor, axis, indices) for every tensor DepGraph knows, for units 0 and size-1; ours may hold more members (a buffer, a free matmul parameter), which only makes it more conservative. Counts on the corpus: ResNet 8 (the `fc` output group is excluded as output-reaching), U-Net 12, YOLO 8, ViT 2 (the two MLP groups; DepGraph is wrong on attention, and its 27-member residual-stream group, with `cls_token` and `pos_embed`, fails its own forward check because the corpus ViT derives the head width from the stream width, so a dedicated test compares it outside the gate), CRNN 1 (the second CNN group runs into the LSTM). Excluded groups are printed.

## Limits

Static shapes only. Grouped (non-depthwise) convolutions, `index`/gather, data-dependent reshapes and every op not in the table are opaque (open groups, listed). Group-norm and the within-block constraint are recorded, not enforced (G5's `permute_units` must honour `constraints`). A weight-tied or `declared` role override is G5/G6.
