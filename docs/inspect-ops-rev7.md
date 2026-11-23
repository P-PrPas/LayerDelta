# inspect-ops rev7: graph, groups and group selectors (v0.3-G4)

The contract the architecture navigator (G8) builds on. Additive over [rev6](inspect-ops.md); design context in [v0.3-graph-semantics.md](v0.3-graph-semantics.md) section 5 and 7.1 and in [architecture-navigator.md](design/architecture-navigator.md) section 7. Code: `graphview.py` (the view), `group_ops.py` (group selectors), `graph_ops.py` (graph/groups/node envelopes), `v2.py` (routes), `client.py`.

## Compatibility and capabilities

- **With no graph attached, every rev6 response is byte-identical.** `describe` gains keys only when a graph is attached. `OP_REVISION` stays 1. All envelopes are the v1 envelope (`schema_version, resolved, result, fingerprints, side_effects, truncated, cursor`); every cursor is HMAC-signed over HTTP.
- A graph belongs to a **table signature** (names, dtypes, shapes). A checkpoint inherits the newest graph its signature carries, so every child of an attached checkpoint has the same graph and groups without doing anything.

| Route | Capability | Notes |
|---|---|---|
| `GET /checkpoints/{id}/graph` | `model:read` | attach card when nothing is attached; summary + nodes or modules when attached |
| `GET /checkpoints/{id}/graph/nodes/{node_id}` | `model:read` | one node, its rule or why it is opaque |
| `GET /checkpoints/{id}/groups` | `model:read` | paged groups with filters |
| `POST /checkpoints/{id}/graphs` | **human only** (403 for an agent token) | attach: returns `202` and an attach job |
| `GET /graph-attachments/{attach_id}` | **human only** | attach job status |
| `describe`, `read-tensor`, `stats`, `diff`, `query` | unchanged | group selectors need what the route needs (`weights:read` for values) |

Structure (graph, groups, nodes) is `model:read`: it carries names, shapes, roles and boundaries, never values. `no_graph` (HTTP 422, field `checkpoint`, or `selector.group` / `where.group` for selectors) answers every graph-dependent call when nothing is attached.

## describe with a graph

`result.graph = {ir_sha256, provider, table_signature, attached_at, providers: [{name, enabled, reason, models?}], reusable: [{ir_sha256, provider, checkpoints}]}`.
Each row gains `roles: [{axis, role, provenance, basis, factors?, k?}]` (`axis: null` for a 0-d tensor), `binding: "bound" | "unused" | "unbound"` and `groups: [{group_id, axis}]`. `coverage` becomes `{unit: "axes", derived, declared, inferred, unknown, opaque_ops: [{op, count}], tensors: {bound, unused, unbound}, groups: {closed, open}}`.

Roles: `out_unit`, `in_unit`, `spatial`, `channel`, `vocab`, `position`, `scalar`, `counter`, `packed` (an `out_unit` axis made of factors: `factors: [{size, name}]`, `k` the packed size), `unknown`. `provenance` is `derived` or `unknown`; `basis` is `rule@node` (`conv.kernel@n12`, `norm.stat@n7`), `unbound` or `counter.buffer@graph`.

## GET .../graph

Query: `view=nodes|modules` (default nodes), `module_prefix`, `op` (glob over the aten op name, nodes only), `limit` (nodes 500/2000, modules 500/2000), `cursor`.

- **Nothing attached**: `result = {graph: null, providers, reusable}` (the attach card). `providers`: `builtin` (enabled, `models` lists the registered names), `torch_export` and `onnx` (`enabled: false`, with a `reason`). `reusable`: graphs attached to another layout that shares this table's **family signature** (names, shapes, dtype family). Each entry `{ir_sha256, provider, checkpoints}` is an **offer only**: nothing is attached until a human accepts it (below). A layout that differs in a name or a shape is never offered.
- **Attached**: `result = {graph: {...as in describe}, coverage, warnings, counts: {nodes, params, inputs, outputs, groups}, inputs, outputs, view, matched, offset, nodes | modules}`.
  - node row: `{id, op, module_path, module_type, outputs: [{value, kind, shape?}], rule, opaque_reason}`. `rule` is the aten op name the rule table applied; `opaque_reason` is `no_rule` (op not in the table, or an unknown torch release) or `untracked_dimensions` (a rule gave up, for example a reshape that mixes factors). Exactly one of them is `null`.
  - module row: `{path, parent, type, node_count, tensor_count, axes: {derived, declared, unknown}, opaque_count, open_groups}`; counts cover the module's whole subtree; the root is `path: ""`. `module_prefix` keeps a subtree. A `module_path` is only a real submodule of the exported model (never a Dynamo source string).

`GET .../graph/nodes/{node_id}`: the node row plus `args`, `kwargs`, `parameters` (tensors it consumes) and `groups` (ids of groups whose members it uses or whose boundary it is). Unknown id: `unknown_node` (field `node_id`).

## GET .../groups

Filters: `tensor` (exact name), `node` (a node id: groups whose boundary is that node, whose activation spans touch its values, or whose members it consumes), `module_prefix` (members under that module path), `closed` (true/false), `kind`; `limit` (100, max 500), `cursor`. Result `{graph, matched, offset, groups: [...]}`, ordered as derived (deterministic).

Group record:

```
{group_id, size, closed, kind, label, constraints, notes: [str],
 boundary: [{node, op, reason, output?}],
 members: [{tensor, axis, role, offset, block, period, repeat}],
 spans: [{value, dim, offset, block, period, repeat}], spans_total, spans_truncated}
```

- `group_id`: `g_` + 24 hex of sha256(ir_sha256, size, canonical member list): deterministic, independent of tensor values and of dtype, never an ordinal.
- Units and indices: `indices(u) = { offset + u*block + k*period + j | k < repeat, j < block }` for `u < size`, for **members** (indices into a parameter axis) and **spans** (indices into dimension `dim` of activation `value`; `value` ids are the graph's `v<k>`).
- `closed`: true only when reordering the units cannot change the function. Otherwise `boundary` says why: `graph_output` (+ `output`, the graph output index), `graph_input`, `opaque_op` (+ the node and op), `data_dimension` (tied to positions or a frequency axis), `indexed_unit`, `recurrent_state_used`, `recurrent_initial_state`, `non_affine_member`, `overlapping_group`. Every group that reaches a graph output is open.
- `kind`: `channel`, `head`, `qk_dim`, `v_dim`, `hidden`, `position`, `vocab`. `label`: `<module path> <kind words>`, from the first member that produces the units (`blocks.0.attn.qkv heads`). `notes`: plain sentences, one per boundary reason, plus the norm-statistics and group-normalisation warnings. `constraints`: `{kind: "none"}` or `{kind: "within_blocks", n, of}`: units may only move inside blocks of `n` (the gcd of every group-norm block size seen, listed in `of`).
- `spans` is capped at 200 per record (`spans_total` is the real count).

## Group selectors

`{"group": "g_...", "units": [3, 5] | {"start": 2, "stop": 5}, "members": {"roles": ["out_unit"], "exclude": ["norm_stats"]}}`. Keys: `group` (required), `units` (default all), `members` (optional filter), `checkpoint` (must match the call).

- Expansion: each member's `axis` restricted to the union of `indices(u)` of the chosen units, every other axis whole. `resolved.selector = {group, units: [{start, stop}], members, expanded: [tensor-local request forms]}`: `{group, units, members}` from it is re-submittable and gives the same `selection_id`; `expanded` replays with tensor-local selectors and no graph. `fingerprints.selection = "sel_" + sha256({group_id, ir_sha256, units, members})[:32]`: a pure value.
- `members.roles` keeps members whose role is listed (a `packed` axis counts as `out_unit`); `members.exclude: ["norm_stats"]` drops normalisation running statistics. A filter that leaves nothing is `no_members`. Unknown group: `unknown_group`; unit out of range: `index_out_of_range`; unknown filter key: `unknown_field`.
- **read-tensor**: `result = {group: {group_id, size, closed, kind, unit_count}, total_elems, offset, returned_elems, members: [{tensor, axis, role, offset, total_elems, result}]}`; each member `result` is the tensor-local read result. `max_elems` and the cursor page over the members' concatenated elements.
- **stats**: per member, per unit: `members: [{tensor, axis, role, dtype, units: [{unit, count, l1, l2, ...}]}]` (same fields as tensor stats, over that unit's slice of the member). `limit`/cursor page over (member, unit) pairs. `reduce_over_axes` and `bins` are refused (`invalid_argument`).
- **diff** (`selector` is a group): per member per unit rows with the same fields as tensor diff units; both checkpoints must hold the member tensors with equal dtype and shape (`incompatible_tensors`). `values: true` needs a tensor selector (replay `resolved.selector.expanded`).
- **query** `where: {group, units?, members?, stat, op, value}`: ranks the units of the group by one member's statistic. The member is the first `out_unit` member, else the first member, and `matches[0].member_used = {tensor, axis, role, reason}` says which. `matches[0] = {group, member_used, selector, selection_id, units_total, units_truncated, units: [{unit, value}]}`; `selector` is a group selector over the matching units. There is no separate group statistics mode.

## Attaching (human only)

`POST /checkpoints/{id}/graphs` with `{"provider": "builtin", "model": "<registered name>"}` or `{"reuse": "<ir_sha256>"}` (a graph in `reusable`; anything else is `not_offered`). `torch_export` and `onnx` answer `provider_disabled` (422) until they exist. The response is `202` with the job; poll `GET /graph-attachments/{attach_id}`:

`{attach_id, checkpoint_id, status: queued|running|succeeded|failed, stage: queued|building|binding|deriving|done, done, total: 3, unit: "stages", graph: {...} | null, error: {code, message, names[]} | null}`.

A graph that does not fit the checkpoint fails with `graph_mismatch` and the offending tensor `names`. Building runs only code of this project (the builtin registry); a worker, if one is ever added, must not be able to label its graph as another provider's (`provider_mismatch` in the store). **Reuse is never automatic**: the offer is computed from names, shapes and dtype family, and accepting it still proves the binding against the table.

## Client

`AgentClient.graph(checkpoint_id, view=, module_prefix=, op=, limit=, cursor=)`, `.graph_node(checkpoint_id, node_id)`, `.groups(checkpoint_id, tensor=, node=, module_prefix=, closed=, kind=, limit=, cursor=)`. Group selectors go through the existing `read_tensor`, `stats`, `diff` and `query`. Attaching is not on the agent client.

## Where this differs from the section 5 sketch

- `describe` has **no** `graph: null` key when nothing is attached (the byte-identity rule wins); the attach card is `GET .../graph`.
- `reusable` is a list of `{ir_sha256, provider, checkpoints}` (one entry per offered graph).
- `stats` on a group is per member and per unit (a unit may span several indices of a member's axis), not per index.
- Attach also accepts `{"reuse": ir_sha256}`, and `GET groups`/`GET graph/nodes/{id}` answer `no_graph` instead of an empty page.

## Not in G4

Edit ops on group selectors, `permute_units` and the `plan` fields that come with them (`notes`, `moved_units`, value preview) are G5. `repeat_of` (series collapsing) is deferred; a separate group statistics mode is rejected. Activation spans come from the static graph: they say where a group's units sit in each activation, not how large the activation is at run time.
