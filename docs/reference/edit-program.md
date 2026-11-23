# Edit programs, recipe v2 and the child writer (v0.2-T3)

Library level (`layerdelta.edit`): no HTTP routes and no catalog registration yet (T4c). Selectors and the response envelope are the ones in [inspect-ops.md](inspect-ops.md) (rev 5).

## Recipe v2

```json
{"schema_version": 2,
 "base": {"checkpoint_sha256": "<64 hex>", "format": "safetensors", "profile_sha256": null},
 "program": [{"op": "scale", "args": {"k": 0.5},
              "select": {"tensor": "features.block2.conv.weight", "shape": [24, 8, 3, 3],
                         "index": {"0": [{"start": 17, "stop": 18}], "1": [{"start": 0, "stop": 8}], "2": [...], "3": [...]}}}],
 "engine": {"layerdelta_ops_revision": 1, "plugins": {}, "numeric": "cpu-f32-round-per-op"},
 "hypothesis": "…",
 "evaluation": null,
 "recipe_hash": "<sha256 of the canonical JSON of everything above>"}
```

- `freeze_recipe(table, base_id, program, hypothesis, evaluation=None)` builds it: each `select` is parsed against the BASE table and stored in the re-submittable resolved form (every axis, sorted merged ranges, `shape` guard), so equivalent spellings (`[3,1,2]`, `{start:1,stop:4}`) give the same `recipe_hash`. Args are canonical floats (`k: 2` becomes `2.0`, `-0.0` becomes `0.0`).
- `verify_recipe(recipe)` is strict (exact key sets, known ops, exact args, `engine` must equal the engine above, hash must match) and needs no table. A `select` must be exactly the resolved form: keys `tensor, shape, index`; every axis `0..ndim-1` present; each axis a non-empty list of `{start, stop}` with `0 <= start < stop <= shape[i]`, sorted, disjoint and not adjacent; anything else is `noncanonical_recipe` (so one selection has one hash). `evaluation` must be `null` or a JSON object (no NaN/Inf). When a recipe is resolved against a base table, its selector must also equal that table's own resolution.
- `recipe_hash = sha256(canonical_json(recipe without recipe_hash))` (sorted keys, no spaces, no NaN).
- Selectors are always resolved against the base tensor table, never against an intermediate state, so resolution does not depend on op order.

## Ops

| op | args | new value of each selected element `x` (float32) |
|---|---|---|
| `zero` | none | `+0.0` |
| `scale` | `k` | `x * k` |
| `set` | `value` or `blob` | `value`, or the blob's element at the same position |
| `add` | `value` or `blob` | `x + value` or `x + blob` |
| `clamp` | `min` and/or `max` | `min(max(x, min), max)` |
| `lerp` | `source`, `t` in [0, 1] | `x + (b - x) * t`, with `b` from the source checkpoint; `t = 0` gives `x`, `t = 1` gives `b` exactly |
| `copy_from` | `source` | `b`, the same selection of the same tensor in the source checkpoint |

- Numbers must be finite and inside float32 range. `blob` is `sha256:<64 hex>` of a flat little-endian float32 array holding the selected elements in row-major selection order; `source` is a checkpoint id (64 hex). The caller supplies `blobs={blob_id: bytes}` and `sources={checkpoint_id: TensorTable}`; blobs are verified against their hash and size, sources must have the same dtype and shape for that tensor.
- Only `F32`, `F16`, `BF16` tensors can be edited (`read_only_dtype`). A sharded base is `unsupported_base` for now.
- **Numeric contract `cpu-f32-round-per-op`.** Ops run in order on a per-tensor float32 working copy. After EVERY op the result is rounded into the stored dtype (F32 unchanged, F16 IEEE round-to-nearest-even, BF16 round-to-nearest-even on the dropped 16 bits) and written back, so the next op sees exactly what a stored child would hold. Consequences: a program `[a, b]` equals `apply(apply(a), b)` bit for bit (history, undo and step-by-step replay agree), and overflow is detected at the op that causes it (`dtype_overflow`, field `program[i]`). Overlapping writes are allowed and reported. Only the union of the selections is re-encoded; every other byte (including NaN payloads and `-0.0`) is copied from the parent unchanged.
- Compute is elementwise numpy float32 on the CPU (no reductions), so the same recipe on the same bytes gives the same child, bit for bit.

## plan, validate, apply

- `plan(table, recipe, base_id=, sources=, blobs=, limits=)` is a dry run with no writes. It runs the whole program and returns per op `selection_id`/`selected_elems`, per tensor `touched_elems` (union), `changed_elems` (bitwise), `before_sha`, `after_sha`, `overlaps` (pairs of ops and the shared element count), `touched_elems_total`, `changed_elems_total`, `noop` and `child_bytes`.
- `validate(...)` is plan reported as data: `{"valid": bool, "checks": [...], "error": {code, message, field} | null}`.
- `apply(table, recipe, out_path, base_id=, ..., allow_noop=False)` runs `plan` first (so a bad program writes nothing), refuses a program that changes no stored value (`no_op`; with `allow_noop` the child is byte-identical to the parent, e.g. `scale` by 1), writes the child to a hidden partial file next to `out_path`, verifies it with the T1 reader, then renames it into place. A crash or error at any point leaves no child and no partial file; publication never replaces an existing file (`output_exists`; a hard link makes the child visible, the partial is then dropped). A hard crash can leave a hidden `.<name>.partial-<id>` file: the service sweeps those at start (T4c). The directory is not fsynced after the rename (Linux durability only). The child has the parent's exact header and layout; untouched tensors are copied byte ranges.
- The returned envelope has `side_effects: [{"kind": "checkpoint_created", "id": <child sha>, "bytes": n}]` and `result.manifest`:

```json
{"schema_version": 1, "parent": "<base id>", "child_sha256": "…", "recipe_hash": "…", "child_bytes": 212200,
 "engine": {…},
 "touched": [{"tensor": "…", "elems": 72, "changed_elems": 72, "before_sha": "…", "after_sha": "…"}]}
```

`before_sha`/`after_sha` are sha256 of the whole tensor's stored bytes; `elems` is the size of the selection union.

The `apply` result also carries `result.diff_hint = {checkpoint_id: <parent>, other: <child>, selectors: [...], selectors_truncated}` (v0.4): the pair to give `diff`, and the resolved selector of each op, deduplicated and in program order (at most 20). `diff` takes one selector per call; with none it compares the whole table and lists only the tensors that changed. A replay returns the same hint.

## Coupled-group edits and ops revision 2 (v0.3-G5)

A program entry may name a **group** instead of a tensor: `select: {"group": "g_...", "units": [..] | {start, stop}, "members": {roles, exclude}}` (the selector of [inspect-ops-rev7.md](inspect-ops-rev7.md)). `freeze_recipe(..., graph=view)` needs a `GraphView` of the base table for it and **expands the entry into one tensor-local op per member**: `select` stays the canonical tensor-local selector of that member's slice, and the group becomes provenance, `via: {group, ir_sha256, units, members, member: {tensor, axis}, perm?}` (inside the hashed recipe, but never needed to run it). So verifying, planning and applying a recipe never need the graph, and the `apply` result equals a hand-written tensor-local program: zeroing unit 3 of a group gives the same child bytes as zeroing the same slices in the state dict by hand (every member, normalisation running statistics included; use `members` to leave some out).

- **`permute_units`** `{"perm": [..]}` on a group selector: the whole group (no `units` or `members` filter), `perm[j]` = the old unit that moves to position `j`. It is expanded per member into `{op: permute_units, args: {axis, perm}}` where `perm` is the index-level gather along `axis` of that member's selection (it handles `block`, `period` and cat offsets), a **pure gather**: values are moved, never recomputed, no rounding step changes a bit. A group that is open (`group_not_closed`, field `program[i].select.group`; graph outputs and inputs, opaque ops, data dimensions), a permutation that crosses group-normalisation blocks (`constraint_violation`, field `program[i].args.perm`) and a `perm` that is not a permutation of `0..size-1` (`invalid_permutation`) are refused. Only this op needs a closed group: `zero`, `scale`, `set`, `add`, `clamp`, `lerp` and `copy_from` on an open group are allowed and `plan` lists a `group_open` warning with the boundary. `blob` arguments are not available through a group (one blob holds values for one selection).
- A tensor-local `permute_units {axis, perm}` is the replay form (it is what a recipe holds); a user program may only reach it through a group selector, or through a submitted recipe whose `via` re-expands identically (below). `verify_recipe` requires every `permute_units` to carry its `via` with the unit `perm`.
- **A submitted recipe is checked against the graph.** `via` is provenance, and execution never reads it, so nothing stops a client from submitting a recipe with a made-up one. Whenever a graph whose `ir_sha256` equals the `via`'s is available (over HTTP: always, when the checkpoint has one), `plan`, `validate` and `apply` expand each group edit again from its `via` and require the recipe's ops to equal the result exactly (`noncanonical_recipe`, field `program[i].via`, also for an unknown or now-open group). `plan` reports `via_verified` per group: `true` after that check, `false` when no matching graph was available (the recipe still replays: it is a valid tensor-local program).
- **Recipe size.** A group `permute_units` stores one index-level `perm` per member and axis, so a recipe grows with members x axis length (a 4096-wide residual stream shared by about 300 members is roughly 1.2 million integers). Keep that in mind for request body limits.
- **Engine.** New recipes that use a group, `via` or `permute_units` carry `engine.layerdelta_ops_revision: 2` (`edit.ENGINE_V2`). Everything else still freezes with revision 1 and the same hash as in v0.2. `verify_recipe` accepts both (`SUPPORTED_ENGINES`); a revision 1 recipe that uses revision 2 features is `unsupported_engine`. `fingerprints.op_revision` is the recipe's revision.
- **plan** reports, for a group-expanded recipe: each op's `via`; `groups` (one per group edit: units, members, and, when a graph is passed, `closed`, `boundary`, `label`, `notes`); `moved_units` (`{group, count, total, units: [{from, to}] (at most 1000), truncated}`) and `warnings`. The plan itself never depends on the graph.
- Over HTTP, `plan`, `validate` and `apply` take group selectors in `program` (`weights:read`, `weights:write` as before); with no graph attached they answer `no_graph` (`validate` reports it as `valid: false`).

## Validation checks (all before any write)

Schema and hash; selectors in range with the guard shape; editable dtype; blobs and sources present and consistent; finite at every op (`non_finite`, field `program[i]`); no overflow when rounding to F16/BF16 (`dtype_overflow`); shape preserved (the header is reused verbatim); the changed bytes are a subset of the selection union (asserted per tensor); caps (`Limits`: 256 ops, 2^30 selected elements, 2^27 elements per touched tensor, 256 MiB per blob). Error codes are the `InspectError` codes of inspect-ops plus `unsupported_engine`, `noncanonical_recipe`, `recipe_hash_mismatch`, `base_mismatch`, `blob_missing`, `blob_mismatch`, `blob_size`, `source_missing`, `incompatible_source`, `empty_selection`, `no_op`, `output_exists`, `child_invalid`, `nondeterministic_edit`.

## v1 -> v2

`translate_v1(recipe_v1, table)` maps a weight-mode v1 recipe (`{mode: weight, target_id, channel_index, operation: zero|scale}` plus its `mapping`) to a one-op program: `select = {tensor: mapping.parameter_name, index: {mapping.channel_axis: [channel_index]}}`, `base.checkpoint_sha256 = dependencies.checkpoint_sha256`, `hypothesis` and `evaluation` carried over. The v1 `recipe_hash` is verified first. The table is required: it fixes the canonical resolved selector, so one v1 recipe has exactly one v2 hash. Activation-mode v1 recipes raise `unsupported_mode` and stay on the v1 path until v0.4. Equivalence is tested on synthetic random-init parents and, when the reference fixture exists, on the real trained checkpoint (`test_real_reference_checkpoint_v1_translation_equals_v01_export`; set `LAYERDELTA_FIXTURE_DIR` to a built `.layerdelta/reference`, or `LAYERDELTA_REQUIRE_FIXTURE=1` to make a missing fixture a failure, as CI does; a fixture whose digest differs from the pinned one in the example recipe is reported with a warning and the recipe is rebased onto the built checkpoint, because the claim is about the SAME parent - set `LAYERDELTA_REQUIRE_PINNED_DIGEST=1` to fail on drift instead): for `examples/recipes/zero-block2-channel17.json` and a block1/scale variant the v2 child is byte-identical to the v0.1 `checkpoint_export` child.

## Known limits

- Single-file base only; the child is a single file.
- Memory: a touched tensor costs about `elements x (4 work + 1 mask + 2 to 4 new bits + itemsize parent bytes)` at once, and `apply` computes each touched tensor twice (validation pass, write pass). `Limits.max_tensor_elems` bounds it.
- Noise ops are not implemented. `permute_units` is (ops revision 2, below).
