# Edit programs, recipe v2 and the child writer (v0.2-T3)

Library level (`layerdelta.edit`): no HTTP routes and no catalog registration yet (T4c). Selectors and the response envelope are the ones in [inspect-ops.md](inspect-ops.md) (rev 5).

## Recipe v2

```json
{"schema_version": 2,
 "base": {"checkpoint_sha256": "<64 hex>", "format": "safetensors", "profile_sha256": null},
 "program": [{"op": "scale", "args": {"k": 0.5},
              "select": {"tensor": "features.block2.conv.weight", "shape": [24, 8, 3, 3],
                         "index": {"0": [{"start": 17, "stop": 18}], "1": [{"start": 0, "stop": 8}], "2": [...], "3": [...]}}}],
 "engine": {"layerdelta_ops_revision": 1, "plugins": {}, "numeric": "cpu-f32"},
 "hypothesis": "…",
 "evaluation": null,
 "recipe_hash": "<sha256 of the canonical JSON of everything above>"}
```

- `freeze_recipe(table, base_id, program, hypothesis, evaluation=None)` builds it: each `select` is parsed against the BASE table and stored in the re-submittable resolved form (every axis, sorted merged ranges, `shape` guard), so equivalent spellings (`[3,1,2]`, `{start:1,stop:4}`) give the same `recipe_hash`. Args are canonical floats (`k: 2` becomes `2.0`, `-0.0` becomes `0.0`).
- `verify_recipe(recipe)` is strict (exact key sets, known ops, exact args, `engine` must equal the engine above, hash must match) and needs no table.
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
- Ops run in order on a per-tensor float32 working copy. Overlapping writes are allowed and reported. Values are rounded once per tensor, at the end, into the stored dtype: F32 unchanged, F16 IEEE round-to-nearest-even, BF16 round-to-nearest-even on the dropped 16 bits. Only the union of the selections is re-encoded; every other byte (including NaN payloads and `-0.0`) is copied from the parent unchanged.
- Compute is elementwise numpy float32 on the CPU (no reductions), so the same recipe on the same bytes gives the same child, bit for bit.

## plan, validate, apply

- `plan(table, recipe, base_id=, sources=, blobs=, limits=)` is a dry run with no writes. It runs the whole program and returns per op `selection_id`/`selected_elems`, per tensor `touched_elems` (union), `changed_elems` (bitwise), `before_sha`, `after_sha`, `overlaps` (pairs of ops and the shared element count), `touched_elems_total`, `changed_elems_total`, `noop` and `child_bytes`.
- `validate(...)` is plan reported as data: `{"valid": bool, "checks": [...], "error": {code, message, field} | null}`.
- `apply(table, recipe, out_path, base_id=, ..., allow_noop=False)` runs `plan` first (so a bad program writes nothing), refuses a program that changes no stored value (`no_op`; with `allow_noop` the child is byte-identical to the parent, e.g. `scale` by 1), writes the child to a hidden partial file next to `out_path`, verifies it with the T1 reader, then renames it into place. A crash or error at any point leaves no child and no partial file. The child has the parent's exact header and layout; untouched tensors are copied byte ranges.
- The returned envelope has `side_effects: [{"kind": "checkpoint_created", "id": <child sha>, "bytes": n}]` and `result.manifest`:

```json
{"schema_version": 1, "parent": "<base id>", "child_sha256": "…", "recipe_hash": "…", "child_bytes": 212200,
 "engine": {…},
 "touched": [{"tensor": "…", "elems": 72, "changed_elems": 72, "before_sha": "…", "after_sha": "…"}]}
```

`before_sha`/`after_sha` are sha256 of the whole tensor's stored bytes; `elems` is the size of the selection union.

## Validation checks (all before any write)

Schema and hash; selectors in range with the guard shape; editable dtype; blobs and sources present and consistent; finite at every op (`non_finite`, field `program[i]`); no overflow when rounding to F16/BF16 (`dtype_overflow`); shape preserved (the header is reused verbatim); the changed bytes are a subset of the selection union (asserted per tensor); caps (`Limits`: 256 ops, 2^30 selected elements, 2^27 elements per touched tensor, 256 MiB per blob). Error codes are the `InspectError` codes of inspect-ops plus `unsupported_engine`, `noncanonical_recipe`, `recipe_hash_mismatch`, `base_mismatch`, `blob_missing`, `blob_mismatch`, `blob_size`, `source_missing`, `incompatible_source`, `empty_selection`, `no_op`, `output_exists`, `child_invalid`, `nondeterministic_edit`.

## v1 -> v2

`translate_v1(recipe_v1, table=None)` maps a weight-mode v1 recipe (`{mode: weight, target_id, channel_index, operation: zero|scale}` plus its `mapping`) to a one-op program: `select = {tensor: mapping.parameter_name, index: {mapping.channel_axis: [channel_index]}}`, `base.checkpoint_sha256 = dependencies.checkpoint_sha256`, `hypothesis` and `evaluation` carried over. The v1 `recipe_hash` is verified first. Pass `table` to get the canonical resolved form. Activation-mode v1 recipes raise `unsupported_mode` and stay on the v1 path until v0.4. Equivalence is tested: for `examples/recipes/zero-block2-channel17.json` (and block1/scale variants) the v2 child is byte-identical to the v0.1 `checkpoint_export` child.

## Known limits

- Single-file base only; the child is a single file.
- A touched tensor is held in memory as float32 (about 3 to 4 times its elements), and apply computes each touched tensor twice (validation pass, write pass).
- Ops that are not in the first set (noise, permute_units) are not implemented.
- Sequential ops are not re-rounded between ops (one rounding at the end), so applying two ops as two separate children is not bit-identical to one two-op program on F16/BF16.
