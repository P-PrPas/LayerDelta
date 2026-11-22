# v0.2-T2 (rev 5, matches the implementation): selector + inspect ops (library level, no HTTP yet)

**Changelog rev5 (PR #13 review)**
- Cursors are bound to the checkpoint(s): the request id includes `checkpoint_id` (both ids for `diff`), or the parsed file identity (name, size, mtime, inode, device) when no id is given. A cursor from checkpoint A is `invalid_cursor` on B.
- `resolved.selector` is now a re-submittable selector (`{tensor, shape, index: {axis: [{start, stop}, ...]}}`, every axis). `parse_selector` accepts range lists and an optional `shape` guard; feeding `resolved.selector` back yields the same `selection_id`. Index lists come back sorted, merged and deduplicated, never in request order.
- `query` match `units` are capped at 1000 with `units_total` / `units_truncated`; the match selector still covers every matching unit.
- `read_tensor` (and `diff` values) compute `range` / `nonfinite_count` on the first page only by default; later pages omit them unless `include_range: true`.
- 64-bit ints (`I64`, `U64`) are emitted as decimal strings (JSON numbers are not exact above 2^53 in JS). Stats over ints still accumulate in float64.
- Renames: stats unit field `nonfinite` -> `nonfinite_count`; describe row `identical_group` -> `identical_bytes_group`; stats `unit` removed (use `index`); `diff` fingerprints `checkpoint` / `checkpoint_b` -> `checkpoint_a` / `checkpoint_b`.
- Cost note (see §6): whole-model `diff` and predicate `query` recompute everything on every page.

Status: implemented in `layerdelta.selector` / `layerdelta.inspect_ops` (T2); aligned with docs/design/tensor-browser.md §7 (decisions in §7 below) and the binding PR #11 review decisions (a)-(e) (folded in rev 3). T4c wraps these 1:1 as HTTP routes. Field names here are what the code will use.
Modules: `layerdelta.selector` (selector parsing/canonicalization/enumeration) and `layerdelta.inspect_ops` (`describe`, `read_tensor`, `stats`, `diff`, `query`).

## 1. Selector (tensor-local only in v0.2)

```json
{"checkpoint": "sha256:<hex>",            // optional; if present must equal the checkpoint the call is bound to
 "tensor": "features.block2.conv.weight",  // required, exact name from describe
 "index": {"0": [17],                      // optional; key = axis as decimal string "0".."ndim-1"
           "1": {"start": 0, "stop": 8}}}  // omitted axis = the whole axis
```
`IndexSpec` per axis is either
- a non-empty list of ints, each `0 <= i < dim` (duplicates collapse, order ignored), or
- `{"start": int, "stop": int}` with `0 <= start < stop <= dim` (half-open, step 1 only). This is the rectangular-window form for 2D browsing: deterministic, no cursor needed.

Rules: no negative indices; no extra keys (`unknown_field`); `group`/`units` keys (v0.3 coupled selectors) are rejected as `unsupported_selector`; a 0-d tensor takes `index: {}` or no `index`; max 100,000 indices in one list.

Also accepted per axis: a list of `{start, stop}` ranges (unioned), and an optional `"shape": [...]` guard that must equal the tensor shape (`invalid_argument`, field `selector.shape`). Lists are sorted, merged and deduplicated; the request order is not preserved.

Canonical form (what `selection_id` hashes; the echoed `resolved.selector` is the same selection as a request, see below):
```json
{"tensor": "features.block2.conv.weight", "shape": [24, 8, 3, 3],
 "index": {"0": [[17, 18]], "1": [[0, 8]], "2": [[0, 3]], "3": [[0, 3]]}}   // every axis, sorted, merged [start,stop) ranges
```
`resolved.selector` (re-submittable, same `selection_id`):
```json
{"tensor": "features.block2.conv.weight", "shape": [24, 8, 3, 3],
 "index": {"0": [{"start": 17, "stop": 18}], "1": [{"start": 0, "stop": 8}], "2": [{"start": 0, "stop": 3}], "3": [{"start": 0, "stop": 3}]}}
```
`selection_id = "sel_" + sha256(canonical JSON, sorted keys, no spaces)[:32]`. It does not include the checkpoint id (a selection is a pure value); the checkpoint rides in `fingerprints`. Elements are always enumerated in row-major order over the selected indices.

## 2. Response envelope (all five ops)

```json
{"schema_version": 1,
 "resolved": { ...inputs with defaults filled in; resolved.selector is re-submittable... },
 "result": { ...op specific... },
 "fingerprints": {"checkpoint": "sha256:…" | null, "selection": "sel_…" | null, "op_revision": 1},
 "side_effects": [],
 "truncated": false,
 "cursor": null}
```
- `side_effects` is always `[]` for these ops. `diff` has `fingerprints.checkpoint_a` and `checkpoint_b` instead of `checkpoint`.
- Output is canonical JSON (`to_json`: sorted keys, no spaces, `allow_nan=false`); repeated calls on the same bytes are byte-identical. Nothing in a response depends on cache state (e.g. describe never reports a lazily cached sha256).
- `truncated: true` iff more data follows; then `cursor` is a non-null opaque string. Pass it back with the same request to continue. A cursor is bound to op + canonical request + checkpoint (both checkpoints for `diff`; the parsed file identity when no `checkpoint_id` is given), so reuse on a different request or checkpoint is `invalid_cursor` (field `cursor`). Unsigned at library level; T4c may HMAC.
- Every op takes an optional `checkpoint_id` (the id the caller bound the table to; echoed in fingerprints).

## 3. Errors (structured, raised as `InspectError`; HTTP layer maps to 422)

```json
{"code": "index_out_of_range", "message": "index 24 is outside axis 0 of size 24", "field": "selector.index.0[0]"}
```
Codes: `unknown_tensor` (field `selector.tensor`), `unknown_field`, `unsupported_selector`, `invalid_axis` (field `selector.index.<k>`), `invalid_index` (negative, non-int, bool, empty list), `invalid_range`, `index_out_of_range`, `cap_exceeded`, `invalid_cursor`, `invalid_argument` (any other request field), `incompatible_tensors`. `field` is a dotted path into the request (`selector.index.1.stop`, `max_elems`, `where.stat`).

## 4. Float formatting

- F16/BF16/F32/F8 values are JSON numbers using the shortest decimal that round-trips in the tensor's own dtype (F16 `0.1` stays `0.1`, not `0.0999755859375`). F64 uses Python repr. `-0.0` is preserved.
- Non-finite values appear as the strings `"NaN"`, `"Inf"`, `"-Inf"` (the designer's encoding) wherever a value array can contain them.
- `I8..I32`, `U8..U32` are JSON ints; `I64`/`U64` are **decimal strings** (exact; JS numbers are not exact above 2^53); BOOL is `true`/`false`. Stats and diff over integers accumulate in float64, so beyond 2^53 they are approximate (`accumulate_dtype` says so).
- Stats/diff numbers are accumulated in float64 **from the stored values** (widened exactly) with numpy pairwise sums over fixed 1 Mi-element chunks (thread-count independent), printed with Python repr. Every stats/diff `result` declares `"accumulate_dtype": "float64"`. Undefined or non-finite metrics are explicit `null`, never `NaN`.

## 5. Ops

### describe
Request: `{"name_glob": "features.*"?, "limit": 5000 (1..20000), "cursor": null}` (fnmatch, case-sensitive). Never reads tensor data, never hashes.
```json
"result": {"checkpoint": {"format": "safetensors", "sharded": false, "shard_count": 1, "file_bytes": 212200,
                          "tensor_count": 8, "element_count": 53050,  // elements of all tensors, buffers included (not "parameters")
                           "metadata": {"format": "pt"}},
  "coverage": {"unit": "axes", "derived": 0, "declared": 0, "inferred": 0, "unknown": 21, "opaque_ops": []},   // counts AXES (not tensors) by role provenance
  "identical_bytes": null,                                                                           // deferred job, see §7
  "dtypes": {"F32": 8}, "matched": 8, "offset": 0,
  "tensors": [{"name": "…", "dtype": "F32", "shape": [24,8,3,3], "numel": 1728, "nbytes": 6912,
               "editable": true, "read_only_reason": null | "dtype_not_editable",
               "shard": null | "model-00001-of-00002.safetensors", "tensor_sha256": null, "identical_bytes_group": null, "roles": []}]}
```
Tensors are sorted by name; `roles` is always `[]` in v0.2: no inferred out/in roles, every axis is `unknown` (binding decision (a); name patterns are wrong for ConvTranspose, GPT-2 Conv1D, Keras layouts). The whole table of a 10k-tensor model fits one response (default page 5000, max 20000); larger models page with `cursor`/`name_glob`.

### read_tensor
Request: `{"selector": Selector, "max_elems": 1000 (1..10000), "include_range": null, "cursor": null}`
```json
"result": {"tensor": "…", "dtype": "F32", "shape": [24,8,3,3], "selection_shape": [1,8,3,3],
           "total_elems": 72, "offset": 0, "returned_elems": 72, "values": [0.1, -0.25, "NaN"],
           "nonfinite_count": 0,
           "range": {"min": -0.21, "max": 0.19, "abs_max": 0.21, "abs_p99": 0.17}}}   // range/nonfinite_count: first page only unless include_range
```
- `values` is `selection[offset : offset+returned_elems]` flat row-major; concatenating pages equals one unbounded read. `max_elems > 10000` → `cap_exceeded`.
- `range` and `nonfinite_count` cover the whole selection (not the page) and are computed on the first page; `include_range: null` (default) means "only when offset is 0", `true` forces it on any page, `false` never (it rescans the whole selection, so it is not repeated per page; clients keep the first-page range) (finite elements for range; `null` if none), so the scale does not jump between windows. `abs_p99` is the exact nearest-rank 99th percentile of |x| over finite elements (ascending, element at position ceil(0.99·n)−1; no interpolation). Computed when `total_elems <= 16,777,216`; above that `range` is `null` and `range_skipped: "selection_too_large"`.

### stats
Request: `{"selector": Selector, "reduce_over_axes": [1,2,3] | null (null = all axes → one summary), "bins": 0 (0, or 1..256), "limit": 2048 (1..2048), "cursor": null}`
Axes not reduced define "units" (reduce over 1,2,3 → one row per out-channel).
```json
"result": {"tensor": "…", "dtype": "F32", "reduce_over_axes": [1,2,3], "unit_axes": [0], "unit_count": 24, "offset": 0,
  "accumulate_dtype": "float64",
  "units": [{"index": {"0": 17}, "count": 72, "nonfinite_count": 0, "min": …, "max": …, "mean": …, "std": …,
             "l1": …, "l2": …, "zero_frac": …, "constant": false}],
  "histogram": {"edges": [… bins+1 …], "counts": [[… bins …], …]}}      // only if bins > 0; counts[i] belongs to units[i]
```
- `index` has the real position on every kept axis (a single-key map when one axis is kept; `{}` when everything is reduced). min/max/mean/std(population)/l1/l2 are over finite elements (`null` if none); `zero_frac = zeros/count`. `constant` is true when `count > 0`, `nonfinite_count == 0` and every element is equal (all-zero bias, all-one norm weight): constant units/tensors are flagged here, separately from any identical-bytes grouping. Zero L2 is reported as a number, never labelled "dead".
- `histogram.edges` are shared by every unit: equal-width over the finite min/max of the whole selection (last bin closed). Needs a pre-pass; requires `total selected elems <= 67,108,864` else `cap_exceeded` (field `bins`).

### diff
`diff(table_a, table_b, request)`. Request: `{"selector": Selector | null, "name_glob": …?, "unit_axis": 0 | null, "values": false, "include_same": false, "limit": 500 (1..1000), "max_elems": 1000, "cursor": null}`

Whole-model mode (no selector): compares every tensor name present in either checkpoint, bitwise. Cost is a full read of both files; callers should narrow with `name_glob`. `unit_axis` may be any axis; tensors with fewer axes get no per-unit field. The listing is paged by `limit` + `cursor` (every call recomputes the whole comparison, so `summary` is always complete); a negative or non-integer `unit_axis` is `invalid_axis`.
```json
"result": {"accumulate_dtype": "float64", "summary": {"tensors_total": 8, "changed": 1, "same": 7, "only_a": [], "only_b": [], "shape_mismatch": [], "dtype_mismatch": [],
                       "changed_elems_total": 72, "max_abs": 0.21},
  "tensors": [{"name": "…", "status": "changed", "dtype": "F32", "shape": […], "numel": 1728,
               "changed_elems": 72, "max_abs": …, "rel_l2": …, "cosine": …, "basis": "values",
               "unit_axis": 0, "changed_units": {"total": 1, "listed": [17], "truncated": false}}]}
```
- `status ∈ same | changed | only_a | only_b | shape_mismatch | dtype_mismatch`. `tensors` lists every non-`same` tensor (all tensors if `include_same`), sorted by name, at most `limit` (then `truncated: true`; `summary` still covers everything). `only_*`/mismatch entries carry just `name`, `status` and the differing dtype/shape.
- **`changed` means bitwise inequality** of the stored bytes: `-0.0` vs `0.0`, different NaN payloads and rounding-only bf16/f16 changes are changed (with delta 0). `max_abs`, `rel_l2 = ‖a−b‖₂/‖a‖₂`, `cosine = a·b/(‖a‖‖b‖)` are over elements finite in both; **`rel_l2` is `null` when ‖a‖ = 0 and `cosine` is `null` when either norm is 0** (e.g. a zero parent unit), and `max_abs`/`rel_l2` are `null` when no element is finite in both. Equal tensors give `rel_l2: 0` and `cosine: 1` (an all-zero equal tensor still gives `rel_l2: 0`; its `cosine` is `null` because the norm is 0). Only a non-zero difference against an all-zero `a` makes `rel_l2` `null`. Diff aligns by exact name only: a renamed tensor is one `only_a` plus one `only_b`. `basis` is always `"values"` in v0.2 (hash/manifest fast path is T3/T4b). `listed` holds at most 1000 changed unit indices.

Selection mode (selector given; dtype and shape must match, else `incompatible_tensors`): same per-tensor metrics for that selection, plus
- with `unit_axis` (default 0 for ndim ≥ 1, `null` disables): `units: [{"unit": 17, "changed_elems": 72, "max_abs": …, "rel_l2": …, "cosine": …}]`, every unit of the selection along that axis, paginated by `limit` (≤ 1000) + `cursor`.
- with `values: true`: no unit rows are returned (`unit_axis` is ignored); returns, row-major and aligned element for element, `values` (delta `b − a` computed in float64 from the stored values, printed with Python repr; `delta_dtype: "float64"`; non-finite as `"NaN"`/`"Inf"`/`"-Inf"`) and `changed` (list of 0/1, bitwise inequality, so a cell can be `changed: 1` with delta 0), plus `total_elems/returned_elems/offset/truncated/cursor` as read_tensor (`max_elems ≤ 10000`). `range` (`min,max,abs_max,abs_p99`) of the delta over the whole selection is included. Parity rule: the server owns the subtraction and the changed-mask.

### query
Request: `{"name_glob"?, "dtype": ["F32"]?, "min_ndim"?, "max_ndim"?, "editable"?: bool, "min_bytes"?, "max_bytes"?,
 "where": {"stat": "l2", "axis": 0, "op": "lt", "value": 0.01}?, "limit": 100 (1..500), "cursor": null, "max_scan_elems": 134217728}`
`where.stat ∈ l1, l2, max_abs, mean, std, min, max, zero_frac`; `op ∈ lt, le, gt, ge`; stat is computed per unit along `where.axis` over matched tensors; scanning more than `max_scan_elems` (hard cap 2^30) → `cap_exceeded`.
```json
"result": {"matched": 2, "offset": 0,
  "matches": [{"name": "…", "selector": {"tensor": "…", "index": {"0": [3, 9]}}, "selection_id": "sel_…",
               "total_elems": 144, "units_total": 2, "units_truncated": false,
               "units": [{"unit": 9, "value": 0.002}, {"unit": 3, "value": 0.004}]}]}   // units capped at 1000, ranked; the selector covers all matching units
```
Without `where`: whole-tensor matches ordered by name (`units` absent). With `where`: only tensors with ≥ 1 matching unit, by name; units ordered by value (ascending for lt/le, descending for gt/ge), ties by unit index. `fingerprints.selection` is null (each match carries its own `selection_id`). This is the one code path the browser may use for filtering.

## 6. Cost and caps

Cost per call (no result cache yet): `read_tensor` reads the page, plus one range pass on page 1; `stats` and `query` with `where` scan the selection / matched tensors; whole-model `diff` and predicate `query` compute the full answer on every page and then slice it, so paging them is O(model) per page. Narrow with `name_glob`/`dtype`/`selector` until the T3/T4b hash and manifest fast path exists.

### Caps
read page ≤ 10,000 values (default 1,000) · range pass ≤ 2^24 elements · stats ≤ 2,048 units/page, ≤ 256 bins, histogram pre-pass ≤ 2^26 elements · describe ≤ 20,000 tensors/page · diff ≤ 1,000 tensors listed, ≤ 1,000 units listed/page; query ≤ 1,000 units listed per match, ≤ 10,000 delta values/page · query ≤ 500 matches/page, scan ≤ 2^30 elements · selector lists ≤ 100,000 indices.

## 7. Decisions on the designer's asks (tensor-browser §7)

| # | Ask | Decision |
|---|---|---|
| 7.1 | `range {min,max,abs_max,abs_p99}` over whole selection | **Accept all four** + `nonfinite_count` (binding decision (e)); exact, capped at 2^24 elements (p99 = exact nearest-rank on a float32 abs buffer, ≤ 64 MiB). |
| 7.1 | NaN/Inf encoding | **Accept** designer's strings `"NaN"`, `"Inf"`, `"-Inf"`. |
| 7.1 | Rectangular `{start,stop}` windows, cursor for flat reads | **Accept** as is (selector). Cursor works for any selection. Field names adopted: `selection_shape`, `returned_elems`, `total_elems`. |
| 7.2/7.4 | Shared 16-bin histogram edges | **Accept** (`histogram.edges` shared, `counts[unit]`, any `bins` 1..256; ask for 16). |
| 7.4 | Unit pagination | **Accept** (`limit`/`cursor`, up to 2,048 units/page). |
| 7.4 | Server-side downsampled `profile` for 50k+ units | **Defer to T4c**: purely a view concern; 50k units is ≈ 25 pages at the cap. Needs a metric and bucketing rule the UI has not frozen. When added it must return a per-bucket min-max envelope (not a max-only aggregate) so weak/zero units stay visible. |
| 7.2 | `read_only_reason`, `shard`, `shard_count`, `file_bytes`, `coverage` counts | **Accept** (`dtype_not_editable` is the only reason in v0.2). `param_count` is renamed **`element_count`** (it includes buffers); `coverage.unit` states that it counts axes. |
| 7.2 | `roles[]` with `basis` | **Reject for T2**: `roles` is always `[]`; roles come from v0.3 derived/declared sources, with provenance and basis. |
| 7.2 | `tied {state, hashed, candidates, groups}` pollable, `tensor_sha256`, `tied_group` | **Renamed `identical_bytes` (result) / `identical_bytes_group` (row)** (binding decision (b): identical bytes are not proof of tying; zero biases and ones-weights match trivially). **Defer to T4b/T4c**: needs a pollable job (hashing is lazy and slow; stages size → 64 KiB edge pre-filter → full hash, progress = candidates per stage), and describe must stay data-free and cache-independent. `TensorTable.duplicates()` (T1) is the building block; fields are present but `null`. Constant tensors are flagged separately (`stats.constant`) and the job must exclude constant groups from summary counts. |
| 7.1/7.2 | `display_name`, `origin`, `parent_id`, `recipe_hash`, `created_at`, `list_checkpoints` | **Defer to T4b** (checkpoint store metadata, not tensor-table facts). |
| 7.2 | Import progress stages | **Defer to T4b** (import job). |
| 7.5 | Per-tensor `status` incl. only_a/only_b/shape_mismatch/dtype_mismatch; summary names | **Accept** (names adopted verbatim). |
| 7.5 | `basis: sha256|values` | **Accept field, always `"values"`** in v0.2; the manifest/hash fast path arrives with T3/T4b. |
| 7.5 | Per-unit rows when `unit_axis` given, paginated | **Accept** (selection mode). Whole-model mode lists changed unit indices only. |
| 7.5 | `diff(values: true)` for a window | **Accept** (selection mode, server-side f64 delta **and** a `changed` mask, decision (c)). |
| 7.6 | Name-guessed roles in T2? | **No** (binding decision (a)). T2 emits no inferred roles; every axis is `unknown`, `coverage.unknown` counts them. The frontend must not guess either; 'inferred' styling is reserved for v0.3 profiles/heuristics with a documented basis. |
| PR #11 | Precision, null metrics, scale | **Accept** (d): values are shortest-in-stored-dtype, stats/diff accumulate in float64 from stored values and declare `accumulate_dtype`; undefined `rel_l2`/`cosine` are explicit `null`; `range.min >= 0` lets the UI choose a sequential scale. |
| 7.6 | query filters `name_glob`, `dtype[]`, rank, `editable`, `min/max_bytes` | **Accept** (rank = `min_ndim`/`max_ndim`). **Defer `tied`** filter with tied detection. |
