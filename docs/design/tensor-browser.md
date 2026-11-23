# Tensor browser — design spec (v0.2-T7a)

Owner: Senior UX/UI Designer · Status: **proposed, rev 3** (PM review, AI Scientist data-semantics review and inspect contract rev 5 folded in) · Implements: Roadmap v0.2 acceptance 6 · Built by: T7b (Frontend) on T2/T4c payloads
Mockup: [`tensor-browser-mockup.html`](tensor-browser-mockup.html) (static, open in a browser; uses the Desktop fonts from `desktop/public/fonts`)

## 1. Purpose and non-goals

The tensor browser is the human view of v0.2 layer A: any imported safetensors checkpoint, at tensor-name granularity, without running model code. It answers four questions:

1. **What is in this file?** Thousands of tensors, grouped by name prefix, with dtype, shape, size, editability, identical-bytes and constant flags.
2. **What do the values look like?** A slice of one tensor as a heatmap or number grid, plus per-unit statistics along one axis.
3. **What changed between parent and child?** Which tensors and which units, and by how much.
4. **How much of this is known?** v0.2 has no graph and assigns **no axis roles**: every axis is `axis i · unknown`. Name-based guessing is wrong too often to show (ConvTranspose axis 0 is *in*, GPT-2 `Conv1D` is `[in, out]`, Keras kernels are `[kh, kw, I, O]`), so the UI shows axes by index and never implies a meaning.

Non-goals for v0.2: any editing UI (v0.2 edits go through API/recipes), architecture graph and unit-level editing (v0.3: plan → effect preview → apply), activations and evaluation (v0.4), full program/recipe builder and multi-child diff (v0.6).

**Parity rule.** Every number on screen comes from a v2 payload (`describe`, `read_tensor`, `stats`, `diff`, `query`) that an agent can request too. The browser does not compute statistics or guess roles on its own. This includes Δ values and the changed mask, which `diff` returns server-side. The only client-side work is grouping by name, filtering, sorting, colour mapping and choosing which already-returned values to draw (e.g. the min–max envelope in §4.4 C). Each view can copy its exact selector, so a human can give an agent what they are looking at.

## 2. What we keep from the v0.1 workbench

| Keep | Change |
|---|---|
| Palette (`--p-bg #161c1a`, mint accent `#9fc7b8`, hairline `--bench-line`), dark-only, Instrument Sans for UI text, DM Mono for every identifier and number | Higher density: the workbench is a 15px document layout, and the browser is a 13px tool layout with 28px rows. Big display headings (`h1` 49px) are dropped inside the browser. Its chrome uses the context bar only, so the data gets the space |
| Left rail + context bar shell; new rail entry **Tensors** above *Model explorer* (the existing screens stay as the MNIST fixture flow) | Panels are flush, split by 1px lines, not floating rounded cards. Cards don't work for a three-pane tool |
| Hand-drawn CSS nav glyphs (`.nav-sign`), the `meta-text` mono captions, `role=status` / `role=alert` patterns, skip link, reduced-motion rule | Adds a data-scale family chosen from the value range: diverging for signed data, sequential for non-negative, categorical for bool. The workbench only had sequential grey and activation ramps |

What makes it feel premium is restraint and exactness: mono for all machine values, tabular numerals, right-aligned numbers, one accent, colour used only for data and state. The details carry the craft: a byte-size micro-bar in every row, compact folder chains, a selector line that is always visible, and axis chips that say `unknown` rather than guess. There is no decoration that doesn't encode something.

Principles taken from reference tools (not copied):
- **VS Code explorer / Figma layers:** compact folders (`encoder.down1.conv` as one row when each level has one child), indent guides, type-to-filter, keyboard-first tree.
- **Netron:** the tensor's shape and dtype are the identity. Show them inline in the row, not hidden behind a click.
- **TensorBoard histograms / W&B tables:** distributions as small multiples beat single scalars. Per-unit stats get a profile strip and not only a table.
- **Spreadsheet virtualisation:** never render 10k DOM rows. The tree and grid are windowed.

## 3. Information architecture

```
Rail ─ Tensors (new) · Model explorer · Experiment editor · Run comparison · History · Final eval
Context bar ─ Workspace / <checkpoint name>   [Checkpoint ▾]  [Compare with ▾ | none]  [Import…]
Tensors screen
 ├─ A  Tensor tree          (left, 340–480px, resizable)
 │    search/filter field · facet chips · summary line · virtualised tree
 ├─ B  Inspector            (centre, flexible)
 │    selector line · tensor facts · axis chips (all `unknown` in v0.2; provenance styling reserved for v0.3)
 │    view tabs: Heatmap | Values | (Mosaic for rank-4 with small spatial dims)
 │    axis mapper (rows axis, cols axis, fixed indices) · scale legend · truncation bar
 └─ C  Units panel          (bottom of B, collapsible; or right column ≥1600px)
      unit axis picker · profile strip (one bar per position, min–max envelope when positions > pixels) · sortable stats table
Compare mode (Compare with ≠ none) re-skins A, B and C. It is not a separate screen:
 A gains a Δ column and a "changed only" default filter, B shows parent | child | Δ with a changed mask, C shows per-position diff
```

Compare mode is the same screen on purpose. The user's question moves from "what is this tensor" to "what happened to it", and the selection, the axes and the scroll position stay where they were.

## 4. Screens and states

### 4.1 Import / empty (no checkpoint in workspace)

The centre shows one import card:
- Heading "Open a checkpoint". Path field (mono, full width), **Choose file…** (native picker through the existing desktop bridge) and **Import** (primary).
- Accepted: `.safetensors`, sharded `*.safetensors.index.json`. Refused with a reason before upload: `.pt/.pth/.bin` ("Pickle files can run code when loaded. Convert with `layerdelta convert` yourself, then import the .safetensors."). Drag-and-drop of a file onto the window fills the path.
- One line under the field: "LayerDelta reads only the header and the bytes you inspect. Nothing in the file is executed."
- If other checkpoints exist, list them under the card (name, short sha, tensors, size, imported at).

### 4.2 Loading

Import is staged, and each stage is a row in a `role=status` checklist. The stages are the T4b import job's (`GET /checkpoint-imports/{id}`: `stage`, `done`, `total`, `unit`), shown before the first poll as *"Submitting checkpoint…"*:
1. Validating: the T1 reader checks the header, extents and shards in place (`done / total` files)
2. Copying: the bytes are copied into the workspace and hashed on the way (`done / total` bytes, as a progress bar), so the checkpoint id is the sha256 of exactly the bytes kept
3. Verifying: the copy must parse to the same tensor table (files)
4. Publishing: atomic rename into the catalog → **the entry opens and the tree becomes usable here**. If identical bytes are already in the catalog (`existing: true`), the original entry opens with *"Already in this workspace"* and the reason (same bytes, first imported at …).
5. Finding identical bytes. This runs in the background in three filter stages: group by byte size → compare the first and last 64 KiB → full sha256 of the survivors. Progress shows the candidates left after each stage (`size 1,903 → edges 212 → hashed 40 / 212`). Until it finishes, the Identical facet and flag show *pending* (hollow glyph) and not "none". **v0.2 state** (the job is deferred to T4b/T4c and `identical_bytes` is `null`): stage 5 is not shown, the Identical and Constant facets are disabled with the tooltip *"Not checked yet"*, and no row shows `=b` or `const`.

Stage 5 does not block anything. A multi-GB file can be browsed while the hashing runs.

### 4.3 Errors

Errors are shown inline in the import card. Each has a title, a sentence, the stable `code` in mono and one recovery action. The T1 codes map like this:

| Code(s) | Title | Recovery |
|---|---|---|
| `file_size`, `header_size` | File is larger than the workspace limit | Shows the limit and the file size. Link to the setting |
| `header_json`, `duplicate_field`, `tensor_record`, `metadata`, `dtype_unknown`, `shape` | Header is not valid safetensors | "Re-export the checkpoint", plus a copy-details action |
| `extent`, `extent_mismatch`, `extent_overlap_or_gap`, `data_size` | Tensor data does not match the header | Same as above; the file may be truncated |
| `duplicate_tensor` | Two tensors share a name | Same |
| `index_invalid`, `index_mismatch`, `shard_missing` | Sharded checkpoint is incomplete | Lists the missing shard names |
| `file_changed` | File changed on disk while reading | **Re-import** |

Read errors inside the inspector (e.g. a 422 on a selector) show in the panel and not as a toast. The tree stays usable.

### 4.4 Main browse view (the mockup)

**A · Tensor tree**
- The filter field (`/` to focus) matches name substrings and globs (`*.attn.*`). Facet tokens can be typed or toggled as chips: `dtype:bf16`, `rank:4`, `editable:no`, `size:>10MB`, `identical:yes` and `constant:yes` (both disabled until the identical-bytes job exists), and in compare mode `changed:yes`. Chips and typed tokens stay in sync. Filtering is client-side over the `describe` table (or through `query`, which supports the same filters except identical/constant).
- Summary line, live: `10,412 tensors · 1.21 B elements · 2.42 GB`, plus `· 37 identical-bytes groups` once that job exists, or with a filter: `214 of 10,412 match`. "Elements" (from `element_count`), not "params", because the file also holds buffers such as running stats. Constant tensors (all elements equal: zero biases, ones norm weights, constant buffers) are **excluded** from the identical-bytes count, since equal bytes there say nothing about tying.
- Grouping: names split on `.`. Numeric segments sort naturally (`layers.2` before `layers.10`). Single-child chains collapse into one row (`encoder.down1.conv`). On first load only depth 1 is expanded.
- **Group row:** chevron, segment name, tensor count, total size, and in compare mode the count of changed tensors.
- **Leaf row** (28px), as columns: leaf name · dtype chip · shape (`64×32×3×3`) · size, right-aligned with a log-scaled micro-bar · flags. The flags are a lock (read-only dtype: int/bool/fp8/f64), `=b n` (identical bytes with n−1 other tensors; hover lists them and says *"Same bytes. They may or may not be tied in the model."*), `const v` (constant tensor, with its value) and Δ in compare mode. `=b` and `const` appear only once the identical-bytes job exists. The UI never uses the word "tied".
- The row tooltip and the inspector show the full name. Long names are middle-truncated (`encoder…conv1.weight`) and never end-truncated, because the suffix (`.weight`/`.bias`) is what identifies a tensor.

**B · Inspector**
- **Selector line**, mono and always visible: `decoder.up2.conv.weight [17, :, :, :]`. It can be edited with NumPy-style slice syntax. Next to it are *Copy selector* (the JSON selector from Architecture_Target §3.1) and the `selection_id` once one is resolved.
- **Facts:** dtype, shape, numel, bytes, editable (yes / no + reason), identical-bytes group (with links to the members, once available), constant (yes + value, from a whole-tensor `stats` call with `reduce_over_axes: null`), `tensor_sha256` (shows *not computed* with a "Compute" action, because it is lazy).
- **Axis chips:** one chip per axis, `axis 0`, `axis 1` … all styled `unknown`, because `roles` is always `[]` in v0.2. A single quiet line next to them reads *"Axis meaning is unknown without a graph (v0.3). Axes are shown by index."* No out/in labels and no "guessed" chips anywhere in v0.2. See §6.
- **Axis mapper:** rows axis ▾, cols axis ▾, and an index stepper (`− 17 +`, with direct entry) for every other axis. Defaults by rank:
  - rank 1 → a single strip, rows = 1
  - rank 2 → rows = axis 0, cols = axis 1
  - rank 4 → **Mosaic** when exactly two axes have size ≤ 7. Those two form each small cell and the other two become mosaic rows and cols (window). This is a layout choice from shape alone, works for `[O, I, kh, kw]` and `[kh, kw, I, O]` alike, and labels nothing. Otherwise a Heatmap of the last two axes, others fixed at 0
  - rank 3 and rank ≥ 5 → the last two axes, others fixed at 0
- **Heatmap:** one square cell per value, nearest-neighbour, integer zoom (fit / 1× / 2× / 4× / 8×). Hover shows a crosshair readout `[9, 4, 1, 2] = −0.03125 · bf16 · ulp 0.000244`. Click pins the readout. **The scale is picked from `range`, which covers the whole selection, so it does not jump between windows:** signed data (min < 0 < max) → diverging and symmetric, **±`abs_p99` by default** so one outlier cannot wash out the map (cells beyond it get a small notch mark), with ±`abs_max` one click away; min ≥ 0 → sequential from 0 to the same p99 / max; max ≤ 0 → sequential, reversed; bool → categorical (two colours + legend words); constant → no scale, "All values = 1.0". An option **Centre at mean** re-centres the diverging scale (useful for norm weights around 1). When the selection is over 2^24 elements, `range` is `null` (`range_skipped`): the scale then comes from the loaded window and the legend says *"Scale from this window only"*.
- **Values:** a virtualised number grid in DM Mono, tabular. Each value is printed as the backend sent it: the shortest decimal that round-trips **in the stored dtype** (so bf16 never shows false f32 digits; `-0.0` is kept). Dense mode trims to 4 significant digits; hover and copy give the full value, and hover adds the ulp at that value. Optional cell tint uses the same scale at 35% strength. Non-finite values show as `NaN` / `+Inf` in the alert colour, never as a colour.
- **Truncation bar**, shown whenever `truncated: true`: `Showing rows 0–99 of 2,048 · cols 0–99 of 1,024 (10,000 of 2,097,152 values)` (a read page is at most 10,000 values), a window minimap (a small rectangle inside the full extent) and **Previous / Next window** controls. For 2D windows the browser asks for rectangular index ranges, which are deterministic and resumable. For flat reads it follows `cursor`. The bar never hides: a truncated view must never look complete.

**C · Units panel**
- Unit axis ▾, default axis 0, labelled *"positions along axis 0"*. Never "channel", "output unit" or similar. Every other axis is reduced. Stats are computed by the backend in float64 from the stored values (`accumulate_dtype`), shown in the panel header as `accumulated in float64`.
- **Rank ≤ 1:** there is no other axis to reduce, so each position is a single value and per-position stats (std 0, constant, 1-bar histogram) are noise. For rank 0 and rank 1 the panel is collapsed by default, with the note *"Each position is a single value. See Heatmap or Values."*; no `stats` call is made until the user expands it.
- **Profile strip:** one bar per position (L2 by default; switchable to mean/std/zero fraction/max |x|). It is 64px tall and spans the panel width, drawn on a canvas. When there are more positions than pixels, each pixel draws a **min–max envelope** of the positions it covers: a solid bar from 0 to the minimum plus a lighter band from the minimum to the maximum, so the strip still reads as bars when the positions are uniform, and one weak or all-zero position among thousands dips visibly (max-aggregation would hide it). Click a bar to select that position, which updates the fixed index in B. A position with L2 = 0 is labelled **"all zero"**, never "dead": dead is a functional claim that needs activations (v0.4). The server-side downsample is deferred, so the strip draws from loaded `stats` pages (2,048 positions each) and hatches the not-yet-loaded extent.
- **Stats table** (virtualised, sortable): position · L2 · mean · std · min · max · zero % · non-finite · constant · a 16-bin histogram sparkline on shared edges. Sorting by L2 ascending is one click ("Lowest L2 first").

### 4.5 Compare mode

Turned on by choosing **Compare with** in the context bar. If the open checkpoint is a child with a manifest, its parent is preselected and labelled "parent" (needs `parent_id`, deferred to T4b). Otherwise any checkpoint can be picked, labelled "other". Shortcut `d`.

- **Summary banner** above the tree: `3 of 10,412 tensors changed · 1,153 values · max |Δ| 0.084`, and when tensors don't align, `2 only in parent · 1 only in child · 1 shape changed`, followed by *"Tensors are matched by name only. A renamed tensor shows as one only-in-parent and one only-in-child."* Alignment problems use the alert colour and say what they mean ("Names match but shapes differ. These tensors can't be compared value by value."). **Changed means bitwise inequality of the stored bytes**, and the banner's info tooltip says so. Whole-model diff reads both files in full, so the banner shows a `role=status` progress line while it runs.
- **Tree:** the "changed only" filter is on by default and can be toggled. A Δ column shows a rel-L2 micro-bar plus `changed_elems`. Status glyphs carry text labels too: `=` same, `Δ` changed, `+` only in child, `−` only in parent, `≠` shape/dtype mismatch. Group rows show their changed count.
- **Inspector:** three aligned panes, **Parent | Child | Δ (child − parent)**, at the same axes, window and zoom. Parent and child share one scale. Δ values and the `changed` mask come from `diff(values: true)` (server-side, parity). Δ has its own symmetric scale, and unchanged cells get no colour, so a single changed position looks like a single stripe. **Changed cells are also outlined** (1px `--diff-mask`), and changed cells whose Δ is 0 (−0 ↔ +0, NaN-payload changes, rounding-only bf16/f16 changes) are hatched, so a change is never shown by colour alone. Hovering one pane shows the crosshair in all three.
- **Units panel:** columns become position · changed values · max |Δ| · rel L2 · cosine, sorted by changed values (descending). `rel_l2 = ‖Δ‖ / ‖parent‖` is `null` when the parent position is all zero; `cosine` is `null` when either side is all zero. Both display as `—` with a tooltip giving the reason (e.g. *"child is all zero, cosine undefined"*) and sort last. The profile strip shows max |Δ| per position, so the edited position is the only bar.
- When a tensor is `same`, the panes show "No change (bitwise equal)". In v0.2 that is always decided from values (`basis: "values"`); a hash or manifest fast path arrives with T3/T4b.

## 5. Component inventory

| Component | Notes | States |
|---|---|---|
| `ImportCard` | path field, file picker, accepted-format line, error slot | idle · validating-path · importing (stage list) · error(code) · refused-format |
| `StageList` | `role=status`, ✓ / spinner / pending glyph per stage, background stage marked "continues in background" | pending · running · done · failed |
| `TreeFilter` | text input + facet chips, token parser, match count | empty · typing (debounced 120ms) · no-results ("No tensors match `rank:5`. Clear filter") |
| `TensorTree` | ARIA `treegrid`, windowed (overscan 20), compact folders, indent guides | loading skeleton (12 grey rows, no shimmer under reduced motion) · ready · filtered · empty |
| `TreeRow` (group / leaf) | columns above; selected = accent left rule + `#26342d` ground (matches the rail's `aria-current`) | default · hover · focus-visible · selected · selected+compare · identical-pending |
| `DtypeChip` | mono 11px, outline only; read-only dtypes add a lock glyph | editable · read-only |
| `SizeBar` | 36×3px, log10 bytes scaled to the checkpoint's largest tensor | — |
| `FlagGlyph` | lock, `=b n`, `const v`, Δ, `+ − ≠`; always paired with an accessible name | — |
| `SelectorLine` | editable mono field, validation inline, copy buttons | valid · invalid (field path from 422) · resolving |
| `AxisChip` | `axis i` (+ role from v0.3), provenance styling (§6) | unknown (only state in v0.2) · inferred · derived · declared |
| `AxisMapper` | two selects + steppers; prevents rows = cols | — |
| `HeatmapCanvas` | canvas, integer zoom, crosshair, pinned readout, keyboard cursor | loading (previous image dimmed 50% + status) · ready · truncated · error · all-zero ("All values are 0", scale hidden) · constant |
| `ValueGrid` | windowed table, `aria-rowcount/colcount` | same as heatmap |
| `ScaleLegend` | gradient bar, domain labels, scale mode toggle | diverging ±p99 (notches) · ±max · centre at mean · sequential · categorical (bool) · none (constant) · window-only (range skipped) · shared (compare) |
| `TruncationBar` | extent text, minimap, prev/next | hidden only if `truncated=false` |
| `UnitProfile` | canvas strip, min–max envelope when positions > pixels, hatched unloaded extent, click/arrow to select | — |
| `UnitTable` | windowed, sortable headers (`aria-sort`) | inspect columns · diff columns |
| `CompareBanner` | counts, alignment issues | identical · changed · misaligned |
| `ProvenanceNote` | one quiet line by the axis chips in v0.2; becomes the dismissible explainer for inferred roles in v0.3 | — |

## 6. Provenance (non-negotiable)

Every role carries a provenance (Architecture_Target §2.2). The UI renders the four levels differently, and these renderings are fixed so v0.3 can add roles without a redesign. **In v0.2 `roles` is always `[]`, so only the first row ever appears.**

| Provenance | Chip | Text | v0.2? |
|---|---|---|---|
| `unknown` | plain, muted | `axis 2` | yes |
| `inferred` (profile heuristic with a documented `basis`) | **dashed** underline, `?` after the role, `guessed` tag in the amber tone; tooltip shows the `basis` | `axis 0 · out? guessed` | v0.3 |
| `derived` (graph rule) | solid underline, small graph glyph | `axis 0 · out` | v0.3 |
| `declared` (profile) | solid, profile glyph | `axis 0 · out` | v0.3 |

Rules:
- An inferred role may label things but must never drive a hidden behaviour. For example, a default taken from it must say *"default from a guessed role"* (v0.3).
- In v0.2 the words "unit" and "channel" are not used for axis positions. The units panel says "positions along axis i".
- `coverage` from `describe` counts **axes** by provenance (`coverage.unit: "axes"`) and shows in the checkpoint facts as `Axis roles: 0 of 31,236 axes known`. In v0.2 that is always 0, and showing it makes the fact obvious.
- Copy selector never embeds a role. Selectors address axes by index only.

## 7. Data needed from the API (contract for T2 / T4c)

All v2 responses have the envelope `schema_version, resolved, result, fingerprints, side_effects, truncated, cursor` (Architecture_Target §4). Fields go in `result` unless noted and use the inspect contract rev 5 names (`docs/inspect-ops.md`). Status: **accepted** (in rev 5) or **deferred** (owner in brackets). The UI must work with every deferred field absent or `null`. Errors arrive as `{code, message, field}`; the inspector shows `message` and highlights `field` in the selector line.

### 7.1 Checkpoint list — **deferred (T4b)**
`checkpoint_id` (sha256), `display_name`, `origin: imported | applied`, `parent_id | null`, `recipe_hash | null`, `created_at`. Until then, the checkpoint picker shows the short sha and file name, and compare mode has no "parent" preselection.

### 7.2 `describe(name_glob?, limit, cursor)`
- **Accepted:** `checkpoint {format, sharded, shard_count, file_bytes, tensor_count, element_count, metadata}`; `coverage {unit: "axes", derived, declared, inferred, unknown, opaque_ops}`; `dtypes` (count per dtype, used for facet chip counts); `tensors[]` sorted by name: `name`, `dtype`, `shape`, `numel`, `nbytes`, `editable`, `read_only_reason` (`dtype_not_editable` or `null`), `shard`, `tensor_sha256` (always `null`, describe never hashes), `identical_bytes_group` (`null`), `roles` (always `[]`). The browser asks with `limit: 20000`, so a 10k-tensor model comes in one page; larger models page with `cursor`.
- **Deferred (T4b/T4c), present as `null`:** `identical_bytes` groups from a pollable job with staged progress (size → 64 KiB edges → full hash, candidates per stage) that excludes constant groups from its counts and flags constant tensors.
- **Accepted (T4b):** import progress from `GET /checkpoint-imports/{id}`: `status`, `stage` (`validating | copying | verifying | publishing`), `done`, `total`, `unit` (`files | bytes`), `checkpoint_id` and `existing` on success, `error {code, message}` on failure (§4.2, §4.3).

### 7.3 `read_tensor(selector, max_elems ≤ 10000, cursor)`
- **Accepted:** `resolved.selector` (canonical, every axis as merged `[start, stop)` ranges), `fingerprints.selection` (`sel_…`), `tensor`, `dtype`, `shape`, `selection_shape`, `total_elems`, `offset`, `returned_elems`, `values` (flat row-major; shortest decimal that round-trips in the stored dtype; `-0.0` kept; non-finite as `"NaN"`, `"Inf"`, `"-Inf"`; `I64`/`U64` as exact decimal strings, which the grid prints as-is and never converts to JS numbers), `nonfinite_count`, `range {min, max, abs_max, abs_p99}` over the whole selection (exact nearest-rank p99), or `range: null` with `range_skipped: "selection_too_large"` above 2^24 elements.
- `range` and `nonfinite_count` come on the first page only (`include_range: null`). On later pages they are **absent, meaning not computed** (not `null`, not zero): the browser keeps the first-page values for the scale and the non-finite count. Only `range: null` + `range_skipped` means the backend declined.
- The browser asks for rectangular windows with `{start, stop}` in the selector, so 2D paging needs no cursor; `cursor` is used for flat reads. The ulp on hover is derived from the stored dtype and the value (a display property, not a statistic).

### 7.4 `stats(selector, reduce_over_axes, bins: 16, limit ≤ 2048, cursor)`
- **Accepted:** `accumulate_dtype: "float64"`, `unit_axes`, `unit_count`, `offset`; `units[]`: `index` (real position on each kept axis, e.g. `{"0": 17}`; `{}` when everything is reduced; there is no `unit` field), `count`, `nonfinite_count`, `min`, `max`, `mean`, `std`, `l1`, `l2`, `zero_frac`, `constant` (metrics `null` when no finite element); `histogram {edges (shared, bins + 1), counts[i] for units[i]}`. The tensor-level constant flag in the facts uses `reduce_over_axes: null` (one summary row).
- **Deferred (T4c):** server-side profile downsample. When added it must return a per-bucket min–max envelope. Until then the strip pages through `units` (§4.4 C).

### 7.5 `diff(a, b, request)`
- **Accepted, both modes:** `fingerprints.checkpoint_a` (parent) and `fingerprints.checkpoint_b` (child) instead of `fingerprints.checkpoint`.
- **Accepted, whole-model mode** (no selector): `accumulate_dtype`, `summary {tensors_total, changed, same, only_a[], only_b[], shape_mismatch[], dtype_mismatch[], changed_elems_total, max_abs}`; `tensors[]` (non-`same` only, by name, ≤ 1000 per page): `name`, `status: same | changed | only_a | only_b | shape_mismatch | dtype_mismatch`, `dtype`, `shape`, `numel`, `changed_elems`, `max_abs`, `rel_l2`, `cosine`, `basis` (always `"values"` in v0.2), `changed_units {total, listed (≤ 1000), truncated}`. It reads both files in full, so the browser narrows with `name_glob` where it can and shows progress.
- **Accepted, selection mode** (selector given): the same metrics for the selection, plus `units[]` (`unit`, `changed_elems`, `max_abs`, `rel_l2`, `cosine`, paginated) when `unit_axis` is set, **or** with `values: true` (`unit_axis` is then ignored) the window's `values` (Δ = child − parent, float64, `delta_dtype`), `changed` (0/1 per cell) and `range` of Δ over the whole selection, with the read_tensor paging fields and the same first-page-only rule for `range`. The units panel and the Δ pane are therefore two calls.
- **Semantics (accepted):** tensors are aligned **by exact name only**. `changed` = bitwise inequality of stored bytes, so −0/+0, NaN-payload and rounding-only changes count as changed with Δ = 0. `rel_l2 = ‖a − b‖ / ‖a‖` is `null` when ‖a‖ = 0; `cosine` is `null` when either norm is 0; both are `null`, never `NaN`.

### 7.6 `query(predicate)`
Accepted: `name_glob`, `dtype[]`, `min_ndim`/`max_ndim`, `editable`, `min_bytes`/`max_bytes`, and a stat predicate `where {stat, axis, op, value}` returning per-match `selector` + `selection_id` and matching positions. The tree filter runs client-side over `describe` by default; `query` backs **Copy as query** and the "Lowest L2 first" search across tensors. The identical-bytes filter is deferred with that job.

## 8. Keyboard and accessibility

| Key | Action |
|---|---|
| `/` | focus filter · `Esc` clears filter, then returns focus to tree |
| `↑ ↓` / `Home End` / `PgUp PgDn` | move in tree (treegrid pattern; roving tabindex) |
| `→ ←` | expand/collapse group; on a leaf `→` moves into the inspector |
| `Enter` | open the tensor in the inspector |
| `*` | expand all siblings (VS Code convention) |
| `h` / `v` / `m` | Heatmap / Values / Mosaic |
| `[` `]` | previous/next index on the first fixed axis |
| `Shift+[` `Shift+]` | previous/next window when truncated |
| arrows inside heatmap | move the value cursor; the readout is announced |
| `d` | toggle compare mode |
| `Ctrl+C` in inspector (no text selected) | copy the selector JSON |
| `?` | shortcut sheet |

- Single-letter shortcuts are off while a text field has focus. They can be turned off in settings (WCAG 2.1.4).
- The heatmap canvas has `role=img` with a computed label (`"Heatmap of decoder.up2.conv.weight[9, :, :, :], 128 by 9, diverging scale ±0.17 (p99), range −0.21 to 0.19"`). **Values** is the full text alternative, one key away. Pinned readouts go to a polite live region.
- Colour is never the only signal. Diff status has glyph + text, changed cells have an outline (hatched when Δ = 0), read-only has lock + text, identical bytes and constant have `=b` / `const` text, and clipped cells have notch marks.
- Contrast: body text ≥ 7:1 on the ground, muted meta text ≥ 4.5:1 (`#a6b1a8` on `#191f1c` = 7.4:1), focus ring `#c6f2e0` 2px, as in the existing Desktop.
- Targets: rows are 28px, above the WCAG 2.2 AA 24px minimum. Controls in the toolbar are ≥ 32px. Under 880px width the panes stack (tree as a drawer), though the Desktop is primarily ≥ 1280px.
- Reduced motion: no skeleton shimmer, no animated transitions between windows.
- Loading and progress use `role=status`, failures use `role=alert`. The tree announces `214 of 10,412 tensors match` after filtering (debounced).

## 9. Tokens

Existing tokens are reused as-is (`style.css` `:root`, `workspace.css` `.workbench-shell`). New tokens, scoped to `.tensor-browser`:

```css
--tb-row-h: 28px;            --tb-indent: 14px;          --tb-font: 13px;      --tb-meta: 11.5px;
--tb-pane-line: #39413b;     --tb-row-hover: #1f2823;    --tb-row-selected: #26342d;   --tb-guide: #2b332e;
/* signed data: CVD-safe blue–orange diverging, dark centre so zero recedes into the ground */
--dv-neg-3: #7fb3ff; --dv-neg-2: #4f7fc9; --dv-neg-1: #2c4366;
--dv-zero:  #161c1a;
--dv-pos-1: #6b4425; --dv-pos-2: #c47a35; --dv-pos-3: #ffc27a;
--dv-nonfinite: #ff5fa2;     /* never part of the ramp */
--prov-guessed: #e9c48a;     /* reserved for v0.3 inferred roles (dashed underline + tag) */
--diff-changed: #ffc27a;  --diff-added: #9fc7b8;  --diff-removed: #ffc0a5;  --diff-mismatch: #ff8f7a;
--diff-mask: #ffe2b8;       /* 1px outline on changed cells; hatched when Δ = 0 */
/* non-negative data: sequential, ground → mint; bool: two categorical colours with legend words */
--seq-0: #161c1a; --seq-1: #24443a; --seq-2: #4f8a74; --seq-3: #c6f2e0;
--cat-false: #3a4640; --cat-true: #9fc7b8;
```

The ramps are interpolated in OKLab between the stops. In the diverging ramp zero sits at the ground colour on purpose: weights are signed and mostly near zero, so structure (all-zero positions, sparse rows) stands out, and the Δ pane reads as "only what changed". Non-negative data (e.g. running variance) gets the sequential ramp, so half the diverging ramp is not wasted. Nothing else on screen uses the data colours, so they keep their meaning.

## 10. Risks and open points

- **Describe payload size.** 10k tensors fit one page (`limit` up to 20,000). Bigger models page with `cursor`; the tree then loads groups lazily and filtering should move to `query`.
- **Identical bytes and constant flags are deferred.** v0.2 ships with those facets disabled. Staged filtering (size → 64 KiB edges → full hash) keeps I/O low, but a multi-GB file still takes time, so the T4b/T4c job needs progress.
- **Diff cost.** Whole-model diff reads both files in full (`basis` is always `"values"` in v0.2). The browser narrows with `name_glob` and shows progress; the manifest/hash fast path (T3/T4b) will make it interactive.
- **Roles.** Resolved for v0.2: `roles` is always `[]` and every axis shows `unknown`. The frontend must never guess on its own, because humans and agents would then see different semantics. Inferred roles return in v0.3 only with a documented `basis`.
- **BF16/F16 display.** Resolved: values are the shortest decimal that round-trips in the stored dtype, the ulp shows on hover, and stats/Δ declare `accumulate_dtype: "float64"`.
- **Very large selections.** Above 2^24 elements `range` is skipped, and the scale falls back to the loaded window (labelled). With the profile downsample deferred, a 50k-position axis needs about 25 `stats` pages; the strip hatches what is not loaded rather than pretending.
