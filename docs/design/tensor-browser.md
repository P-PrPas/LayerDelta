# Tensor browser — design spec (v0.2-T7a)

Owner: Senior UX/UI Designer · Status: **proposed** · Implements: Roadmap v0.2 acceptance 6 · Built by: T7b (Frontend) on T2/T4c payloads
Mockup: [`tensor-browser-mockup.html`](tensor-browser-mockup.html) (static, open in a browser; uses the Desktop fonts from `desktop/public/fonts`)

## 1. Purpose and non-goals

The tensor browser is the human view of v0.2 layer A: any imported safetensors checkpoint, at tensor-name granularity, without running model code. It answers four questions:

1. **What is in this file?** Thousands of tensors, grouped by name prefix, with dtype, shape, size, editability and tied-weight flags.
2. **What do the values look like?** A slice of one tensor as a heatmap or number grid, plus per-unit statistics along one axis.
3. **What changed between parent and child?** Which tensors and which units, and by how much.
4. **How much of this is known, and how much is guessed?** v0.2 has no graph. Any axis meaning is a guess from the tensor's name and rank, and the UI must say so wherever the guess appears.

Non-goals for v0.2: architecture graph (v0.3), editing UI and program builder (v0.6; v0.2 edits go through API/recipes), activations and evaluation (v0.4), multi-child diff (v0.6).

**Parity rule.** Every number on screen comes from a v2 payload (`describe`, `read_tensor`, `stats`, `diff`, `query`) that an agent can request too. The browser does not compute statistics or guess roles on its own. The only client-side work is grouping by name, filtering, sorting and colour mapping. Each view can copy its exact selector, so a human can give an agent what they are looking at.

## 2. What we keep from the v0.1 workbench

| Keep | Change |
|---|---|
| Palette (`--p-bg #161c1a`, mint accent `#9fc7b8`, hairline `--bench-line`), dark-only, Instrument Sans for UI text, DM Mono for every identifier and number | Higher density: the workbench is a 15px document layout, and the browser is a 13px tool layout with 28px rows. Big display headings (`h1` 49px) are dropped inside the browser. Its chrome uses the context bar only, so the data gets the space |
| Left rail + context bar shell; new rail entry **Tensors** above *Model explorer* (the existing screens stay as the MNIST fixture flow) | Panels are flush, split by 1px lines, not floating rounded cards. Cards don't work for a three-pane tool |
| Hand-drawn CSS nav glyphs (`.nav-sign`), the `meta-text` mono captions, `role=status` / `role=alert` patterns, skip link, reduced-motion rule | Adds a signed, diverging data scale. The workbench only had sequential grey and activation ramps |

What makes it feel premium is restraint and exactness: mono for all machine values, tabular numerals, right-aligned numbers, one accent, colour used only for data and state. The details carry the craft: a byte-size micro-bar in every row, compact folder chains, a selector line that is always visible, and dashed "guessed" underlines. There is no decoration that doesn't encode something.

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
 │    selector line · tensor facts · axis roles (with provenance)
 │    view tabs: Heatmap | Values | (Mosaic for rank-4 with small spatial dims)
 │    axis mapper (rows axis, cols axis, fixed indices) · scale legend · truncation bar
 └─ C  Units panel          (bottom of B, collapsible; or right column ≥1600px)
      unit axis picker · profile strip (one bar per unit) · sortable stats table
Compare mode (Compare with ≠ none) re-skins A, B and C. It is not a separate screen:
 A gains a Δ column and a "changed only" default filter, B shows parent | child | Δ, C shows per-unit diff
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

Import is staged, and each stage is a row in a `role=status` checklist:
1. Reading header (bytes / header size)
2. Validating extents (count of tensors checked)
3. Building tensor table → **the tree becomes usable here**
4. Checking for tied weights. This runs in the background: only equal-size tensors are hashed, and the progress shows `hashed / candidates`. Until it finishes, the Tied facet and flag show *pending* (a hollow link glyph) and not "none".

Stage 4 does not block anything. A multi-GB file can be browsed while the hashing runs.

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
- The filter field (`/` to focus) matches name substrings and globs (`*.attn.*`). Facet tokens can be typed or toggled as chips: `dtype:bf16`, `rank:4`, `editable:no`, `tied:yes`, `size:>10MB`, and in compare mode `changed:yes`. Chips and typed tokens stay in sync. Filtering is client-side over the `describe` table.
- Summary line, live: `10,412 tensors · 1.21 B params · 2.42 GB · 37 tied groups`, or with a filter: `214 of 10,412 match`.
- Grouping: names split on `.`. Numeric segments sort naturally (`layers.2` before `layers.10`). Single-child chains collapse into one row (`encoder.down1.conv`). On first load only depth 1 is expanded.
- **Group row:** chevron, segment name, tensor count, total size, and in compare mode the count of changed tensors.
- **Leaf row** (28px), as columns: leaf name · dtype chip · shape (`64×32×3×3`) · size, right-aligned with a log-scaled micro-bar · flags. The flags are a lock (read-only dtype: int/bool/fp8/f64), a link (tied, with group size; hover lists the members) and Δ in compare mode.
- The row tooltip and the inspector show the full name. Long names are middle-truncated (`encoder…conv1.weight`) and never end-truncated, because the suffix (`.weight`/`.bias`) is what identifies a tensor.

**B · Inspector**
- **Selector line**, mono and always visible: `decoder.up2.conv.weight [17, :, :, :]`. It can be edited with NumPy-style slice syntax. Next to it are *Copy selector* (the JSON selector from Architecture_Target §3.1) and the `selection_id` once one is resolved.
- **Facts:** dtype, shape, numel, bytes, editable (yes / no + reason), tied group (with links to the members), `tensor_sha256` (shows *not computed* with a "Compute" action, because it is lazy).
- **Axis roles** are one chip per axis. In v0.2 every chip is either `axis 2 · unknown` or `axis 0 · out? ` with a **dashed underline and a "guessed" tag**. The tooltip reads *"Guessed from the name `…conv.weight` and rank 4. No graph is attached, so LayerDelta cannot confirm what this axis means. Treat units as positions along axis 0, not as verified channels."* A one-time banner on the first guessed tensor explains the same thing and links to v0.3. See §6.
- **Axis mapper:** rows axis ▾, cols axis ▾, and an index stepper (`− 17 +`, with direct entry) for every other axis. Defaults by rank:
  - rank 1 → a single strip, rows = 1
  - rank 2 → rows = axis 0, cols = axis 1
  - rank 4 → **Mosaic** when the last two dims are ≤ 7. Each cell is a k×k kernel, mosaic rows = axis 0 window and cols = axis 1 window. Otherwise a Heatmap of axes 2×3 at fixed [0, 0]
  - rank 3 and rank ≥ 5 → the last two axes, others fixed at 0
- **Heatmap:** one square cell per value, nearest-neighbour, integer zoom (fit / 1× / 2× / 4× / 8×). Hover shows a crosshair readout `[17, 4, 1, 2] = −0.0312`. Click pins the readout. The scale is diverging and symmetric around 0. The legend shows the domain, with a choice of `±max|x|` (default) or `±p99|x|` (clipped cells get a small notch mark).
- **Values:** a virtualised number grid in DM Mono, tabular, 4 significant digits, full precision on hover and copy. Optional cell tint uses the same scale at 35% strength. Non-finite values show as `NaN` / `+Inf` in the alert colour, never as a colour.
- **Truncation bar**, shown whenever `truncated: true`: `Showing rows 0–255 of 2,048 · cols 0–255 of 1,024 (65,536 of 2,097,152 values)`, a window minimap (a small rectangle inside the full extent) and **Previous / Next window** controls. For 2D windows the browser asks for rectangular index ranges, which are deterministic and resumable. For flat reads it follows `cursor`. The bar never hides: a truncated view must never look complete.

**C · Units panel**
- Unit axis ▾. The default is axis 0 when it is guessed `out?`, otherwise axis 0 labelled plainly. Every other axis is reduced.
- **Profile strip:** one bar per unit (L2 by default; switchable to mean/std/zero fraction/max |x|). It is 64px tall and spans the panel width. It is a canvas, with one pixel per unit beyond ~800 units, aggregated by max. Dead units read as gaps right away. Click a bar to select the unit, which updates the fixed index in B. The selected unit is outlined.
- **Stats table** (virtualised, sortable): unit · L2 · mean · std · min · max · zero % · non-finite · a 16-bin histogram sparkline on shared edges. Sorting by L2 ascending is one click ("Weakest first").

### 4.5 Compare mode

Turned on by choosing **Compare with** in the context bar. If the open checkpoint is a child with a manifest, its parent is preselected and labelled "parent". Otherwise any checkpoint can be picked, labelled "other". Shortcut `d`.

- **Summary banner** above the tree: `3 of 10,412 tensors changed · 1 unit (guessed) · 1,153 values · max |Δ| 0.084`, and when tensors don't align, `2 only in parent · 1 only in child · 1 shape changed`. Alignment problems use the alert colour and say what they mean ("Names match but shapes differ. These tensors can't be compared value by value.").
- **Tree:** the "changed only" filter is on by default and can be toggled. A Δ column shows a rel-L2 micro-bar plus `changed_elems`. Status glyphs carry text labels too: `=` same, `Δ` changed, `+` only in child, `−` only in parent, `≠` shape/dtype mismatch. Group rows show their changed count.
- **Inspector:** three aligned panes, **Parent | Child | Δ (child − parent)**, at the same axes, window and zoom. Parent and child share one scale. Δ has its own symmetric scale, and unchanged cells get no colour, so a single changed unit looks like a single stripe. Hovering one pane shows the crosshair in all three.
- **Units panel:** columns become unit · changed values · max |Δ| · rel L2 · cosine, sorted by changed values (descending). The profile strip shows max |Δ| per unit, so the edited unit is the only bar.
- When a pair is identical by hash, the panes show "Identical bytes (sha256 match)", with no values read.

## 5. Component inventory

| Component | Notes | States |
|---|---|---|
| `ImportCard` | path field, file picker, accepted-format line, error slot | idle · validating-path · importing (stage list) · error(code) · refused-format |
| `StageList` | `role=status`, ✓ / spinner / pending glyph per stage, background stage marked "continues in background" | pending · running · done · failed |
| `TreeFilter` | text input + facet chips, token parser, match count | empty · typing (debounced 120ms) · no-results ("No tensors match `rank:5`. Clear filter") |
| `TensorTree` | ARIA `treegrid`, windowed (overscan 20), compact folders, indent guides | loading skeleton (12 grey rows, no shimmer under reduced motion) · ready · filtered · empty |
| `TreeRow` (group / leaf) | columns above; selected = accent left rule + `#26342d` ground (matches the rail's `aria-current`) | default · hover · focus-visible · selected · selected+compare · tied-pending |
| `DtypeChip` | mono 11px, outline only; read-only dtypes add a lock glyph | editable · read-only |
| `SizeBar` | 36×3px, log10 bytes scaled to the checkpoint's largest tensor | — |
| `FlagGlyph` | lock, link, Δ, `+ − ≠`; always paired with an accessible name | — |
| `SelectorLine` | editable mono field, validation inline, copy buttons | valid · invalid (field path from 422) · resolving |
| `AxisChip` | `axis i · role`, provenance styling (§6) | unknown · inferred(guessed) · derived · declared |
| `AxisMapper` | two selects + steppers; prevents rows = cols | — |
| `HeatmapCanvas` | canvas, integer zoom, crosshair, pinned readout, keyboard cursor | loading (previous image dimmed 50% + status) · ready · truncated · error · all-zero ("All values are 0", scale hidden) · constant |
| `ValueGrid` | windowed table, `aria-rowcount/colcount` | same as heatmap |
| `ScaleLegend` | gradient bar, domain labels, scale mode toggle | symmetric-max · p99 (clip notches) · shared (compare) |
| `TruncationBar` | extent text, minimap, prev/next | hidden only if `truncated=false` |
| `UnitProfile` | canvas strip, click/arrow to select | — |
| `UnitTable` | windowed, sortable headers (`aria-sort`) | inspect columns · diff columns |
| `CompareBanner` | counts, alignment issues | identical · changed · misaligned |
| `ProvenanceNote` | dismissible once per workspace, re-openable from any guessed chip | — |

## 6. Provenance (non-negotiable)

Every role carries a provenance (Architecture_Target §2.2). The UI renders the four levels differently, and these renderings are fixed so v0.3 can add the stronger ones without a redesign:

| Provenance | Chip | Text | v0.2? |
|---|---|---|---|
| `unknown` | plain, muted | `axis 2` | yes |
| `inferred` (heuristic, from the name) | **dashed** underline, `?` after the role, `guessed` tag in the alert-amber tone | `axis 0 · out? guessed` | yes |
| `derived` (graph rule) | solid underline, small graph glyph | `axis 0 · out` | v0.3 |
| `declared` (profile) | solid, profile glyph | `axis 0 · out` | v0.3 |

Rules:
- A guessed role may label things but must never drive a hidden behaviour. For example, the unit axis default may use it, but the axis picker states *"default from a guessed role"*.
- Wherever the word "unit" or "channel" appears for a guessed axis, it is written "unit (guessed)" the first time on that screen.
- `coverage` from `describe` shows in the checkpoint facts as `Axis roles: 0 derived · 812 guessed · 9,600 unknown`. In v0.2 the derived count is always 0, and showing it makes that fact obvious.
- Copy selector never embeds a role. Selectors address axes by index only.

## 7. Data needed from the API (contract for T2 / T4c)

All v2 responses have the envelope `schema_version, resolved, result, fingerprints, side_effects, truncated, cursor` (Architecture_Target §4). The fields below go in `result` unless noted. Fields marked **(ask)** are not in the current drafts.

### 7.1 `list_checkpoints`
`checkpoint_id` (sha256), `display_name` **(ask)**, `format`, `sharded`, `shard_count` **(ask)**, `file_bytes`, `tensor_count`, `param_count` **(ask)**, `parent_id | null` and `recipe_hash | null` (from the child manifest), `origin: imported | applied` **(ask)**, `created_at`, plus `next_cursor`.

### 7.2 `describe(checkpoint)`
- Checkpoint: the fields from 7.1 plus `metadata` (safetensors `__metadata__`, string map), `coverage {derived, declared, inferred, unknown, opaque_ops: []}`.
- `tensors[]`, **the whole table in one response** (10k tensors ≈ 1–2 MB JSON; see the risks below): `name`, `dtype`, `shape[]`, `nbytes`, `numel` (convenience), `editable`, `read_only_reason | null` (`dtype_not_editable` …) **(ask)**, `shard | null`, `tensor_sha256 | null` (null = not computed yet), `tied_group | null` (group id), `roles[]: {axis, role, provenance, basis}`. `basis` is a human-readable rule id, e.g. `name:*.conv*.weight+rank4` **(ask)**.
- `tied {state: pending | done, hashed, candidates, groups: [{id, members[]}]}` **(ask: needs to be pollable or a separate route, because hashing is lazy)**.
- Import progress (stages 1–4) **(ask: the T4b import job should expose `stage`, `done`, `total`, and on failure the T1 `code` + message)**.

### 7.3 `read_tensor(selector, max_elems)`
`resolved.selector` (canonical), `selection_id`, `dtype`, `selection_shape[]` (shape after indexing), `values` (row-major flat array over the selection; non-finite encoded as the strings `"NaN"`, `"Inf"`, `"-Inf"`) **(ask: encoding)**, `returned_elems`, `total_elems`, `truncated`, `cursor`. Also `range {min, max, abs_max, abs_p99}` **over the whole selection, not just the returned page** **(ask)**, so the scale doesn't jump between windows. And `nonfinite_count`.
The browser asks for rectangular windows by putting `{start, stop}` ranges in the selector, so 2D paging is deterministic without a cursor. `cursor` is used only for flat reads.

### 7.4 `stats(selector, reduce_over_axes)`
Per unit, in order: `unit` (index along the kept axis), `min`, `max`, `mean`, `std`, `l1`, `l2`, `zero_frac`, `nonfinite`. Also `histogram {edges[] (shared across units, 17 edges), counts[unit][16]}` **(ask: shared edges)**. Units are paginated with `cursor`, because a vocab axis can have 50k+ units, and the profile strip needs either all units or a server-side downsample `profile {metric, buckets[]}` **(ask)**.

### 7.5 `diff(a, b, selector?, unit_axis?)`
- Summary: `tensors_total`, `changed`, `same`, `only_a[]`, `only_b[]`, `shape_mismatch[]`, `dtype_mismatch[]`, `changed_elems_total`, `max_abs`.
- Per tensor: `name`, `status: same | changed | only_a | only_b | shape_mismatch | dtype_mismatch`, `changed_elems`, `numel`, `max_abs`, `rel_l2`, `cosine`, `basis: sha256 | values` (whether "same" came from a hash match) **(ask)**.
- Per unit, when `unit_axis` is given: `unit`, `changed_elems`, `max_abs`, `rel_l2`, `cosine`. Paginated.
- The Δ heatmap uses `read_tensor` on both sides plus a client subtraction for display only. **(ask: or should `diff` return delta values for a window, to keep the parity rule strict? Recommended: yes, `diff(..., values: true)` with the same window/truncation fields as 7.3.)**

### 7.6 `query(predicate)`
v0.2 browser use is optional. The filter runs client-side over `describe`. If the backend wants one code path, the browser can send `name_glob`, `dtype[]`, `rank`, `editable`, `tied`, `min_bytes`/`max_bytes` and get back `selection_id` + names, which also makes "Copy as query" possible.

## 8. Keyboard and accessibility

| Key | Action |
|---|---|
| `/` | focus filter · `Esc` clears filter, then returns focus to tree |
| `↑ ↓` / `Home End` / `PgUp PgDn` | move in tree (treegrid pattern; roving tabindex) |
| `→ ←` | expand/collapse group; on a leaf `→` moves into the inspector |
| `Enter` | open the tensor in the inspector |
| `*` | expand all siblings (VS Code convention) |
| `h` / `v` / `m` | Heatmap / Values / Mosaic |
| `[` `]` | previous/next index on the first fixed axis (unit) |
| `Shift+[` `Shift+]` | previous/next window when truncated |
| arrows inside heatmap | move the value cursor; the readout is announced |
| `d` | toggle compare mode |
| `Ctrl+C` in inspector (no text selected) | copy the selector JSON |
| `?` | shortcut sheet |

- Single-letter shortcuts are off while a text field has focus. They can be turned off in settings (WCAG 2.1.4).
- The heatmap canvas has `role=img` with a computed label (`"Heatmap of decoder.up2.conv.weight[17, :, :, :], 32 by 9, range −0.21 to 0.19"`). **Values** is the full text alternative, one key away. Pinned readouts go to a polite live region.
- Colour is never the only signal. Diff status has glyph + text, read-only has lock + text, guessed has dashed + `?` + the word "guessed", and clipping has notch marks.
- Contrast: body text ≥ 7:1 on the ground, muted meta text ≥ 4.5:1 (`#a6b1a8` on `#191f1c` = 7.4:1), focus ring `#c6f2e0` 2px, as in the existing Desktop.
- Targets: rows are 28px, above the WCAG 2.2 AA 24px minimum. Controls in the toolbar are ≥ 32px. Under 880px width the panes stack (tree as a drawer), though the Desktop is primarily ≥ 1280px.
- Reduced motion: no skeleton shimmer, no animated transitions between windows.
- Loading and progress use `role=status`, failures use `role=alert`. The tree announces `214 of 10,412 tensors match` after filtering (debounced).

## 9. Tokens

Existing tokens are reused as-is (`style.css` `:root`, `workspace.css` `.workbench-shell`). New tokens, scoped to `.tensor-browser`:

```css
--tb-row-h: 28px;            --tb-indent: 14px;          --tb-font: 13px;      --tb-meta: 11.5px;
--tb-pane-line: #39413b;     --tb-row-hover: #1f2823;    --tb-row-selected: #26342d;   --tb-guide: #2b332e;
/* signed data scale: CVD-safe blue–orange diverging, dark centre so zero recedes into the ground */
--dv-neg-3: #7fb3ff; --dv-neg-2: #4f7fc9; --dv-neg-1: #2c4366;
--dv-zero:  #161c1a;
--dv-pos-1: #6b4425; --dv-pos-2: #c47a35; --dv-pos-3: #ffc27a;
--dv-nonfinite: #ff5fa2;     /* never part of the ramp */
--prov-guessed: #e9c48a;     /* dashed underline + tag; also the "guessed" banner rule */
--diff-changed: #ffc27a;  --diff-added: #9fc7b8;  --diff-removed: #ffc0a5;  --diff-mismatch: #ff8f7a;
```

The colour ramp is interpolated in OKLab between the stops. Zero sits at the ground colour on purpose. Weights are signed and mostly near zero, so structure (dead units, sparse rows) stands out, and the Δ pane reads as "only what changed". Nothing else on screen uses the data colours, so they keep their meaning.

## 10. Risks and open points

- **Describe payload size.** 10k tensors are fine. 100k+ (very large LLM shards) would need a paginated `describe` and server-side `query` for filtering. The tree design already works with lazily loaded groups, but T2 must decide.
- **Tied detection cost.** Hashing equal-size tensors on a multi-GB file takes time. The UI handles a *pending* state, but the backend must expose progress (7.2).
- **Diff cost.** Comparing 10k tensors by values is expensive. Hash-first (and the child manifest's `touched` list as a fast path) keeps it interactive. The UI shows `basis` so a "same" from a hash is honest.
- **Role guessing ownership.** If T2 emits no `inferred` roles at all in v0.2, every axis shows `unknown` and the units panel defaults to axis 0 with a plain label. That works. What must **not** happen is the frontend guessing on its own, because humans and agents would then see different semantics.
- **BF16/F16 display.** Values are shown as the backend's shortest f32 repr of the stored value. The facts panel says "stored as bf16" so precision isn't overstated.
