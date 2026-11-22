# Architecture navigator: design spec (v0.3-G8a)

Owner: Senior UX/UI Designer · Status: **proposed, rev 1** · Implements: Roadmap v0.3 "Desktop" line, task G8 of [v0.3-graph-semantics.md](../v0.3-graph-semantics.md) · Built by: G8 Frontend (after G4, G5) · Mockup: [architecture-navigator-mockup.html](architecture-navigator-mockup.html) · Extends: [tensor-browser.md](tensor-browser.md) (same shell, selectors, tokens and rules)

## 1. Purpose and non-goals

v0.2 shows tensors whose axes are all `unknown`. v0.3 attaches a graph and derives, by rule, what each axis means (roles), which slices must move together (coupling groups), where understanding stops (opaque boundaries), and how much is understood (coverage). The navigator answers three questions on models with thousands of nodes:

1. **Where am I in the model?** A module outline and a dataflow canvas that collapse and expand by module, with search and a minimap.
2. **What moves together?** Select a group or a unit, and every member lights up across tensors: conv out rows, BN vectors, next conv in columns, concat offsets, packed qkv rows.
3. **What happens if I edit this unit?** `plan` → effect preview → `apply`, with refusals and `side_effects` shown before anything is written.

The navigator is honest about what it does not know. A role is shown with its provenance and basis. An opaque op is drawn as a boundary, never as a normal node. Coverage is a number that is always visible.

Non-goals for v0.3: activations and evaluation in the graph (v0.4); running user code from the Desktop (the `torch_export` provider is blocked by Q1, and attaching it is a human action outside the agent's reach); a graph editor (the graph is derived and read-only); multi-child diff and the full program builder (v0.6); name-based role guesses (`inferred` stays unused, decision of PR #11).

## 2. Principles, derived from references

| Reference | What works there | Principle here |
|---|---|---|
| Netron | Faithful op-level graph, immediate node properties | Show the real aten ops when asked, but **start at module level**. An op graph of a 10k-node model is a hairball |
| Google Model Explorer | Hierarchical layers collapsed by default, expand in place, per-node overlays, works on very large graphs | Compound nodes from the module tree, **expand in place** with stable layout. Overlays (coverage, group membership) are data layers, not new screens |
| TensorBoard graph | Grouping by name scope, series collapsing, colour by attribute | Collapse repeated siblings (`layers.0…23`) into one series node with a count; colour is reserved for one overlay at a time |
| VS Code outline + minimap, Figma layers | Outline and canvas stay in sync; the outline is the keyboard path | The **outline is the primary, accessible navigator**. The canvas is a visual companion; every canvas action has an outline or inspector equivalent |
| Torch-Pruning / DepGraph diagrams | Dependency groups drawn as highlighted chains across layers | A group is one highlight across tensors, with member slices drawn as **index strips** so offsets and packing are visible, not just named |
| Node editors (Blender, Houdini) | Ports, typed edges, frame nodes | Edges carry their channel count; a collapsed module shows its in/out port counts; frames (compound nodes) never hide an opaque op inside |

## 3. Information architecture

Same Desktop shell as the tensor browser. The rail gains **Architecture** directly below **Tensors**. Both share the context bar (checkpoint picker, Compare with, Import…, `?`), and the context bar gains a **graph status** item:

`graph: builtin · unet-like · 948 axes · 912 derived (96.2%) · 1 opaque op · 2 open groups`, or `no graph attached` with *Attach…*.

```
┌ rail ┬ context bar: crumb · checkpoint · compare · graph status · Import… · ? ─────────────────────┐
│      ├ A · Outline (320px) ┬ B · Graph canvas (flex) ──────────────┬ C · Inspector (400px) ─────┤
│      │ search + tokens     │ breadcrumb (module path) · overlay ▾  │ tabs: Node | Group | Tensor│
│      │ coverage summary    │ compound nodes, expand in place       │ roles, members, strips      │
│      │ module tree         │ edges with channel counts             │ ─────────────────────────── │
│      │ (virtualised        │ minimap (bottom right)                │ Edit units: plan → preview  │
│      │  treegrid)          │ legend                                │ → apply                     │
└──────┴─────────────────────┴───────────────────────────────────────┴────────────────────────────┘
```

- **A · Outline**: the module tree from the graph (`module` path of each node, §7 gap 1), with node and tensor counts, a coverage micro-bar per module and flags (opaque, open group, unbound). This is the keyboard path through the model.
- **B · Canvas**: left-to-right dataflow of the current scope. The scope starts at the root, with top-level modules collapsed.
- **C · Inspector**: context for the selection. Node: op, rule, roles of the tensors it reads. Group: members, strips, constraints, closed/open. Tensor: the v0.2 facts with roles, plus *Open in Tensors*.
- **Selection model**: one selection at a time (a module, a node, a group, or a tensor axis) plus an optional **unit** (`u`) within a group. The selection is in the URL (`?ckpt=…&sel=group:g_12&u=5`), so it survives reloads and can be shared within the workspace.

## 4. Screens and states

### 4.1 No graph attached

The canvas shows an attach card. The outline falls back to the v0.2 name-prefix tree (labelled *"from tensor names, no graph"*), so the screen is still useful.

- **Heading** "Attach a graph". One line: *"A graph tells LayerDelta what each axis means and which slices move together. It is read from the model's structure, not guessed from names."*
- **Providers** as radio cards, each with its status from the server (§7 gap 9):
  - `builtin`: models shipped with LayerDelta and its test models. *Runs only LayerDelta's code.*
  - `onnx`: choose an `.onnx` file. *Parsed, never executed.* If the dependency is missing: disabled, *"ONNX support is not installed in this workspace."*
  - `torch_export`: *"Runs your model code in an isolated worker."* Disabled until the owner decides Q1, with the reason as text (*"Not enabled in this workspace"*), never hidden. When enabled, it needs an explicit per-project opt-in checkbox. It is never offered to an agent token, and the card says so.
- **Reuse**: if the server already has a graph for this checkpoint's `table_signature`, the card says *"A graph built for 3 checkpoints with the same tensor table can be attached."* with one button.

### 4.2 Attaching

A `role=status` checklist, the same pattern as import (tensor-browser §4.2): **Building** (provider), **Binding** (`done / total` params), **Deriving** (roles, groups, coverage) → the outline and canvas open. Errors are inline with the stable code and one recovery action:

| Code | Title | Recovery |
|---|---|---|
| `graph_mismatch` | The graph does not fit this checkpoint | Lists the names that are missing or have a different shape (first 20, then "and N more", copy all) |
| `provider_disabled` | This graph source is not enabled | Text reason (Q1, missing dependency); no action |
| `provider_failed`, `provider_timeout` | Building the graph failed | Shows the provider message; *Try again*. A failed build leaves no graph |
| `unsupported_ir` | Graph format not supported | Copy details |

### 4.3 Navigator (main view, the mockup)

**A · Outline**
- Search `/` matches module paths, node ids, tensor names and tokens: `op:conv2d`, `role:out_unit`, `prov:unknown`, `group:g_12`, `opaque`, `open`, `unbound`, `unused`. Results expand their ancestors. Matching modules get a mint tick on the canvas, so search and canvas agree.
- Rows: chevron · module name (compact folders as in the tensor browser) · type tag (`Conv2d`, `Sequential`, `opaque`) · node count · coverage micro-bar (derived share of the axes in the module) · flags: `⊘` opaque op inside, `◌` open group touches it, `∅` unbound tensor.
- **Series collapsing**: siblings with the same structure (same op sequence and shapes, e.g. `blocks.0…11`) show as one row `blocks.[0–11] ×12` that expands to the members.
- Two pseudo-folders at the bottom, always present when non-empty: **Not in graph (unbound)** and **Unused by forward**, with counts. They are never hidden in a filter menu, because "unbound" is a real, common state (spike finding 2).

**B · Canvas**
- **Layout**: layered left to right, computed per compound scope and cached by `(ir_sha256, scope, expanded set)`. Expanding a module grows it in place. Its neighbours shift but keep their relative order, so the mental map survives. No physics, no animation under reduced motion.
- **Level of detail**: a collapsed module is a frame with its name, type, node count and port counts (`in 16 · out 32`), plus a coverage hairline along its bottom edge. Expanded, it shows its children. Op nodes show the op short name (`conv2d`, `cat`, `bn`) and the bound tensors as small chips under them. Below 60% zoom, op labels hide and frames keep their names.
- **Edges** carry the channel count of the dimension that matters for coupling (`16`, `32`). Edges into `cat` show their offset (`@0`, `@16`).
- **Breadcrumb** above the canvas: `root / dec2 / 0`. `Backspace` goes up a scope; double-click (or `Enter`) on a frame goes into it.
- **Minimap** bottom right (160×100): the whole current layout as blocks, a viewport rectangle, opaque ops as hatched dots and group members as mint dots. It is drag-to-pan, and hidden below 1100px width.
- **Overlay ▾** (one at a time, so colour keeps its meaning): `none` · `coverage` (frames tinted by derived share, sequential ramp) · `group` (default when a group is selected) · `binding` (bound / unused / unbound).

**C · Inspector**: see 4.4–4.6.

### 4.4 Coupling highlight

Selecting a group (from a role chip, a node's tensor chip, the outline token `group:`, or the Group tab list) turns on the group overlay:

- Every **member node** gets a 2px mint outline (`--grp-hi`). Every **edge in the group's unit space** (needs §7 gap 2) is drawn 2px mint. Everything else drops to 35% opacity. Collapsed frames that contain members show a mint count badge (`3 members`), so a highlight is never lost inside a collapsed module.
- **Group tab** in the inspector:
  - Title from the group's `label` (§7 gap 4), e.g. *enc2 channels*, with `g_12` in mono, size (`16 units`), `closed` or `open`, and constraints in words (*"permute within blocks of 4 (GroupNorm)"*, *"heads move as blocks of 8"*).
  - **Member strips**: one row per member, `tensor · axis · role chip`, and a strip of `shape[axis]` cells. For the current unit `u`, the cells in `indices(u) = {offset + u·block + k·period + j}` light up. This makes concat offsets (`dec2.0.weight axis 1`: cells 0–15 belong to enc2, 16–31 to up2) and packed qkv (`repeat 3`, three lit blocks one period apart) visible at a glance. Strips longer than 256 cells compress into buckets and keep the lit cells exact (a lit bucket is never averaged away).
  - **Unit picker**: `− 5 +` stepper and a profile strip of a per-unit metric (L2 of the first `out_unit` member by default, from `stats` with the group selector or `query where.group`). This is the same profile component as the v0.2 units panel, so units are chosen the same way everywhere. Hovering a strip cell sets `u` to the unit that owns it.
  - **Boundary** (open groups): each boundary as `node · op · reason`, with *Show on canvas*. The tab says *"This group touches an op LayerDelta does not understand. Edits that rely on the whole group are refused."*
  - **Notes** from the server, verbatim (e.g. *"LayerNorm: zeroing a unit does not remove it; it still enters the mean and variance."*).
- Units are **always** shown as positions along the group's unit space. The word "unit" is used here because the group is derived. It is still never "neuron" or "channel 5 of the network".

### 4.5 Role chips and provenance

The v0.2 axis chip (`axis 0 · unknown`) becomes a role chip with three parts: `axis 0 · out_unit · derived`.

| Provenance | Chip style | Basis shown |
|---|---|---|
| `derived` | solid mint left rule, role in ink | `rule:conv.weight@n7`, linked: click selects node `n7` on the canvas |
| `declared` | dashed rule in `--prov-declared` | the profile entry (file and key); v0.3 schema only, rare |
| `inferred` | not used in v0.3 | never rendered |
| `unknown` | hollow, muted, role text `unknown` | *why*: `no rule for aten.cumsum (n40)`, `unbound: not in the graph`, or `unused by forward` |

- Packed roles render segmented: `axis 0 · packed 3 × heads 2 × d_head 8 · derived`. Each segment can be hovered to show its factor.
- A tensor used by several nodes (tied weights) shows one chip per axis plus a `tied ×2` tag listing the uses. The graph declares ties (`ties`), so the v0.2 "identical bytes" hint becomes a fact when a graph says so; identical bytes alone still never imply a tie.
- Chips are buttons: activating a chip with a group selects that group (4.4).

### 4.6 Opaque boundaries, coverage and binding

- **Opaque op node**: drawn with the hatch pattern (`--opaque-hatch`), the label `opaque · aten.cumsum`, and a `⊘` glyph. Edges into and out of it are cut: a 6px gap with a `⊘` marker, dashed beyond. A frame containing an opaque op shows `⊘ 1` on its header and can never look fully covered (its hairline has a hatched segment).
- **Coverage**: the context bar shows the number. Clicking it opens a coverage sheet:
  - axes `derived / declared / inferred / unknown` with counts and a stacked bar;
  - `opaque_ops` with counts and *Show on canvas*;
  - tensors `bound / unused / unbound`;
  - groups `closed / open`.
  Every count is a filter link into the outline. Coverage is never rounded to 100% when any axis is unknown (`99.9%` shows as `99.9%`, and `947 of 948` is always next to it).
- **Unbound and unused tensors**: listed in their outline folders; in the Tensor tab they show `binding: unbound (not in the graph)` and all axes `unknown` with that basis. They remain fully inspectable and editable as tensors (v0.2 behaviour); only group addressing is unavailable.

### 4.7 Unit-level edit: plan → effect preview → apply

Lives at the bottom of the Group tab (and the Tensor tab for single-tensor edits). Three steps in one panel, never a modal, always reversible until *Apply*:

1. **Compose**: op `zero` · `scale k` · `permute_units` (the only op claimed function-preserving, labelled so). Units: the current `u`, a list, or a range. For `permute_units` the panel offers generators: *swap u ↔ v*, *sort by metric ▾ (descending)*, *paste permutation*; the result is shown as a mapping strip (moved units drawn as arcs). A **hypothesis** field is required, as in v0.1.
2. **Plan** (calls `plan`, no writes): the effect preview shows
   - one row per member: `tensor · axis · selected elems · changed elems (bitwise) · before_sha…after_sha`, each with *Show slice* (opens the tensor browser compare at that selector once applied, or the parent slice before);
   - totals `touched / changed / child bytes`, overlaps, `noop`;
   - the expanded selectors (mono, collapsible): what will be recorded in the recipe, with `via: {group, ir_sha256, units}`;
   - refusals inline with their code and what to do: `group_not_closed` (*"Edits on an open group are refused. Edit single tensors instead, or see the boundary."*), `constraint_violation` (*"Units 3 and 6 are in different GroupNorm blocks"*), `invalid_permutation`, `read_only_dtype`, `dtype_overflow`, `no_op`.
   - Value preview: v0.3 shows counts and hashes from `plan`; a before/after mini heatmap per member needs plan preview values (§7 gap 6). Until then the panel says *"Values are shown after apply, in Compare."* It never computes them client-side (parity rule).
3. **Apply** (primary button, enabled only after a successful plan of the same program; any change returns to step 1): calls `apply`. The panel then shows the envelope's **side effects** as a list, exactly as returned: `checkpoint_created · a3f1…c09 · 212.2 kB`, plus `recipe_hash`. Actions: *Open child*, *Compare with parent* (opens tensor-browser compare mode on the first member, child vs parent), *Copy recipe*. The child inherits the graph through its table signature (A8), so the navigator stays on the same view for the child.

### 4.8 Integration with the tensor browser

- **Same selectors**: the selector line accepts and displays group form `g_12[5]` (expanding to the member list in a popover), and tensor-local form as before. *Copy selector* copies the canonical JSON (group form plus `expanded`).
- **Open in Tensors** from any member, chip or node tensor opens the tensor browser on that tensor with the member's index preselected (`{tensor, index:{axis:[indices(u)]}}`). **Show in Architecture** from the tensor browser's facts selects the tensor's node and its axis-0 group.
- With a graph attached, the tensor browser's axis chips become role chips (4.5), the units panel title becomes *"units of g_12 along axis 0 (out_unit)"*, and the mosaic default uses `spatial` roles instead of the shape heuristic. Without a graph, nothing changes (rev6 byte-identity, A9).
- Compare mode stays tensor-level. In the navigator, compare adds one overlay: frames and nodes with changed tensors get the Δ glyph and the changed count, from the whole-model diff.

## 5. Component inventory

| Component | Notes | States |
|---|---|---|
| `GraphStatus` (context bar) | provider, coverage %, opaque count, open groups; opens coverage sheet | none · attaching · attached · stale (graph for another signature) |
| `AttachCard` | provider radio cards with server status, reuse offer | idle · attaching · error |
| `ModuleOutline` | WAI-ARIA treegrid, virtualised, series collapsing, pseudo-folders | loading · ready · filtered · empty |
| `GraphCanvas` | layered layout per scope, LOD, breadcrumb, overlays | loading layout · ready · scope empty · too large (falls back to outline + "expand a module") |
| `FrameNode` / `OpNode` / `OpaqueNode` | ports, counts, coverage hairline, member badge | default · hover · focus · selected · member · dimmed |
| `Minimap` | blocks, viewport, member dots | — |
| `RoleChip` | provenance styles, packed segments, basis link | derived · declared · unknown · tied |
| `MemberStrip` | `shape[axis]` cells, `indices(u)` lit, bucketed above 256 | — |
| `UnitPicker` | stepper + profile strip (shared with v0.2 `UnitProfile`) | loading · ready |
| `EditPanel` | compose → plan → apply, refusal messages, side effects | compose · planning · planned · refused · applying · applied |
| `CoverageSheet` | stacked bar, counts as filters | — |

## 6. Wording rules

- "unit" only for positions in a **derived** group's unit space. Elsewhere it stays "position along axis i" (tensor-browser §6).
- Never "neuron", "feature", "dead". "All zero" stays as in v0.2.
- "Tied" only when the graph declares it. Identical bytes stay "identical bytes".
- "Opaque" means "LayerDelta has no rule for this op". It is never "unsupported model" or "error".
- "Function-preserving" is said of `permute_units` on a closed group only.

## 7. API fields (contract for G4 / rev7)

The [rev7 sketch](../v0.3-graph-semantics.md) §5 covers most of what the navigator needs. **Used as sketched:** `describe.graph`, row `roles [{axis, role, factors?, provenance, basis}]`, row `binding`, row `groups [{group_id, axis}]`, `coverage.tensors/groups/opaque_ops`; `GET …/graph` (IR summary + paged nodes); `GET …/groups` (`group_id, size, closed, boundary, constraints, members`); group selectors in `read_tensor/stats/diff`, `query where.group`; edit ops on group selectors, `permute_units`, `plan` reporting `group` and `members`, recipe `via`; `side_effects` from `apply`; errors `no_graph`, `group_not_closed`, `constraint_violation`, `invalid_permutation`, `graph_mismatch`.

**Gaps** (requests to G4, priority **P1** = navigator cannot work without it, **P2** = degraded without it):

| # | Field / route | Why | Priority |
|---|---|---|---|
| 1 | IR node `module` (module path from `nn_module_stack`) and `module_type`; for ONNX, the node name scope | Compound frames and the outline. Without it, frames fall back to parameter-name prefixes, which leave parameter-free ops (`cat`, `add`, `relu`) without a home | **P1** |
| 2 | Group `spans`: the activation value dimensions in the group's unit space, `[{value, dim, offset, block, period, repeat}]` | Highlighting edges, not just parameter nodes; cat offsets on edges | **P1** |
| 3 | `GET …/graph?view=modules`: module tree with per-module `node_count`, `tensor_count`, `axes {derived, declared, unknown}`, `opaque_count`, `open_groups`; and nodes filtered by `module_prefix` | Outline and coverage bars without loading every node of a 10k-node graph | **P1** |
| 4 | Group `label` (human-readable, e.g. from the producing module: `enc2 channels`), `kind` (`channel`, `head`, `qk_dim`, `v_dim`, `hidden`), `notes[]` | Group titles, LayerNorm and BN notes (§8 of the semantics doc already plans per-group notes) | P2 |
| 5 | Groups route filters `tensor=`, `node=`, `module_prefix=`; `GET …/graph/nodes/{id}` with `rule` id or `opaque_reason` | "Which groups touch this tensor" from a chip; basis links resolve to a node | **P1** |
| 6 | `plan` preview values: optional `preview: {member index, window selector, max_elems ≤ 10000}` returning before/after values for that window | Value preview before apply without client-side maths. Without it, values appear after apply (4.7) | P2 |
| 7 | `plan` for `permute_units`: `moved_units` (count and list ≤ 1000) | Mapping strip and a sanity check (*"12 of 16 units move"*) | P2 |
| 8 | `describe.graph` gains `providers: [{name, enabled, reason}]` and `reusable: {ir_sha256, checkpoints}` for this signature | Attach card states and the reuse offer | **P1** |
| 9 | Attach job `GET …/graph-attachments/{id}`: `status, stage (building, binding, deriving), done, total, unit, error {code, message, names[]}` | Attaching checklist, `graph_mismatch` names | **P1** |
| 10 | Series hint: IR or modules view `repeat_of` (module structurally identical to a sibling) | Series collapsing without client-side graph hashing | P2 |
| 11 | `stats` over a group selector with `reduce_over_members: "first_out"` (or document which member `query where.group` ranks by) | The unit profile strip for a group | P2 |

## 8. Keyboard and accessibility

| Key | Action |
|---|---|
| `/` | focus search · `Esc` clears, then returns to the outline |
| `↑ ↓ ← → Home End` | outline treegrid (roving tabindex; `→` expands, `←` collapses or goes to parent) |
| `Enter` | outline: select and centre on canvas · canvas frame: go into scope |
| `Backspace` | canvas: up one scope |
| `g` | select the group of the focused role chip or node output |
| `[` `]` | previous / next unit in the selected group |
| `o` | cycle overlay |
| `t` | open the selection in Tensors |
| `p` / `Ctrl+Enter` | plan / apply in the edit panel (apply asks nothing more, the plan is the confirmation) |
| `?` | shortcut sheet |

- The **outline and inspector carry everything** the canvas shows: every frame is an outline row, every member is an inspector row, every highlight is also a text list. The canvas is `role=application` only while focused, with arrow keys moving between nodes in layout order and a live region announcing *"conv2d n7 in enc2.0, member of g_12, unit 5"*.
- Colour is never the only signal: members have outline + badge + list, opaque has hatch + glyph + text, provenance has rule style + word, open groups have `◌` + text.
- Contrast, targets and reduced motion as in tensor-browser §8 (rows 28px, controls ≥ 32px, no animated layout).

## 9. Tokens

All tensor-browser tokens are reused. New, scoped to `.arch-nav`:

```css
--grp-hi: #c6f2e0;            /* member outline, unit-space edges, lit strip cells */
--grp-dim: 0.35;              /* opacity of non-members while a group is selected */
--prov-derived: #9fc7b8;      /* solid left rule */
--prov-declared: #c9b6f2;     /* dashed left rule; distinct hue from data ramps and diff colours */
--prov-unknown: #78897b;      /* hollow chip, muted text */
--opaque-line: #ff8f7a;       /* opaque node border and cut-edge marker (same hue as --diff-mismatch: "cannot follow") */
--opaque-hatch: repeating-linear-gradient(135deg, #ff8f7a33 0 2px, transparent 2px 7px);
--edge: #4a564f;  --edge-label: #a6b1a8;
--frame-bg: #1b2320;  --frame-line: #3d4842;  --frame-head: #202a26;
```

`--prov-guessed` (reserved in v0.2 for inferred roles) stays unused, because v0.3 does not infer.

## 10. Risks

- **Graph size.** Layout of a fully expanded 10k-node graph is slow and unreadable. Mitigation: per-scope layout, collapsed by default, series collapsing, and a hard cap (e.g. 600 visible nodes) above which the canvas asks to expand a narrower scope; the outline always works.
- **Module paths missing** (gap 1) for ONNX or partial exports. Mitigation: fall back to name scopes, then to parameter prefixes, and say so in the outline header.
- **Wrong groups claimed as derived.** The semantics doc's A3 permutation tests are the guard; the UI shows basis and boundary so a wrong group can be traced to a rule.
- **Q1 undecided.** Most real user models need `torch_export`. Until Q1, the attach card is honest that only `builtin` (and `onnx` once installed) are available, and the navigator is demonstrated on the corpus and fixture.
- **Edit panel trust.** Users may read "plan" as "safe". The preview leads with refusals and side effects, and apply always creates a new child (never in place), which the copy says.
