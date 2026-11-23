# LayerDelta — Target architecture (v0.2+)

Status: **adopted direction** · authored by the Senior AI Scientist, adopted by the PM on 2026-11-23 with one change: the human Desktop surface ships with every milestone (see [Roadmap](../roadmap/Roadmap.md)) instead of all at v0.6. The current, implemented system is described in [Architecture.md](Architecture.md).

## 0. Executive summary

1. **Split the product into three layers that today are fused:** (A) *tensor space* — the parameter file itself; (B) *graph semantics* — what each tensor axis means and which axes are coupled; (C) *execution/evaluation* — running the model on data for a task. v0.1 needs all three for every operation because `ReferenceCNN` is the only source of truth. Weight inspection, comparison and editing need only A plus (optionally) B. Only activation edits and metrics need C.
2. **Highest-leverage first milestone (v0.2 "Tensor-space core"):** import *any* safetensors checkpoint without executing code, and give humans and agents deterministic inspect / diff / select / transform / validate / export over it, with recipes v2 that replay to a byte-identical child. That alone works for CNN, U-Net, OCR, detection, segmentation and transformers on day one, at "tensor-name" granularity. Graph semantics (axis roles, coupling, U-Net skips) arrive in v0.3 on top of the same selector model.
3. **Do not write our own graph IR or a model-by-model adapter zoo.** Use `torch.export` (PyTorch, primary) and ONNX (secondary, no-code-execution path) as graph sources; derive axis roles and coupling with a small per-op rule table (pluggable). Model "adapters" shrink to optional declarative *profiles* (labels, packed-tensor splits).
4. **v0.2 plan as written should be reordered:** keep the `weights:read` leak fix and a minimal event trace; replace Conv-specific `preview_weights` with the generic `read_tensor`; **park the two-arm benchmark (M5.3)** — it measures one 2-layer MNIST CNN with n≈47 target samples and the scripted pilot already showed no signal (−0.20 pp); it does not inform the new direction.

---

## 1. Where v0.1 is hardwired (trace)

| Site | Hardwiring | Disposition |
|---|---|---|
| `model.py` `ReferenceCNN`, `target_catalog` | Targets = (conv kernel, post-ReLU) pairs found by walking `features.*.conv` | Becomes a **test fixture**. Catalog replaced by generic tensor table + roles |
| `model.py` `describe_model` | Forward hooks on a 1×28×28 zero input | Generalized to graph introspection (v0.3) |
| `checkpoint.py` `load_checkpoint` | Builds `ReferenceCNN()`, strict key/shape match, `F32` only, 1 MiB / 16 KiB caps, whole-file read | **Keep the validation discipline** (header parse, unique fields, contiguous non-overlapping extents, finite check); drop the model construction; switch to bounded header parse + mmap/`safe_open` per tensor; configurable caps; F16/BF16/F32 (+ int types read-only) |
| `domain.py` `_targets`, `normalize_intervention`, `_dependencies` | One target, one channel, `zero|scale`, scale ∈ [0,2]; dependency block names MNIST, blur, adapter id | Recipe **schema v2** (edit program); v1 kept importable via translation |
| `engine.py` `_channel_view` | `ndim == 4 and float32`; one channel axis | Generic selector (per-axis index sets) |
| `engine.py` `preview` | `target.module_path.rsplit(".",1)` to pair relu↔conv; image 1×1×28×28 | Retire; replaced by `read_tensor` / `stats` (weights) and a later generic activation probe |
| `engine.py` `infer` | 28×28 input, 10 logits | Task evaluator plugin (v0.4) |
| `evaluation.py` `compare(…, truth, confused_with)` | 10-class digits, `normal|blurred`, `exploration|holdout` | Classification evaluator plugin; other tasks get their own |
| `job.py` `checkpoint_path`, `run` | Everything resolved via `workspace/"reference"` fixture; fixture seed/digest mutated for uploaded checkpoints | Checkpoint catalog independent of the fixture |
| `api.py` `/previews` | `targets` dict from fixture; `sample_id` required | New generic endpoints alongside; old ones frozen |

Keep untouched (they are the real asset): `canonical_hash`, SQLite store with idempotency/quota/holdout audit, isolated child process + publish protocol (`store._publish`), scoped tokens and loopback/Origin checks, immutable content-addressed checkpoints, paired evaluation *concept*.

---

## 2. Extensible import architecture

### 2.1 Framework coverage (ordered)

| Order | Format | Why | Code execution? |
|---|---|---|---|
| 1 | **safetensors** (single + sharded `*.index.json`) | De-facto weight interchange (HF, timm, diffusers, ultralytics export); already our format; header is JSON so it is cheap and safe to inspect | **No** |
| 2 | **PyTorch module + `torch.export`** | Gives graph + param↔node binding for semantics and execution; covers the families the PM listed | **Yes** — user-supplied module factory; runs only inside the isolated worker, opt-in per project, never for agent tokens |
| 3 | **ONNX** (`initializers` + graph) | Graph semantics without Python code; many detection/OCR models ship as ONNX | No (protobuf parse; inference via onnxruntime later) |
| — | `.pt/.pth/.bin` pickle | **Refuse** (arbitrary code on load). Offer a one-shot conversion CLI the user runs themselves, with a warning | — |
| later | GGUF, TF/Keras, quantized formats (GPTQ/AWQ/int8+scale) | Read-only first; quantized tensors are not linearly editable | — |

### 2.2 Representation

```
Checkpoint   := content hash (sha256 of file set) + format + [Tensor]
Tensor       := name, dtype, shape, byte extent, tensor_sha256 (lazy), roles?: AxisRoles
AxisRoles    := per axis one of {out_unit, in_unit, spatial, vocab, head, gate, packed(k), batch-free, unknown}
                + provenance {declared(profile) | derived(graph rule) | inferred(heuristic)} 
CouplingGroup:= id, unit_count, members: [(tensor, axis, offset, stride)], boundary: closed | opaque(op)
Graph (opt.) := nodes (op, inputs, outputs, param bindings) from torch.export / ONNX
```

- Layer A (tensor table) exists for every checkpoint. Layer B (roles, coupling groups) exists only when a graph source or profile is attached; otherwise roles are `unknown` and edits are tensor-local only.
- **Provenance is mandatory** on every role. Agents must see whether "axis 0 = output channel" was derived from a graph or guessed from a name.

### 2.3 Deriving targets and channel axes generically (op rule table)

Rules keyed by graph op (aten op for torch.export, ONNX op type). Each rule maps parameter axes to roles and emits coupling edges between the producer's `out_unit` and consumers' `in_unit`.

| Op | Weight layout → roles | Coupling notes |
|---|---|---|
| conv{1,2,3}d | `[O, I/g, k…]` → out_unit, in_unit(per group), spatial | groups: in_unit coupled per group; depthwise (g=I=O) ties out and in |
| conv_transpose | `[I, O/g, k…]` → **axis 1 is out_unit** | Classic trap; must come from the rule, never from "axis 0 = out" |
| linear / matmul with param | `[O, I]` | Flatten before linear: in_unit = C×H×W, coupled to channel c as a strided block (offset c·H·W, stride 1, len H·W) — requires shape propagation |
| batch/layer/group/instance norm | `[C]` weight, bias, running stats | BN tied to preceding conv out_unit; LayerNorm ties hidden dim across the whole residual stream; GroupNorm couples units in groups |
| add (residual) | — | **Ties** out_units of both branches into one group (ResNet, U-Net decoder adds) |
| cat (U-Net skip) | — | Consumer in_unit splits into **offset partitions**: [0,C_skip) coupled to encoder, [C_skip, C_skip+C_up) to decoder path |
| multi-head attention | packed `in_proj [3E,E]` or separate q/k/v | `packed(3)` + `head` roles: E = H·d; head h = rows h·d…(h+1)·d of q,k,v and **columns** of out_proj |
| LSTM/GRU (CRNN-OCR) | `[4H,I]` / `[3H,I]` | `gate(k)` packing; hidden unit couples across gates and recurrent weight both axes |
| embedding | `[V, D]` | vocab, hidden |
| anything unknown | — | **opaque boundary**: tensor-local edits allowed; coupled edits (prune/permute a unit) that would cross it are **refused** with `unsupported_op:<name>` |

This is essentially the dependency-graph idea from structured pruning (Torch-Pruning "DepGraph", Fang et al. 2023). **Recommendation: evaluate vendoring/depending on Torch-Pruning's dependency graph** before writing ours; our rule table then becomes a thin adapter plus the provenance/opaque-boundary policy.

### 2.4 Adapters vs generic introspection; plugins

- Default path = generic introspection. No per-model Python classes.
- **Profile** (optional, declarative JSON/TOML, versioned, hashed into the recipe): human labels ("encoder.down2"), packed-tensor splits the graph cannot reveal (e.g. fused qkv in a custom kernel), role overrides, and the input spec needed to run `torch.export` (example shape/dtype).
- **Plugin entry points** (Python `importlib.metadata` entry points, discovered only in the worker):
  - `layerdelta.op_rules` — extra op → roles/coupling rules.
  - `layerdelta.evaluators` — task evaluators (v0.4).
  - `layerdelta.formats` — extra readers (GGUF, ONNX writer).
  Each plugin declares a name + version; versions enter fingerprints so recipes stay reproducible.
- Unsupported-op handling = the opaque boundary above, plus `describe` reports `coverage: {tensors_with_roles / total, opaque_ops: [...]}` so users/agents know how much of the model is semantically understood.

---

## 3. Generic weight-space operation model

### 3.1 Selector (the single addressing primitive)

```json
{"checkpoint": "sha256:…", "tensor": "decoder.up2.conv.weight",
 "index": {"0": [17], "1": {"start": 0, "stop": 32}}}          // tensor-local
{"checkpoint": "sha256:…", "group": "cg:enc2.out", "units": [17]} // coupled (v0.3)
```
A group selector expands deterministically to tensor-local selectors (all coupled tensors/axes). Selections are canonicalized (sorted, ranges merged) and hashed → `selection_id`.

### 3.2 Operation families

| Family | Ops | Side effects |
|---|---|---|
| **inspect** | `describe` (tensor table, roles, groups, coverage), `read_tensor(selector, max_elems)` with cursor, `stats(selector, reduce_over_axes)` → min/max/mean/std/L1/L2/zero_frac/histogram per unit | none, cacheable by (checkpoint, selector, op_version) |
| **compare** | `diff(a, b, selector?)` → per tensor: changed_elems, max_abs, rel L2, cosine; per unit when roles exist; `align(a,b)` name/shape matching report | none |
| **select** | `query(predicate)`: by name glob, role, group, and stat predicates (e.g. "out_units of conv tensors with L2 < τ"); returns ranked, deterministic `selection_id` | none (selection stored as a pure value) |
| **transform** | `zero`, `scale(k)`, `set(value|blob)`, `add(delta blob)`, `clamp`, `noise(seed,σ)` (seeded, deterministic on CPU), `permute_units(group, perm)` (function-preserving), `copy_from(other_ckpt, selector)` (transplant/interpolate `lerp(a,b,t)`) | **planned** only; materialized by `apply` |
| **validate** | static: finite, dtype range overflow (F16/BF16), shape preserved, *touched-mask ⊆ selection*, coupled-edit consistency, no edit crosses opaque boundary; dynamic (v0.4): evaluator delta vs guardrails | none |
| **apply/export** | `plan(program)` = dry run → effect report; `apply(program)` → new immutable child checkpoint + manifest | writes one child; reported |
| **undo/reproduce** | Parents are immutable, so undo = "use parent id". Reproduce = replay recipe on parent → must yield identical child sha | none / one child |

### 3.3 Recipe and fingerprint v2

```json
{"schema_version": 2,
 "base": {"checkpoint_sha256": "…", "format": "safetensors", "profile_sha256": null},
 "program": [{"op": "scale", "args": {"k": 0.5}, "select": {…}}, …],
 "engine": {"layerdelta_ops_revision": 1, "plugins": {"op_rules": "…@1"}, "numeric": "cpu-f32"},
 "hypothesis": "…",
 "evaluation": null | {"evaluator": "classification@1", "dataset_sha256": "…", "slices": […]}}
recipe_hash = canonical_hash(recipe without recipe_hash)
child manifest = {parent, child_sha256, recipe_hash, touched: [{tensor, elems, before_sha, after_sha}]}
```
- Program = ordered list; ops apply sequentially on a working copy; each op's selector is resolved against the **base** tensor table (not the intermediate) to keep resolution order-independent; overlapping writes are allowed but reported.
- Determinism: transforms run on CPU in the tensor's stored dtype with explicitly specified rounding (compute in f32, cast back) → bitwise-reproducible child; GPU never used for `apply`.
- v1 recipes: a translator maps `{mode: weight, target_id, channel_index, operation}` → v2 program of one op with selector `{tensor: "<conv>.weight", index: {"0": [c]}}`. Acceptance: child bytes equal v0.1 export for `examples/recipes/zero-block2-channel17.json`. Activation-mode v1 recipes stay on the v1 path until v0.4.

### 3.4 Safety checks (all at backend, before any write)

Selector bounds & caps (`max_elems` per read, per program, per child bytes); program length cap; no write to base; finite + dtype-overflow check; touched-mask equality check after apply (the v0.1 "diff limited to slice" check, generalized); refuse coupled edits across opaque ops; refuse editing quantized/packed tensors without a declared split; budget/quota per agent session; holdout-gated evaluation unchanged.

---

## 4. Agent tool surface

Principles: every tool is **one HTTP endpoint ≡ one typed client method ≡ (later) one MCP tool**; requests strict-schema (`extra=forbid` as in `schemas.Body`); every response has

```json
{"schema_version": 1, "resolved": {…canonical inputs…}, "result": {…},
 "fingerprints": {"checkpoint": "…", "selection": "…", "op_revision": 1},
 "side_effects": [] | [{"kind": "checkpoint_created", "id": "…", "bytes": …},
                       {"kind": "quota_charged", "runs": 1}],
 "truncated": false, "cursor": null}
```
Pure tools return `side_effects: []` always; that makes "safe to call freely" machine-checkable.

| Tool | Capability | Pure? |
|---|---|---|
| `list_checkpoints`, `describe(checkpoint)` | `model:read` | yes |
| `read_tensor`, `stats`, `diff`, `query` | `weights:read` | yes |
| `plan(program)` | `weights:read` | yes (dry run, effect report) |
| `apply(program, idem_key)` | `weights:write` + quota | no → child id |
| `validate(checkpoint, checks)` | `model:read` | yes |
| `evaluate(checkpoint, evaluator, split)` (v0.4) | `eval:exploration` | async job, quota |
| `history/trace(session)` | session | yes |

Composability: outputs return ids (`selection_id`, `checkpoint_id`) that are valid inputs elsewhere; agents never pass paths or raw tensors back in (deltas go through a content-addressed blob upload with size cap).

**Fit with the v0.2 plan:**
- M5.1 step 2 (`weights:read` capability, strip kernel stats from activation preview and from `_patch_kernel` report for unprivileged tokens) — **keep, do first, small**; it is a real leak in shipped code.
- M5.1 `preview_weights` (Conv-specific `[C_out, C_in/groups, K_h, K_w]`) — **replace** by generic `read_tensor`/`stats`; Conv is just one role layout.
- M5.1 heatmap — keep later, rendered from `read_tensor` payload (v0.6 UI).
- M5.2 trace — **keep a minimal version in v0.2** (events table: session, seq, tool, request id, status, ids/fingerprints; never tensor values/tokens). Cheap and makes every later agent experiment auditable.
- M5.3 benchmark + provider adapter — **park** until v0.5; re-scope to multi-family tasks. D1/D4/D5 decisions become moot for now; D6 (no tensor egress by default) stays as policy.

---

## 5. Milestone ladder (capability-driven, no dates)

### v0.2 — Tensor-space core  ← **highest leverage**
- **Goal:** any safetensors model is inspectable, comparable and editable without code execution.
- **User value:** works today for U-Net/YOLO/TrOCR/ViT checkpoints people already have; agents get deterministic weight-space programming.
- **Capability:** streaming loader, tensor table, selectors, inspect/diff/query, edit programs, recipe v2, child export + manifest, `weights:read/write`, minimal trace.
- **Acceptance:** see §6.
- **Dependencies:** none beyond current store/API.
- **Risks:** memory on multi-GB files (mitigate: mmap/`safe_open`, per-tensor streaming, child write by copying untouched byte ranges); BF16/F16 rounding in transforms; shared/tied tensors (safetensors forbids aliasing → tied weights appear once or duplicated; edits to one copy silently untie) → detect equal-hash duplicate tensors and warn.

### v0.3 — Graph semantics & coupling
- **Goal:** axis roles + coupling groups from `torch.export` and ONNX via the op rule table; profiles.
- **Value:** "channel 17 of encoder block 2" means the same unit across conv, BN, next conv, skip concat; function-preserving ops (`permute_units`), structured prune masks.
- **Acceptance:** on a corpus (ResNet-18, small U-Net with concat skips, ConvTranspose decoder, ViT-tiny, CRNN w/ LSTM, one detector backbone+neck): (1) every param bound to a node or flagged; (2) `permute_units` on every closed group leaves outputs equal within 1e-5 on random inputs; (3) zeroing a unit via group equals zeroing it in the original module by hand; (4) ConvTranspose out axis = 1; (5) opaque ops listed, crossing edits refused.
- **Dependencies:** v0.2 selectors. **Risks:** `torch.export` fails on data-dependent control flow (detection post-processing/NMS, some OCR decoders) → export the backbone/neck only, mark the rest opaque; ONNX constant folding fuses BN into conv so ONNX names ≠ training checkpoint names (need name-mapping report); Torch-Pruning dependency maturity/licensing.

### v0.4 — Execution & task evaluators
- **Goal:** run edited models on data; paired before/after evaluation through plugins: classification (current MNIST becomes one fixture), segmentation (mIoU/Dice), detection (mAP via pycocotools-compatible), OCR (CER/WER).
- **Value:** closes the loop "edit → measure" for each family; generic activation probes/edits via graph nodes.
- **Acceptance:** MNIST fixture results unchanged (G2–G9 regression tests pass via the plugin); each evaluator has a golden oracle on a tiny dataset; holdout gating unchanged.
- **Dependencies:** v0.3 for activation targets (weight-only eval works with v0.2 + a runnable module). **Risks:** datasets/preprocessing are the true per-model hardcoding; trust boundary of user module code; GPU nondeterminism (keep "predicted class / integer counts" replay criterion).

### v0.5 — Agent surface hardening + research benchmark
- MCP bridge mapping the same tool table; per-session budgets on read volume; trace timeline; provider adapter with D6 egress policy; *then* a benchmark across ≥3 families with pre-registered N. Acceptance: same as G12/G13 generalized; tool-schema contract tests; MCP ≡ HTTP parity test.

### v0.6 — Desktop generalization
- Tensor browser, role-aware heatmaps, diff views, recipe editor, all from the same payloads. Human pilot is scheduled here (just before release), per PM.

### Later
ONNX write-back, quantized formats (edit in dequantized space + requantize with error report), merges (task arithmetic, model soups), training-in-the-loop.

### Keep / generalize / retire summary
- **Keep:** store/supervisor/child isolation, publish protocol, idempotency/quota/holdout, scoped tokens, canonical hashing, safetensors header validation rules, paired-evaluation idea, existing tests as regression.
- **Generalize:** `Target` → selector + roles; `normalize_intervention` → op registry; recipe v1 → v2 (v1 translator); `load_checkpoint` → streaming reader; `describe_model` → graph introspection; `evaluation.compare` → classification evaluator plugin; `job.checkpoint_path` → catalog not tied to fixture.
- **Retire/freeze:** `ReferenceCNN` as product core (stays as fixture), `engine.preview` relu↔conv pairing, hard 28×28/10-logit checks outside the MNIST plugin, Conv-specific `preview_weights`, M5.3 benchmark in its current form.
- **Still worth it from v0.2:** `weights:read` split (yes, first), minimal trace (yes), `preview_weights` (only as generic `read_tensor`), heatmap (later), benchmark/provider (park).

---

## 6. First milestone (v0.2 Tensor-space core) — engineering tasks

**T1 · Streaming safetensors reader + tensor table**
- Bounded header parse reusing `checkpoint.py` rules (unique fields, contiguous non-overlapping extents, size caps now configurable, default e.g. 16 GiB file / 100 MiB header); sharded index support; dtypes F32/F16/BF16 editable, ints/bool read-only; lazy per-tensor sha256; duplicate-tensor (tied) detection. No model construction.
- Accept: reference checkpoint table equals `ReferenceCNN().state_dict()` names/shapes; all existing `test_checkpoint.py` malformed-input cases rejected with the same codes; a ~2 GB synthetic file is described with RSS < 200 MB.
- Tests: port malformed-header cases; BF16 round-trip; sharded index with missing shard rejected; overlapping/gapped extents rejected; RSS bound test (marked slow).

**T2 · Selector + pure inspect ops (`describe`, `read_tensor`, `stats`, `diff`, `query`)**
- Canonical selector schema, `selection_id`, caps + cursor pagination, deterministic float formatting (JSON numbers from f32 → shortest repr).
- Accept: `read_tensor` values equal `tensor[index]` for first/last index on each axis; out-of-range/negative/unknown tensor → 422 with field path; `diff(x, x)` all-zero; `diff(parent, v0.1 child)` reports exactly 1 changed out-unit; repeated calls byte-identical.
- Tests: property tests (hypothesis lib if present, else parametrized) on random shapes; pagination concatenation equals full read.

**T3 · Edit program, transforms, validator, child writer (recipe v2)**
- Ops: zero, scale, set, add(blob), clamp, lerp/copy_from; `plan` effect report; `apply` writes child by copying untouched byte ranges and re-encoding touched tensors; manifest with per-tensor before/after sha; v1→v2 translator.
- Accept: **v1 translator child sha == v0.1 export sha for `zero-block2-channel17`**; touched-mask == selection for every op; scale(1) child sha == parent sha (or no-op detected); non-finite/overflow → refused before write; replay of recipe on another workspace gives same child sha.
- Tests: golden v1 equivalence; per-op touched-mask test; BF16 scale rounding golden; crash-during-write leaves no visible child (reuse publish protocol tests).

**T4 · API/client/store integration with capabilities + side-effect reporting**
- New `/api/v1/projects/{p}/checkpoints/{id}/…` endpoints and `AgentClient` methods; `model:read`, `weights:read`, `weights:write` on tokens; strip kernel stats from legacy `/previews` and from agent-visible reports without `weights:read` (the M5.1 leak fix); `apply` idempotent + quota; legacy endpoints unchanged.
- Accept: token without `weights:read` gets 403 from every weight endpoint **and** sees no weight stats via previews/reports/catalog (enumerate all GET routes in a test); idempotent `apply` returns same child and charges once; every response has `side_effects`, pure tools always `[]`.
- Tests: route-enumeration leak test; idempotency/concurrency (reuse `test_m3_http.py` patterns); schema contract snapshot.

**T5 · Minimal session trace**
- `events` table (session, seq, tool, request_id, status/code, ids & fingerprints, selection_id; no values/tokens/images); survives restart.
- Accept: from an `apply` event you reach its `plan`, inputs' selections and resulting child; grep of DB for the agent token and for any tensor value returns nothing.
- Tests: restart persistence; redaction test.

**T6 · Cross-family corpus test (the proof of generality)**
- Generate small random-init checkpoints in-test (no downloads): torchvision ResNet-18, a toy U-Net with concat skips + ConvTranspose, ViT-tiny (packed qkv), CRNN (conv + LSTM + linear), a YOLO-like backbone/neck; save as safetensors.
- Accept: each loads, describes, diffs, applies a 3-op program and replays to identical sha, with zero model-specific code in `src/`.
- Tests: one parametrized test over the corpus; runs on CPU in < 60 s.

Order: T1 → T2 → T3 → T4 (T5 parallel with T4) → T6 continuous from T2 onward.

---

## 7. Scrutiny of this proposal (scrutinize skill, applied to myself)

**Intent in one sentence:** make weight inspection/editing model-agnostic by separating tensor space from graph semantics and execution.

**Simpler alternatives considered**
- *Do nothing / keep adapters:* each family costs a model class, loader, catalog, evaluator, fixture (adapter-guide §1–5). Linear cost per model; contradicts the new goal. Rejected.
- *Just be a library over existing tools (safetensors + Torch-Pruning + torch.export) with an MCP server, no desktop/store:* **genuinely competitive** for the "agents edit weights" goal and much cheaper. The store/holdout/quota machinery earns its keep only if paired evaluation and auditability remain core value. I recommend keeping it, but PM should treat "library + MCP first, desktop later" as an explicit decision, not a default.
- *Use ONNX as the single IR:* tempting (no code exec) but ONNX export constant-folds/fuses (BN→conv), renaming and merging the very parameters users want to edit. Fine as secondary source, wrong as primary.

**Assumptions and what is speculative**
1. *"Tensor-level editing without graph semantics is useful."* Plausible for diff/inspection and whole-unit edits when names are conventional; **weak** for meaningful structured edits — without roles, "channel" is a guess. Risk: v0.2 looks generic but users still can't do semantically correct U-Net edits until v0.3. Mitigation: provenance labels; T6 corpus proves mechanics, not meaning.
2. *"Agents benefit from weight-space programming."* **Unproven.** The only data point (scripted pilot, one task) showed −0.20 pp and identical choices across arms. This whole direction rests on a hypothesis we have not tested; v0.5 benchmark is where it gets tested, and it could come back null.
3. *Coupling derivation via op rules covers the listed families.* **Partly speculative.** Residual/concat/BN/attention are well understood (DepGraph); detection heads, deformable conv, custom CUDA ops, and dynamic control flow will be opaque. Coverage %, not a yes/no, is the honest metric.
4. *`torch.export` works on user models.* Often fails on data-dependent control flow (NMS, beam search, some OCR decoders). Export-a-subgraph is a workaround, not a guarantee.
5. *Bitwise reproducibility.* Holds for CPU elementwise ops on stored dtype; does **not** extend to evaluation on GPU — keep v0.1's class/count replay criterion for metrics.
6. *Memory/perf on large checkpoints.* safetensors mmap on Windows behaves differently from Linux (file locks, page cache); the "copy untouched byte ranges" child writer must be validated on Windows specifically.
7. *Tied weights.* safetensors cannot express aliasing; HF stores tied weights once and relies on code to tie. Without code we cannot know an edit to `lm_head` also edits `embed_tokens`. Need profile-level declaration; flag as known gap.
8. *Security.* Refusing pickle is right, but v0.3/v0.4 still execute user module code for `torch.export`/inference; that is a new trust boundary vs v0.1 ("checkpoint supplies tensors only", adapter-guide). Needs explicit opt-in and isolation, and agents must not be able to trigger it.

**Seams I traced and where the plan touches reality**
- `job.run` mutates `fixture["training"]["seed"]`/artifacts for uploaded checkpoints and routes everything through `workspace/"reference"` — the checkpoint catalog must be decoupled before T4, otherwise new endpoints inherit fixture coupling.
- `api.submit_preview` validates targets against the fixture's `targets` dict — the generic endpoints must not reuse that path.
- `engine.intervention` restores weight edits by clone/copy within one process — v2 `apply` writes children instead, so the v1 in-process weight path remains only for legacy runs; two weight-edit implementations will coexist until v0.4. The v1→v2 sha-equivalence test (T3) is what keeps them honest.

**Verdict:** fix-then-ship as a direction. The single biggest open question is not technical but scientific: whether weight-space tools help agents at all (assumption 2). v0.2 is still the right first step because it is cheap, model-agnostic, and is the instrument needed to test that question on more than one 2-layer MNIST CNN.
