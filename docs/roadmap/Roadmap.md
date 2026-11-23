# LayerDelta — Roadmap and Project Status

This document is the primary source for milestones and current status · Last updated 23 Nov 2026

## Product direction

Make neural-network weights easy to **understand, inspect and edit**, for both humans and AI agents, with a full audit trail. The work covers several model families (CNN, U-Net, OCR, detection, segmentation and others) without writing code for each model. The target architecture has three layers ([Architecture_Target](../architecture/Architecture_Target.md)):

| Layer | Meaning | Must run the model's code? |
|---|---|---|
| A · Tensor space | All parameter files: read, compare, select, transform, verify and export | No |
| B · Graph semantics | The meaning of each axis (out/in unit, head, gate) and the groups of axes that are tied together (residual, U-Net skip, BN), taken from `torch.export`/ONNX through a pluggable per-op rule table | Export only |
| C · Execution | Run the model on data and measure the before/after result with a task evaluator (classification, segmentation, detection, OCR) | Yes |

**Milestone principle:** milestones are split by capability, not by date. Every milestone delivers the same capability through both the **API/agent** and the **Desktop**, using the same payload. The `0.x` number only increases when the system can do something new.

**Release policy:** do not release, publish, tag a release or declare production readiness until the project owner gives an explicit instruction. The human pilot is moved into the *Release readiness* milestone and is not a dependency of current work.

## Status summary

| Milestone | Status |
|---|---|
| v0.1 Reference workbench | **Done** (internal; build `0.1.0rc1` not published) |
| v0.2 Tensor-space core | **Done** (capability complete, internal; not published) |
| v0.3 Graph semantics & coupling | **Done** (internal; not published) · [board](Implementation_Plan_v0.3.md) |
| v0.4 Agent weight-space programming | **In progress** · backend done: MCP bridge, budgets, egress classes, headless runner, supervision, lineage (S1, S3 to S6); S2 content and S7 cookbook open ([board](Implementation_Plan_v0.4.md)) |
| v0.5 Evaluators + benchmark · v0.6 Human editing | Not started (agent-first order) |
| Release readiness | Parked until the project owner instructs otherwise |

Task details and owners for the current milestone are in [Implementation_Plan_v0.3](Implementation_Plan_v0.3.md); the completed v0.2 task board is in [Implementation_Plan_v0.2](Implementation_Plan_v0.2.md). Questions waiting for a decision from the project owner are in [QUESTION.md](../../QUESTION.md).

## Milestones

### v0.1 · Reference workbench — Done

- **What it does:** works with a single hardwired MNIST CNN. Experiments: zero or scale one activation or kernel element. Paired comparison of results, replay, export of a child checkpoint, the scoped Agent API, and holdout audit. Runs on Windows CPU/CUDA.
- **Evidence:** G1–G9 pass ([m1](../history/m1-measurements.md)–[m4](../history/m4-validation.md)). The test suite of 88 cases passes.
- **Carried forward to the next release:** store/supervisor, child-process isolation, publish protocol, idempotency/quota/holdout, scoped token, canonical hash, safetensors header validation rules.
- **To retire or freeze:** `ReferenceCNN` as the core of the product (kept as a test fixture), and the relu↔conv matching in `engine.preview`.

### v0.2 · Tensor-space core — Done

- **Goal:** any safetensors checkpoint can be inspected, compared and edited, without running code from the file.
- **Value to users:** existing checkpoints of U-Net, YOLO, TrOCR and ViT can be opened immediately at the tensor-name level. Agents get a deterministic set of weight-space tools.
- **Capabilities:** streaming reader and tensor table, selectors, `describe`/`read_tensor`/`stats`/`diff`/`query`, edit program + recipe v2 + child writer, the `weights:read`/`weights:write` capabilities (including closing the places where weights leaked in the v0.1 code), minimal trace, Desktop tensor browser.
- **Acceptance:**
  1. A corpus of toy models defined in `tests/` uses seeded initialization and adds no dependency (no timm or ultralytics): ResNet-18-like, U-Net with concat skips and ConvTranspose, ViT-tiny, CRNN, and a YOLO-style backbone. All of them must load, describe, diff and apply a 3-op program, and replay to the same child SHA. `src/` contains no model-specific code.
  2. When recipe v1 `zero-block2-channel17` is converted, the child SHA equals the child that v0.1 exported. (The writer must keep the parent's header layout.)
  3. A non-MNIST checkpoint can be imported both from a local path and from a streamed upload, without loading the whole file into memory. The catalog is not tied to the reference fixture.
  4. A token without `weights:read` cannot see weight values or statistics **directly** through any route. (Tested by walking every route. The activation map plus the image can still be used to recover weights algebraically; see R2.)
  5. The response of every **v2** tool contains `resolved` and `fingerprints`, and a tool with no side effects always returns `side_effects: []`. Legacy routes keep their existing format.
  6. Desktop shows the tensor table, slices and diff from the same payload as the v2 API. It has smoke tests on the reference fixture and on at least one corpus model, and the UX/UI Designer has approved it.
  7. All existing v0.1 tests still pass.
- **Dependencies:** nothing new beyond the existing store/API.
- **Risks:** memory with multi-GB files (mmap on Windows); rounding of BF16/F16; tied weights that safetensors cannot represent; "channel" as a concept that has no graph semantics yet, which is only a guess from the name (it must show its provenance).

**Priority (owner decision, 2026-11-23): the agent surface comes first.** Each milestone ships agent-facing capability first. Desktop work is limited to what helps a human supervise or review agent work, plus work already in flight.

### v0.3 · Graph semantics & coupling — Done

- **Goal:** get axis roles and coupling groups from `torch.export` (primary; user code is opt-in per project in an isolated worker, per Q1) and ONNX (no-code alternative), using a pluggable per-op rule table. Unknown ops become opaque boundaries.
- **Value:** "channel 17 of encoder block 2" means the same unit in the conv, the BN, the next conv and the skip concat. Agents and humans can make structured edits correctly, for example `permute_units` that preserves the function.
- **Agent surface:** graph/groups/node routes and group selectors (G4, done); group edits and `permute_units` (G5).
- **Desktop:** a read-only architecture navigator (G8b phase 1). Unit-level editing from the navigator moves to v0.6.
- **Acceptance:** A1–A11 of the [v0.3 design](../architecture/v0.3-graph-semantics.md) on the T6 corpus, plus a user module attached end to end through `torch_export` (G6) and a corpus family through ONNX (G7).
- **Risks:** `torch.export` fails on data-dependent control flow (NMS, beam search). The workaround is to export only the backbone/neck. ONNX fuses BN into conv, so names no longer match the checkpoint.

### v0.4 · Agent weight-space programming

- **Goal:** agents use LayerDelta as a complete, composable tooling layer for weight-space work.
- **Capabilities:** an MCP bridge (1 endpoint ≡ 1 client method ≡ 1 MCP tool) with contract tests; tool schemas and structured errors designed for LLM use; composable multi-step workflows (query → plan → apply → diff) with side effects reported; per-session read and write budgets; the trace timeline; provider adapters with an outbound data policy (by default no tensors or images leave the machine); an agent cookbook with runnable examples.
- **Desktop (supervision only):** a session/trace viewer, so a human can review what an agent read, planned and applied.
- **Acceptance:** MCP results match HTTP (parity test); tool schema contract tests; G12 trace criteria in generalized form; a scripted agent completes a multi-family edit task through MCP alone.
- **Risks:** the main assumption, that weight tools help agents, **has no evidence yet** (the v0.1 pilot showed −0.20 pp, n = 1). It is tested in v0.5.

### v0.5 · Evaluators as agent tools + research benchmark

- **Goal:** agents can measure the effect of an edit on any supported family. Evaluator plugins cover classification (MNIST becomes one fixture), segmentation (mIoU/Dice), detection (mAP) and OCR (CER/WER), plus a generic activation probe/edit on graph nodes. All of it is exposed as agent tools.
- **Research:** a benchmark of whether seeing weights helps an agent, across at least 3 model families with a pre-registered N. A null result is also a valid result to report. The original spec is in [LayerDelta_Agent_Extension](../history/LayerDelta_Agent_Extension.md).
- **Acceptance:** MNIST results do not change through the plugin; every evaluator has a golden oracle on a small dataset; holdout gating as before; benchmark report with uncertainty.
- **Risks:** dataset and preprocessing still need model-specific code; GPU non-determinism (class and count are the replay criteria).

### v0.6 · Human editing workflow at scale (lower priority)

- **Goal:** complete weight editing without writing code: unit-level edits from the navigator, a program/recipe editor, history and undo, a multi-child diff, safety preview before apply, models with billions of parameters, accessibility.
- **Acceptance:** to be defined with the UX/UI Designer when v0.5 is finished.

### Release readiness — Parked (starts only when the project owner instructs)

Packaging of the new capabilities, the human pilot under the [protocol](../history/human-pilot.md), testing on a machine without WebView2, manual testing of the folder picker, release notes, and the license/provenance of the new artifacts.

## Risk register

| Risk | Impact | Mitigation | Milestone |
|---|---|---|---|
| R1 No evidence yet that weight tools help agents | The research direction may give a null result | Assumption (not yet tested, because the pilot is parked): the v0.2 tools are useful to people even if the research result is null. The v0.5 benchmark uses several model families and pre-registration. | v0.5 |
| R2 Weights leak to tokens that lack permission (already present in the code) | Permissions do not work in practice, and the benchmark is unfair | v0.2-T4a closes the direct path (kernel stats). v0.2-T4c closes the activation map and original image of the preview (which reveal weights algebraically) for tokens without `weights:read`. Every route is covered by tests that walk all routes. Still open: the benchmark runner (v0.5) must stop the agent from accessing the workspace filesystem (`report.json`, preview cache). | v0.2 → v0.5 |
| R3 Graph/op coverage is incomplete | Users wrongly believe the system knows the meaning of each axis | Mandatory provenance, coverage numbers, opaque boundaries | v0.3 |
| R4 Running the user's code is a new trust boundary | Security | Opt-in, run in an isolated worker, the agent cannot trigger it. Owner chose Q1 option 3; shipped in G6. | v0.3–v0.4 |
| R5 Large checkpoints on Windows (mmap, file locks) | Cannot be used with real models | Test the RSS bound in T1, verify the child writer on Windows | v0.2 |
| R6 Two weight-edit implementations (v1 in-process, v2 child writer) | Results do not match | Test SHA equality between v1 and v2 | v0.2–v0.4 |

## Changes in this round (23 Nov 2026)

- Changed the roadmap from a date-based "v0.1 → release" plan to a capability ladder from v0.2 to v0.6.
- Moved the human pilot, the WebView2/folder-picker testing and the tag/release into *Release readiness* (parked).
- Redefined v0.2: it was a weight preview for Conv only plus a benchmark. It is now a tensor-space core that works for every model. What remains is closing the `weights:read` gap and a minimal trace. `preview_weights` is replaced by `read_tensor`, and the benchmark moves to v0.5.
- The decision about the 9→4 / 8→3 benchmark tasks is postponed to v0.5, because the benchmark will use several model families.
