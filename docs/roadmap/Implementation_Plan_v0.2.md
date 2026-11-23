# LayerDelta v0.2 — Tensor-space core: task board

> **Status: capability complete (2026-11-23).** All seven v0.2 acceptance items have evidence (see the Roadmap). This is an internal milestone: nothing was tagged or published. Hardening (#33) closed the picker/CSP/import-cancel follow-ups.

Milestone goal and acceptance criteria are in [Roadmap](Roadmap.md#v02--tensor-space-core--done). Technical specification per task is in [Architecture_Target §3–§6](../architecture/Architecture_Target.md). This file tracks owners, branches, dependencies and status. It replaces the earlier M5.0–M5.3 plan (Conv-only `preview_weights`, two-arm benchmark), which was superseded on 2026-11-23: the leak fix and minimal trace are kept as T4/T5, `preview_weights` becomes generic `read_tensor` in T2, and the benchmark moved to v0.5.

## Team routing

| Work type | Owner | Reviewer |
|---|---|---|
| AI, backend, weight-space, agent tools, tests | Senior AI Engineer | Senior AI Scientist (uses `scrutinize` for high-impact changes) |
| Desktop / frontend implementation | Senior Frontend Developer | Senior UX/UI Designer |
| UX/UI design direction | Senior UX/UI Designer | PM |
| Docs, roadmap, coordination | PM | Senior AI Scientist (technical content) |

One task has one owner, and parallel implementation work uses separate worktrees under `../LayerDelta-wt/<task>`. If a review goes back and forth more than three times, the reviewer stops and does a root-cause analysis first.

## Tasks

| ID | Task | Owner | Branch | Depends on | Reviewer | Status |
|---|---|---|---|---|---|---|
| T1 | Streaming safetensors reader + tensor table (sharded, F32/F16/BF16, lazy sha256, tied-duplicate detection, RSS bound) | AI Engineer | `feat/v0.2-tensor-reader` · PR #9 | — | AI Scientist | **Done** (merged e07f82d) |
| T2 | Selector schema + pure inspect ops `describe`, `read_tensor`, `stats`, `diff`, `query` (caps, cursor, deterministic output) | AI Engineer | `feat/v0.2-inspect-ops` · PR #13 | T1 | AI Scientist | **Done** (merged 2b94033; contract [inspect-ops.md](../reference/inspect-ops.md) rev 5) |
| T3 | Edit program, transforms, validator, child writer, recipe v2, v1→v2 translator | AI Engineer | `feat/v0.2-edit-program` · PR #20 | T2 | AI Scientist | **Done** (merged 77d2b10; [edit-program.md](../reference/edit-program.md), numeric `cpu-f32-round-per-op`) |
| T4a | Weight-access leak fix: `weights:read` capability on tokens; strip kernel statistics from legacy `/previews` and agent-visible reports when absent; route-enumeration leak test | AI Engineer | `feat/v0.2-weights-read` · PR #10 | — (fixes a known bug in v0.1 code) | AI Scientist | **Done** (merged 8428d4e) |
| T4b | Checkpoint import + catalog: import by local path (Desktop/CLI) and streamed HTTP upload with configurable caps, validated by the T1 reader; catalog decoupled from the reference fixture (`job.checkpoint_path`) | AI Engineer | `feat/v0.2-checkpoint-import` · PR #18 | T1 | AI Scientist | **Done** (merged ce956a9) |
| T4c | v2 API/client: `model:read`/`weights:write`, generic inspect routes, idempotent `apply`, `side_effects` on every v2 response; legacy routes unchanged | AI Engineer | `feat/v0.2-v2-api` · PR #22 | T2, T3, T4a, T4b | AI Scientist | **Done** (merged 26ba04c; contract rev 6) |
| T5 | Minimal session trace (`events` table, redaction, restart persistence) | AI Engineer | `feat/v0.2-trace` · PR #28 | T4c routes | AI Scientist | **Done** (merged) |
| T6 | Cross-family random-init corpus test (ResNet-18, U-Net + ConvTranspose, ViT-tiny, CRNN, YOLO-like) | AI Engineer | `feat/v0.2-corpus` · PR #25 | T2 | AI Scientist | **Done** (merged fe6c8ca; found and fixed an empty-selection bug in T3) |
| T7a | Tensor browser design: information architecture for large tensor tables, slice/heatmap and diff views | UX/UI Designer | `design/v0.2-tensor-browser` · PR #11 | T2 schema draft | PM + AI Scientist (data semantics) | **Done** (merged 49933ee) |
| T7b | Desktop tensor browser implementation from T2/T4c payloads | Frontend Developer | #15 · #23 · #32 | T4c, T7a | UX/UI Designer | **Done** (phase 1 mock #15, live import + picker #23, live inspect/diff #32); fit-to-pane polish in `feat/v0.2-browser-fit` |

Order: T4a can start anytime (no dependencies). T1 → {T2, T4b} in parallel → T3 → T4c (T5 runs alongside T4c) → T6 runs continuously starting from T2. T7a starts as soon as the T2 schema is drafted, and T7b starts after T4c.

T3 note: v1→v2 child SHA equality holds only because the fixture parent and the v0.1 child are both written by `safetensors.save_file` with no metadata (`fixture.py`, job export). The v2 writer must preserve the parent's header layout.

## Tracked follow-ups (from reviews)

| From | Item | Lands in |
|---|---|---|
| PR #10 review | Record the capability set on the token/session record and write it into every event, so past runs and benchmark arms stay provable | Per-principal record done in T4c (#22); per-event snapshot in T5 |
| PR #22 review | Compatibility break to state in future release notes: agent tokens without `weights:read` no longer receive activation maps, images or kernel stats from the legacy preview (`agent_demo` needs `--agent-weights-read`), and `/agent-session` capabilities now include `model:read` | Release readiness |
| PR #23 review | The native picker probes `GET /holdout-audit` as its session check. Add a dedicated human-only session route | Done in hardening (#33) |
| PR #23 review | Serve the Desktop index with a strict CSP (`script-src 'self'`, no inline), because any page script can call the native bridge | Done in hardening (#33) |
| PR #18 review | Import has no cancel route or stall timeout, so a hung UNC/network path blocks the queue | Done in hardening (#33) |
| PR #10 review | Activation maps plus the source image let an agent recover weights exactly. Gate activation values/image behind a capability or cap previews per session | T4c (decision), v0.5 benchmark gate |
| PR #10 review | `report.json` and the preview cache on disk keep weight stats. The benchmark runner must deny agents filesystem access to the workspace | v0.5 |
| PR #13 review | Cap the query match `selector` (or return only `selection_id` when truncated) and map an over-cap error to field `where` | Done in T4c (#22) |
| PR #13 review | Document that an absent range key means not computed and `null` means no finite values | Done in T4c (#22) |
| PR #13 review | Sign cursors with an HMAC | Done in T4c (#22) |
| PR #17 note | Unify naming: stats rows use `index`, but diff `units[]` and query `units[]` still use `unit` | Done in T4c (#22) |
| PR #11 review | Backend field questions from the tensor browser spec §7 are resolved in [inspect-ops.md](../reference/inspect-ops.md) (T2 schema rev 4) | Done in T2 |

## Definition of done (every task)

Acceptance criteria pass, the full existing suite and any new tests pass, the desktop build passes for UI tasks, the reviewer approves with no open critical/high issues, affected docs are updated, PR checks pass, and the PM has recorded the result here.

## Test environment note

The shared `.venv` has an editable install that points at the main checkout. Inside a worktree, set `PYTHONPATH=<worktree>/src` and check that `layerdelta.__file__` resolves into the worktree. If `%TEMP%` cannot be written, pass `--basetemp` to pytest.
