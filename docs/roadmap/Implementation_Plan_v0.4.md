# LayerDelta v0.4: agent weight-space programming task board

The milestone goal and acceptance criteria are in the [Roadmap](Roadmap.md). Each task's technical specification is in [v0.4-agent-surface.md](../architecture/v0.4-agent-surface.md) (tasks S1–S7, acceptance T1–T9). PM decisions are recorded on PR #44:
- the `mcp` SDK is a pinned optional extra;
- `weights:values` is split out of `weights:read`, and the MCP bridge defaults to egress class 1;
- the approval gate is part of S5, off by default.

The session viewer UI is [session-trace-viewer.md](../design/session-trace-viewer.md) (PR #45). This file tracks owners, branches and status. Team routing, the definition of done and the test environment note are the same as in [Implementation_Plan_v0.2](Implementation_Plan_v0.2.md).

| ID | Task | Owner | Branch / PR | Depends on | Reviewer | Status |
|---|---|---|---|---|---|---|
| D | Agent-surface design | AI Scientist | PR #44 | — | PM | **Done** |
| S1 | Route registry; generated client and MCP `tools/list`; `layerdelta-mcp` stdio bridge (results, errors, size guard); parity (T1) and tool-schema contract (T2) tests | AI Engineer | — | — | AI Scientist | In review |
| S2 | Tool descriptions, examples and LLM defaults (design §2); error catalog with `hint`/`retryable` (T3) | AI Scientist (content), AI Engineer (wiring) | — | S1 | AI Engineer | Not started |
| S3 | Per-session budgets in the backend (`values`, `scan`, `applies`, `blob_bytes`): metering, refusal before reading, headers, `session_info`, CLI flags (T4) | AI Engineer | — | — | AI Scientist | In review |
| S4 | `weights:values` capability split; egress class in `connection.json`; headless runner with provider adapters (stub, local, Anthropic, OpenAI-compatible) (T6) | AI Engineer | — | S1 | AI Scientist | S4a #56 and #58 merged; S4b in review |
| S5 | Trace fields (`tool`, `via`, `budget_charged`); revoke route; budget change; approval gate (default off); viewer backend routes (PR #45 gaps 1–6); Desktop session viewer (T5, T8) | AI Engineer (backend), Frontend Developer (viewer), UX/UI Designer (spec, PR #45) | — | S1, S3 | AI Scientist (backend), UX/UI Designer (UI) | Spec in review (PR #45) |
| S6 | Lineage in `get_checkpoint` (`parent`, `recipe_hash`, `children`) and `diff_hint` on apply (design §3) | AI Engineer | — | — | AI Scientist | Not started |
| S7 | Agent cookbook and runnable examples; scripted multi-family agent over MCP (T7); live smoke with a real MCP host (T9) | AI Scientist | — | S1–S6 | PM | Not started |

S1, S3 and S6 can start in parallel. S2 starts once the registry exists. S7 starts with deterministic scripts over HTTP and switches to MCP when S1 lands.

## Tracked follow-ups (from reviews)

| From | Item | Lands in |
|---|---|---|
| PR #45 review | The viewer renders agent-originated text (tensor names, module paths, hypotheses) as plain text only. Tensor names are never written into the trace. Targets resolve through a separate `selections` table with a batch route. | S5 |
