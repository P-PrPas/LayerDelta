# LayerDelta v0.3 — Graph semantics & coupling: task board

Milestone goal and acceptance criteria are in the [Roadmap](Roadmap.md#v03--graph-semantics--coupling). Each task's technical specification is in [v0.3-graph-semantics.md](../architecture/v0.3-graph-semantics.md) (tasks G1–G8, acceptance A1–A11, constraints in §7.1). The UI design is [architecture-navigator.md](../design/architecture-navigator.md). This file tracks owners, branches and status. Team routing, the definition of done and the test environment note are the same as in [Implementation_Plan_v0.2](Implementation_Plan_v0.2.md).

| ID | Task | Owner | Branch / PR | Depends on | Reviewer | Status |
|---|---|---|---|---|---|---|
| G1 | DepGraph verification spike + ADR | AI Scientist | PR #29 | — | PM | **Done** ([ADR 0001](../architecture/adr/0001-depgraph-oracle.md): dev-only scoped oracle) |
| G2 | GraphIR, provider seam, `builtin` provider, binding, graph store | AI Engineer | `feat/v0.3-graph-ir` · PR #35 | — | AI Scientist | **In review** |
| G3 | Rule table v1, factor tracking, coupling groups, opaque/graph-output boundaries, coverage, DepGraph oracle harness | AI Engineer | `feat/v0.3-rules` | G2 | AI Scientist | **In progress** |
| G4 | inspect-ops rev 7: graph/groups routes, group selectors, navigator fields (§7.1) | AI Engineer | — | G3 | AI Scientist | Not started |
| G5 | Group selectors in edit programs, `permute_units`, engine revision set | AI Engineer | — | G3 (G4 for HTTP) | AI Scientist | Not started |
| G6 | `torch_export` provider for user code (isolated worker, opt-in, human-only) | AI Engineer | — | G2 | AI Scientist | Not started (unblocked: owner chose Q1 option 3) |
| G7 | `onnx` provider (bind by name + shape, honest coverage) | AI Engineer | — | G2, G3 | AI Scientist | Not started (`onnx` dependency decision at start) |
| G8a | Architecture navigator design | UX/UI Designer | PR #30 | — | PM + AI Scientist | **Done** |
| G8b | Architecture navigator, read-only (phase 1). Unit editing from the navigator moved to v0.6 (agent-first priority) | Frontend Developer | `feat/v0.3-navigator` | G4, G8a | UX/UI Designer | **In progress** |

## Tracked follow-ups (from reviews)

| From | Item | Lands in |
|---|---|---|
| PR #35 review | Family signature (names + shapes + dtype family), used only to *offer* graph reuse and never to auto-attach | G4 |
| PR #35 review | The `aten.lstm` decoder handles `proj_size>0` (`w_hr` per direction) and `has_biases=False`, with a corpus case | G3 |
| PR #35 review | The graph store records the provider name from the dispatcher call site, never from worker JSON | G2 fix / G6 |
