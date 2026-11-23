# LayerDelta documentation

This is the index for all project documentation. If you are new, read the [README](../README.md) first, then pick the section that matches your task.

## Guides: using LayerDelta

| Document | What it covers |
|---|---|
| [Windows quickstart](guides/windows-quickstart.md) | Install the packaged Windows app, run your first experiment, and reproduce it in a second workspace |
| [Agent integration](guides/agent-integration.md) | How an AI agent uses the HTTP API: tools, budget, the experiment loop, error handling and reporting |

## Reference: contracts and formats

| Document | What it covers |
|---|---|
| [API specification v1](reference/api.md) | HTTP routes, schemas, capabilities, idempotency, quotas, errors and the trace endpoint |
| [Inspect operations](reference/inspect-ops.md) | Selectors and read-only inspection operations over tensors (library level) |
| [Edit programs and recipe v2](reference/edit-program.md) | Edit programs, recipe v2, and the child checkpoint writer (library level) |
| [Adapter guide](reference/adapter-guide.md) | How checkpoints are accepted for the supported architecture, and what is needed to add another one |
| [GraphIR and graph providers](reference/graph-ir.md) | The graph intermediate representation, graph providers, binding and the graph store (v0.3) |

## Architecture

| Document | What it covers |
|---|---|
| [Architecture (current)](architecture/Architecture.md) | The system as implemented in v0.1: components, persistence, run lifecycle and recovery |
| [Target architecture](architecture/Architecture_Target.md) | The adopted direction for v0.2 and later: model-agnostic tensor-space tooling |
| [v0.3 graph semantics](architecture/v0.3-graph-semantics.md) | Technical design for axis roles and coupling groups derived from the model graph |
| [ADR 0001: DepGraph oracle](architecture/adr/0001-depgraph-oracle.md) | Decision to use Torch-Pruning DepGraph only as a development test oracle |

## Roadmap and planning

| Document | What it covers |
|---|---|
| [Roadmap](roadmap/Roadmap.md) | Product direction, milestones, status, acceptance criteria and risks |
| [v0.2 implementation plan](roadmap/Implementation_Plan_v0.2.md) | Task board for the tensor-space core: owners, dependencies and status |
| [v0.3 implementation plan](roadmap/Implementation_Plan_v0.3.md) | Task board for graph semantics and coupling: owners, PRs and status |
| [Open questions](../QUESTION.md) | Decisions waiting for the project owner (kept at the repository root) |

## Design specs

| Document | What it covers |
|---|---|
| [Tensor browser](design/tensor-browser.md) | Design spec for the Desktop tensor browser, with the [static mockup](design/tensor-browser-mockup.html) and [screenshots](design/screens/) |
| [Architecture navigator](design/architecture-navigator.md) | Design spec for the Desktop architecture navigator, with the [static mockup](design/architecture-navigator-mockup.html) |

## Images

- [`images/run-comparison.png`](images/run-comparison.png): run comparison screen used in the README.

## History

Records of earlier milestones, the product requirements, and planning documents that have been superseded. They are kept for traceability and are not maintained as current guidance. See [history/README.md](history/README.md) for the list.
