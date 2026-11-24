# Open questions for the project owner

The team keeps working while these are open. Each question records what can proceed in the meantime.

## Open

_None._

## Resolved

- **Q3 · Live benchmark provider: Antigravity, model `hydrogen`, effort high** (owner, 2026-11-24). Chosen to keep cost low. Scope is narrowed: one architecture, a plain CNN first. The comparison is agent weight-space editing *without* LayerDelta versus *with* LayerDelta on that same CNN. Class-1 aggregates may go to this provider. The owner gave no explicit spend cap: the team posts the estimated cost from the dry run here before starting the live run.
- **Q4 · Datasets: MNIST** (owner, 2026-11-24). MNIST plus the in-repo synthetic sets is enough. No licensed real datasets for v0.5.
- **Q1 · Running user-supplied model code: Option 3** (owner, 2026-11-23). `torch.export` on user code is the primary graph path, opt-in per project, run in an isolated worker, and never triggerable by agent tokens. ONNX is the no-code alternative. Pickle files stay refused. This unblocks v0.3 task G6.
- **Q2 · Priority: the agent surface comes first** (owner, 2026-11-23). Desktop work continues only where it supports the agent workflow or is already in flight (the read-only architecture navigator). Human editing UI moves later. See the [Roadmap](docs/roadmap/Roadmap.md).

## Parked (not questions)

- Release, publish, tagging and the human pilot are parked under *Release readiness* in the [Roadmap](docs/roadmap/Roadmap.md) until the owner explicitly asks.
