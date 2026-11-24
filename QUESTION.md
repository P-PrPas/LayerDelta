# Open questions for the project owner

The team keeps working while these are open. Each question records what can proceed in the meantime.

## Open

- **Q3 · The live v0.5 benchmark needs a model provider** (asked 2026-11-24; see [v0.5 design §5](docs/architecture/v0.5-evaluators-benchmark.md)). The live run sends class-1 tool results (aggregate statistics, metrics, sample IDs; never raw weights or pixels) to a model provider, and costs money. Please decide:
  - which provider and model to use (one model is enough for the primary result);
  - a spend cap;
  - whether class-1 aggregates may go to a remote provider, or the live run uses a local model only.

  The planned size is about 111 session pairs (222 sessions), each with an equal step and completion-token budget. The final N comes from the dry run and can only go up. *Meanwhile:* everything up to and including the dry run (scripted controls and a local loopback model) proceeds.
- **Q4 · Real datasets for evaluators and the benchmark** (asked 2026-11-24). v0.5 uses MNIST (already in use since v0.1) and synthetic datasets generated in the repo, which need no licence. Evaluating on real segmentation, detection or OCR data (for example Pascal VOC, COCO or IAM) needs a licence decision for each dataset. It also matters whether the data stays local-only or ships in tests. *Meanwhile:* the synthetic datasets are used, and the benchmark's claim is scoped to them.

## Resolved

- **Q1 · Running user-supplied model code: Option 3** (owner, 2026-11-23). `torch.export` on user code is the primary graph path, opt-in per project, run in an isolated worker, and never triggerable by agent tokens. ONNX is the no-code alternative. Pickle files stay refused. This unblocks v0.3 task G6.
- **Q2 · Priority: the agent surface comes first** (owner, 2026-11-23). Desktop work continues only where it supports the agent workflow or is already in flight (the read-only architecture navigator). Human editing UI moves later. See the [Roadmap](docs/roadmap/Roadmap.md).

## Parked (not questions)

- Release, publish, tagging and the human pilot are parked under *Release readiness* in the [Roadmap](docs/roadmap/Roadmap.md) until the owner explicitly asks.
