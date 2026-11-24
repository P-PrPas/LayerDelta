# Open questions for the project owner

The team keeps working while these are open. Each question records what can proceed in the meantime.

## Open

- **Q6 · Where the benchmark runs** (asked 2026-11-25; follows Q5). The owner chose audit-only isolation (Q5) and will not create a separate Windows user on this machine. Proposed instead: run the sessions on the remote Linux machine `malren` (16 cores, 6 GB RAM, `agy` already installed at `~/.local/bin/agy`), driven with `hc run --on malren`. The session driver wipes and restores the Antigravity state of the account it runs in before every session, so the account used on `malren` must not hold anyone's personal Antigravity history. *Waiting for:* the owner's OK on `malren`, and which account there. *Meanwhile:* nothing starts. The dry run, the pre-registration's first freeze and the live run wait on this.

## Resolved

- **Q5 · Benchmark session isolation: (b) audit-only** (owner, 2026-11-25). No OS isolation. Every out-of-bounds read, fetch, web or browser tool call, download and training signature is logged by the session driver (PR #102). The counted violations follow the pre-registration §6: rerun once, then exclude, by the same rule for both arms. The violation rate per arm is reported next to the results. Where the sessions run is Q6.
- **Q3 · Live benchmark provider: Antigravity, model `hydrogen`, effort high** (owner, 2026-11-24). Chosen to keep cost low. Scope is narrowed: one architecture, a plain CNN first. The comparison is agent weight-space editing *without* LayerDelta versus *with* LayerDelta on that same CNN. Class-1 aggregates may go to this provider. The owner gave no explicit spend cap: the team posts the estimated cost from the dry run here before starting the live run.
- **Q4 · Datasets: MNIST** (owner, 2026-11-24). MNIST plus the in-repo synthetic sets is enough. No licensed real datasets for v0.5.
- **Q1 · Running user-supplied model code: Option 3** (owner, 2026-11-23). `torch.export` on user code is the primary graph path, opt-in per project, run in an isolated worker, and never triggerable by agent tokens. ONNX is the no-code alternative. Pickle files stay refused. This unblocks v0.3 task G6.
- **Q2 · Priority: the agent surface comes first** (owner, 2026-11-23). Desktop work continues only where it supports the agent workflow or is already in flight (the read-only architecture navigator). Human editing UI moves later. See the [Roadmap](docs/roadmap/Roadmap.md).

## Parked (not questions)

- Release, publish, tagging and the human pilot are parked under *Release readiness* in the [Roadmap](docs/roadmap/Roadmap.md) until the owner explicitly asks.
