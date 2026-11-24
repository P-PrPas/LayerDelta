# Open questions for the project owner

The team keeps working while these are open. Each question records what can proceed in the meantime.

## Open

- **Q5 · Benchmark session isolation with Antigravity** (asked 2026-11-24; see [E7s spike](docs/research/v0.5-e7s-antigravity-spike.md), PR #92). `agy` runs headless with `gemini-4-hydrogen-high`, so the driver works. But even with `--sandbox`, a headless session read a file outside its directory and fetched a URL (HTTP 200). It also has web search, URL read and browser tools, and its config cannot be redirected by environment variables. Options:
  - **(a) Enforced isolation:** a dedicated low-privilege Windows account for benchmark sessions, which the owner logs in to Antigravity once, plus an outbound firewall rule for that account (about a day of work). Needs the owner's OK to use `--dangerously-skip-permissions` inside that account only.
  - **(b) Audit-only:** no OS isolation. Every out-of-bounds read, fetch or tool call is logged and the violation rate is reported per arm next to the results.

  *Meanwhile:* the dry run with the local loopback model and E4e continue. Only the live run waits.

## Resolved

- **Q3 · Live benchmark provider: Antigravity, model `hydrogen`, effort high** (owner, 2026-11-24). Chosen to keep cost low. Scope is narrowed: one architecture, a plain CNN first. The comparison is agent weight-space editing *without* LayerDelta versus *with* LayerDelta on that same CNN. Class-1 aggregates may go to this provider. The owner gave no explicit spend cap: the team posts the estimated cost from the dry run here before starting the live run.
- **Q4 · Datasets: MNIST** (owner, 2026-11-24). MNIST plus the in-repo synthetic sets is enough. No licensed real datasets for v0.5.
- **Q1 · Running user-supplied model code: Option 3** (owner, 2026-11-23). `torch.export` on user code is the primary graph path, opt-in per project, run in an isolated worker, and never triggerable by agent tokens. ONNX is the no-code alternative. Pickle files stay refused. This unblocks v0.3 task G6.
- **Q2 · Priority: the agent surface comes first** (owner, 2026-11-23). Desktop work continues only where it supports the agent workflow or is already in flight (the read-only architecture navigator). Human editing UI moves later. See the [Roadmap](docs/roadmap/Roadmap.md).

## Parked (not questions)

- Release, publish, tagging and the human pilot are parked under *Release readiness* in the [Roadmap](docs/roadmap/Roadmap.md) until the owner explicitly asks.
