# Open questions for the project owner

The team keeps working while these are open. Each question records what can proceed in the meantime.

## Open

- **Q5 · Benchmark session isolation with Antigravity** (asked 2026-11-24; see [E7s spike](docs/research/v0.5-e7s-antigravity-spike.md), PR #92). `agy` runs headless with `gemini-4-hydrogen-high`, so the driver works. But even with `--sandbox`, a headless session read a file outside its directory and fetched a URL (HTTP 200). It also has web search, URL read and browser tools, and its config cannot be redirected by environment variables. Options:
  - **(a) Enforced isolation:** the benchmark Windows user below can read only its session folders, and its outbound network is limited to the model provider. This is about a day of setup, with administrator rights. Needs the owner's OK to use `--dangerously-skip-permissions` for that user only.
  - **(b) Audit-only:** no OS isolation. Every out-of-bounds read, fetch or tool call is logged, and the violation rate is reported per arm next to the results.

  **Both options now need a dedicated benchmark account: a separate Windows user, not a second Antigravity subscription.** Antigravity keeps its state (conversations, its "brain" memory, history) in the Windows user's profile, and environment variables cannot move it (spike, PR #92). The session driver (PR #102) therefore wipes that state before every session and restores it afterwards, so it must never run in a person's own Windows profile. It refuses unless the account is confirmed as the benchmark account. What the owner does once:
  1. Create a local **standard** (not administrator) Windows user for the benchmark, for example `ld-bench` (Settings → Accounts → Other users).
  2. Sign in to Windows as that user once, start Antigravity (`agy`), sign in with your Antigravity account, then sign out.
  3. Tell the team the user name. The harness runs each session as that user, with `AGY_BENCH_ACCOUNT=1` set in its profile.
  4. For (a) only: also allow the folder permissions and the outbound restriction for that user, and decide on `--dangerously-skip-permissions`.

  *Meanwhile:* the benchmark code is merged: tasks (#91), analysis and controls (#98, #101), the driver (#102) and the pre-registration draft (#103). **The dry run (10 pairs with `hydrogen`) and the pre-registration's first freeze wait on this choice and the account.** The live run then waits on the dry run's cost estimate (Q3).

## Resolved

- **Q3 · Live benchmark provider: Antigravity, model `hydrogen`, effort high** (owner, 2026-11-24). Chosen to keep cost low. Scope is narrowed: one architecture, a plain CNN first. The comparison is agent weight-space editing *without* LayerDelta versus *with* LayerDelta on that same CNN. Class-1 aggregates may go to this provider. The owner gave no explicit spend cap: the team posts the estimated cost from the dry run here before starting the live run.
- **Q4 · Datasets: MNIST** (owner, 2026-11-24). MNIST plus the in-repo synthetic sets is enough. No licensed real datasets for v0.5.
- **Q1 · Running user-supplied model code: Option 3** (owner, 2026-11-23). `torch.export` on user code is the primary graph path, opt-in per project, run in an isolated worker, and never triggerable by agent tokens. ONNX is the no-code alternative. Pickle files stay refused. This unblocks v0.3 task G6.
- **Q2 · Priority: the agent surface comes first** (owner, 2026-11-23). Desktop work continues only where it supports the agent workflow or is already in flight (the read-only architecture navigator). Human editing UI moves later. See the [Roadmap](docs/roadmap/Roadmap.md).

## Parked (not questions)

- Release, publish, tagging and the human pilot are parked under *Release readiness* in the [Roadmap](docs/roadmap/Roadmap.md) until the owner explicitly asks.
