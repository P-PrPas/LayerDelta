# Open questions for the project owner

The team keeps working while these are open. Each question records what can proceed in the meantime.

## Q1 · May LayerDelta execute user-supplied model code? (needed before v0.3 graph import)

**Context.** v0.2 reads safetensors only and never executes code from a file. Graph semantics (v0.3) and evaluation of non-MNIST models (v0.4) need a model definition. With PyTorch that means running the user's module factory and `torch.export`, which is a new trust boundary: v0.1 promised that a checkpoint supplies tensors only.

**Options.**
1. *Opt-in user code.* Run it per project inside the isolated worker only. Agent tokens can never trigger it, and pickle files stay refused.
2. *ONNX only.* No Python execution. ONNX fuses and renames parameters (for example BN folded into conv), so names stop matching the training checkpoint and some edits cannot be expressed.
3. *Both:* `torch.export` on user code is the primary path for fidelity, opt-in per project. ONNX is the no-code alternative for users who decline to run code.

**Recommendation.** Option 3. This matches [Architecture_Target §2.1](docs/architecture/Architecture_Target.md).

**Consequences.** Option 1 covers the most model families but carries the highest security responsibility. Option 2 is safest but loses fidelity in exactly the place users want to edit. Option 3 costs a little more to build and leaves the choice with the user.

**Meanwhile.** In the [v0.3 design](docs/architecture/v0.3-graph-semantics.md), only task G6 (the `torch_export` provider for user code) waits on this answer. G1–G5, G7 and G8 use our own test models or ONNX and proceed now. If you choose ONNX-only, G7 becomes the primary provider and nothing else changes.

## Q2 · Priority between the agent surface and the human Desktop surface

**Context.** The AI Scientist pointed out that "library + MCP first, Desktop later" is the cheaper path if agent weight-space programming is the primary goal. The product direction gives equal weight to human understanding and editing.

**PM decision (reversible, pending your confirmation).** Ship both surfaces from one payload in every milestone. Desktop work runs in parallel on separate worktrees and does not block backend tasks.

**Meanwhile.** Continue as decided. If you prefer agent-first, T7a/T7b move to a later milestone and nothing else changes.

## Parked (not questions)

- Release, publish, tagging and the human pilot are parked under *Release readiness* in the [Roadmap](docs/roadmap/Roadmap.md) until the owner explicitly asks.
