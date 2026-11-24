# LayerDelta v0.5: evaluators as agent tools and the research benchmark task board

The milestone goal is in the [Roadmap](Roadmap.md). Each task's technical specification is in [v0.5-evaluators-benchmark.md](../architecture/v0.5-evaluators-benchmark.md) (tasks E1–E10, acceptance V1–V10). This file tracks owners, branches and status. Team routing, the definition of done and the test environment note are the same as in [Implementation_Plan_v0.2](Implementation_Plan_v0.2.md).

The priority is the agent surface first (Q2): Desktop work is limited to E9, which is supervision.

| ID | Task | Owner | Branch / PR | Depends on | Reviewer | Status |
|---|---|---|---|---|---|---|
| D | v0.5 design and this board; benchmark amended to the owner's Q3/Q4 decision (with against without LayerDelta, one CNN, MNIST) | AI Scientist | #70, #73, this PR | — | AI Engineer | Merged; amendment in review |
| E1 | Executable artifacts per provider (`builtin`; a `.pt2` written by the export worker at attach, with its SHA-256 and torch version recorded; `onnx` via the `eval-onnx` extra, loaded in the worker only, with G7 limits); weight and constant binding; isolated evaluation worker that re-hashes before load; the batch-size decision (dynamic dim 0, or pad and trim) | AI Engineer | — | D | AI Scientist | Merged (#75) |
| E2 | Human-only dataset import; content-addressed manifests; closed format set; declarative preprocessing; hash splits; `mnist-v1` as the first dataset | AI Engineer | — | D | AI Scientist | Merged (#76) |
| E2s | Seeded synthetic datasets (`shapes-seg`, `boxes-det`, `glyph-ocr`) and their spec | AI Scientist | — | D | AI Engineer | Merged (#80) |
| E3 | Evaluator plugins and decoders (classification, segmentation, detection with in-tree NMS, OCR with CTC); metrics; golden oracles (V2); `pycocotools` as a dev-only oracle extra if it installs on CI | AI Scientist | — | D | AI Engineer | Merged (#77, #79) |
| E4 | `evaluate`, `get_evaluation`, `evaluation_changes` (registry, content, errors); the E9 contract additions (metric metadata, categories and per-sample scores, trace links, human-only sample image and label routes; design §3.2); paired bootstrap; `evals` and `eval_samples` budgets; trace links; generalised human-only final evaluation; legacy MNIST routes over the plugin, byte-identical (V1) | AI Engineer | — | E1, E2, E3 | AI Scientist | Merged (#78 evaluate, get_evaluation, evaluation_changes; #81 sample routes; #82 final evaluation, resampler, MNIST conformance) |
| E5 | `probe` and `get_probe` (K rule in class 1, raw values need `weights:values`) and activation interventions in `evaluate`, per provider | AI Engineer | — | E4 | AI Scientist | E5a probe (#83) and E5b activation edit (#84) merged; E5c group selectors and `/interventions/resolve` in review; ONNX probes and edits not built (probe_unsupported) |
| E6 | Structure-only profile for a token without `weights:read`: plan, validate, apply and diff without `changed_elems`, `touched_elems`, shas or `noop`; status-only diff; no-op refusal off; W and legacy tools refused; allow-list route walk test (V8) | AI Engineer | — | D | AI Scientist | Merged (#74) |
| E7a | Benchmark harness in `tools/bench/` for the owner's scope (one CNN, MNIST; design §5): task generator (S-nat confusion pairs on blurred, planted S-vis and S-inv faults on `ReferenceCNN`, minimum gap at generation); arm R's session kit (`start.safetensors`, standalone `model.py`, `exploration.npz`, task text); final-checkpoint checks and holdout scoring as a person; the definition of an edit and its checks (no gradient training: transcript check, changed-elements cap); transcript audit for isolation and the violation rule; scripted controls (harness symmetry, A/A calibration, positive control); report generator | AI Scientist | — | E4e | AI Engineer | Not started |
| E7s | One-day spike, gating E7b: show with evidence whether Antigravity can (i) launch non-interactively with a prompt and working directory, (ii) take per-session MCP configuration (arm L only) and auto-approve tool permissions, (iii) select `hydrogen` and the effort, (iv) export transcripts with tool calls, file reads and token counts, (v) run under another OS user or a sandbox, (vi) stop on limits and kill its process tree, (vii) run concurrently. The result goes into design §5 before E7b is sized; if (i) or (ii) fail, ask the owner | AI Engineer | — | — | AI Scientist | Not started |
| E7b | Antigravity session driver: start a session for an arm (model `hydrogen`, effort high) in a fresh directory, with the `layerdelta-mcp` bridge at class 1 for arm L only; a dedicated low-privilege OS account per session where feasible (design §5.2); wall-time, step and token limits; export the transcript and the token count. Fallback if the host cannot run headless: drive it through its terminal | AI Engineer | — | E7s | AI Scientist | Not started |
| E8 | Pre-registration (`docs/research/v0.5-benchmark-prereg.md`); dry run with `hydrogen` at a small N, then the cost estimate (N × cost per pair, with its range) in QUESTION.md before the live run (owner, Q3); live run; report | AI Scientist | — | E7a, E7b | AI Engineer | Not started |
| E4e | `mnist-v1` in `evaluate` and final evaluations: the evaluation worker reads its samples from the verified idx archives (normal and blurred conditions); the classification evaluator with the 8 → 3 confusion pair within blurred; `reference_cnn` graphs on imported checkpoints. Arm L needs it to evaluate, and both arms need it to be scored on holdout | AI Engineer | — | E4 | AI Scientist | Not started |
| E9 | Desktop session viewer: evaluation and probe events, metric deltas with intervals, fixed and broken samples (with pixels, for the human only) | Frontend Developer (UI), UX/UI Designer (spec) | — | E4 | UX/UI Designer | Not started |
| E10 | `docs/reference/evaluators.md` and `datasets.md`; cookbook recipes (measure an edit, probe a group, the repair loop) running in CI | AI Scientist | — | E4, E5 | AI Engineer | Not started |

## Order

1. E1, E2, E2s, E3 and E6 can start in parallel. E6 is independent of execution. The AI Engineer lands E6 before E4, because both edit the registry's capability table.
2. E4 joins them. Then E5, E9 and E10.
3. For the benchmark (owner scope, design §5): E7s (the host spike) comes first and gates E7b. E4e (AI Engineer) and E7a (AI Scientist) run in parallel with it. E7a's scripted controls need E4e. Its arm-R kit and task generator do not.
4. E8: the pre-registration and the dry run need E7a and E7b. The cost estimate goes into QUESTION.md before the live run.

## Owner decisions

Both are recorded in [QUESTION.md](../../QUESTION.md) (PR #85) and applied in design §5:
- **Q3**: Antigravity, model `hydrogen`, effort high. One plain CNN first. The comparison is with against without LayerDelta. Class-1 aggregates may go to the provider. There is no cap: the dry run's cost estimate goes into QUESTION.md before the live run.
- **Q4**: MNIST only. The synthetic datasets stay as evaluator fixtures (E2s) and are not used in the benchmark.
