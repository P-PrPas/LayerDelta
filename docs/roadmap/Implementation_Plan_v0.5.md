# LayerDelta v0.5: evaluators as agent tools and the research benchmark task board

The milestone goal is in the [Roadmap](Roadmap.md). Each task's technical specification is in [v0.5-evaluators-benchmark.md](../architecture/v0.5-evaluators-benchmark.md) (tasks E1–E10, acceptance V1–V10). This file tracks owners, branches and status. Team routing, the definition of done and the test environment note are the same as in [Implementation_Plan_v0.2](Implementation_Plan_v0.2.md).

The priority is the agent surface first (Q2): Desktop work is limited to E9, which is supervision.

| ID | Task | Owner | Branch / PR | Depends on | Reviewer | Status |
|---|---|---|---|---|---|---|
| D | v0.5 design and this board | AI Scientist | this PR | — | AI Engineer | In review |
| E1 | Executable artifacts per provider (`builtin`; a `.pt2` written by the export worker at attach, with its SHA-256 and torch version recorded; `onnx` via the `eval-onnx` extra, loaded in the worker only, with G7 limits); weight and constant binding; isolated evaluation worker that re-hashes before load; the batch-size decision (dynamic dim 0, or pad and trim) | AI Engineer | — | D | AI Scientist | In review |
| E2 | Human-only dataset import; content-addressed manifests; closed format set; declarative preprocessing; hash splits; `mnist-v1` as the first dataset | AI Engineer | — | D | AI Scientist | In review |
| E2s | Seeded synthetic datasets (`shapes-seg`, `boxes-det`, `glyph-ocr`) and their spec | AI Scientist | — | D | AI Engineer | Not started |
| E3 | Evaluator plugins and decoders (classification, segmentation, detection with in-tree NMS, OCR with CTC); metrics; golden oracles (V2); `pycocotools` as a dev-only oracle extra if it installs on CI | AI Scientist | — | D | AI Engineer | Not started |
| E4 | `evaluate`, `get_evaluation`, `evaluation_changes` (registry, content, errors); the E9 contract additions (metric metadata, categories and per-sample scores, trace links, human-only sample image and label routes; design §3.2); paired bootstrap; `evals` and `eval_samples` budgets; trace links; generalised human-only final evaluation; legacy MNIST routes over the plugin, byte-identical (V1) | AI Engineer | — | E1, E2, E3 | AI Scientist | Not started |
| E5 | `probe` and `get_probe` (K rule in class 1, raw values need `weights:values`) and activation interventions in `evaluate`, per provider | AI Engineer | — | E4 | AI Scientist | Not started |
| E6 | Structure-only profile for a token without `weights:read`: plan, validate, apply and diff without `changed_elems`, `touched_elems`, shas or `noop`; status-only diff; no-op refusal off; W and legacy tools refused; allow-list route walk test (V8) | AI Engineer | — | D | AI Scientist | In review |
| E7 | Benchmark harness in `tools/bench/`: seeded base-model training, task and fault generators (S-vis, S-inv, S-nat), arm profiles, pairing and order, out-of-arm trace check, holdout via human credential, report generator; scripted A/A and positive controls | AI Scientist | — | E4, E6 (E5 for arm W probes) | AI Engineer | Not started |
| E8 | Pre-registration (`docs/research/v0.5-benchmark-prereg.md`); dry run with a local provider; live run; report | AI Scientist | — | E7 | AI Engineer | Not started; the live run waits for Q3 |
| E9 | Desktop session viewer: evaluation and probe events, metric deltas with intervals, fixed and broken samples (with pixels, for the human only) | Frontend Developer (UI), UX/UI Designer (spec) | — | E4 | UX/UI Designer | Not started |
| E10 | `docs/reference/evaluators.md` and `datasets.md`; cookbook recipes (measure an edit, probe a group, the repair loop) running in CI | AI Scientist | — | E4, E5 | AI Engineer | Not started |

## Order

1. E1, E2, E2s, E3 and E6 can start in parallel. E6 is independent of execution. The AI Engineer lands E6 before E4, because both edit the registry's capability table.
2. E4 joins them. Then E5, E9 and E10.
3. E7 can build the harness and scripted controls against E4 and E6 before E5 lands.
4. E8 needs E7. Its pre-registration and dry run need no owner decision. Its live run needs Q3.

## Questions for the owner

Both are in [QUESTION.md](../../QUESTION.md). Work up to the dry run continues without them.
- **Q3**: provider and model for the live benchmark, the spend cap, and class-1 aggregates to a remote provider.
- **Q4**: real datasets and their licences.
