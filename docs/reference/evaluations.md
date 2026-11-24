# Evaluations: `evaluate`, `get_evaluation`, `evaluation_changes` (v0.5-E4)

Design: [v0.5 design §3.2](../architecture/v0.5-evaluators-benchmark.md). Code: `evalservice.py` (plan, ledger, job, changes), `eval_worker.py` (decode, preprocess, forward, decode outputs, in an isolated worker), `bootstrap.py` (paired intervals, categories), `evaluators/` ([evaluators.md](evaluators.md)), [executables.md](executables.md) and [datasets.md](datasets.md) for what it runs and on what.

```
POST /checkpoints/{checkpoint_id}/evaluations   tool evaluate            eval:run, Idempotency-Key   202 {evaluation_id, status: queued, ...}
GET  /evaluations/{evaluation_id}               tool get_evaluation      model:read                  status, metrics, counts
GET  /evaluations/{evaluation_id}/changes       tool evaluation_changes  model:read                  paged samples whose result changed
```

`eval:run` is a capability of its own (`layerdelta --agent-eval`); a person has it. All three tools are in the `structure` profile: an evaluation reports the function's outputs, which both benchmark arms may have.

## The request

`{dataset_id, evaluator, split: "exploration", baseline?, limit_samples?, graph?}`.

- `evaluator` is a task name (`classification`, `segmentation`, `detection`, `ocr`) or a spec. The dataset's own classes, ignore index, categories (and their ids) and alphabet are filled in where the spec leaves them out, and the filled spec is what is hashed. A task that is not the dataset's is `dataset_task_mismatch`.
- `baseline`: absent means the checkpoint's parent (S6 lineage); an import with no parent is `no_baseline`, never a quiet "no change". `null` gives absolute metrics (no delta, no interval). Otherwise a checkpoint id with the same layout.
- `split`: only `exploration`. Anything else is 403 `forbidden`; holdout is reached through a human-only final evaluation.
- `limit_samples` takes the first n samples in sample-id-hash order, deterministically. `graph` picks one of the graphs attached to the layout (default: the newest).
- Refused before any charge or work: unknown checkpoint, dataset or baseline (404), no graph or executable (`no_graph`, `executable_missing`, `executable_mismatch`), a baseline of another layout, `mnist-v1` (`dataset_unsupported`: its samples are idx archives; the v0.1 run routes serve it).

## Cost, keys and the cache

Charged at submission, **before any work**, both or neither: `evals` 1 and `eval_samples` n, plus n for the baseline unless it is a cache hit. A refusal is 429 `budget_exhausted` with the field of the dimension that ran out. A repeated `Idempotency-Key` with the same request returns the same evaluation (`replayed: true`) and is not charged again; with a different request it is `idempotency_conflict`.

Baselines are cached by `(checkpoint sha, fingerprint)` in `evalcache/`. The **fingerprint** hashes the filled evaluator spec, the exploration split hash and the subset, the preprocessing spec and its implementation version (preprocess, torch, Pillow), the executable's fingerprint, device and dtype. The same fingerprint gives the same records and the same intervals.

## The result

`get_evaluation` has the job fields (`status`: queued, running, succeeded, failed; `created_at`, `started_at`, `finished_at`, `error_code`, `message`), the ids and `fingerprint`, `samples`, `baseline_cached`, a `diff_hint`, and, once succeeded:

- `metrics`: each `{name, primary, unit: rate|count|score, better: higher|lower, baseline, value, delta, ci}`, with `per_class` rows of the same shape plus `class` and `n`. `delta = value - baseline` always; `better` gives the direction. `ci` is the 95% paired-bootstrap interval of the delta, or null (no baseline, or the metric is undefined on more than 5% of resamples).
- `counts`: `{samples, fixed, broken, changed}`.
- `resamples`: 2,000 where the evaluator's metrics decompose into per-sample sums (the bootstrap is a matrix product); 200 for detection, whose mAP is recomputed on every resample. The seed comes from the fingerprint, and both evaluations use the same resampled indices (that is what makes it paired).

`evaluation_changes` pages the samples whose record changed: `{sample_id, label, condition, before, after, category, score_before, score_after}`. `category`: `fixed` (incorrect to correct), `broken` (correct to incorrect), `changed` (the record changed, correctness did not). Correctness and score come from the evaluator spec. `category` filters; `sort=score_change` lists the worst change first (a fall of a higher-is-better score, a rise of CER). Records are model outputs on exploration samples only (a class, per-class pixel counts, boxes in source coordinates, a decoded string); there are no pixels.

## Who can read what, and the trace

A token reads only the evaluations of its own session (anything else is 404); a person reads all. Every evaluate and read is a trace event with `evaluation_id`, `dataset_id`, `fingerprint`, `baseline_id`, `baseline_cached`, `eval_samples` and the charge, ids and numbers only. A restart ends a running evaluation (`interrupted`).

## Not yet

Activation interventions (`intervention`) and `probe` are E5; the human-only sample image and label routes and the generalised final evaluation (holdout) are E4b and E4c; the legacy MNIST routes over the classification evaluator (V1, byte-identical) are E4d.
