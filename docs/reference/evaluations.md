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
- `intervention` (v0.5-E5b, optional): an activation edit applied to the checkpoint's run only, no checkpoint made: `{node_id, units, op: zero|scale, factor?, unit_axis: 1}` or `{group, units, op, factor?}` (a coupling group id resolves to its layer's node and unit axis, as for a probe; `units` are then indices of the group's units). `zero` sets the listed units of the node's output to 0 and `scale` multiplies them by `factor` (finite), while the program runs; the evaluation is fingerprinted with it (the baseline's run is not edited and is cached under the fingerprint without it, so to ablate, give `baseline` as the same checkpoint and compare). It needs `weights:read` (ablating a unit and watching the output is a probe by another name: the structure profile gets 403), the node must be one floating-point tensor with a batch and a unit axis in a `builtin` or `torch_export` graph (`probe_unsupported` otherwise), and `units` are 1 to 4,096 distinct indices in range. The evaluation body echoes the edit; the trace carries only `intervention_id` (a hash), never a node or a unit.
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
- `resamples`: 2,000 for every task (null without a baseline), through `Evaluator.resampler`: parts are summed, and detection matches once and repeats per-image rows. The seed comes from the fingerprint, and both evaluations use the same resampled indices (that is what makes it paired).

`evaluation_changes` pages the samples whose record changed: `{sample_id, label, condition, before, after, category, score_before, score_after}`. `category`: `fixed` (incorrect to correct), `broken` (correct to incorrect), `changed` (the record changed, correctness did not). Correctness and score come from the evaluator spec. `category` filters; `sort=score_change` lists the worst change first (a fall of a higher-is-better score, a rise of CER). Records are model outputs on exploration samples only (a class, per-class pixel counts, boxes in source coordinates, a decoded string); there are no pixels.

## Who can read what, and the trace

A token reads only the evaluations of its own session (anything else is 404); a person reads all. Every evaluate and read is a trace event with `evaluation_id`, `dataset_id`, `fingerprint`, `baseline_id`, `baseline_cached`, `eval_samples` and the charge, ids and numbers only. A restart ends a running evaluation (`interrupted`).

## The final evaluation: the one road to holdout (v0.5-E4c)

```
POST /checkpoints/{checkpoint_id}/final-evaluations   human only, Idempotency-Key   {dataset_id, evaluator, recipe_hash, confirmed: true, baseline?, graph?}
```

A person confirms the **frozen recipe hash** of the child (`confirmed` must be true and `recipe_hash` must be the one the checkpoint's lineage records, else `dependency_mismatch`, 409; an imported root checkpoint has none, so it is confirmed with `recipe_hash: null`, and the audit records null, which is how the benchmark gets the holdout score of a starting model) and the job evaluates the whole **holdout** split of the dataset, through the same worker, metrics and bootstrap as `evaluate`. It is never charged. The evaluation is read with `get_evaluation` and `evaluation_changes` by a person; its `split` is `holdout` and its body carries `recipe_hash`.

- **Agents:** the route is 403 for every token, `evaluate` answers 403 for `split: "holdout"`, and a token reads only its own session's evaluations, so a person's final evaluation is a 404 to it, as is its `changes`. The holdout fingerprint contains the holdout split hash, so no agent run or cached baseline is ever reused for it (and the reverse).
- **Audit:** every final evaluation is listed by `GET /holdout-audit` (human only) next to the v0.1 final evaluations, as `{run_id: evaluation_id, actor, recipe_hash, created_at, kind: "evaluation", dataset_id, checkpoint_id, fingerprint, status}`.

## Not yet

Activation interventions (`intervention`) and `probe` are E5. The legacy MNIST run routes are unchanged; the classification evaluator is pinned to their numbers by `tests/test_mnist_conformance.py` (V1).
