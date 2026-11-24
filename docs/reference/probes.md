# Activation probes: `probe` and `get_probe` (v0.5-E5a)

Design: [v0.5 design §4.1](../architecture/v0.5-evaluators-benchmark.md). Code: `probeservice.py` (plan, ledger, job), `probe_worker.py` (isolated worker), `graph_run.py` (run an exported program node by node), [executables.md](executables.md), [datasets.md](datasets.md), [evaluations.md](evaluations.md) (keys, budgets, restarts work the same way).

```
POST /checkpoints/{checkpoint_id}/probes   tool probe      eval:run + weights:read, Idempotency-Key   202 {probe_id}
GET  /probes/{probe_id}                    tool get_probe  weights:read                               status, aggregates
```

A probe records what graph nodes output on exploration samples and returns **per-unit aggregates**. Activations on known inputs derive from the weights (Roadmap R2), so a probe needs `weights:read`; the benchmark's control arm has none, so it has no probe (the `structure` profile refuses both routes on the server).

## The request

`{dataset_id, nodes, unit_axis: 1, split_by: null, bins: 0, values: false, limit_samples?, graph?}`.

- `nodes`: 1 to 8 distinct node ids of the graph (`n0`, `n1`, ... from `graph`). A node must output one floating-point tensor with a batch axis and the unit axis; otherwise `probe_unsupported`, never a partial answer.
- `unit_axis`: the axis whose indices are the units (1: channels of `[N, C, H, W]`, features of `[N, F]`). Statistics are over every sample and every other position.
- `split_by`: `label` (classification datasets) or `condition`; a group is its own set of aggregates.
- `bins`: a histogram per unit on one range per node (two passes); 0 for none.
- `values: true`: the raw activations of the first batch of samples (capped at 65,536 elements per node). Needs `weights:values` (egress class 2): a class-1 token gets 403.
- Only `builtin` and `torch_export` graphs can be probed (both are exported programs); an ONNX graph is `probe_unsupported`. A `{"group": ...}` selector is not yet accepted.

## The K rule and the cost

For a session without `weights:values` (class 1) every unit's statistics must aggregate at least **K** elements, counted as samples x positions per unit (default K = 8). A split group below K is skipped and listed in `skipped_small`; a probe in which every group is below K is refused with `aggregate_too_small` (403) before anything is charged. A probe costs `evals` 1 and `eval_samples` n (the samples run), both or neither, at submission; a repeated key is the same probe and is charged once.

## The result

`get_probe` has the job fields (`status`: queued, running, succeeded, failed; `error_code`, `message`), `checkpoint_id`, `dataset_id`, `samples`, and when it succeeded `nodes`: per node `{node_id, unit_axis, units, groups: {name: {n, mean, std, l2, frac_positive, histogram?: {edges, counts}}}}` (`n` is the elements per unit; `std` is the population standard deviation; `l2` is the norm of the unit's values), `skipped_small`, and `values` for a session with `weights:values` that asked. A token reads only the probes of its own session (anything else is 404). The trace carries `probe_id`, `dataset_id`, the sample count and the charge.

## How it runs

The worker loads the executable's `ExportedProgram` (the registered model exported afresh, or the saved `.pt2` re-hashed first), substitutes the checkpoint's weights by name and runs it through `torch.fx.Interpreter`, tapping the nodes. IR node ids are the program's `call_function` nodes in order, so an id of the graph is the same node here (`graph_run.node_ids`). Aggregates are accumulated in float64, so the same inputs give the same numbers.

## Not yet

Activation interventions in `evaluate` (E5b: `graph_run.run(change=...)` is the hook for them); ONNX graphs and group selectors (E5c).
