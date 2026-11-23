# Agent session budgets (v0.4-S3)

Design: [v0.4-agent-surface.md](../architecture/v0.4-agent-surface.md) section 4. Code: `budget.py` (torch-free), the meter hooks in `tensors.py`, `inspect_ops.py`, the middleware and routes in `api.py` and `v2.py`.

Budgets limit what an agent may read and write in one session. They are metered and refused in the backend, so neither the MCP bridge nor a raw HTTP client can get around them. One server process is one session; usage resets when the service restarts (the apply quota is the exception: it lives in the workspace ledger).

## Set them

```
layerdelta --workspace .layerdelta --agent-budget values=5000,scan=2000000,applies=3,blob_bytes=1000000
```

A dimension that is left out is unlimited but still counted and reported. `applies` is the existing `--quota` under its budget name (the flag overrides `--quota`). Nothing an agent sends can raise or reset a budget; there is no route for it. (The human's change and revoke routes belong to the session viewer, v0.4-S5.)

| Dimension | Charged by | Unit |
|---|---|---|
| `values` | `read_tensor` and group reads, `diff` with `values: true` | elements returned |
| `scan` | everything that reads checkpoint bytes: `stats`, `query`, `diff`, `read_tensor` (including its range pass over the whole selection), group tools, `plan`, `validate` | elements read from the file |
| `applies` | `apply` (a keyed replay is free) | children created |
| `blob_bytes` | `upload_blob` | bytes in the request |

`scan` is charged chunk by chunk **before** each chunk is read, at the single place that reads tensor bytes, so a new tool cannot forget to meter itself. A tensor hash that is already cached costs nothing. `apply` charges `applies` when it makes a child; its planning reads are checked against `scan` (a probe cannot go on after the budget is gone) but charged only if no child comes of it, and writing the child is not metered. A person (the human session) is never charged. An `apply` that makes no child (an error, `no_op`, a refusal) is charged what it read, otherwise a refused apply would be a free probe of the weights; a successful apply and a replay are charged `applies` only.

Not metered: the legacy MNIST routes (`/previews` returns activations and kernels). They are not MCP tools and S4 gates them behind the `weights:values` capability.

## Refusal

A call that would pass a limit answers **429 `budget_exhausted`** with `field` = the dimension (`values`, `scan` or `blob_bytes`) and returns no result. Units already read for that call stay charged for `scan` (a read cannot be un-read); `values` and `blob_bytes` are refused before anything is counted. The message holds numbers only. The apply quota keeps its own code, `quota_exhausted`.

## Where usage is reported

- `GET /agent-session` (`session_info`): the quota keys `limit`, `charged`, `remaining` are unchanged; new `budgets: {values, scan, blob_bytes, applies}`, each `{limit | null, used, remaining | null}`.
- Response header on every agent response: `LayerDelta-Budget: values=used/limit;scan=used/limit;applies=used/limit;blob_bytes=used/limit` (`-` for no limit). Bodies are byte-identical to before. The MCP bridge copies the header into the result's `_meta` as `layerdelta/budget`.
- Trace: an event that charged anything has `budget_charged: {dimension: amount}` (numbers only, stored in the event detail).
