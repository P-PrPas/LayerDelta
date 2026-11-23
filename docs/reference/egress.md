# Egress classes and the `weights:values` capability (v0.4-S4)

Design: [v0.4-agent-surface.md](../architecture/v0.4-agent-surface.md) section 6. Code: `Credentials` and the gates in `api.py` and `v2.py`; tests in `tests/test_egress.py`.

Data leaves the machine in two ways: an external MCP host forwards every tool result to its model provider, and LayerDelta's own headless runner (S4b) calls a provider itself. LayerDelta cannot filter what a host forwards, so the control is **what the backend releases to the token**.

| Class | Released | Capability |
|---|---|---|
| 0 | metadata: IDs, names, dtypes, shapes, roles, groups, coverage, trace | `model:read` (always) |
| 1 | plus aggregates: `stats`, `query` rankings, `diff` summaries, `plan`/`validate` effect counts | `weights:read` |
| 2 | plus raw element values: `read_tensor` (tensor and group), `diff` with `values: true`, legacy previews (activation maps, kernels, images) | `weights:values` |
| 3 | inbound: blobs, programs (not egress) | `weights:write` |

`weights:values` is split out of `weights:read`. For v0.2 compatibility a token with `weights:read` keeps values unless the service is started with `--agent-no-values` (`Credentials(weights_values=False)`), which makes the session class 1. `weights:values` requires `weights:read`. A refused call answers `403 forbidden` with the capability in the message; responses are never redacted in flight. The human session has every capability.

`GET /agent-session` reports `capabilities` and `egress_class`; `connection.json` (`--agent-token-file`) carries `egress_class` beside `url`, `project_id` and `token`. `layerdelta-mcp` prints a warning to stderr when the class is 2, because the values go to the host's model provider.

## What class 1 is not

Class 1 keeps bulk values out of results; it is not secrecy. A statistic over one element is that element (per-unit `stats` of a tensor with one-element units returns each value; see the recorded known-limit test), a bisection with `query` or the `apply` no-op check can recover values at the cost of budget, and an agent that can `apply` can write a known value and `diff` it. Budgets ([budgets.md](budgets.md)) bound the cost of all of these. A minimum aggregate size for class 1 is an open design question.
