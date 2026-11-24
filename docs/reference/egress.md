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

`GET /agent-session` reports `capabilities` and `egress_class`; `connection.json` (`--agent-token-file`) carries `egress_class` beside `url`, `project_id` and `token`. `layerdelta-mcp` **refuses to start** at class 2 (or without `egress_class`) unless `--allow-values` is given, because an MCP host sends every result to its model provider; the backend default stays v0.2 (a `weights:read` token has values). Decision 2 on PR #44: the bridge defaults to class 1.

## Class 1 is a volume control, not secrecy (v0.4-S4a.2)

Aggregates over arbitrary selections leak exact values through differencing and bisection, so class 0 is the secrecy class. What class 1 prevents is bulk extraction. For a token **without** `weights:values`, every statistic must aggregate at least **K** elements (`--agent-min-aggregate K`, default 8; `create_app(min_aggregate=)`). A person and a class-2 token are not subject to it.

| Tool | Rule |
|---|---|
| `stats` (tensor selector) | the elements of each unit (the whole selection when `reduce_over_axes` is omitted) must be at least K; otherwise `403 aggregate_too_small`, `field: reduce_over_axes`. With `bins`, each unit needs K elements **per bucket** (`field: bins`): a shared histogram puts every element of a small unit in its own bucket, which is its value to about log2(bins) bits |
| `stats`, `diff` (group selector) | the elements of one unit of one member must be at least K; a group with a bias member (one element per unit) is refused |
| `diff` with a selector | the selection must be at least K (`field: selector`). An **explicit** `unit_axis` whose units are smaller is refused (`field: unit_axis`). When `unit_axis` is left out and the default breakdown would be too small, it is resolved to none and echoed as `resolved.unit_axis: null` (a default being resolved, not a redaction) |
| `diff` without a selector | tensors with fewer than K elements are listed as `status: skipped_small` and left out of the comparison (`summary.skipped_small`, `skipped_small_count`); a per-unit breakdown smaller than K is dropped for that tensor and the row says `units_omitted: aggregate_too_small` |
| `query` with `where` | a tensor (or a group's ranking member) whose units have fewer than K elements is not evaluated and is reported in `result.skipped_small` and `skipped_small_count`. Metadata queries are unchanged. |

The error is `403 aggregate_too_small` with the argument to widen in `field`; the message holds numbers only.

**Residual leakage, bounded by budgets** ([budgets.md](budgets.md)): `min` and `max` are exact values, at most two per K scanned elements; differencing (two aggregates that differ by one element), the `apply` no-op check and `apply set` followed by `diff` recover values at budget cost; hashes of one- or two-element tensors (`plan`, `validate`) can be brute-forced.

## The structure profile: a session that cannot read weights (v0.5-E6)

`layerdelta --agent-profile structure --agent-weights-write` starts a session whose token has `model:read` and `weights:write` and **no** `weights:read` (egress class 0). It is the control arm of the v0.5 benchmark, and the server, not the tool list, enforces it:

- **Routes.** A profile is a set of routes (`registry.profile_routes`; a route is in `structure` when it is marked `structure=True` in the registry). Every other route answers 403 `forbidden` to this token: `stats`, `query`, `read-tensor`, the legacy MNIST routes, the not-exposed routes. The `tools` profile (`--agent-profile tools`) is every agent tool and no legacy route; it is the benchmark's weights arm. `connection.json` carries `profile`, and `layerdelta-mcp` and `layerdelta-agent` offer only that profile's tools (a tool outside it is `unknown_tool` before any request).
- **plan, validate, apply, diff** are allowed, and their results are rebuilt from an allow-list (`structure.py`), so a field added to a response later is invisible to this profile until it is listed. Removed: `changed_elems`, `touched_elems` (and their totals), `before_sha`, `after_sha`, `noop`, `manifest.touched`, `max_abs`, `changed_units`. `diff` compares whole tensors only and answers a `status` per tensor (a selector, `values` or `include_range` is refused 403).
- **No `no_op` refusal.** A session without `weights:read` always has `allow_noop` set, because "this changed nothing" would tell it whether a region is all zero. `--agent-allow-noop` does the same for any agent token, so both benchmark arms behave alike.
- **Residual leaks, bounded by the budgets.** (1) One bit per apply: a program that changed nothing yields a child with the parent's bytes, and `existing` on an apply says the child's bytes were already in the catalog (the parent after a no-op, or an earlier child). (2) One bit per plan, validate or apply from a value-dependent refusal (`dtype_overflow` and `non_finite` depend on magnitudes). (3) One bit per tensor per `diff` (`changed` or `same`). The `applies` and `scan` budgets bound all three, and the benchmark harness counts them in the trace.
- **Tests.** `tests/test_structure_profile.py` walks every registry route: outside the profile is refused on the server, inside it the response has exactly the allow-listed keys and none of the fields above.
- **Not covered by this profile:** a graph attached from an ONNX file may carry small constants (up to 64 elements, as `onnx.*` node kwargs) that come from that file, not from the checkpoint; the benchmark uses builtin graphs. `describe` never fills `tensor_sha256`, `identical_bytes_group` or `identical_bytes` for a session without `weights:read`.
