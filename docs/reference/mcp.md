# MCP bridge and the tool registry (v0.4-S1)

Design: [v0.4-agent-surface.md](../architecture/v0.4-agent-surface.md) section 1. Code: `registry.py` (the route table and the snapshot generator, torch-free), `agent_tools.json` (the committed snapshot), `agent_bridge.py` (`layerdelta-mcp`), `AgentClient.call` in `client.py`.

## Run it

```powershell
layerdelta --workspace .layerdelta --agent-token-file connection.json      # the service writes {url, project_id, token}
pip install "layerdelta[mcp]"                                                # optional extra: mcp==1.30.0
layerdelta-mcp --connection connection.json [--legacy-tools] [--max-result-bytes 200000] [--timeout 120] [--allow-values]
```

It refuses to start when `connection.json` says `egress_class` 2 (raw tensor values) unless `--allow-values` is passed; start the service with `--agent-no-values` for class 1 ([egress.md](egress.md)).

The bridge speaks MCP over stdio to the host and HTTP to the loopback service with the agent token. It reads `connection.json` only, never the workspace, never the human cookie, and imports no torch. It adds no authority: capabilities, quota, human-only routes and caps are the backend's, so a raw HTTP client cannot get around anything the bridge does.

## The registry

`registry.ROUTES` has one row per HTTP route under `/api/v1/projects/{project_id}`: `method, path, exposure, tool, capability, pure, key, summary`. `exposure` is one of:

| exposure | meaning |
|---|---|
| `tool` | an MCP tool and an `AgentClient.call` name, always on |
| `legacy` | the v0.1 MNIST tools (`propose`, `run`, `status`, `compare`, `changes`, `cancel`), only with `--legacy-tools` |
| `human_only` | the backend answers an agent token with 403 (attach, import, export, final evaluation, holdout audit, upload): never a tool |
| `not_exposed` | allowed to an agent over HTTP, not a tool in v0.4 (catalog, samples, previews, recipe import and read, run list and report) |

Tools: `session_info`, `list_checkpoints`, `get_checkpoint`, `describe`, `graph`, `graph_node`, `groups`, `stats`, `query`, `diff`, `read_tensor`, `plan`, `validate`, `apply`, `upload_blob`, `trace`.

`registry.generate(app)` builds the snapshot from the app's own OpenAPI document, so input schemas come from the pydantic request models the backend validates with (`additionalProperties: false`). It also fails if the app has a route the registry lacks, or the reverse. A tool's arguments are its path parameters, query parameters and body fields in one flat object, plus `key` (the idempotency key, required for `apply` and `run`) and, for `upload_blob`, `data_base64` (at most 1 MiB decoded). `project_id` is never an argument.

Tool descriptions, examples, the LLM-sized defaults and the error catalog live in `agent_content.json` (v0.4-S2): `registry.generate` takes each tool's `description` from it, puts `examples` on the input schema and each default on its property (`default`); the bridge sends the defaults explicitly, so a call over MCP and the same call over HTTP with the same documented defaults give the same bytes. Changing it needs a `registry.VERSION` bump and a snapshot rewrite like any schema change.

`agent_tools.json` is committed. `tests/test_agent_registry.py::test_the_snapshot_is_current` regenerates it and fails on any difference: bump `registry.VERSION` and rewrite it with `LAYERDELTA_WRITE_SNAPSHOT=1 pytest tests/test_agent_registry.py`.

## Results and errors

- `structuredContent` is the HTTP response body, unchanged (canonical JSON equal). `content` is one line of IDs and counts (`describe on <checkpoint_id>, truncated: pass cursor for more`), never a tensor name or other free text.
- Annotations: `readOnlyHint` is the registry's `pure`; `apply` is `destructiveHint: false` (a new checkpoint, never an overwrite) and `idempotentHint: true` with the same `key`.
- A failure is a tool result with `isError: true` and `structuredContent = {error: {code, message, field, hint, retryable, status}}`. `code`, `message`, `field` and `status` are the backend's. `hint` and `retryable` come from the error catalog in `agent_content.json` ([errors.md](errors.md)); a code the catalog does not know keeps them `null`. The bridge's own refusals use the same shape: `unknown_tool`, `invalid_argument` (a malformed call, caught before anything is sent: a missing path parameter or key, a bad `data_base64`, an unknown argument on a GET tool), `backend_unreachable` (retryable), `request_timeout` (with a key: "the request may have completed: retry with the same key to get its result"), `backend_error` (the service answered without a structured error) and `result_too_large`.
- Unknown fields on a POST tool reach the backend, which refuses them with its own error, so the code and field are the same as over HTTP.
- A result over `--max-result-bytes` (default 200 000) is refused with `result_too_large` and a hint naming the argument to lower (`limit`, `max_elems`); the bridge never truncates.

## Tests

`tests/test_mcp_bridge.py` runs a scripted set of valid and invalid calls through the bridge and over plain HTTP against a live loopback service and compares body, error code, field and status (T1). One test drives the real stdio process with the MCP SDK (skipped without the extra). `tests/test_agent_registry.py` covers T2: the tool list is the registry filtered to agent routes, no human-only route is a tool, every schema is closed, annotations follow `pure`, and every human-only route refuses an agent token.

Not in S1: tool descriptions and LLM defaults, the error catalog (S2); budgets (S3); the `weights:values` split and egress class (S4); trace `tool`/`via` fields (S5).
