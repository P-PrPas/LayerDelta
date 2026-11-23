# Headless agent runner (v0.4-S4b)

Design: [v0.4-agent-surface.md](../architecture/v0.4-agent-surface.md) section 6. Code: `agent_runner.py` (console script `layerdelta-agent`), reusing `agent_bridge.execute` so a tool call behaves exactly as over MCP. Egress classes: [egress.md](egress.md).

```
layerdelta --workspace .layerdelta --agent-no-values --agent-token-file connection.json     # class 1 session
layerdelta-agent --connection connection.json --provider local --base-url http://127.0.0.1:8080/v1 --model qwen --task "find the weakest units of block1 and zero them" --max-steps 20
ANTHROPIC_API_KEY=... layerdelta-agent --connection connection.json --provider anthropic --model <model id> --task "..."
OPENAI_API_KEY=...    layerdelta-agent --connection connection.json --provider openai    --model <model id> --task "..."
```

## Providers and the data policy

LayerDelta owns the provider call here, so it also owns the policy. A provider has a locality:

| Provider | Locality | Notes |
|---|---|---|
| `local` (any OpenAI-compatible endpoint: llama.cpp, Ollama, vLLM) | local: `--base-url` must be loopback (`127.0.0.1`, `localhost`, `::1`), otherwise the runner refuses to start | `LAYERDELTA_LOCAL_API_KEY` optional; never `OPENAI_API_KEY`, so a typo or a pasted URL cannot send it elsewhere |
| `openai` | remote | `OPENAI_API_KEY` |
| `anthropic` (Messages API) | remote | `ANTHROPIC_API_KEY` |
| `StubProvider` (tests, dry runs) | local | scripted replies; records everything it is sent |

A **remote** provider is refused at start when the session can read raw values (`egress_class` 2, or none in `connection.json`) unless `--allow-values-to-remote` is given; start the service with `--agent-no-values` for class 1. A local provider may see whatever the session's class releases. Keys come from the environment only (never an argument or a file) and are not printed. The runner sends the provider only what the token already received (the tool results as returned) and the tool schemas; never the token, the endpoint or the workspace.

## The loop and the report

`run(client, provider, task, max_steps=20)`: the provider answers with tool calls or text; each call runs through `agent_bridge.execute` (backend errors, `budget_exhausted`, `forbidden` and `aggregate_too_small` are results the model sees); text ends the run. It also stops on `budget_exhausted` or `quota_exhausted` (the model is not asked to keep going) and at `max_steps`. A tool call whose arguments are not a JSON object comes back to the model as an `invalid_arguments` tool result (nothing is sent to the service). A provider failure (HTTP error, transport error) ends the run with `stop_reason: provider_error` and `provider_error: {status}` (status only; a response body can echo the prompt) and keeps the report and the children found so far. The report is `{stop_reason, steps, calls: [{name, is_error, code}], children, final, usage, egress_class, provider_locality}`: `children` lists only children the apply results created **and** that the catalog confirms with `get_checkpoint`; `usage` is the last `LayerDelta-Budget` header.

The adapter interface is `send(messages, tools) -> Reply(text, tool_calls)` with neutral messages (`user`, `assistant` with `tool_calls`, `tool` results); the OpenAI-compatible and Anthropic adapters translate to and from their wire formats (tested against mock transports).
