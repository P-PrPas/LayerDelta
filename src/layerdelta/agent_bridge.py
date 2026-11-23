"""`layerdelta-mcp`: the MCP bridge of the agent surface (v0.4-S1). MCP over stdio to the host, HTTP to the loopback backend.

It is a client with the agent token and nothing more: every check that matters (capabilities, quota, human-only routes, caps)
happens in the backend, so a raw HTTP client cannot get around anything the bridge does. It reads `connection.json`
(`{url, project_id, token}`) and never the workspace; it imports no torch. The tool list is the committed snapshot of the route
registry (registry.py), so it is the registry filtered to agent routes, by construction.

Results: `structuredContent` is the HTTP response body, unchanged; the session's usage (the `LayerDelta-Budget` header) is copied
into the result's `_meta` as `layerdelta/budget`. `content` is one short line of IDs and counts, never free text
from a checkpoint (tensor names, hypotheses). Errors: `isError` with `{error: {code, message, field, hint, retryable, status}}`;
`hint` and `retryable` are filled from the error catalog (agent_content.json, docs/reference/errors.md); a code it does not know keeps them null. A result above `--max-result-bytes`
is refused with `result_too_large`, never truncated.

`execute` is the whole behaviour and needs no MCP SDK; only `main` imports it (optional extra `layerdelta[mcp]`).
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

from . import registry
from .client import AgentClient, ClientError
from .selector import canonical_json

MAX_RESULT_BYTES = 200_000
_PAGING = ("limit", "max_elems", "max_scan_elems")


def tool_list(legacy: bool = False) -> list[dict]:
    return [t for t in registry.load()["tools"] if legacy or not t["legacy"]]


def check_egress(connection: dict, allow_values: bool) -> None:
    """The bridge forwards every result to an external model provider, so it refuses a class-2 session (raw values) unless the person
    said so. A connection.json without `egress_class` is treated as class 2."""
    if connection.get("egress_class", 2) >= 2 and not allow_values:
        raise SystemExit("layerdelta-mcp: this session can read raw tensor values, and an MCP host sends every result to its model "
                         "provider. Start the service with --agent-no-values (egress class 1: aggregates only), or pass "
                         "--allow-values here to accept that the values leave this machine.")


def _error(code: str, message: str, status: int | None, field: str | None = None, *, hint: str | None = None,
           retryable: bool | None = None) -> dict:
    known = registry.agent_content()["errors"].get(code)  # the error catalog; a code it does not know keeps null hint and retryable
    if known is not None:
        hint = known["hint"] if hint is None else hint
        retryable = known["retryable"] if retryable is None else retryable
    return {"error": {"code": code, "message": message, "field": field, "hint": hint, "retryable": retryable, "status": status}}


def execute(client: AgentClient, tool: str, arguments: dict, *, max_result_bytes: int = MAX_RESULT_BYTES,
            legacy: bool = False) -> tuple[dict, bool, str]:
    """Run one tool call: (structured result, is_error, one line of text)."""
    if tool not in {t["name"] for t in tool_list(legacy)}:
        return _error("unknown_tool", f"no tool named {tool!r}", None), True, f"{tool}: unknown tool"
    props = registry.tools_by_name()[tool]["inputSchema"]["properties"]
    arguments = {**{k: v["default"] for k, v in props.items() if "default" in v and k not in arguments}, **arguments}  # defaults, explicit
    try:
        body = client.call(tool, **arguments)
    except ClientError as exc:
        return _error(exc.code, str(exc), exc.status, exc.field), True, f"{tool} failed: {exc.code}"
    except httpx.TimeoutException:
        retry = "retry with the same key to get its result" if "key" in arguments else "the call may have been processed"
        return _error("request_timeout", "the service did not answer in time", None, hint="the request may have completed: " + retry,
                      retryable="key" in arguments), True, f"{tool}: timed out"
    except httpx.TransportError as exc:
        return _error("backend_unreachable", f"the LayerDelta service did not answer ({type(exc).__name__})", None,
                      hint="check that the service is running and connection.json is current", retryable=True), True, f"{tool}: backend unreachable"
    size = len(canonical_json(body).encode())
    if size > max_result_bytes:
        paging = [a for a in _PAGING if a in registry.tools_by_name()[tool]["inputSchema"]["properties"]]
        return (_error("result_too_large", f"the result is {size} bytes; the limit is {max_result_bytes}", None,
                       paging[0] if paging else None,
                       hint=("lower `" + paging[0] + "` or narrow the selection") if paging else "narrow the request", retryable=False),
                True, f"{tool}: result too large")
    return body, False, _line(tool, arguments, body)


def _line(tool: str, arguments: dict, body: dict) -> str:
    """IDs and counts only: nothing here can carry text from a checkpoint."""
    target = arguments.get("checkpoint_id") or arguments.get("run_id")
    text = tool + (f" on {target}" if isinstance(target, str) and len(target) <= 80 else "")
    if isinstance(body, dict) and body.get("truncated"):
        text += ", truncated" + (": pass `cursor` for more" if body.get("cursor") else "")
    return text


async def serve(connection: dict, *, legacy: bool, max_result_bytes: int, timeout: float = 120) -> None:
    import anyio
    import mcp.types as types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    tools = tool_list(legacy)
    server = Server("layerdelta")
    client = AgentClient(connection["url"], connection["project_id"], connection["token"], timeout=timeout, mutate_timeout=timeout,
                         egress_class=connection.get("egress_class"), via="mcp")

    @server.list_tools()
    async def list_tools():
        return [types.Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"],
                           annotations=types.ToolAnnotations(**t["annotations"])) for t in tools]

    @server.call_tool(validate_input=False)  # the backend validates, so errors and parity are the backend's own
    async def call_tool(name: str, arguments: dict):
        def run():
            outcome = execute(client, name, arguments or {}, max_result_bytes=max_result_bytes, legacy=legacy)
            return outcome, client.last_budget

        (body, is_error, line), usage = await anyio.to_thread.run_sync(run)
        return types.CallToolResult(content=[types.TextContent(type="text", text=line)], structuredContent=body, isError=is_error,
                                    _meta={"layerdelta/budget": usage} if usage else None)

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="layerdelta-mcp", description="MCP bridge to a running LayerDelta service")
    parser.add_argument("--connection", type=Path, required=True, help="connection.json written by `layerdelta --agent-token-file`")
    parser.add_argument("--legacy-tools", action="store_true", help="also offer the v0.1 MNIST run tools")
    parser.add_argument("--allow-values", action="store_true",
                        help="accept a session that can read raw tensor values (egress class 2); the default refuses it")
    parser.add_argument("--max-result-bytes", type=int, default=MAX_RESULT_BYTES)
    parser.add_argument("--timeout", type=float, default=120, help="seconds to wait for the service on one call (apply may wait up to 60)")
    args = parser.parse_args(argv)
    connection = json.loads(args.connection.read_bytes())
    if not {"url", "project_id", "token"} <= set(connection):
        raise SystemExit("connection.json must have url, project_id and token")
    check_egress(connection, args.allow_values)
    try:
        import anyio
        import mcp  # noqa: F401
    except ImportError:
        raise SystemExit('layerdelta-mcp needs the optional MCP SDK: pip install "layerdelta[mcp]"') from None
    print(f"layerdelta-mcp: {len(tool_list(args.legacy_tools))} tools for {connection['url']} "
          f"(egress class {connection.get('egress_class', '?')})", file=sys.stderr)
    anyio.run(lambda: serve(connection, legacy=args.legacy_tools, max_result_bytes=args.max_result_bytes,
                                  timeout=args.timeout))


if __name__ == "__main__":
    main()
