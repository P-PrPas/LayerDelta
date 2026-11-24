"""The route registry (v0.4-S1): the one table the agent surface is generated from. Torch-free.

Every HTTP route under `/api/v1/projects/{project_id}` has one row. `exposure` says what an agent may do with it:

- `tool`: a tool of the MCP bridge (and of `AgentClient.call`), always on.
- `legacy`: the v0.1 MNIST run tools, only with `layerdelta-mcp --legacy-tools`.
- `human_only`: the backend refuses an agent token (403); never a tool.
- `not_exposed`: allowed over HTTP for an agent but not offered as a tool in v0.4.

`generate(app)` builds the tool snapshot (`agent_tools.json`: input schemas, annotations and the request recipe of each tool) from
the app's own OpenAPI document, so schemas come from the same pydantic request models the backend validates with. The snapshot is
committed; a test regenerates it and fails on any difference, so it changes only together with a `VERSION` bump.
Rewrite it with `LAYERDELTA_WRITE_SNAPSHOT=1 pytest tests/test_agent_registry.py`; the bridge only reads the file and imports no torch.
"""

import json
from dataclasses import dataclass
from pathlib import Path

SNAPSHOT = Path(__file__).with_name("agent_tools.json")
CONTENT = Path(__file__).with_name("agent_content.json")  # tool descriptions, examples, LLM defaults and the error catalog (v0.4-S2)
VERSION = 3  # bump together with any change to the snapshot (or to agent_content.json: descriptions, examples, defaults)
HUMAN_CAPABILITIES = ["model:read", "weights:read", "weights:values", "weights:write"]  # the human session has all of them
BASE = "/api/v1/projects/{project_id}"
BLOB_CALL_MAX = 1 << 20  # bytes of one upload_blob call over MCP


@dataclass(frozen=True)
class Route:
    method: str
    path: str                        # relative to BASE
    exposure: str                    # tool | legacy | human_only | not_exposed
    tool: str | None = None
    capability: str | None = None    # what the backend requires today (v0.4-S4 splits weights:values out of weights:read)
    pure: bool = True                # no side effects: the tool is read-only
    key: bool = False                # the route requires an Idempotency-Key
    structure: bool = False          # offered by the `structure` profile: no route of it returns a value-derived field (v0.5-E6)


def _tool(method, path, tool, capability, summary, **kw):
    """`summary` is a reading aid for this table; what a model reads comes from agent_content.json."""
    return Route(method, path, "tool", tool, capability, **kw)


def _legacy(method, path, tool, summary, **kw):
    return Route(method, path, "legacy", tool, "agent", **kw)


ROUTES: tuple[Route, ...] = (
    _tool("GET", "/agent-session", "session_info", "agent", "Your capabilities and apply quota.", structure=True),
    _tool("GET", "/checkpoints", "list_checkpoints", "agent", "List checkpoints (paged).", structure=True),
    _tool("GET", "/checkpoints/{checkpoint_id}", "get_checkpoint", "agent", "One checkpoint's metadata.", structure=True),
    _tool("GET", "/checkpoints/{checkpoint_id}/describe", "describe", "model:read",
          "Tensor table of a checkpoint with roles and groups when a graph is attached.", structure=True),
    _tool("GET", "/checkpoints/{checkpoint_id}/graph", "graph", "model:read", "The attached graph (structure only), or the attach card.",
          structure=True),
    _tool("GET", "/checkpoints/{checkpoint_id}/graph/nodes/{node_id}", "graph_node", "model:read", "One graph node.", structure=True),
    _tool("GET", "/checkpoints/{checkpoint_id}/groups", "groups", "model:read", "Coupling groups; closed ones can be permuted.",
          structure=True),
    _tool("POST", "/checkpoints/{checkpoint_id}/stats", "stats", "weights:read", "Statistics of a selection (aggregates)."),
    _tool("POST", "/checkpoints/{checkpoint_id}/query", "query", "weights:read", "Find tensors or groups by name, shape or statistic."),
    _tool("POST", "/checkpoints/{checkpoint_id}/diff", "diff", "weights:read", "Compare a checkpoint with another (values: true needs weights:values).",
          structure=True),
    _tool("POST", "/checkpoints/{checkpoint_id}/read-tensor", "read_tensor", "weights:values", "Read raw tensor values (capped)."),
    _tool("POST", "/checkpoints/{checkpoint_id}/plan", "plan", "weights:read", "Dry-run an edit program; returns a recipe.",
          structure=True),
    _tool("POST", "/checkpoints/{checkpoint_id}/validate", "validate", "weights:read", "Check an edit program or recipe.", structure=True),
    _tool("POST", "/checkpoints/{checkpoint_id}/apply", "apply", "weights:write", "Apply a recipe: creates a child checkpoint.",
          pure=False, key=True, structure=True),
    _tool("POST", "/blobs", "upload_blob", "weights:write", "Upload a float32 array (base64) for set/add ops.", pure=False,
          structure=True),
    _tool("GET", "/trace", "trace", "model:read", "Your session's event timeline.", structure=True),
    # v0.1 MNIST tools
    _legacy("POST", "/recipes", "propose", "Record an MNIST intervention as a recipe.", pure=False),
    _legacy("POST", "/runs", "run", "Submit an MNIST run.", pure=False, key=True),
    _legacy("GET", "/runs/{run_id}", "status", "Status of a run."),
    _legacy("GET", "/runs/{run_id}/comparison", "compare", "Baseline-versus-intervention comparison of a run."),
    _legacy("GET", "/runs/{run_id}/changes", "changes", "Per-sample changes of a run (paged)."),
    _legacy("POST", "/runs/{run_id}/cancel", "cancel", "Cancel a run.", pure=False),
    # a person decides these
    Route("POST", "/checkpoints", "human_only"),
    Route("POST", "/checkpoint-imports", "human_only"),
    Route("POST", "/checkpoint-imports/upload", "human_only"),
    Route("GET", "/checkpoint-imports/{import_id}", "human_only"),
    Route("POST", "/checkpoint-imports/{import_id}/cancel", "human_only"),
    Route("POST", "/checkpoint-exports", "human_only"),
    Route("GET", "/checkpoint-exports/{run_id}", "human_only"),
    Route("GET", "/checkpoint-exports/{run_id}/file", "human_only"),
    Route("POST", "/checkpoints/{checkpoint_id}/graphs", "human_only"),
    Route("GET", "/graph-attachments/{attach_id}", "human_only"),
    Route("POST", "/final-evaluations", "human_only"),
    Route("GET", "/holdout-audit", "human_only"),
    Route("GET", "/whoami", "human_only"),
    Route("GET", "/trace/sessions", "human_only"),
    Route("POST", "/agent-sessions/{session}/revoke", "human_only"),
    Route("POST", "/agent-sessions/{session}/budget", "human_only"),
    Route("GET", "/proposals", "human_only"),
    Route("POST", "/proposals/{proposal_id}/approve", "human_only"),
    Route("POST", "/proposals/{proposal_id}/reject", "human_only"),
    Route("POST", "/selections/resolve", "human_only"),
    Route("GET", "/recipes/by-hash/{recipe_hash}", "human_only"),
    # allowed to an agent over HTTP, not a tool in v0.4
    Route("GET", "/catalog", "not_exposed"),
    Route("GET", "/samples", "not_exposed"),
    Route("POST", "/previews", "not_exposed"),
    Route("GET", "/previews/{run_id}", "not_exposed"),
    Route("POST", "/recipes/import", "not_exposed"),
    Route("GET", "/recipes/{recipe_id}", "not_exposed"),
    Route("GET", "/runs", "not_exposed"),
    Route("GET", "/runs/{run_id}/report", "not_exposed"),
)
OUTSIDE = {("GET", "/api/v1/health"), ("POST", "/api/v1/session")}  # not under a project: no agent tool, no registry row


PROFILES = ("structure", "tools")


def profile_routes(profile: str) -> frozenset:
    """(method, path template) of the routes a session profile may call. `tools`: every agent tool (no legacy MNIST routes, nothing
    that is not a tool). `structure`: the tools whose responses carry no value-derived field (v0.5 benchmark arm C)."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}")
    return frozenset((r.method, BASE + r.path) for r in ROUTES
                     if (r.structure if profile == "structure" else r.exposure == "tool"))


def profile_tools(profile: str | None) -> set | None:
    """Tool names of a profile; None (no profile) offers everything."""
    if profile is None:
        return None
    return {r.tool for r in ROUTES if (r.method, BASE + r.path) in profile_routes(profile)}


def by_tool() -> dict[str, Route]:
    return {r.tool: r for r in ROUTES if r.tool}


# ---------------------------------------------------------------- snapshot generation

def _clean(schema, *, drop_null=False):
    """Pydantic's schema without titles (noise for a model); optionally `X | null` collapsed to X (query parameters)."""
    if isinstance(schema, list):
        return [_clean(s, drop_null=drop_null) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: _clean(v, drop_null=drop_null) for k, v in schema.items() if k not in ("title", "examples") or not isinstance(v, (str, list))}
    options = out.get("anyOf")
    if drop_null and options and len(options) == 2 and {"type": "null"} in options:
        return {**{k: v for k, v in out.items() if k != "anyOf"}, **next(o for o in options if o != {"type": "null"})}
    return out


def _refs(schema, found: set):
    if isinstance(schema, dict):
        ref = schema.get("$ref")
        if isinstance(ref, str):
            found.add(ref.rsplit("/", 1)[-1])
        for v in schema.values():
            _refs(v, found)
    elif isinstance(schema, list):
        for v in schema:
            _refs(v, found)


def _retarget(schema):
    if isinstance(schema, dict):
        return {k: (v.replace("#/components/schemas/", "#/$defs/") if k == "$ref" and isinstance(v, str) else _retarget(v))
                for k, v in schema.items()}
    if isinstance(schema, list):
        return [_retarget(v) for v in schema]
    return schema


def generate(app) -> dict:
    """The tool snapshot for every `tool` and `legacy` route, from the app's OpenAPI document. Also checks the registry covers the app."""
    spec = app.openapi()
    components = spec.get("components", {}).get("schemas", {})
    operations = {(m.upper(), p): op for p, item in spec["paths"].items() for m, op in item.items()}
    expected = {(r.method, BASE + r.path) for r in ROUTES} | OUTSIDE
    if set(operations) != expected:
        raise ValueError(f"the registry and the app disagree: only in the app {sorted(set(operations) - expected)}, "
                         f"only in the registry {sorted(expected - set(operations))}")
    tools = []
    for route in ROUTES:
        if route.exposure not in ("tool", "legacy"):
            continue
        op = operations[(route.method, BASE + route.path)]
        props, required, defs = {}, [], set()
        path_params, query = [], []
        for p in op.get("parameters", []):
            if p["name"] == "project_id" or p["in"] == "header":
                continue
            props[p["name"]] = _clean(p["schema"], drop_null=True)
            if p["in"] == "path":
                path_params.append(p["name"])
                required.append(p["name"])
            else:
                query.append(p["name"])
        body = None
        content = op.get("requestBody", {}).get("content", {})
        if "application/json" in content:
            body = "json"
            ref = content["application/json"]["schema"].get("$ref")
            model = components[ref.rsplit("/", 1)[-1]] if ref else content["application/json"]["schema"]
            for name, schema in model.get("properties", {}).items():
                if name in props:
                    raise ValueError(f"{route.tool}: body field {name!r} collides with a path or query parameter")
                props[name] = _clean(schema)
            required += model.get("required", [])
            _refs(model.get("properties", {}), defs)
        elif route.tool == "upload_blob":
            body = "octet"
            props["data_base64"] = {"type": "string", "maxLength": BLOB_CALL_MAX * 4 // 3 + 4,
                                    "description": "Base64 of a flat little-endian float32 array, at most 1 MiB decoded."}
            required.append("data_base64")
        if route.key:
            props["key"] = {"type": "string", "minLength": 1, "maxLength": 200,
                            "description": "Idempotency key: reuse it to retry after a lost response (same result, charged once)."}
            required.append("key")
        closure, todo = {}, set(defs)
        while todo:
            name = todo.pop()
            if name not in closure:
                closure[name] = _clean(components[name])
                more: set = set()
                _refs(closure[name], more)
                todo |= more - set(closure)
        if route.tool not in agent_content()["tools"]:
            raise ValueError(f"agent_content.json has no entry for the tool {route.tool!r}")
        described = agent_content()["tools"][route.tool]
        for name, value in described.get("defaults", {}).items():  # LLM-sized defaults live in the schema; the bridge sends them explicitly
            if name not in props:
                raise ValueError(f"{route.tool}: default for unknown argument {name!r}")
            props[name]["default"] = value
        schema = {"type": "object", "properties": props, "required": sorted(set(required)), "additionalProperties": False}
        if described.get("examples"):
            schema["examples"] = described["examples"]
        if closure:
            schema["$defs"] = closure
        tools.append({
            "name": route.tool, "legacy": route.exposure == "legacy", "description": described["description"],
            "annotations": {"readOnlyHint": route.pure, "destructiveHint": False, "idempotentHint": route.pure or route.key or route.tool == "upload_blob"},
            "inputSchema": _retarget(schema),
            "route": {"method": route.method, "path": route.path, "path_params": path_params, "query": query, "body": body, "key": route.key,
                      "capability": route.capability}})
    return {"version": VERSION, "tools": tools}


def load() -> dict:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def snapshot_text(snapshot: dict) -> str:
    return json.dumps(snapshot, indent=1, sort_keys=True) + "\n"


def agent_content() -> dict:
    global _CONTENT
    if _CONTENT is None:
        _CONTENT = json.loads(CONTENT.read_text(encoding="utf-8"))
    return _CONTENT


_CONTENT = None


def tools_by_name() -> dict[str, dict]:
    global _BY_NAME
    if _BY_NAME is None:
        _BY_NAME = {t["name"]: t for t in load()["tools"]}
    return _BY_NAME


_BY_NAME = None
