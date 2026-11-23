"""The MCP bridge (v0.4-S1, acceptance T1): the same call gives the same bytes over MCP as over HTTP, errors included.

`execute` (the whole behaviour of the bridge) is tested against a live loopback service; one test drives the real stdio process
with the MCP SDK and is skipped without the optional extra (`pip install "layerdelta[mcp]"`).
"""

import base64
import json
import os
import socket
import sys
import threading
import time

import httpx
import pytest
import uvicorn

from layerdelta import agent_bridge, api
from layerdelta.client import AgentClient
from layerdelta.selector import canonical_json
from test_v2_api import BASE, BIAS, PROGRAM, REF, make_workspace  # noqa: F401

C = f"/checkpoints/{REF}"
PRE = "/api/v1/projects/demo"


@pytest.fixture
def live(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path, monkeypatch)
    creds = api.Credentials("demo", weights_read=True, weights_write=True)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(api.create_app(workspace, creds, quota=5), log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started and time.monotonic() < deadline:
        time.sleep(.05)
    assert server.started
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    connection = {"url": url, "project_id": "demo", "token": creds.agent}
    path = tmp_path / "connection.json"
    path.write_text(json.dumps(connection), encoding="utf-8")
    raw = httpx.Client(base_url=url + PRE, headers={"Authorization": "Bearer " + creds.agent}, trust_env=False)
    with AgentClient(**connection) as client:
        yield client, raw, path
    raw.close()
    server.should_exit = True
    thread.join(30)


# (tool, arguments, the same request over plain HTTP): valid and invalid calls, one per tool where one can be pure
SCRIPT = [
    ("session_info", {}, lambda h: h.get("/agent-session")),
    ("list_checkpoints", {}, lambda h: h.get("/checkpoints")),
    ("get_checkpoint", {"checkpoint_id": REF}, lambda h: h.get(C)),
    ("get_checkpoint", {"checkpoint_id": "f" * 64}, lambda h: h.get("/checkpoints/" + "f" * 64)),
    ("describe", {"checkpoint_id": REF, "limit": 3}, lambda h: h.get(C + "/describe", params={"limit": 3})),
    ("describe", {"checkpoint_id": REF, "limit": 0}, lambda h: h.get(C + "/describe", params={"limit": 0})),
    ("describe", {"checkpoint_id": REF, "name_glob": "*.bias"}, lambda h: h.get(C + "/describe", params={"name_glob": "*.bias"})),
    ("graph", {"checkpoint_id": REF}, lambda h: h.get(C + "/graph")),
    ("groups", {"checkpoint_id": REF}, lambda h: h.get(C + "/groups")),
    ("graph_node", {"checkpoint_id": REF, "node_id": "n0"}, lambda h: h.get(C + "/graph/nodes/n0")),
    ("stats", {"checkpoint_id": REF, "selector": BIAS}, lambda h: h.post(C + "/stats", json={"selector": BIAS})),
    ("stats", {"checkpoint_id": REF, "selector": {"tensor": "nope"}}, lambda h: h.post(C + "/stats", json={"selector": {"tensor": "nope"}})),
    ("stats", {"checkpoint_id": REF, "selector": BIAS, "bogus": 1}, lambda h: h.post(C + "/stats", json={"selector": BIAS, "bogus": 1})),
    ("query", {"checkpoint_id": REF, "limit": 2}, lambda h: h.post(C + "/query", json={"limit": 2})),
    ("diff", {"checkpoint_id": REF, "other": REF}, lambda h: h.post(C + "/diff", json={"other": REF})),
    ("read_tensor", {"checkpoint_id": REF, "selector": BIAS, "max_elems": 4},
     lambda h: h.post(C + "/read-tensor", json={"selector": BIAS, "max_elems": 4})),
    ("plan", {"checkpoint_id": REF, **PROGRAM}, lambda h: h.post(C + "/plan", json=PROGRAM)),
    ("validate", {"checkpoint_id": REF, **PROGRAM}, lambda h: h.post(C + "/validate", json=PROGRAM)),
    ("validate", {"checkpoint_id": REF, "hypothesis": "x"}, lambda h: h.post(C + "/validate", json={"hypothesis": "x"})),
    ("trace", {"limit": 5}, lambda h: h.get("/trace", params={"limit": 5})),
    ("trace", {"limit": 0}, lambda h: h.get("/trace", params={"limit": 0})),
]


def test_every_scripted_call_gives_the_http_body_and_error(live):
    client, raw, _ = live
    for tool, arguments, over_http in SCRIPT:
        structured, is_error, line = agent_bridge.execute(client, tool, arguments)
        response = over_http(raw)
        body = response.json()
        if tool == "trace":  # the trace grows with every call: compare the shape, not the event list
            body, structured = {k: v for k, v in body.items() if k != "result"}, {k: v for k, v in structured.items() if k != "result"}
        if tool == "session_info":
            body, structured = {k: v for k, v in body.items() if k != "charged"}, {k: v for k, v in structured.items() if k != "charged"}
        assert is_error is (response.status_code >= 400), (tool, arguments)
        if is_error:
            assert {k: structured["error"][k] for k in ("code", "field")} == {k: body["error"].get(k) for k in ("code", "field")}, (tool, arguments)
            assert structured["error"]["status"] == response.status_code and set(structured["error"]) >= {"hint", "retryable", "message"}
            assert "failed" in line
        else:
            assert canonical_json(structured) == canonical_json(body), (tool, arguments)
        assert len(line) < 200 and "\n" not in line


def test_apply_replays_by_key_and_blobs_travel_as_base64(live):
    client, raw, _ = live
    first, error, line = agent_bridge.execute(client, "apply", {"checkpoint_id": REF, "key": "mcp-1", **PROGRAM})
    assert not error and [e["kind"] for e in first["side_effects"]] == ["checkpoint_created", "quota_charged"], first
    again, error, _ = agent_bridge.execute(client, "apply", {"checkpoint_id": REF, "key": "mcp-1", **PROGRAM})
    assert not error and again["result"]["replayed"] is True and again["side_effects"] == [] and again["fingerprints"] == first["fingerprints"]
    missing, error, _ = agent_bridge.execute(client, "apply", {"checkpoint_id": REF, **PROGRAM})
    assert error and missing["error"]["code"] == "invalid_argument" and missing["error"]["field"] == "key"
    blob, error, _ = agent_bridge.execute(client, "upload_blob", {"data_base64": base64.b64encode(bytes(8)).decode()})
    assert not error and blob["side_effects"][0]["kind"] == "blob_stored"
    odd, error, _ = agent_bridge.execute(client, "upload_blob", {"data_base64": base64.b64encode(bytes(7)).decode()})
    assert error and odd["error"]["code"] == "blob_size"  # the backend's own refusal, unchanged
    capped, error, _ = agent_bridge.execute(client, "upload_blob", {"data_base64": base64.b64encode(bytes(1 << 20 + 4)).decode()})
    assert error and capped["error"]["field"] == "data_base64"


def test_bridge_refusals_are_results_not_crashes(live):
    client, raw, _ = live
    for tool, arguments, code in (("nope", {}, "unknown_tool"), ("run", {"kind": "baseline", "key": "k"}, "unknown_tool"),  # legacy is off
                                  ("describe", {}, "invalid_argument"), ("describe", {"checkpoint_id": REF, "x": 1}, "invalid_argument")):
        structured, is_error, _ = agent_bridge.execute(client, tool, arguments)
        assert is_error and structured["error"]["code"] == code, (tool, arguments)
    structured, is_error, _ = agent_bridge.execute(client, "status", {"run_id": "r1"}, legacy=True)  # on with the flag: the backend answers
    assert is_error and structured["error"]["code"] == "not_found" and structured["error"]["status"] == 404
    big, is_error, line = agent_bridge.execute(client, "describe", {"checkpoint_id": REF}, max_result_bytes=200)
    assert is_error and big["error"]["code"] == "result_too_large" and big["error"]["field"] == "limit" and "limit" in big["error"]["hint"]
    assert "describe" in line and REF not in line and "tensor" not in line


def test_failures_that_are_not_backend_errors_are_still_structured_results():
    answers = {"/checkpoints/a": httpx.Response(500, text="<html>proxy page</html>"),
               "/checkpoints/b": httpx.Response(404, json={"detail": "no error key"}),
               "/checkpoints/c": httpx.Response(200, text="not json")}

    def handle(request):
        path = request.url.path.removeprefix("/api/v1/projects/demo")
        if request.method == "POST":
            raise httpx.ReadTimeout("slow", request=request)
        return answers[path]

    with AgentClient("http://127.0.0.1:1", "demo", "t", transport=httpx.MockTransport(handle)) as client:
        for argument, code, status in (("..", "invalid_argument", 422), ("a/b", "invalid_argument", 422), ("a", "backend_error", 500),
                                       ("b", "backend_error", 404), ("c", "backend_error", 200)):
            structured, is_error, _ = agent_bridge.execute(client, "get_checkpoint", {"checkpoint_id": argument})
            assert is_error and (structured["error"]["code"], structured["error"]["status"]) == (code, status), argument
        assert structured["error"]["field"] is None
        timed_out, is_error, _ = agent_bridge.execute(client, "apply", {"checkpoint_id": REF, "key": "k", **PROGRAM})
        assert is_error and timed_out["error"]["code"] == "request_timeout" and timed_out["error"]["retryable"] is True
        assert "retry with the same key" in timed_out["error"]["hint"]
        unkeyed, _, _ = agent_bridge.execute(client, "plan", {"checkpoint_id": REF, **PROGRAM})
        assert unkeyed["error"]["code"] == "request_timeout" and unkeyed["error"]["retryable"] is False


def test_the_bridge_refuses_a_session_that_can_read_values_unless_told(tmp_path, capsys):
    path = tmp_path / "connection.json"
    for connection, allowed in (({"egress_class": 2}, False), ({}, False), ({"egress_class": 1}, False), ({"egress_class": 2}, True)):
        path.write_text(json.dumps({"url": "http://127.0.0.1:1", "project_id": "p", "token": "t", **connection}), encoding="utf-8")
        refused = connection.get("egress_class", 2) >= 2 and not allowed
        if refused:
            with pytest.raises(SystemExit, match="--allow-values"):
                agent_bridge.main(["--connection", str(path)])  # refuses before it imports the SDK or starts anything
        else:
            agent_bridge.check_egress(json.loads(path.read_text()), allowed)
    assert AgentClient("http://127.0.0.1:1", "p", "t", egress_class=1).egress_class == 1  # connection.json can be splatted into AgentClient


def test_the_text_line_carries_ids_and_counts_only(live):
    client, raw, _ = live
    structured, is_error, line = agent_bridge.execute(client, "describe", {"checkpoint_id": REF, "limit": 2})
    assert not is_error and line == f"describe on {REF}, truncated: pass `cursor` for more"
    assert not any(t["name"] in line for t in structured["result"]["tensors"])
    dead = AgentClient("http://127.0.0.1:9", "demo", "t")
    unreachable, is_error, _ = agent_bridge.execute(dead, "session_info", {})
    assert is_error and unreachable["error"]["code"] == "backend_unreachable" and unreachable["error"]["retryable"] is True


def test_the_real_stdio_bridge_lists_and_calls_tools(live):
    pytest.importorskip("mcp")
    import anyio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    client, raw, connection = live
    params = StdioServerParameters(command=sys.executable, args=["-m", "layerdelta.agent_bridge", "--connection", str(connection), "--allow-values"],
                                   env=dict(os.environ))

    async def session():
        async with stdio_client(params) as (read, write), ClientSession(read, write) as mcp_session:
            await mcp_session.initialize()
            listed = await mcp_session.list_tools()
            ok = await mcp_session.call_tool("describe", {"checkpoint_id": REF, "limit": 2})
            bad = await mcp_session.call_tool("describe", {"checkpoint_id": REF, "limit": 0})
            return listed, ok, bad

    listed, ok, bad = anyio.run(session)
    names = [t.name for t in listed.tools]
    assert names == [t["name"] for t in agent_bridge.tool_list()] and "run" not in names
    describe = next(t for t in listed.tools if t.name == "describe")
    assert describe.annotations.readOnlyHint is True and describe.inputSchema["additionalProperties"] is False
    assert ok.meta["layerdelta/budget"].startswith("values=") and "applies=0/5" in ok.meta["layerdelta/budget"]
    assert not ok.isError and canonical_json(ok.structuredContent) == canonical_json(raw.get(C + "/describe", params={"limit": 2}).json())
    assert bad.isError and bad.structuredContent["error"]["code"] == "invalid_argument" and bad.structuredContent["error"]["field"] == "limit"
