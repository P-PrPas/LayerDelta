"""The headless agent runner (v0.4-S4b, acceptance T6): provider adapters and the egress policy."""

import json
import socket
import threading
import time

import httpx
import pytest
import uvicorn
from safetensors.torch import load_file

from layerdelta import agent_runner, api
from layerdelta.agent_runner import Anthropic, OpenAICompatible, Reply, StubProvider
from layerdelta.client import AgentClient
from test_egress import contains_elements, runs  # noqa: F401
from test_v2_api import BIAS, PROGRAM, REF, make_workspace  # noqa: F401


@pytest.fixture
def live(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path, monkeypatch)
    started = []

    def make(quota=5, **caps):
        creds = api.Credentials("demo", **{"weights_read": True, "weights_write": True, **caps})
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(api.create_app(workspace, creds, quota=quota), log_level="warning"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        deadline = time.monotonic() + 30
        while not server.started and time.monotonic() < deadline:
            time.sleep(.05)
        started.append((server, thread))
        client = AgentClient(f"http://127.0.0.1:{sock.getsockname()[1]}", "demo", creds.agent, egress_class=creds.egress_class)
        return client, creds

    make.workspace = workspace
    yield make
    for server, thread in started:
        server.should_exit = True
        thread.join(30)


def call(i, name, **arguments):
    return {"id": f"c{i}", "name": name, "arguments": arguments}


def test_a_class_one_run_sends_the_provider_no_element_array_and_no_credential(live):
    client, creds = live(weights_values=False)
    stub = StubProvider([
        Reply(tool_calls=[call(1, "session_info"), call(2, "describe", checkpoint_id=REF, limit=4)]),
        Reply(tool_calls=[call(3, "stats", checkpoint_id=REF, selector=BIAS), call(4, "read_tensor", checkpoint_id=REF, selector=BIAS),
                          call(5, "stats", checkpoint_id=REF, selector=BIAS, reduce_over_axes=[])]),
        Reply(text="The bias is small; the raw values are not available to me.")])
    report = agent_runner.run(client, stub, "inspect the classifier bias")
    assert report["stop_reason"] == "final" and report["steps"] == 5 and report["children"] == []
    assert [(c["name"], c["code"]) for c in report["calls"]][3:] == [("read_tensor", "forbidden"), ("stats", "aggregate_too_small")]
    state = load_file(live.workspace / "reference" / "model.safetensors")
    flats = runs(state)
    sent = json.dumps(stub.sent)
    assert not contains_elements(stub.sent, flats)  # nothing the provider received is a run of tensor elements
    assert "forbidden" in sent and creds.agent not in sent and "127.0.0.1" not in sent  # the errors are data; the token and endpoint are not sent
    assert len(stub.sent) == 3 and stub.sent[0]["tools"] == [t["name"] for t in agent_runner.agent_bridge.tool_list()]


def test_the_report_lists_only_children_that_exist_and_the_quota_stops_the_run(live):
    client, creds = live(quota=1)
    other = {"program": [{"op": "scale", "args": {"k": 3.0}, "select": BIAS}], "hypothesis": "triple"}
    stub = StubProvider([
        Reply(tool_calls=[call(1, "apply", checkpoint_id=REF, key="a", **PROGRAM)]),
        Reply(tool_calls=[call(2, "apply", checkpoint_id=REF, key="b", **other)]),
        Reply(text="never reached")])
    report = agent_runner.run(client, stub, "double then triple the bias")
    assert report["stop_reason"] == "quota_exhausted" and report["steps"] == 2 and len(stub.sent) == 2
    assert [c["code"] for c in report["calls"]] == [None, "quota_exhausted"] and len(report["children"]) == 1
    listed = client.call("list_checkpoints")["items"]
    assert report["children"][0] in {c["checkpoint_id"] for c in listed}  # it exists
    assert report["usage"].startswith("values=")


def test_max_steps_and_unknown_tools(live):
    client, creds = live()
    stub = StubProvider([Reply(tool_calls=[call(i, "no_such_tool")]) for i in range(5)])
    report = agent_runner.run(client, stub, "loop", max_steps=3)
    assert report["stop_reason"] == "max_steps" and [c["code"] for c in report["calls"]] == ["unknown_tool"] * 3


def test_a_remote_provider_does_not_get_a_values_session_unless_allowed():
    remote = OpenAICompatible("https://api.openai.com/v1", "m", api_key="k")
    local = OpenAICompatible("http://127.0.0.1:8080/v1", "m")
    assert (remote.locality, local.locality, Anthropic("m", api_key="k").locality, StubProvider([]).locality) == ("remote", "local", "remote", "local")
    for cls in (2, None):
        with pytest.raises(SystemExit, match="--agent-no-values"):
            agent_runner.check_provider(remote, cls, False)
    agent_runner.check_provider(remote, 1, False)
    agent_runner.check_provider(remote, 0, False)
    agent_runner.check_provider(remote, 2, True)
    agent_runner.check_provider(local, 2, False)


def test_the_openai_compatible_adapter_speaks_tools_both_ways():
    seen = {}

    def handle(request):
        seen["url"], seen["auth"], seen["body"] = str(request.url), request.headers.get("authorization"), json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "describe", "arguments": json.dumps({"checkpoint_id": REF})}}]}}]})

    provider = OpenAICompatible("http://127.0.0.1:8080/v1/", "qwen", api_key="sk-secret", transport=httpx.MockTransport(handle))
    tools = agent_runner.agent_bridge.tool_list()
    reply = provider.send([{"role": "user", "text": "go"},
                           {"role": "assistant", "text": None, "tool_calls": [call(0, "session_info")]},
                           {"role": "tool", "tool_call_id": "c0", "name": "session_info", "content": "{}", "is_error": False}], tools)
    assert reply.tool_calls == [{"id": "t1", "name": "describe", "arguments": {"checkpoint_id": REF}}]
    assert seen["url"] == "http://127.0.0.1:8080/v1/chat/completions" and seen["auth"] == "Bearer sk-secret"
    roles = [m["role"] for m in seen["body"]["messages"]]
    assert roles == ["system", "user", "assistant", "tool"] and seen["body"]["tools"][0]["function"]["parameters"]["type"] == "object"


def test_the_anthropic_adapter_speaks_tools_both_ways():
    seen = {}

    def handle(request):
        seen["headers"], seen["body"] = dict(request.headers), json.loads(request.content)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "looking"},
                                                     {"type": "tool_use", "id": "u1", "name": "stats", "input": {"checkpoint_id": REF}}]})

    provider = Anthropic("claude-x", api_key="key-secret", transport=httpx.MockTransport(handle))
    reply = provider.send([{"role": "user", "text": "go"},
                           {"role": "assistant", "text": "ok", "tool_calls": [call(1, "describe", checkpoint_id=REF), call(2, "graph", checkpoint_id=REF)]},
                           {"role": "tool", "tool_call_id": "c1", "name": "describe", "content": "{}", "is_error": False},
                           {"role": "tool", "tool_call_id": "c2", "name": "graph", "content": "{}", "is_error": True}],
                          agent_runner.agent_bridge.tool_list())
    assert reply.text == "looking" and reply.tool_calls == [{"id": "u1", "name": "stats", "arguments": {"checkpoint_id": REF}}]
    assert seen["headers"]["x-api-key"] == "key-secret" and seen["body"]["tools"][0]["input_schema"]["type"] == "object"
    last = seen["body"]["messages"][-1]
    assert last["role"] == "user" and [b["tool_use_id"] for b in last["content"]] == ["c1", "c2"] and last["content"][1]["is_error"] is True


def test_a_malformed_tool_call_is_a_tool_result_and_a_provider_error_keeps_the_report(live):
    client, creds = live()
    sent = []

    def handle(request):
        sent.append(json.loads(request.content))
        if len(sent) == 1:
            calls = [{"id": "a", "type": "function", "function": {"name": "describe", "arguments": "{not json"}},
                     {"id": "b", "type": "function", "function": {"name": "describe", "arguments": "[1, 2]"}},
                     {"id": "c", "type": "function", "function": {"name": "apply", "arguments": json.dumps({"checkpoint_id": REF, "key": "k", **PROGRAM})}}]
            return httpx.Response(200, json={"choices": [{"message": {"content": None, "tool_calls": calls}}]})
        return httpx.Response(429, text="rate limited: your prompt was: SECRET-PROMPT")

    provider = OpenAICompatible("http://127.0.0.1:8080/v1", "m", transport=httpx.MockTransport(handle))
    report = agent_runner.run(client, provider, "go")
    assert [c["code"] for c in report["calls"]] == ["invalid_arguments", "invalid_arguments", None]  # the good call still ran
    assert report["stop_reason"] == "provider_error" and report["provider_error"] == {"status": 429}
    assert len(report["children"]) == 1 and "SECRET-PROMPT" not in json.dumps(report)  # the child found before the failure is reported
    assert report["egress_class"] == 2 and report["provider_locality"] == "local"
    tool_messages = [m for m in sent[1]["messages"] if m["role"] == "tool"]
    assert "invalid_arguments" in tool_messages[0]["content"] and all("{not json" not in str(m) for m in sent[1]["messages"] if m["role"] == "assistant")
    down = OpenAICompatible("http://127.0.0.1:1/v1", "m", transport=httpx.MockTransport(lambda r: (_ for _ in ()).throw(httpx.ConnectError("no", request=r))))
    assert agent_runner.run(client, down, "go")["provider_error"] == {"status": None}


def test_local_means_loopback_and_never_uses_the_openai_key(monkeypatch, tmp_path):
    path = tmp_path / "connection.json"
    path.write_text(json.dumps({"url": "http://127.0.0.1:1", "project_id": "p", "token": "t", "egress_class": 1}), encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-leak")
    for url in ("https://api.example.com/v1", "http://192.168.1.5:8080/v1", "http://localhost.evil.com/v1"):
        with pytest.raises(SystemExit, match="loopback"):
            agent_runner.main(["--connection", str(path), "--provider", "local", "--base-url", url, "--model", "m", "--task", "t"])
    args = type("A", (), {"provider": "local", "base_url": "http://localhost:8080/v1", "model": "m"})()
    provider = agent_runner.make_provider(args)
    assert "authorization" not in provider.http.headers  # no key, and in particular not OPENAI_API_KEY
    monkeypatch.setenv("LAYERDELTA_LOCAL_API_KEY", "local-key")
    assert agent_runner.make_provider(args).http.headers["authorization"] == "Bearer local-key"


def test_keys_come_from_the_environment_only(monkeypatch, tmp_path):
    path = tmp_path / "connection.json"
    path.write_text(json.dumps({"url": "http://127.0.0.1:1", "project_id": "p", "token": "t", "egress_class": 1}), encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    for provider, env in (("anthropic", "ANTHROPIC_API_KEY"), ("openai", "OPENAI_API_KEY")):
        with pytest.raises(SystemExit, match=env):
            agent_runner.main(["--connection", str(path), "--provider", provider, "--model", "m", "--task", "t"])
    with pytest.raises(SystemExit, match="--base-url"):
        agent_runner.main(["--connection", str(path), "--provider", "local", "--model", "m", "--task", "t"])
