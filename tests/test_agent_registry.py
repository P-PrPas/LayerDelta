"""The route registry and the tool snapshot (v0.4-S1, acceptance T2): the agent surface is generated from one table."""

import os
import re

import pytest
from fastapi.testclient import TestClient

from layerdelta import api, registry
from layerdelta.client import AgentClient, ClientError
from test_v2_api import BASE, REF, make_workspace  # noqa: F401

PREFIX = "/api/v1/projects/demo"


@pytest.fixture
def app(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path, monkeypatch)
    creds = api.Credentials("demo", weights_read=True, weights_write=True)
    return api.create_app(workspace, creds), creds


def test_the_snapshot_is_current(app):
    generated = registry.generate(app[0])  # also fails if the app has a route the registry does not know, or the reverse
    text = registry.snapshot_text(generated)
    if os.environ.get("LAYERDELTA_WRITE_SNAPSHOT"):
        registry.SNAPSHOT.write_text(text, encoding="utf-8")
    assert registry.SNAPSHOT.read_text(encoding="utf-8").replace("\r\n", "\n") == text, (
        "agent_tools.json is stale: bump registry.VERSION and rewrite it with LAYERDELTA_WRITE_SNAPSHOT=1")
    assert generated["version"] == registry.VERSION


def test_tools_are_the_registry_filtered_to_agent_routes():
    tools = registry.load()["tools"]
    names = [t["name"] for t in tools]
    assert len(names) == len(set(names))
    expected = [r.tool for r in registry.ROUTES if r.exposure in ("tool", "legacy")]
    assert names == expected
    human = {r.tool for r in registry.ROUTES if r.exposure == "human_only"}
    assert not human & set(names) and all(r.tool is None for r in registry.ROUTES if r.exposure in ("human_only", "not_exposed"))
    assert {t["name"] for t in tools if t["legacy"]} == {"propose", "run", "status", "compare", "changes", "cancel"}
    assert {"session_info", "describe", "read_tensor", "plan", "validate", "apply", "upload_blob", "trace"} <= set(names)


def test_every_schema_is_closed_and_annotations_follow_the_registry():
    for tool in registry.load()["tools"]:
        schema, route = tool["inputSchema"], registry.by_tool()[tool["name"]]
        assert schema["type"] == "object" and schema["additionalProperties"] is False, tool["name"]
        assert set(schema["required"]) <= set(schema["properties"]), tool["name"]
        assert 0 < len(tool["description"]) <= 600
        assert tool["annotations"]["readOnlyHint"] is route.pure and tool["annotations"]["destructiveHint"] is False
        assert ("key" in schema["properties"]) is route.key and (not route.key or "key" in schema["required"])
        for ref in re.findall(r'"\$ref": "([^"]+)"', str(schema).replace("'", '"')):
            assert ref.startswith("#/$defs/") and ref.rsplit("/", 1)[-1] in schema["$defs"], (tool["name"], ref)
    describe = registry.tools_by_name()["describe"]["inputSchema"]
    assert describe["required"] == ["checkpoint_id"] and "project_id" not in describe["properties"] and "limit" in describe["properties"]
    assert registry.tools_by_name()["apply"]["annotations"]["idempotentHint"] is True


def test_human_only_routes_refuse_an_agent_token(app):
    application, creds = app
    agent = {"Authorization": "Bearer " + creds.agent}
    with TestClient(application, base_url="http://127.0.0.1") as client:
        for route in (r for r in registry.ROUTES if r.exposure == "human_only"):
            path = PREFIX + re.sub(r"\{[a-z_]+\}", REF, route.path)
            response = client.request(route.method, path, headers=agent, json={} if route.method == "POST" else None)
            assert response.status_code == 403, (route.method, route.path, response.status_code)


def test_call_refuses_malformed_calls_before_sending_anything():
    client = AgentClient("http://127.0.0.1:9", "demo", "token")  # nothing listens: a request would raise a transport error
    for tool, args, field in (("describe", {}, "checkpoint_id"), ("describe", {"checkpoint_id": REF, "bogus": 1}, "bogus"),
                              ("apply", {"checkpoint_id": REF}, "key"), ("upload_blob", {"data_base64": "!!"}, "data_base64"),
                              ("upload_blob", {"data_base64": ""}, "data_base64")):
        with pytest.raises(ClientError) as info:
            client.call(tool, **args)
        assert info.value.code == "invalid_argument" and info.value.field == field, (tool, args)
    with pytest.raises(ClientError) as info:
        client.call("no_such_tool")
    assert info.value.code == "unknown_tool"
