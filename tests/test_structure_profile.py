"""The structure profile (v0.5-E6, acceptance V8): a session without weights:read can plan, validate, apply and diff, and nothing it is
told depends on what the weights hold, except the residual bits named in docs/reference/egress.md.

The route walk is two-sided: every route outside the profile is refused ON THE SERVER, and every route inside it answers with exactly the
allow-listed fields (a field added to a response later fails here until someone lists it).
"""

import json
import re

import pytest
from fastapi.testclient import TestClient

from layerdelta import agent_bridge, api, registry, structure
from layerdelta.client import AgentClient
from test_v2_api import BASE, BIAS, REF, make_workspace  # noqa: F401

C = f"/checkpoints/{REF}"
SCALE = lambda k: {"program": [{"op": "scale", "args": {"k": k}, "select": BIAS}], "hypothesis": "test"}  # noqa: E731
LEAKY = {"changed_elems", "touched_elems", "touched_elems_total", "changed_elems_total", "before_sha", "after_sha", "noop", "max_abs",
         "changed_units", "touched", "mean", "std", "l2", "min", "max", "histogram", "values"}


@pytest.fixture
def server(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path, monkeypatch)
    opened = []

    def make(profile="structure", **flags):
        caps = {"weights_write": True} if profile == "structure" else {"weights_read": True, "weights_write": True, "weights_values": False}
        creds = api.Credentials("demo", profile=profile, **caps)
        client = TestClient(api.create_app(workspace, creds, quota=500, **flags), base_url="http://127.0.0.1")
        client.__enter__()
        opened.append(client)
        csrf = client.post("/api/v1/session", json={"ticket": creds.ticket}).json()["csrf"]
        return client, {"Authorization": "Bearer " + creds.agent}, {"X-CSRF-Token": csrf}, creds

    yield make
    for client in opened:
        client.__exit__(None, None, None)


def keys(value):
    """Every key anywhere in a JSON value."""
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in keys(v)}
    return set()


def test_the_profile_validates_its_credentials():
    assert api.Credentials("p", weights_write=True, profile="structure").capabilities == {"model:read", "weights:write"}
    assert api.Credentials("p", weights_write=True, profile="structure").egress_class == 0
    with pytest.raises(ValueError, match="no weights:read"):
        api.Credentials("p", weights_read=True, profile="structure")
    with pytest.raises(ValueError, match="weights_write requires weights_read"):
        api.Credentials("p", weights_write=True)  # outside the profile the v0.4 rule stands
    with pytest.raises(ValueError, match="profile must be"):
        api.Credentials("p", profile="admin")


def test_route_walk_everything_outside_the_profile_is_refused_by_the_server(server):
    client, agent, human, creds = server()
    inside = registry.profile_routes("structure")
    assert {r.tool for r in registry.ROUTES if (r.method, registry.BASE + r.path) in inside} == registry.profile_tools("structure")
    refused = checked = 0
    for route in registry.ROUTES:
        path = re.sub(r"\{(\w+)\}", lambda m: REF if m.group(1) == "checkpoint_id" else "x", BASE + route.path)
        if (route.method, registry.BASE + route.path) in inside:
            checked += 1
            continue
        answer = client.request(route.method, path, headers=agent, json={} if route.method == "POST" else None)
        assert answer.status_code == 403, (route.method, route.path, answer.status_code)
        assert answer.json()["error"]["code"] == "forbidden"
        refused += 1
    assert checked >= 12 and refused >= 20
    for tool in ("stats", "query", "read_tensor", "run", "compare", "propose"):  # absent from the profile by name
        assert tool not in registry.profile_tools("structure")


def test_responses_inside_the_profile_have_exactly_the_allow_listed_fields(server):
    client, agent, human, creds = server()
    ask = lambda path, body=None: (client.post(BASE + path, headers=agent, json=body) if body is not None  # noqa: E731
                                   else client.get(BASE + path, headers=agent))
    planned = ask(C + "/plan", SCALE(2.0))
    assert planned.status_code == 200
    body = planned.json()
    assert set(body["result"]) <= set(structure.PLAN) and not keys(body) & LEAKY
    assert set(body["result"]["tensors"][0]) == {"tensor", "dtype", "ops"}
    assert set(body["result"]["ops"][0]) <= set(structure.OP)

    valid = ask(C + "/validate", SCALE(2.0)).json()
    assert set(valid["result"]) <= set(structure.VALIDATE) and valid["result"]["valid"] is True and not keys(valid) & LEAKY

    applied = client.post(BASE + C + "/apply", headers={**agent, "Idempotency-Key": "k1"}, json=SCALE(2.0))
    assert applied.status_code == 200, applied.text
    done = applied.json()
    assert set(done["result"]) <= set(structure.APPLY) and not keys(done) & LEAKY
    assert set(done["result"]["manifest"]) == set(structure.MANIFEST)
    child = done["result"]["manifest"]["child_sha256"]

    again = client.post(BASE + C + "/apply", headers={**agent, "Idempotency-Key": "k1"}, json=SCALE(2.0)).json()  # the ledger replay
    assert again["result"]["replayed"] is True and not keys(again) & LEAKY and set(again["result"]) <= set(structure.APPLY)

    whole = ask(C + "/diff", {"other": child})
    assert whole.status_code == 200
    diff = whole.json()
    assert set(diff["result"]) <= set(structure.DIFF) and not keys(diff) & LEAKY
    assert all(set(row) <= set(structure.DIFF_ROW) for row in diff["result"]["tensors"])
    assert {row["name"]: row["status"] for row in diff["result"]["tensors"]} == {"classifier.bias": "changed"}

    for body in ({"other": child, "selector": BIAS}, {"other": child, "include_range": True}):
        assert ask(C + "/diff", body).status_code == 403  # selectors and ranges say WHICH units moved and by how much
    assert ask(C + "/describe").status_code == 200 and ask("/trace").status_code == 200 and ask("/agent-session").json()["profile"] == "structure"


def test_the_noop_refusal_is_not_an_oracle(server):
    client, agent, human, creds = server()
    scale_one = client.post(BASE + C + "/apply", headers={**agent, "Idempotency-Key": "noop"}, json=SCALE(1.0))
    assert scale_one.status_code == 200, scale_one.text  # no `no_op` for a session that cannot read weights
    child = scale_one.json()["result"]["manifest"]["child_sha256"]  # the residual bit: the child has the parent's bytes
    same = client.post(BASE + C + "/diff", headers=agent, json={"other": child}).json()["result"]
    assert same["summary"]["changed"] == 0 and same["tensors"] == []


def test_outside_the_profile_no_op_is_still_refused(server):
    client, agent, human, creds = server(profile="tools")
    refused = client.post(BASE + C + "/apply", headers={**agent, "Idempotency-Key": "noop"}, json=SCALE(1.0))
    assert refused.status_code == 422 and refused.json()["error"]["code"] == "no_op"  # the v0.4 behaviour is unchanged elsewhere


def test_the_allow_noop_flag_makes_both_benchmark_arms_behave_alike(server):
    client, agent, human, creds = server(profile="tools", allow_noop=True)
    allowed = client.post(BASE + C + "/apply", headers={**agent, "Idempotency-Key": "noop"}, json=SCALE(1.0))
    assert allowed.status_code == 200 and allowed.json()["result"]["noop"] is True  # a session that CAN read weights still sees the field


def test_the_tools_profile_has_no_legacy_routes_and_keeps_every_tool(server):
    client, agent, human, creds = server(profile="tools")
    assert client.post(BASE + "/runs", headers=agent, json={}).status_code == 403
    assert client.get(BASE + "/catalog", headers=agent).status_code == 403
    assert client.post(BASE + C + "/stats", headers=agent, json={"selector": BIAS}).status_code == 200
    assert registry.profile_tools("tools") == {r.tool for r in registry.ROUTES if r.exposure == "tool"}
    assert registry.profile_tools(None) is None
    with pytest.raises(ValueError):
        registry.profile_routes("root")


def test_the_bridge_and_runner_offer_only_the_profile_and_refuse_the_rest(server):
    names = lambda profile: {t["name"] for t in agent_bridge.tool_list(False, profile)}  # noqa: E731
    assert names(None) >= names("tools") > names("structure")
    assert {"plan", "validate", "apply", "diff", "describe", "graph", "groups", "trace"} <= names("structure")
    assert not {"stats", "query", "read_tensor", "run", "compare"} & names("structure")
    client, agent, human, creds = server()
    transport = None
    with AgentClient("http://127.0.0.1:1", "demo", creds.agent, transport=transport, profile="structure") as local:
        body, is_error, _ = agent_bridge.execute(local, "stats", {"checkpoint_id": REF, "selector": BIAS}, profile="structure")
        assert is_error and body["error"]["code"] == "unknown_tool"  # absent from the list, so refused before any request
    connection = {"url": "http://127.0.0.1:1", "project_id": "demo", "token": creds.agent, "egress_class": 0, "profile": "structure"}
    with AgentClient(**connection) as accepted:  # a connection.json with a profile still builds a client
        assert accepted.profile == "structure"
    assert json.loads(json.dumps(connection))["profile"] == "structure"


# The structure routes that are not edits: each answers exactly these top-level fields, so a field added later fails here first (V8).
ENVELOPE = {"cursor", "fingerprints", "resolved", "result", "schema_version", "side_effects", "truncated"}
EXACT = {
    "describe": {"checkpoint", "coverage", "dtypes", "graph", "identical_bytes", "matched", "offset", "tensors"},  # graph: when one is attached
    "describe_row": {"binding", "dtype", "editable", "groups", "identical_bytes_group", "name", "nbytes", "numel", "read_only_reason",
                     "roles", "shape", "shard", "tensor_sha256"},
    "graph": {"counts", "coverage", "graph", "inputs", "matched", "nodes", "offset", "outputs", "view", "warnings"},
    "graph_node": {"graph", "node"},
    "groups": {"graph", "groups", "matched", "offset"},
    "group": {"boundary", "closed", "constraints", "group_id", "kind", "label", "members", "notes", "size", "spans", "spans_total",
              "spans_truncated"},
    "checkpoint": {"adapter_compatible", "checkpoint_id", "children", "display_name", "element_count", "file_bytes", "format", "imported_at",
                   "origin", "parent_id", "recipe_hash", "shard_count", "sharded", "tensor_count"},
    "session": {"budgets", "capabilities", "charged", "egress_class", "limit", "profile", "queue_limit", "remaining", "timeout_seconds"},
    "trace": {"events", "next_after", "session", "summary"},
    "blob": {"blob_id", "bytes", "elems", "existing"},
}


TRACE_EVENT = {"at", "budget_charged", "capabilities", "charged", "checkpoint_id", "child_id", "control", "duration_ms", "error_code",
               "event_id", "existing", "idempotency_key", "links", "ops", "other_checkpoint_id", "proposal_id", "recipe_hash", "replay_of",
               "replayed", "request_id", "route", "run_id", "selection_id", "selections", "seq", "session", "status", "tool", "via"}


def attach_graph(client, human):
    import time
    started = client.post(BASE + C + "/graphs", headers=human, json={"provider": "builtin", "model": "reference_cnn"})
    for _ in range(300):
        done = client.get(f"{BASE}/graph-attachments/{started.json()['attach_id']}", headers=human).json()
        if done["status"] in ("succeeded", "failed"):
            break
        time.sleep(.1)
    assert done["status"] == "succeeded", done


def test_the_other_structure_routes_have_exactly_their_fields_and_no_hashes(server):
    client, agent, human, creds = server()
    attach_graph(client, human)
    get = lambda path, **q: client.get(BASE + path, headers=agent, params=q)  # noqa: E731
    seen = []

    def check(name, response, result_keys=None):  # the trace's budget summary has a `values` budget (a count of elements read, not values)
        assert response.status_code == 200, (name, response.text)
        body = response.json()
        seen.append(body)
        if result_keys is not None:
            assert set(body) == ENVELOPE and set(body["result"]) == EXACT[name], name
        assert not keys(body) & (LEAKY - {"values"} if name == "trace" else LEAKY), name
        return body

    described = check("describe", get(C + "/describe"), True)["result"]
    assert described["identical_bytes"] is None  # a hash or an identical-bytes tie pins content: never filled for a blind session
    assert all(set(row) == EXACT["describe_row"] and row["tensor_sha256"] is None and row["identical_bytes_group"] is None
               for row in described["tensors"])
    graph = check("graph", get(C + "/graph"), True)["result"]
    check("graph_node", get(C + f"/graph/nodes/{graph['nodes'][0]['id']}"), True)
    groups = check("groups", get(C + "/groups", limit=50), True)["result"]["groups"]
    assert groups and all(set(group) == EXACT["group"] for group in groups)
    assert set(get(C).json()) == EXACT["checkpoint"]
    assert all(set(item) == EXACT["checkpoint"] - {"children"} for item in get("/checkpoints").json()["items"])
    assert set(get("/agent-session").json()) == EXACT["session"] and get("/agent-session").json()["profile"] == "structure"
    applied = client.post(BASE + C + "/apply", headers={**agent, "Idempotency-Key": "t"}, json=SCALE(2.0))
    assert applied.status_code == 200
    traced = check("trace", get("/trace"), True)["result"]
    assert traced["events"] and all(set(event) <= TRACE_EVENT for event in traced["events"]), {k for e in traced["events"] for k in e} - TRACE_EVENT
    blob = client.post(BASE + "/blobs", headers=agent, content=bytes(4))
    assert blob.status_code == 200 or blob.status_code == 201
    assert set(blob.json()["result"]) == EXACT["blob"]


def test_an_approval_gated_apply_of_a_blind_session_answers_the_structure_view(server):
    client, agent, human, creds = server(apply_requires_approval=True)
    first = client.post(BASE + C + "/apply", headers={**agent, "Idempotency-Key": "gate"}, json=SCALE(1.0))
    assert first.status_code == 202 and not keys(first.json()) & LEAKY, first.text  # parked: a no-op program is not refused here either
    pending = client.get(BASE + "/proposals", headers=human).json()
    proposal = (pending.get("result") or pending)["proposals"][0]["proposal_id"] if "proposals" in (pending.get("result") or pending) else None
    assert proposal, pending
    approved = client.post(BASE + f"/proposals/{proposal}/approve", headers=human, json={})
    assert approved.status_code == 200, approved.text
    retried = client.post(BASE + C + "/apply", headers={**agent, "Idempotency-Key": "gate"}, json=SCALE(1.0))
    assert retried.status_code == 200 and "noop" not in keys(retried.json()) and not keys(retried.json()) & LEAKY
    assert set(retried.json()["result"]) <= set(structure.APPLY)


# Every tool of the profile has its exact fields checked in one of these tests. A new structure route fails here until it is listed AND tested.
WALKED = {"session_info", "list_checkpoints", "get_checkpoint", "describe", "graph", "graph_node", "groups", "diff", "plan", "validate", "apply",
          "upload_blob", "trace",                                    # test_the_other_structure_routes... and the edit tests above
          "evaluate", "get_evaluation", "evaluation_changes", "list_datasets", "get_dataset"}  # test_evaluate.py: the structure profile test


def test_every_structure_tool_has_its_fields_walked():
    assert registry.profile_tools("structure") == WALKED
