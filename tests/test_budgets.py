"""Per-session budgets (v0.4-S3, acceptance T4): metered and refused in the backend, for agents only."""

import pytest
from fastapi.testclient import TestClient

from layerdelta import api, budget
from test_v2_api import BASE, BIAS, PROGRAM, REF, make_workspace  # noqa: F401

C = f"/checkpoints/{REF}"


@pytest.fixture
def server(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path, monkeypatch)
    opened = []

    def make(**limits):
        creds = api.Credentials("demo", weights_read=True, weights_write=True)
        applies = limits.pop("applies", 5)
        client = TestClient(api.create_app(workspace, creds, quota=applies, budgets=limits), base_url="http://127.0.0.1")
        client.__enter__()
        opened.append(client)
        csrf = client.post("/api/v1/session", json={"ticket": creds.ticket}).json()["csrf"]
        return client, {"Authorization": "Bearer " + creds.agent}, {"X-CSRF-Token": csrf}

    yield make
    for client in opened:
        client.__exit__(None, None, None)


def used(client, agent):
    return {k: v["used"] for k, v in client.get(BASE + "/agent-session", headers=agent).json()["budgets"].items()}


def post(client, headers, path, body):
    return client.post(BASE + path, headers=headers, json=body)


def test_flag_values_parse_and_bad_ones_are_refused():
    assert budget.parse("values=1000,scan=1e6, applies=3") == {"values": 1000, "scan": 1000000, "applies": 3}
    assert budget.parse("") == {}
    for bad in ("nope=1", "values", "values=x", "values=-1", "values=1,values=2"):
        with pytest.raises(ValueError):
            budget.parse(bad)


def test_session_info_reports_every_dimension_and_the_header_agrees(server):
    client, agent, human = server(values=100, scan=1000)
    info = client.get(BASE + "/agent-session", headers=agent)
    budgets = info.json()["budgets"]
    assert budgets["values"] == {"limit": 100, "used": 0, "remaining": 100} and budgets["blob_bytes"]["limit"] is None
    assert budgets["applies"] == {"limit": 5, "used": 0, "remaining": 5} and info.json()["limit"] == 5  # the quota keys are unchanged
    response = post(client, agent, C + "/read-tensor", {"selector": BIAS, "max_elems": 4, "include_range": False})
    assert response.status_code == 200
    assert response.headers["LayerDelta-Budget"] == "values=4/100;scan=4/1000;applies=0/5;blob_bytes=0/-"
    assert used(client, agent) == {"values": 4, "scan": 4, "blob_bytes": 0, "applies": 0}
    assert "LayerDelta-Budget" not in client.get(BASE + "/checkpoints", headers=human).headers  # a person is not metered
    assert set(response.json()) == {"schema_version", "resolved", "result", "fingerprints", "side_effects", "truncated", "cursor"}


def test_values_are_refused_before_reading_and_the_failed_call_charges_nothing(server):
    client, agent, human = server(values=10)
    for _ in range(2):
        assert post(client, agent, C + "/read-tensor", {"selector": BIAS, "max_elems": 5}).status_code == 200
    refused = post(client, agent, C + "/read-tensor", {"selector": BIAS, "max_elems": 5})
    assert refused.status_code == 429
    assert refused.json()["error"]["code"] == "budget_exhausted" and refused.json()["error"]["field"] == "values"
    assert used(client, agent)["values"] == 10 and refused.headers["LayerDelta-Budget"].startswith("values=10/10")
    # a person reading the same tensor is not limited by the agent's budget
    assert post(client, human, C + "/read-tensor", {"selector": BIAS, "max_elems": 10}).status_code == 200
    assert used(client, agent)["values"] == 10


def test_scan_counts_what_is_read_and_stops_at_the_limit(server):
    client, agent, human = server(scan=25)
    assert post(client, agent, C + "/stats", {"selector": BIAS}).status_code == 200      # 10 elements
    assert post(client, agent, C + "/stats", {"selector": BIAS}).status_code == 200      # 20
    third = post(client, agent, C + "/stats", {"selector": BIAS})                         # would be 30
    assert third.status_code == 429 and third.json()["error"]["field"] == "scan"
    assert used(client, agent)["scan"] == 20  # the refused call read nothing and was not charged
    planned = post(client, agent, C + "/plan", PROGRAM)  # a plan reads the selection and hashes the tensor: more than the 5 left
    assert planned.status_code == 429 and planned.json()["error"]["field"] == "scan" and used(client, agent)["scan"] == 20


def test_the_range_pass_of_a_read_is_a_scan_of_the_whole_selection(server):
    client, agent, human = server()
    post(client, agent, C + "/read-tensor", {"selector": BIAS, "max_elems": 3})  # include_range defaults on for the first page
    assert used(client, agent)["values"] == 3 and used(client, agent)["scan"] == 3 + 10


def test_apply_costs_an_apply_not_a_scan_and_a_replay_costs_nothing(server):
    client, agent, human = server(scan=1000)
    before = used(client, agent)
    keyed = {**agent, "Idempotency-Key": "k1"}
    first = post(client, keyed, C + "/apply", PROGRAM)
    assert first.status_code == 200
    after = used(client, agent)
    assert after["scan"] == before["scan"] and after["applies"] == 1 and first.headers["LayerDelta-Budget"].endswith("applies=1/5;blob_bytes=0/-")
    replay = post(client, keyed, C + "/apply", PROGRAM)
    assert replay.status_code == 200 and replay.json()["result"]["replayed"] is True and used(client, agent) == after


def test_blob_bytes_and_the_apply_budget(server):
    client, agent, human = server(blob_bytes=8, applies=1)
    assert client.post(BASE + "/blobs", headers=agent, content=bytes(8)).status_code == 201
    over = client.post(BASE + "/blobs", headers=agent, content=bytes(4))
    assert over.status_code == 429 and over.json()["error"]["field"] == "blob_bytes"
    assert post(client, {**agent, "Idempotency-Key": "a"}, C + "/apply", PROGRAM).status_code == 200
    second = post(client, {**agent, "Idempotency-Key": "b"}, C + "/apply",
                  {"program": [{"op": "scale", "args": {"k": 3.0}, "select": BIAS}], "hypothesis": "triple"})
    assert second.status_code == 429 and second.json()["error"]["code"] == "quota_exhausted"  # applies is the existing quota, unchanged


def test_every_charge_is_a_trace_field_and_an_agent_cannot_change_a_budget(server):
    client, agent, human = server(values=100, scan=100)
    post(client, agent, C + "/read-tensor", {"selector": BIAS, "max_elems": 3, "include_range": False})
    events = client.get(BASE + "/trace", headers=agent).json()["result"]["events"]
    read = next(e for e in events if e["tool"] == "read_tensor")
    assert read["budget_charged"] == {"values": 3, "scan": 3}
    # no route and no argument raises a budget: the only inputs are the flags
    for path in ("/agent-session", "/budget"):
        assert client.post(BASE + path, headers=agent, json={"values": 10 ** 9}).status_code in (404, 405)
