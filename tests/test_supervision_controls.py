"""What a person can do to a running agent session (v0.4-S5, acceptance T8): revoke, change a budget, approve or reject an apply."""

import pytest
from fastapi.testclient import TestClient

from layerdelta import api
from test_v2_api import BASE, BIAS, PROGRAM, REF, make_workspace  # noqa: F401

C = f"/checkpoints/{REF}"
OTHER = {"program": [{"op": "scale", "args": {"k": 3.0}, "select": BIAS}], "hypothesis": "triple the bias"}


@pytest.fixture
def server(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path, monkeypatch)
    opened = []

    def make(**kw):
        creds = api.Credentials("demo", weights_read=True, weights_write=True)
        client = TestClient(api.create_app(workspace, creds, **{"quota": 5, **kw}), base_url="http://127.0.0.1")
        client.__enter__()
        opened.append(client)
        csrf = client.post("/api/v1/session", json={"ticket": creds.ticket}).json()["csrf"]
        return client, {"Authorization": "Bearer " + creds.agent}, {"X-CSRF-Token": csrf}, creds

    make.workspace = workspace
    yield make
    for client in opened:
        client.__exit__(None, None, None)


def post(client, headers, path, body=None):
    return client.post(BASE + path, headers=headers, json={} if body is None else body)


def events(client, human, session):
    return client.get(BASE + "/trace", headers=human, params={"session": session, "limit": 500}).json()["result"]["events"]


def keyed(agent, key):
    return {**agent, "Idempotency-Key": key}


# ------------------------------------------------------------------ revoke

def test_revoke_stops_the_session_and_leaves_its_children(server):
    client, agent, human, creds = server()
    child = post(client, keyed(agent, "k"), C + "/apply", PROGRAM).json()["result"]["checkpoint"]["checkpoint_id"]
    assert post(client, agent, f"/agent-sessions/{creds.principal}/revoke").status_code == 403  # an agent cannot revoke itself or anyone
    assert post(client, human, "/agent-sessions/human/revoke").status_code == 422
    assert post(client, human, "/agent-sessions/agent-nobody/revoke").status_code == 404
    assert post(client, human, f"/agent-sessions/{creds.principal}/revoke").json()["result"] == {"session": creds.principal, "revoked": True}
    for method, path in (("GET", "/agent-session"), ("GET", C + "/describe"), ("POST", C + "/stats")):
        response = client.request(method, BASE + path, headers=agent, json={"selector": BIAS} if method == "POST" else None)
        assert response.status_code == 401 and response.json()["error"]["code"] == "session_revoked", path
    assert post(client, human, f"/agent-sessions/{creds.principal}/revoke").status_code == 409  # already
    assert client.get(BASE + f"/checkpoints/{child}", headers=human).status_code == 200  # children stay in the catalog
    assert post(client, human, C + "/stats", {"selector": BIAS}).status_code == 200  # the human session is untouched
    log = events(client, human, creds.principal)
    control = [e for e in log if e["control"]]
    assert [(e["tool"], e["status"], e["control"]) for e in control] == [("revoke_session", 200, {"principal": "human", "action": "revoke"})]
    assert any(e["status"] == 401 for e in log) and log.index(control[0]) < next(i for i, e in enumerate(log) if e["status"] == 401)
    row = next(r for r in client.get(BASE + "/trace/sessions", headers=human).json()["result"]["sessions"] if r["session"] == creds.principal)
    assert row["revoked"] is True and row["revoked_at"]


# ------------------------------------------------------------------ budget change

def test_a_person_changes_a_budget_and_it_takes_effect_and_is_traced(server):
    client, agent, human, creds = server(budgets={"scan": 1000})
    stats = {"selector": BIAS}
    assert post(client, agent, C + "/stats", stats).status_code == 200  # 10 scanned
    assert post(client, human, f"/agent-sessions/{creds.principal}/budget", {"dimension": "scan", "limit": 15}).json()["result"] == {
        "session": creds.principal, "dimension": "scan", "from": 1000, "to": 15}
    refused = post(client, agent, C + "/stats", stats)  # lowered near use: 10 + 10 > 15
    assert refused.status_code == 429 and refused.json()["error"]["field"] == "scan"
    post(client, human, f"/agent-sessions/{creds.principal}/budget", {"dimension": "scan", "limit": None})  # no limit
    assert post(client, agent, C + "/stats", stats).status_code == 200
    post(client, human, f"/agent-sessions/{creds.principal}/budget", {"dimension": "values", "limit": 0})
    assert post(client, agent, C + "/read-tensor", {"selector": BIAS, "max_elems": 1, "include_range": False}).status_code == 429
    summary = client.get(BASE + "/trace", headers=agent).json()["result"]["summary"]["budget"]
    assert summary["scan"]["limit"] is None and summary["values"]["limit"] == 0  # the live limits are what the meters show
    controls = [e["control"] for e in events(client, human, creds.principal) if e["control"]]
    assert controls == [{"principal": "human", "action": "budget_change", "dimension": "scan", "from": 1000, "to": 15},
                        {"principal": "human", "action": "budget_change", "dimension": "scan", "from": 15, "to": None},
                        {"principal": "human", "action": "budget_change", "dimension": "values", "from": None, "to": 0}]


def test_the_apply_quota_is_a_budget_too_and_bad_changes_are_refused(server):
    client, agent, human, creds = server(quota=2)
    path = f"/agent-sessions/{creds.principal}/budget"
    assert post(client, human, path, {"dimension": "applies", "limit": 1}).json()["result"]["from"] == 2
    assert post(client, keyed(agent, "a"), C + "/apply", PROGRAM).status_code == 200
    assert post(client, keyed(agent, "b"), C + "/apply", OTHER).json()["error"]["code"] == "quota_exhausted"
    assert client.get(BASE + "/agent-session", headers=agent).json()["budgets"]["applies"] == {"limit": 1, "used": 1, "remaining": 0}
    for body in ({"dimension": "applies", "limit": None}, {"dimension": "nope", "limit": 1}, {"dimension": "scan", "limit": -1},
                 {"dimension": "scan"}, {"dimension": "scan", "limit": 1, "extra": 1}):
        assert post(client, human, path, body).status_code == 422, body
    assert post(client, agent, path, {"dimension": "scan", "limit": 10 ** 9}).status_code == 403  # the agent cannot raise its own budget


# ------------------------------------------------------------------ approval gate

def test_with_the_gate_on_an_agent_apply_waits_for_a_person(server):
    client, agent, human, creds = server(apply_requires_approval=True)
    first = post(client, keyed(agent, "p1"), C + "/apply", PROGRAM)
    assert first.status_code == 202
    proposal = first.json()["result"]["proposal_id"]
    assert first.json()["result"]["status"] == "pending" and first.json()["side_effects"] == [{"kind": "proposal_created", "id": proposal}]
    assert [c["origin"] for c in client.get(BASE + "/checkpoints", headers=human).json()["items"]] == ["fixture"]  # nothing was written
    assert client.get(BASE + "/agent-session", headers=agent).json()["budgets"]["applies"]["used"] == 0  # and nothing charged
    again = post(client, keyed(agent, "p1"), C + "/apply", PROGRAM)  # the same key is the same proposal
    assert again.status_code == 202 and again.json()["result"]["proposal_id"] == proposal and again.json()["side_effects"] == []
    assert post(client, keyed(agent, "p1"), C + "/apply", OTHER).json()["error"]["code"] == "idempotency_conflict"
    bad = post(client, keyed(agent, "p2"), C + "/apply", {"program": [{"op": "scale", "args": {"k": 1.0}, "select": BIAS}], "hypothesis": "x"})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "no_op"  # invalid requests are errors, not proposals
    assert client.get(BASE + "/proposals", headers=agent).status_code == 403
    pending = client.get(BASE + "/proposals", params={"status": "pending"}, headers=human).json()["result"]
    assert pending["approval_gate"] is True and len(pending["proposals"]) == 1
    row = pending["proposals"][0]
    assert (row["proposal_id"], row["session"], row["checkpoint_id"], row["status"]) == (proposal, creds.principal, REF, "pending")
    assert row["seq"] >= 1 and len(row["selections"]) == 1 and len(row["recipe_hash"]) == 64
    for path in (f"/proposals/{proposal}/approve", f"/proposals/{proposal}/reject"):
        assert post(client, agent, path).status_code == 403
    # a person's own apply is not gated
    assert post(client, keyed(human, "h1"), C + "/apply", OTHER).status_code == 200


def test_approving_runs_the_same_apply_and_the_agent_sees_its_result(server):
    client, agent, human, creds = server(apply_requires_approval=True)
    proposal = post(client, keyed(agent, "p1"), C + "/apply", PROGRAM).json()["result"]["proposal_id"]
    approved = post(client, human, f"/proposals/{proposal}/approve")
    assert approved.status_code == 200 and approved.json()["result"]["proposal"] == {"proposal_id": proposal, "status": "approved"}
    child = approved.json()["result"]["manifest"]["child_sha256"]
    assert client.get(BASE + "/agent-session", headers=agent).json()["budgets"]["applies"]["used"] == 1  # charged once, to the agent
    assert [s["kind"] for s in approved.json()["side_effects"]] == ["checkpoint_created", "quota_charged"]
    direct = post(client, keyed(human, "d"), C + "/apply", PROGRAM).json()["result"]  # the same program applied directly: the same child
    assert direct["manifest"]["child_sha256"] == child and direct["existing"] is True
    replay = post(client, keyed(agent, "p1"), C + "/apply", PROGRAM)  # the agent retries its key after the approval
    assert replay.status_code == 200 and replay.json()["result"]["replayed"] is True and replay.json()["result"]["manifest"]["child_sha256"] == child
    assert post(client, human, f"/proposals/{proposal}/approve").json()["error"]["code"] == "not_pending"
    done = client.get(BASE + "/proposals", params={"session": creds.principal, "status": "approved"}, headers=human).json()["result"]["proposals"]
    assert [(p["proposal_id"], p["child_id"]) for p in done] == [(proposal, child)]
    controls = [e["control"] for e in events(client, human, creds.principal) if e["control"]]
    assert controls == [{"principal": "human", "action": "approve", "proposal_id": proposal, "child_id": child}]
    applies = [e for e in events(client, human, creds.principal) if e["tool"] == "apply"]
    assert [e["status"] for e in applies] == [202, 200]  # the proposal, then the agent's replay after the approval
    assert next(e for e in applies if e["status"] == 202)["proposal_id"] == proposal


def test_rejecting_closes_the_proposal_and_the_agent_is_told(server):
    client, agent, human, creds = server(apply_requires_approval=True)
    proposal = post(client, keyed(agent, "r1"), C + "/apply", OTHER).json()["result"]["proposal_id"]
    assert post(client, human, f"/proposals/{proposal}/reject").json()["result"] == {"proposal_id": proposal, "status": "rejected"}
    retry = post(client, keyed(agent, "r1"), C + "/apply", OTHER)
    assert retry.status_code == 409 and retry.json()["error"]["code"] == "proposal_rejected"
    assert post(client, human, f"/proposals/{proposal}/reject").json()["error"]["code"] == "not_pending"
    assert post(client, human, "/proposals/p_" + "0" * 16 + "/approve").status_code == 404
    assert [c["origin"] for c in client.get(BASE + "/checkpoints", headers=human).json()["items"]] == ["fixture"]
    assert [e["control"]["action"] for e in events(client, human, creds.principal) if e["control"]] == ["reject"]
    assert client.get(BASE + "/agent-session", headers=agent).json()["budgets"]["applies"]["used"] == 0


def test_a_gated_apply_still_pays_for_what_it_read(server):
    client, agent, human, creds = server(apply_requires_approval=True, budgets={"scan": 12})
    assert post(client, keyed(agent, "p1"), C + "/apply", PROGRAM).status_code == 202
    assert client.get(BASE + "/agent-session", headers=agent).json()["budgets"]["scan"]["used"] > 0  # planning reads are charged, no free oracle
