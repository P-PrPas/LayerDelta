"""What the session viewer reads (v0.4-S5): trace fields, summary, the sessions list, selection and recipe resolution."""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from layerdelta import api
from layerdelta.checkpoints import Catalog
from test_v2_api import BASE, BIAS, PROGRAM, REF, make_workspace  # noqa: F401

C = f"/checkpoints/{REF}"


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


def post(client, headers, path, body):
    return client.post(BASE + path, headers=headers, json=body)


def events_of(client, headers, **params):
    return client.get(BASE + "/trace", headers=headers, params=params).json()["result"]


def test_events_carry_the_registry_tool_via_and_duration(server):
    client, agent, human, creds = server()
    post(client, agent, C + "/stats", {"selector": BIAS})
    client.get(BASE + "/agent-session", headers={**agent, "X-LayerDelta-Via": "mcp"})
    client.get(BASE + "/agent-session", headers={**agent, "X-LayerDelta-Via": "<script>"})  # anything but "mcp" is http
    events = events_of(client, agent)["events"]
    assert [(e["tool"], e["via"]) for e in events][:3] == [("stats", "http"), ("session_info", "mcp"), ("session_info", "http")]
    assert all(isinstance(e["duration_ms"], int) and e["duration_ms"] >= 0 for e in events)
    summary = events_of(client, agent)["summary"]
    assert summary["via"] == {"http": 3, "mcp": 1} and summary["errors"] == 0 and summary["by_status"] == {"200": 4}  # the trace reads count too
    assert summary["last_status"] == 200 and summary["last_at"]


def test_summary_counts_errors_and_reports_budgets_with_the_live_limits(server):
    client, agent, human, creds = server(budgets={"values": 100, "scan": 1000})
    post(client, agent, C + "/read-tensor", {"selector": BIAS, "max_elems": 4, "include_range": False})
    post(client, agent, C + "/read-tensor", {"selector": {"tensor": "nope"}})
    summary = events_of(client, agent)["summary"]
    assert summary["errors"] == 1 and summary["by_status"] == {"200": 1, "422": 1}
    assert summary["budget"]["values"] == {"limit": 100, "used": 4, "remaining": 96}
    assert summary["budget"]["scan"] == {"limit": 1000, "used": 4, "remaining": 996}
    assert summary["budget"]["blob_bytes"] == {"limit": None, "used": 0, "remaining": None}
    assert summary["budget"]["applies"]["limit"] == 5
    # a person looking at the session sees the same numbers; at all sessions at once there is no single budget
    assert events_of(client, human, session=creds.principal)["summary"]["budget"]["values"]["used"] == 4
    assert "budget" not in events_of(client, human)["summary"]


def test_the_sessions_list_is_human_only_and_complete(server):
    client, agent, human, creds = server()
    post(client, agent, C + "/stats", {"selector": BIAS})
    client.get(BASE + "/checkpoints", headers=human)
    assert client.get(BASE + "/trace/sessions", headers=agent).status_code == 403
    body = client.get(BASE + "/trace/sessions", headers=human).json()["result"]
    rows = {r["session"]: r for r in body["sessions"]}
    assert body["current_agent_session"] == creds.principal and set(rows) == {creds.principal, "human"}
    mine = rows[creds.principal]
    assert (mine["kind"], mine["egress_class"], mine["events"], mine["errors"], mine["via_mix"]) == ("agent", 2, 2, 1, False)  # the stats call and the refused list
    assert mine["capabilities"] == ["model:read", "weights:read", "weights:values", "weights:write"] and mine["last_status"] == 403
    assert rows["human"]["kind"] == "human" and rows["human"]["events"] >= 1


def test_a_selection_id_in_the_trace_resolves_to_its_selector_for_a_human_only(server):
    client, agent, human, creds = server()
    index = {"0": {"start": 0, "stop": 5}}
    post(client, agent, C + "/stats", {"selector": {"tensor": "classifier.bias", "index": index}})
    post(client, agent, C + "/plan", PROGRAM)
    events = events_of(client, agent)["events"]
    ids = [e["selection_id"] for e in events if e["selection_id"]] + [s for e in events for s in e["selections"]]
    assert len(set(ids)) == 2
    assert all("classifier.bias" not in str(e) for e in events)  # the trace itself names no tensor
    assert post(client, agent, "/selections/resolve", {"ids": ids}).status_code == 403
    found = post(client, human, "/selections/resolve", {"ids": [*ids, "sel_" + "f" * 32]}).json()["result"]
    assert found["unknown"] == ["sel_" + "f" * 32]
    selectors = [r["selector"] for rows in found["selections"].values() for r in rows]
    assert {"tensor": "classifier.bias", "index": {"0": [[0, 5]]}} in selectors or any(s.get("tensor") == "classifier.bias" for s in selectors)
    assert all(r["checkpoint_id"] == REF for rows in found["selections"].values() for r in rows)
    assert post(client, human, "/selections/resolve", {"ids": ["sel_" + "0" * 32] * 501}).status_code == 422


def test_a_recipe_hash_in_the_trace_resolves_to_the_frozen_recipe_for_a_human_only(server):
    client, agent, human, creds = server()
    planned = post(client, agent, C + "/plan", PROGRAM).json()["result"]["recipe"]
    applied = post(client, {**agent, "Idempotency-Key": "k"}, C + "/apply", PROGRAM)
    assert applied.status_code == 200
    assert client.get(BASE + f"/recipes/by-hash/{planned['recipe_hash']}", headers=agent).status_code == 403
    got = client.get(BASE + f"/recipes/by-hash/{planned['recipe_hash']}", headers=human).json()["result"]
    assert got["recipe"] == planned and got["recipe"]["hypothesis"] == "double the bias" and got["untrusted_text"] == ["recipe.hypothesis"]
    missing = client.get(BASE + "/recipes/by-hash/" + "0" * 64, headers=human)
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"


def test_a_catalog_from_before_v0_4_gains_the_new_event_columns(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    con = sqlite3.connect(workspace / "catalog.sqlite3")
    con.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT NOT NULL, seq INTEGER NOT NULL, at TEXT NOT NULL, "
                "route TEXT NOT NULL, tool TEXT NOT NULL, request_id TEXT NOT NULL, status INTEGER NOT NULL, error_code TEXT, "
                "checkpoint_id TEXT, other_checkpoint_id TEXT, selection_id TEXT, recipe_hash TEXT, child_id TEXT, run_id TEXT, "
                "idempotency_key TEXT, replayed INTEGER, charged INTEGER NOT NULL, capabilities TEXT NOT NULL, detail TEXT NOT NULL, "
                "UNIQUE(session, seq))")
    con.execute("INSERT INTO events(session,seq,at,route,tool,request_id,status,charged,capabilities,detail) "
                "VALUES ('old',1,'t','GET /x','x','r',200,0,'[]','{}')")
    con.commit()
    con.close()
    Catalog(workspace).start()
    con = sqlite3.connect(workspace / "catalog.sqlite3")
    assert {r[1] for r in con.execute("PRAGMA table_info(events)")} >= {"via", "duration_ms"}
    assert con.execute("SELECT via, duration_ms FROM events").fetchall() == [("http", None)]
    con.close()
