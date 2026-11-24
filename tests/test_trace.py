"""Session trace: one id-only event per request, links read -> plan -> apply -> child, no values, restart-safe."""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from layerdelta import api, trace
from layerdelta.checkpoints import Catalog
from layerdelta.model import ReferenceCNN, describe_model

BASE = "/api/v1/projects/demo"
REF = "a" * 64
BIAS = {"tensor": "classifier.bias"}
PROGRAM = {"program": [{"op": "scale", "args": {"k": 2.0}, "select": BIAS}], "hypothesis": "double the bias"}


def make_workspace(tmp_path, monkeypatch):
    torch.manual_seed(2)
    workspace = tmp_path / "ws"
    (workspace / "reference").mkdir(parents=True)
    save_file(ReferenceCNN().state_dict(), workspace / "reference" / "model.safetensors")
    fixture = {"artifacts": {"model.safetensors": REF, "dataset.json": "b" * 64, "exploration.json": "c" * 64},
               "manifest_sha256": "d" * 64, "training": {"seed": 1}, "model": describe_model(ReferenceCNN())}
    monkeypatch.setattr(api, "load_fixture", lambda _: (fixture, {"dataset_revision_id": "v", "split_sha256": {}, "samples": []}, {}))
    return workspace


class Server:
    def __init__(self, workspace, **caps):
        self.creds = api.Credentials("demo", **caps)
        self.client = TestClient(api.create_app(workspace, self.creds, quota=3), base_url="http://127.0.0.1")
        self.client.__enter__()
        self.ticket = self.creds.ticket  # the server blanks it after the exchange
        csrf = self.client.post("/api/v1/session", json={"ticket": self.creds.ticket}).json()["csrf"]
        self.agent = {"Authorization": "Bearer " + self.creds.agent}
        self.human = {"X-CSRF-Token": csrf}
        self.principal = self.creds.principal

    def post(self, path, body=None, headers=None, **kw):
        return self.client.post(BASE + path, json={} if body is None else body, headers=headers or self.agent, **kw)

    def get(self, path, headers=None, **kw):
        return self.client.get(BASE + path, headers=headers or self.agent, **kw)

    def timeline(self, headers=None, **params):
        return self.get("/trace", headers=headers, params=params).json()

    def close(self):
        self.client.__exit__(None, None, None)


@pytest.fixture
def server(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path, monkeypatch)
    started = []

    def start(**caps):
        s = Server(workspace, **{"weights_read": True, "weights_write": True, **caps})
        started.append(s)
        return s

    start.workspace = workspace
    yield start
    for s in started:
        s.close()


def test_apply_event_reaches_its_plan_reads_and_child_and_replays_are_not_double_counted(server):
    s = server()
    c = f"/checkpoints/{REF}"
    read = s.post(c + "/read-tensor", {"selector": BIAS}).json()
    plan = s.post(c + "/plan", PROGRAM).json()
    applied = s.post(c + "/apply", PROGRAM, headers={**s.agent, "Idempotency-Key": "once"}).json()
    again = s.post(c + "/apply", PROGRAM, headers={**s.agent, "Idempotency-Key": "once"}).json()
    child = applied["result"]["checkpoint"]["checkpoint_id"]
    assert again["result"]["replayed"] is True
    result = s.timeline()["result"]
    events = result["events"]
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1)) and {e["session"] for e in events} == {s.principal}
    by_tool = {e["tool"]: e for e in events if e["status"] == 200 and e["tool"] != "trace"}
    selection = read["fingerprints"]["selection"]
    assert by_tool["read_tensor"]["selection_id"] == selection and by_tool["read_tensor"]["checkpoint_id"] == REF
    assert by_tool["plan"]["recipe_hash"] == plan["result"]["recipe"]["recipe_hash"]
    first, second = [e for e in events if e["tool"] == "apply"]
    # from the apply event: its plan, its input selections and the child it made
    assert first["links"] == {"plan_seqs": [by_tool["plan"]["seq"]], "selection_seqs": [by_tool["read_tensor"]["seq"]],
                              "child_id": child}
    assert first["child_id"] == child and first["selections"] == [selection] and first["ops"] == 1
    assert (first["replayed"], first["charged"], first["replay_of"], first["idempotency_key"]) == (False, 1, None, trace.key_digest("once"))
    assert (second["replayed"], second["charged"], second["replay_of"]) == (True, 0, first["seq"])
    assert second["child_id"] == child and second["recipe_hash"] == first["recipe_hash"]
    assert result["summary"]["applies"] == 1 and result["summary"]["replays"] == 1
    assert result["summary"]["charged_runs"] == s.get("/agent-session").json()["charged"] == 1
    # each event holds the capability snapshot of its principal
    assert all(e["capabilities"] == ["model:read", "weights:read", "weights:values", "weights:write"] for e in events)


def test_concurrent_retries_link_to_one_apply(server):
    s = server()
    body = {"program": [{"op": "scale", "args": {"k": 3.0}, "select": BIAS}], "hypothesis": "race"}
    with ThreadPoolExecutor(5) as pool:
        list(pool.map(lambda _: s.post(f"/checkpoints/{REF}/apply", body, headers={**s.agent, "Idempotency-Key": "race"}), range(5)))
    result = s.timeline()["result"]
    applies = [e for e in result["events"] if e["tool"] == "apply"]
    assert len(applies) == 5 and result["summary"]["applies"] == 1 and result["summary"]["replays"] == 4
    originals = [e for e in applies if e["replay_of"] is None]
    assert len(originals) == 1 and originals[0]["replayed"] is False and originals[0]["charged"] == 1
    assert {e["replay_of"] for e in applies if e is not originals[0]} == {originals[0]["seq"]}
    assert result["summary"]["charged_runs"] == 1


def test_failures_are_recorded_with_status_and_error_code(server):
    s = server(weights_write=False)
    c = f"/checkpoints/{REF}"
    s.post(c + "/read-tensor", {"selector": {"tensor": "classifier.bias", "index": {"0": [10]}}})
    s.post(c + "/apply", PROGRAM, headers={**s.agent, "Idempotency-Key": "k"})
    s.get(f"/checkpoints/{'9' * 64}/describe")
    s.post(c + "/read-tensor", {"selector": BIAS, "unexpected": 1})
    s.get("/runs/run_" + "0" * 32)
    events = {(e["tool"], e["status"]): e for e in s.timeline()["result"]["events"]}
    assert events[("read_tensor", 422)]["error_code"] in ("index_out_of_range", "unsupported_schema")
    assert events[("apply", 403)]["error_code"] == "forbidden" and events[("apply", 403)]["idempotency_key"] == trace.key_digest("k")
    assert events[("describe", 404)]["error_code"] == "not_found" and events[("describe", 404)]["checkpoint_id"] == "9" * 64
    assert any(e["status"] == 422 and e["error_code"] == "unsupported_schema" for e in events.values())
    assert events[("status", 404)]["run_id"] == "run_" + "0" * 32


def test_sessions_are_private_to_the_agent_and_visible_to_the_human(server):
    s = server()
    s.get(f"/checkpoints/{REF}/describe")
    s.get(f"/checkpoints/{REF}/describe", headers=s.human)
    mine = s.timeline()
    assert {e["session"] for e in mine["result"]["events"]} == {s.principal} and mine["result"]["session"] == s.principal
    other = s.get("/trace", params={"session": "human"})
    assert other.status_code == 403 and other.json()["error"]["code"] == "forbidden"
    assert s.get("/trace", params={"session": s.principal}).status_code == 200  # naming itself is fine
    everything = s.timeline(headers=s.human)["result"]
    assert {e["session"] for e in everything["events"]} >= {s.principal, "human"}
    only = s.timeline(headers=s.human, session=s.principal)["result"]
    assert {e["session"] for e in only["events"]} == {s.principal} and only["session"] == s.principal
    human_events = [e for e in everything["events"] if e["session"] == "human"]
    assert human_events and human_events[0]["capabilities"] == ["eval:run", "model:read", "weights:read", "weights:values", "weights:write"]


def test_timeline_pages_by_event_id(server):
    s = server()
    for _ in range(5):
        s.get(f"/checkpoints/{REF}/describe")
    full = s.timeline(limit=500)["result"]["events"]
    seen, after = [], 0
    while True:
        page = s.timeline(limit=2, after=after)
        seen += page["result"]["events"]
        after = page["result"]["next_after"]
        assert page["truncated"] is (after is not None)
        if after is None:
            break
    assert [e["event_id"] for e in seen][:len(full) - 1] == [e["event_id"] for e in full][:len(full) - 1]
    assert s.get("/trace", params={"limit": 0}).status_code == 422


def test_trace_survives_a_restart_and_the_sequence_continues(server):
    first = server()
    first.get(f"/checkpoints/{REF}/describe")
    first.timeline(headers=first.human)  # a human event, too
    old_principal = first.principal
    before = first.timeline(headers=first.human, session="human")["result"]["events"]
    first.close()
    second = server()
    assert second.principal != old_principal
    old = second.timeline(headers=second.human, session=old_principal)["result"]["events"]
    assert [e["tool"] for e in old][:1] == ["describe"] and {e["session"] for e in old} == {old_principal}
    second.get(f"/checkpoints/{REF}/describe", headers=second.human)
    human = second.timeline(headers=second.human, session="human")["result"]["events"]
    assert [e["seq"] for e in human] == list(range(1, len(human) + 1)) and len(human) > len(before)


def test_nothing_secret_or_valued_is_in_the_database(server, tmp_path):
    s = server()
    c = f"/checkpoints/{REF}"
    values = s.post(c + "/read-tensor", {"selector": {"tensor": "features.block1.conv.weight", "index": {"0": [0]}}}).json()["result"]["values"]
    s.post(c + "/stats", {"selector": BIAS})
    blob = bytes(range(40))
    s.client.post(BASE + "/blobs", headers=s.agent, content=blob)
    s.get(f"/checkpoints/{s.creds.agent}/describe")  # a token pasted into a URL: dropped, 404
    s.get("/checkpoints/" + "ab" * 32 + "/describe", headers={**s.agent, "X-Note": "secret-note"})
    s.post(c + "/apply", PROGRAM, headers={**s.agent, "Idempotency-Key": "tok-SECRET-key-value"})
    s.client.post("/api/v1/session", json={"ticket": "x"})
    workspace = server.workspace
    haystack = b"".join(p.read_bytes() for p in workspace.glob("catalog.sqlite3*") if p.is_file() and not p.name.endswith("-shm"))  # -shm is a memory-mapped index (no data) that Windows locks while sqlite is open
    assert len(haystack) > 1000
    for secret in (s.creds.agent, s.creds.csrf, s.creds.cookie, s.ticket, "secret-note", "Bearer", "tok-SECRET-key-value"):
        assert secret.encode() not in haystack, secret
    for value in values[:5]:
        assert repr(value).encode() not in haystack and str(value).encode() not in haystack
    assert blob not in haystack  # bodies are never stored
    import sqlite3
    con = sqlite3.connect(workspace / "catalog.sqlite3")
    trace_text = repr(con.execute("SELECT * FROM events").fetchall()) + repr(con.execute("SELECT * FROM sessions").fetchall())
    assert "features.block1.conv.weight" not in trace_text  # the trace names no tensor; selectors live in their own human-only table
    assert "features.block1.conv.weight" in repr(con.execute("SELECT selector FROM selections").fetchall())
    con.close()


def test_trace_fields_are_whitelisted_and_checked():
    fields, token = trace.begin()
    trace.note(checkpoint_id="ab" * 32, selection_id="sel_" + "0" * 32, selections=["sel_" + "1" * 32, "junk"], ops=2,
               replayed=True, error_code="not_found")
    trace.note(checkpoint_id="not-an-id", child_id="../../etc", run_id="run_x", blob_id="nope", recipe_hash=None,
               error_code="has spaces")
    with pytest.raises(KeyError):
        trace.note(values=[1.0])
    trace.end(token)
    assert fields == {"checkpoint_id": "ab" * 32, "selection_id": "sel_" + "0" * 32, "selections": ["sel_" + "1" * 32],
                      "ops": 2, "replayed": True, "error_code": "not_found"}
    trace.note(checkpoint_id="ab" * 32)  # no active request: a no-op, not an error
    assert trace.clean_key("k" * 200) and trace.clean_key("k" * 201) is None and trace.clean_key("bad\nkey") is None
    assert trace.tool_of("POST", "/checkpoints/{checkpoint_id}/read-tensor") == "read_tensor"
    assert trace.tool_of("GET", "/runs/{run_id}/comparison") == "compare" and trace.tool_of("GET", "/catalog") == "catalog"
    assert trace.tool_of("POST", "/blobs") == "upload_blob" and trace.tool_of("GET", "/agent-session") == "session_info"


def test_request_id_header_matches_the_event_and_the_error_body(server):
    s = server()
    ok = s.get(f"/checkpoints/{REF}/describe")
    bad = s.post(f"/checkpoints/{REF}/read-tensor", {"selector": {"tensor": "nope"}})
    events = {e["request_id"]: e for e in s.timeline()["result"]["events"]}
    assert ok.headers["X-Request-Id"] in events and events[ok.headers["X-Request-Id"]]["status"] == 200
    assert bad.json()["error"]["request_id"] == bad.headers["X-Request-Id"] and events[bad.headers["X-Request-Id"]]["status"] == 422
    assert events[bad.headers["X-Request-Id"]]["error_code"] == "unknown_tensor"


def test_charged_runs_match_the_quota_across_legacy_runs_replays_and_applies(server):
    s = server()
    c = f"/checkpoints/{REF}"
    run = {**s.agent, "Idempotency-Key": "run-1"}
    assert s.post("/runs", {"kind": "baseline"}, headers=run).status_code == 202
    assert s.post("/runs", {"kind": "baseline"}, headers=run).status_code == 202  # a legacy replay: same run, no charge
    s.post(c + "/apply", PROGRAM, headers={**s.agent, "Idempotency-Key": "ap-1"})
    s.post(c + "/apply", PROGRAM, headers={**s.agent, "Idempotency-Key": "ap-1"})
    result = s.timeline()["result"]
    runs = [e for e in result["events"] if e["route"] == "POST /runs"]
    assert [(e["replayed"], e["charged"], e["replay_of"]) for e in runs] == [(False, 1, None), (True, 0, runs[0]["seq"])]
    assert runs[0]["run_id"] == runs[1]["run_id"] and runs[0]["run_id"].startswith("run_")
    assert result["summary"]["charged_runs"] == s.get("/agent-session").json()["charged"] == 2
    assert result["summary"]["replays"] == 2 and result["summary"]["applies"] == 1


def test_timeline_cost_is_the_page_not_the_log(server, monkeypatch):
    s = server()
    for _ in range(60):
        s.get(f"/checkpoints/{REF}/describe")
    built = []
    real = trace._row
    monkeypatch.setattr(trace, "_row", lambda row: built.append(row["id"]) or real(row))
    page = s.timeline(limit=5)["result"]
    assert len(page["events"]) == 5 and len(built) <= 6  # one page (plus the look-ahead row), never the whole log
    assert page["summary"]["events"] >= 60 and page["next_after"] == page["events"][-1]["event_id"]
    human = s.timeline(headers=s.human, limit=3)["result"]  # the human view over all sessions pages the same way
    assert len(human["events"]) == 3
