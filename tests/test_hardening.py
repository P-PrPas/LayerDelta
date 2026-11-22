"""v0.2 hardening: whoami probe, hashed legacy idempotency keys, import cancel and stall timeout."""

import hashlib
import sqlite3
import threading
import time
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from layerdelta import api, checkpoints
from layerdelta.checkpoints import Catalog, ImportControl, ImportFailure, ImportLimits, import_checkpoint
from layerdelta.model import ReferenceCNN, describe_model
from layerdelta.store import Store

BASE = "/api/v1/projects/demo"


def setup(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    (workspace / "reference").mkdir(parents=True)
    save_file(ReferenceCNN().state_dict(), workspace / "reference" / "model.safetensors")
    fixture = {"artifacts": {"model.safetensors": "a" * 64, "dataset.json": "b" * 64, "exploration.json": "c" * 64},
               "manifest_sha256": "d" * 64, "training": {"seed": 1}, "model": describe_model(ReferenceCNN())}
    monkeypatch.setattr(api, "load_fixture", lambda _: (fixture, {"dataset_revision_id": "v", "split_sha256": {}, "samples": []}, {}))
    return workspace


class Server:
    def __init__(self, workspace, limits=None, **caps):
        self.creds = api.Credentials("demo", **caps)
        self.client = TestClient(api.create_app(workspace, self.creds, quota=5, import_limits=limits), base_url="http://127.0.0.1")
        self.client.__enter__()
        csrf = self.client.post("/api/v1/session", json={"ticket": self.creds.ticket}).json()["csrf"]
        self.agent, self.human = {"Authorization": "Bearer " + self.creds.agent}, {"X-CSRF-Token": csrf}

    def close(self):
        self.client.__exit__(None, None, None)


@pytest.fixture
def server(tmp_path, monkeypatch):
    workspace = setup(tmp_path, monkeypatch)
    started = []

    def start(limits=None, **caps):
        s = Server(workspace, limits, **caps)
        started.append(s)
        return s

    start.workspace = workspace
    yield start
    for s in started:
        s.close()


# ------------------------------------------------------------------ whoami

def test_whoami_is_a_human_only_identity_probe(server):
    s = server()
    human = s.client.get(BASE + "/whoami", headers=s.human)
    assert human.status_code == 200
    assert human.json() == {"project_id": "demo", "principal": "human",
                            "capabilities": ["model:read", "weights:read", "weights:write"]}
    agent = s.client.get(BASE + "/whoami", headers=s.agent)
    assert agent.status_code == 403 and agent.json()["error"]["code"] == "forbidden"
    anonymous = TestClient(s.client.app, base_url="http://127.0.0.1").get(BASE + "/whoami")
    assert anonymous.status_code == 401
    assert s.client.get("/api/v1/projects/other/whoami", headers=s.human).status_code == 404


# ------------------------------------------------------------------ legacy idempotency keys are hashed

def test_legacy_idempotency_key_is_stored_as_a_digest(server):
    s = server()
    secret = "tok-SECRET-run-key"
    run = {**s.agent, "Idempotency-Key": secret}
    first = s.client.post(BASE + "/runs", json={"kind": "baseline"}, headers=run)
    again = s.client.post(BASE + "/runs", json={"kind": "baseline"}, headers=run)
    assert first.status_code == again.status_code == 202 and first.json()["run_id"] == again.json()["run_id"]
    other = s.client.post(BASE + "/runs", json={"kind": "baseline"}, headers={**s.agent, "Idempotency-Key": "another"})
    assert other.json()["run_id"] != first.json()["run_id"]
    conflict = s.client.post(BASE + "/runs", json={"kind": "baseline", "device": "cuda"}, headers=run)
    assert conflict.status_code in (409, 422)
    db = server.workspace / "catalog.sqlite3"
    keys = {r[0] for r in sqlite3.connect(db).execute("SELECT key FROM jobs")}
    assert keys == {hashlib.sha256(secret.encode()).hexdigest()[:32], hashlib.sha256(b"another").hexdigest()[:32]}
    assert secret.encode() not in b"".join(p.read_bytes() for p in server.workspace.glob("catalog.sqlite3*") if p.is_file())


def test_a_row_written_with_a_raw_key_is_still_found(tmp_path):
    store = Store(tmp_path, quota=5)
    store.start()
    try:
        job = store.submit("agent-x", "runs", "raw-key", {"checkpoint_id": "a" * 64}, "baseline")
        with store._db(True) as db:  # what a server from before the change would have left behind, in flight
            db.execute("UPDATE jobs SET key='raw-key' WHERE id=?", (job["run_id"],))
        again = store.submit("agent-x", "runs", "raw-key", {"checkpoint_id": "a" * 64}, "baseline")
        assert again["run_id"] == job["run_id"]  # found by the raw key, not submitted again
        with pytest.raises(Exception, match="different request"):
            store.submit("agent-x", "runs", "raw-key", {"checkpoint_id": "b" * 64}, "baseline")
        fresh = store.submit("agent-x", "runs", "new-key", {"checkpoint_id": "a" * 64}, "baseline")
        assert fresh["run_id"] != job["run_id"]
    finally:
        store.close()


# ------------------------------------------------------------------ import cancel and stall

def save_ckpt(tmp_path, name, seed):
    torch.manual_seed(seed)
    path = tmp_path / name
    save_file({"w": torch.randn(64)}, path)
    return path


def wait_status(s, import_id, wanted, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = s.client.get(f"{BASE}/checkpoint-imports/{import_id}", headers=s.human).json()
        if record["status"] in wanted:
            return record
        time.sleep(.02)
    pytest.fail(f"import stayed {record['status']}")


def submit(s, path):
    response = s.client.post(BASE + "/checkpoint-imports", headers=s.human, json={"path": str(path)})
    assert response.status_code == 202, response.text
    return response.json()["import_id"]


@pytest.fixture
def hung(monkeypatch):
    """open_tensors blocks on files named hung*: a path on a disconnected network share."""
    gate, real = threading.Event(), checkpoints.open_tensors

    def maybe_hang(path, **kw):
        if Path(path).name.startswith("hung"):
            gate.wait(30)
        return real(path, **kw)

    monkeypatch.setattr(checkpoints, "open_tensors", maybe_hang)
    yield gate
    gate.set()


def test_a_stalled_import_fails_and_the_queue_moves_on(tmp_path, server, hung):
    s = server(ImportLimits(stall_seconds=0.4))
    stuck = submit(s, save_ckpt(tmp_path, "hung1.safetensors", 1))
    good = submit(s, save_ckpt(tmp_path, "good.safetensors", 2))
    record = wait_status(s, stuck, ("failed",))
    assert record["error"]["code"] == "import_stalled" and "0.4" in record["error"]["message"]
    done = wait_status(s, good, ("succeeded",))
    assert done["checkpoint_id"]
    hung.set()  # the network path comes back: the abandoned worker wakes up, sees it was cancelled and stops
    time.sleep(.5)
    assert s.client.get(f"{BASE}/checkpoint-imports/{stuck}", headers=s.human).json()["status"] == "failed"
    listed = [c["checkpoint_id"] for c in s.client.get(BASE + "/checkpoints", headers=s.human).json()["items"]]
    assert len(listed) == 2 and done["checkpoint_id"] in listed  # the fixture and the good import, never the stalled one
    assert list((server.workspace / "staging").iterdir()) == []


def test_cancel_a_queued_and_a_running_import(tmp_path, server, hung):
    s = server(ImportLimits(stall_seconds=60))
    running = submit(s, save_ckpt(tmp_path, "hung2.safetensors", 3))
    queued = submit(s, save_ckpt(tmp_path, "queued.safetensors", 4))
    later = submit(s, save_ckpt(tmp_path, "later.safetensors", 5))
    wait_status(s, running, ("running",))
    cancelled = s.client.post(f"{BASE}/checkpoint-imports/{queued}/cancel", headers=s.human)
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["error"]["code"] == "cancelled"
    stopped = s.client.post(f"{BASE}/checkpoint-imports/{running}/cancel", headers=s.human).json()
    assert stopped["status"] == "cancelled"  # the dispatcher stops waiting for the blocked read at once
    assert wait_status(s, later, ("succeeded",))["checkpoint_id"]  # the queue was not blocked
    assert s.client.get(f"{BASE}/checkpoint-imports/{queued}", headers=s.human).json()["status"] == "cancelled"
    hung.set()
    time.sleep(.3)
    assert s.client.get(f"{BASE}/checkpoint-imports/{running}", headers=s.human).json()["status"] == "cancelled"
    assert len(s.client.get(BASE + "/checkpoints", headers=s.human).json()["items"]) == 2  # fixture + later only


def test_cancel_route_permissions_and_finished_imports(tmp_path, server):
    s = server()
    done = wait_status(s, submit(s, save_ckpt(tmp_path, "ok.safetensors", 6)), ("succeeded",))
    again = s.client.post(f"{BASE}/checkpoint-imports/{done['import_id']}/cancel", headers=s.human)
    assert again.status_code == 200 and again.json() == done  # a finished import is returned unchanged
    assert s.client.post(f"{BASE}/checkpoint-imports/imp_missing/cancel", headers=s.human).status_code == 404
    assert s.client.post(f"{BASE}/checkpoint-imports/{done['import_id']}/cancel", headers=s.agent).status_code == 403


def test_cancel_during_the_copy_leaves_nothing_behind(tmp_path):
    catalog = Catalog(tmp_path / "ws")
    catalog.start()
    source = save_ckpt(tmp_path, "m.safetensors", 7)
    control = ImportControl()

    def cancel_when_copying(stage, done, total, unit, **kw):
        if stage == "copying":
            assert control.cancel() is True

    with pytest.raises(ImportFailure) as info:
        import_checkpoint(catalog, source, source.name, progress=cancel_when_copying, control=control)
    assert info.value.code == "cancelled" and catalog.list() == []
    assert list((tmp_path / "ws" / "staging").iterdir()) == [] and not any((tmp_path / "ws" / "checkpoints").iterdir())


def test_a_cancel_after_the_commit_point_is_refused_and_the_import_completes(tmp_path):
    catalog = Catalog(tmp_path / "ws")
    catalog.start()
    source = save_ckpt(tmp_path, "m.safetensors", 8)
    control, results = ImportControl(), []

    def try_cancel_while_publishing(stage, done, total, unit, **kw):
        if stage == "publishing":
            results.append(control.cancel())

    entry, existing = import_checkpoint(catalog, source, source.name, progress=try_cancel_while_publishing, control=control)
    assert results and set(results) == {False} and existing is False  # refused at both publishing progress calls and catalog.get(entry["checkpoint_id"]) is not None
    late = ImportControl()
    late.cancel()
    with pytest.raises(ImportFailure):
        late.commit()  # cancelled first: it may never publish


def test_stall_seconds_flag_and_default():
    assert ImportLimits().stall_seconds == 120.0 and ImportLimits(stall_seconds=3).stall_seconds == 3
