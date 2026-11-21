import concurrent.futures
import json
import sqlite3
import sys
import threading
import time

import pytest

from layerdelta.store import Store, StoreError


@pytest.fixture
def store(tmp_path):
    worker = tmp_path / "test_worker.py"
    worker.write_text("""
import argparse, json, sys, time
from pathlib import Path
p = argparse.ArgumentParser()
for field in ('workspace', 'request', 'output'):
    p.add_argument('--' + field, type=Path)
a = p.parse_args()
r = json.loads(a.request.read_bytes())
a.output.mkdir()
payload = r['payload']
time.sleep(payload.get('sleep', 0))
if payload.get('error'):
    (a.output/'error.json').write_text(json.dumps({'code':payload['error'],'message':'fault'}))
    sys.exit(1)
for f in ('report.json','predictions.json','comparison.json','recipe.json'):
    (a.output/f).write_text(json.dumps(r['run_id']))
""", encoding="utf-8")
    service = Store(tmp_path / "งาน lab", quota=2, queue_limit=6, timeout=10,
                    child_command=[sys.executable, str(worker)])
    service.start()
    yield service
    service.close()


def wait(store, job, *states, actor="human"):
    end = time.monotonic() + 20
    while time.monotonic() < end:
        record = store.get(job["run_id"], actor)
        if record["status"] in states:
            return record
        time.sleep(.025)
    pytest.fail(f"Job remained {record}")


def test_atomic_idempotency_and_session_budget(store):
    def submit(_):
        return store.submit("agent-a", "/runs", "same", {}, "baseline")

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        ids = {r["run_id"] for r in pool.map(submit, range(10))}
    assert len(ids) == 1 and store.quota_info("agent-a")["charged"] == 1
    with pytest.raises(StoreError, match="different request"):
        store.submit("agent-a", "/runs", "same", {"device": "cuda"}, "baseline")
    store.submit("agent-a", "/runs", "second", {}, "baseline")
    with pytest.raises(StoreError) as e:
        store.submit("agent-a", "/runs", "third", {}, "baseline")
    assert e.value.code == "quota_exhausted"
    assert submit(None)["run_id"] in ids
    assert store.submit("agent-b", "/runs", "same", {}, "baseline")["run_id"] not in ids
    assert all(job["run_id"] not in ids for job in store.list_jobs("agent-b"))


def test_holdout_scopes_and_transactional_audit(store):
    payload = {"confirmed": True, "recipe_hash": "f"*64, "split_sha256": "a"*64}
    for actor, data in [("agent", payload), ("human", {**payload, "confirmed": False})]:
        with pytest.raises(StoreError):
            store.submit(actor, "/final", "key", data, "final_evaluation", split="holdout")
    assert not store.list_jobs("human") and not store.holdout_audit("human")
    job = store.submit("human", "/final", "key", payload, "final_evaluation", split="holdout")
    assert store.holdout_audit("human")[0]["run_id"] == job["run_id"]
    assert store.holdout_audit("human")[0]["split_manifest_hash"] == "a"*64
    store.submit("human", "/exports", "key", {"source_run_id": job["run_id"], "confirmed": True},
                 "checkpoint_export", split="holdout")
    assert len(store.holdout_audit("human")) == 1
    assert not store.list_jobs("agent")
    for method in [store.get, store.cancel]:
        with pytest.raises(StoreError) as e:
            method(job["run_id"], "agent")
        assert e.value.code == "not_found"
    with pytest.raises(StoreError):
        store.artifact(job["run_id"], "agent", "report.json")


def test_publication_manifest_integrity_and_invisible_staging(store):
    job = store.submit("agent", "/runs", "key", {"sleep": .25}, "baseline")
    run = job["run_id"]
    wait(store, job, "running")
    with pytest.raises(StoreError) as e:
        store.artifact(run, "agent", "report.json")
    assert e.value.code == "run_not_ready"
    wait(store, job, "succeeded")
    file = store.artifact(run, "agent", "report.json")
    assert json.loads(file.read_bytes()) == run
    assert store.cancel(run, "agent")["status"] == "succeeded"
    for filename in ("../catalog.sqlite3", "private.json"):
        with pytest.raises(StoreError) as e:
            store.artifact(run, "agent", filename)
        assert e.value.code == "not_found"
    file.write_bytes(b"damaged")
    with pytest.raises(StoreError) as e:
        store.artifact(run, "agent", "report.json")
    assert e.value.code == "artifact_unavailable"
    (file.parent / "manifest.json").unlink()
    with pytest.raises(StoreError) as e:
        store.artifact(run, "agent", "predictions.json")
    assert e.value.code == "artifact_unavailable"


def test_cancel_before_publish_failure_timeout_and_next_job(store, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    publish = store._publish

    def delayed(*args):
        entered.set()
        assert release.wait(timeout=10)
        publish(*args)

    monkeypatch.setattr(store, "_publish", delayed)
    first = store.submit("agent", "/runs", "1", {}, "intervention")
    assert entered.wait(timeout=10)
    store.cancel(first["run_id"], "agent")
    release.set()
    wait(store, first, "cancelled")
    assert store.quota_info("agent")["charged"] == 1
    monkeypatch.setattr(store, "_publish", publish)
    error = store.submit("human", "/runs", "2", {"error": "gpu_oom"}, "intervention")
    assert wait(store, error, "failed")["error"]["code"] == "gpu_oom"
    store.timeout = .15
    slow = store.submit("human", "/runs", "3", {"sleep": 4}, "baseline")
    assert wait(store, slow, "failed")["error"]["code"] == "timeout"
    store.timeout = 10
    good = store.submit("human", "/runs", "4", {}, "baseline")
    wait(store, good, "succeeded")


def test_single_supervisor_restart_and_crash_after_rename(store):
    with pytest.raises(StoreError) as e:
        Store(store.workspace).start()
    assert e.value.code == "workspace_locked"
    working = store.submit("human", "/previews", "a", {"sleep": 4}, "preview")
    wait(store, working, "running")
    queued = store.submit("human", "/runs", "b", {}, "baseline")
    store.close()
    assert store.get(queued["run_id"], "human")["status"] == "queued"
    run = working["run_id"]
    # Simulate process death after directory publication but before SQLite commits.
    orphan = store.workspace / "runs" / run
    orphan.mkdir()
    (orphan / "report.json").write_bytes(b"{}")
    with sqlite3.connect(store.db_path) as db:
        db.execute("UPDATE jobs SET status='running',error=NULL,finished_at=NULL WHERE id=?", (run,))
    store.start()
    assert not orphan.exists()
    assert store.get(run, "human")["error"]["code"] == "interrupted"
    wait(store, queued, "succeeded")


def test_queue_and_publication_io_failure(store, monkeypatch):
    store.queue_limit = 1
    one = store.submit("human", "/runs", "a", {"sleep": 4}, "baseline")
    wait(store, one, "running")
    two = store.submit("human", "/runs", "b", {}, "baseline")
    with pytest.raises(StoreError) as e:
        store.submit("agent", "/runs", "c", {}, "baseline")
    assert e.value.code == "queue_full"
    assert store.quota_info("agent")["charged"] == 0
    store.cancel(two["run_id"], "human")
    store.cancel(one["run_id"], "human")
    wait(store, one, "cancelled")
    def full(*_):
        raise OSError(28, "disk full")
    monkeypatch.setattr(store, "_publish", full)
    failed = store.submit("agent", "/runs", "c", {}, "baseline")
    assert "disk full" in wait(store, failed, "failed")["error"]["message"]
    with pytest.raises(StoreError) as e:
        store.artifact(failed["run_id"], "agent", "report.json")
    assert e.value.code == "run_not_ready"


def test_recipe_replay_scope(store):
    recipe = {"recipe_hash": "x", "evaluation": {"split": "exploration"}}
    key = store.put_recipe(recipe, "agent-a")
    assert store.get_recipe(key, "human") == recipe and not store.list_recipes("agent-b")
    with pytest.raises(StoreError):
        store.get_recipe(key, "agent-b")
    store.put_recipe(recipe, "human")
    assert store.get_recipe(key, "agent-b") == recipe and store.list_recipes("agent-a") == [recipe]
