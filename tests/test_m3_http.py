"""Authorization and fault behavior through the shared HTTP application and real SQLite supervisor."""

from concurrent.futures import ThreadPoolExecutor
import sys
import time

from fastapi.testclient import TestClient
import pytest

from layerdelta import api
from layerdelta.model import ReferenceCNN, describe_model
from layerdelta.store import Store


@pytest.fixture
def http(tmp_path, monkeypatch):
    fixture = {
        "artifacts": {"model.safetensors": "a"*64, "dataset.json": "b"*64, "exploration.json": "c"*64},
        "manifest_sha256": "d"*64, "training": {"seed": 1}, "model": describe_model(ReferenceCNN()),
    }
    manifest = {"dataset_revision_id": "mnist-v1", "split_sha256": {"holdout": "e"*64}, "samples": [
        {"sample_id": f"{split}/{c}", "source_sample_id": split, "split": split, "condition": c, "label": 2}
        for split in ("holdout", "exploration") for c in ("normal", "blurred")
    ]}
    (tmp_path / "reference").mkdir()
    (tmp_path / "reference" / "model.safetensors").touch()
    monkeypatch.setattr(api, "load_fixture", lambda _: (fixture, manifest, {}))
    worker = tmp_path / "worker.py"
    worker.write_text("""
import hashlib, json, sys
from pathlib import Path
request = json.loads(Path(sys.argv[sys.argv.index('--request')+1]).read_bytes())
output = Path(sys.argv[sys.argv.index('--output')+1])
output.mkdir()
if request['kind'] == 'baseline' and not request['payload'].get('retry_of_run_id'):
    (output/'error.json').write_text(json.dumps({'code':'gpu_oom','message':'simulated worker OOM'}))
    sys.exit(1)
files = ('predictions.json','comparison.json','recipe.json')
for name in files:
    (output/name).write_bytes(b'{}')
report = {'artifacts':{f:{'sha256':hashlib.sha256(b'{}').hexdigest(),'bytes':2} for f in files}}
(output/'report.json').write_text(json.dumps(report))
""", encoding="utf-8")
    monkeypatch.setattr(api, "Store", lambda path, **kw: Store(path, child_command=[sys.executable, str(worker)], **kw))
    creds = api.Credentials("demo")
    app = api.create_app(tmp_path, creds, quota=2)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        token = {"Authorization": "Bearer " + creds.agent}
        csrf = client.post("/api/v1/session", json={"ticket": creds.ticket}).json()["csrf"]
        yield client, "/api/v1/projects/demo", token, {"X-CSRF-Token": csrf}


def finished(client, path, headers):
    deadline = time.monotonic()+20
    while time.monotonic() < deadline:
        response = client.get(path, headers=headers)
        assert response.status_code == 200
        state = response.json()
        if state["status"] in ("failed", "cancelled", "succeeded"):
            return state
        time.sleep(.025)
    pytest.fail(f"Job did not complete in time: {path}")


def test_permission_boundaries_and_hidden_holdout(http):
    client, base, agent, human = http
    edit = {"hypothesis": "identity", "intervention": {"target_id": "features.block1.conv.kernel",
            "mode": "weight", "channel_index": 0, "operation": "scale", "scale": 1}}
    assert client.post(base + "/recipes", json=edit).status_code == 403  # cookie only, no CSRF
    assert client.post(base+"/recipes", json=edit, headers={**human, "Origin": "http://localhost:7000"}).status_code == 403
    assert client.get(base.replace("/demo", "/other") + "/catalog", headers=agent).status_code == 404
    assert client.get(base + "/catalog", headers={**agent, "Host": "other.example"}).status_code == 403
    recipe = client.post(base + "/recipes", headers=agent, json=edit).json()
    import_bad = {**recipe, "hypothesis": "modified after hashing"}
    assert client.post(base+"/recipes/import", headers=agent, json=import_bad).status_code == 422
    confirm = {"confirmed": True, "recipe_id": recipe["recipe_id"], "recipe_hash": recipe["recipe_hash"]}
    action = {"Idempotency-Key": "final"}
    assert client.post(base+"/final-evaluations", headers={**agent, **action}, json=confirm).status_code == 403
    assert client.get(base+"/holdout-audit", headers=agent).status_code == 403
    invalid = client.post(base+"/final-evaluations", headers={**human, **action}, json={**confirm, "confirmed":False})
    assert invalid.status_code == 409
    assert not client.get(base+"/holdout-audit", headers=human).json()["items"]
    result = client.post(base+"/final-evaluations", headers={**human, **action}, json=confirm)
    assert result.status_code == 202
    run = result.json()["run_id"]
    assert finished(client, base+"/runs/"+run, human)["status"] == "succeeded"
    audit = client.get(base+"/holdout-audit", headers=human).json()["items"]
    assert len(audit) == 1 and audit[0]["recipe_hash"] == recipe["recipe_hash"] and audit[0]["split_manifest_hash"] == "e"*64
    for suffix in ("/runs/"+run, "/previews/"+run, f"/runs/{run}/comparison", f"/runs/{run}/changes",
                   f"/runs/{run}/report"):
        assert client.get(base+suffix, headers=agent).status_code == 404
    assert client.post(f"{base}/runs/{run}/cancel", headers=agent).status_code == 404
    export = client.post(base+"/checkpoint-exports", headers={**agent, **action}, json={"confirmed":True,"source_run_id":run})
    assert export.status_code == 403
    assert client.get(base+f"/checkpoint-exports/{run}/file", headers=agent).status_code == 403
    assert not client.get(base+"/runs", headers=agent).json()["items"]
    assert not client.get(base+"/samples?source_sample_id=holdout", headers=agent).json()["items"]
    preview = {"sample_id":"holdout/normal","target_id":"features.block1.relu.output","channel_index":0}
    assert client.post(base+"/previews", headers={**agent, **action}, json=preview).status_code == 404


def test_unavailable_device_oom_retry_and_concurrent_lost_response(http, monkeypatch):
    import torch
    client, base, agent, _ = http
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    def submit(key, **fields):
        return client.post(base+"/runs", json={"kind":"baseline", **fields},
                           headers={**agent, "Idempotency-Key":key})
    unavailable = submit("one", device="cuda")
    assert unavailable.status_code == 409 and unavailable.json()["error"]["code"] == "device_unavailable"
    assert client.get(base+"/agent-session", headers=agent).json()["charged"] == 0
    with ThreadPoolExecutor(4) as pool:
        submissions = list(pool.map(lambda _: submit("one"), range(6)))
    assert all(r.status_code == 202 for r in submissions)
    ids = {r.json()["run_id"] for r in submissions}
    assert len(ids) == 1
    first = ids.pop()
    failed = finished(client, f"{base}/runs/{first}", agent)
    assert failed["status"] == "failed" and failed["error"]["code"] == "gpu_oom"
    assert client.get(f"{base}/runs/{first}/report", headers=agent).status_code == 409
    conflict = submit("one", retry_of_run_id=first)
    assert conflict.status_code == 409 and conflict.json()["error"]["code"] == "idempotency_conflict"
    new = submit("retry", retry_of_run_id=first)
    assert new.status_code == 202
    second = new.json()["run_id"]
    assert second != first and finished(client, f"{base}/runs/{second}", agent)["status"] == "succeeded"
    assert submit("retry", retry_of_run_id=first).json()["run_id"] == second
    assert submit("new").json()["error"]["code"] == "quota_exhausted"
    assert client.get(base+"/agent-session", headers=agent).json()["charged"] == 2
