"""The weights:read capability: agent tokens see no kernel values or statistics unless granted."""

import copy
import json
import sys
import time

from fastapi.testclient import TestClient
import pytest
import torch
from safetensors.torch import save_file

from layerdelta import api
from layerdelta.engine import patch_weight, preview
from layerdelta.model import ReferenceCNN, describe_model
from layerdelta.store import Store

BASE = "/api/v1/projects/demo"
TARGET = "features.block1.conv.kernel"
# Human-only routes: an agent token must get 403 from each of them.
HUMAN_ONLY = {"GET /whoami", "POST /checkpoint-imports/{import_id}/cancel", "POST /checkpoint-imports", "POST /checkpoint-imports/upload", "GET /checkpoint-imports/{import_id}",
              "POST /checkpoints", "GET /holdout-audit", "POST /final-evaluations", "POST /checkpoint-exports",
              "GET /checkpoint-exports/{run_id}", "GET /checkpoint-exports/{run_id}/file",
              "POST /checkpoints/{checkpoint_id}/graphs", "GET /graph-attachments/{attach_id}",  # attaching a graph is a human action
              "POST /dataset-imports", "GET /dataset-imports/{import_id}",  # importing a dataset is a human action (v0.5-E2)
              "GET /trace/sessions", "POST /selections/resolve", "GET /recipes/by-hash/{recipe_hash}",
              "POST /agent-sessions/{session}/revoke", "POST /agent-sessions/{session}/budget", "GET /proposals",
              "POST /proposals/{proposal_id}/approve", "POST /proposals/{proposal_id}/reject"}  # supervision reads (v0.4-S5)
# Structure routes an agent may call (model:read). With no graph attached these two answer no_graph (422), not data.
NEEDS_GRAPH = {"GET /checkpoints/{checkpoint_id}/groups", "GET /checkpoints/{checkpoint_id}/graph/nodes/{node_id}"}
# v2 routes by the capability they need. This fixture never grants weights:write, so WRITE is always 403.
READ = {"POST /checkpoints/{checkpoint_id}/read-tensor", "POST /checkpoints/{checkpoint_id}/stats",
        "POST /checkpoints/{checkpoint_id}/diff", "POST /checkpoints/{checkpoint_id}/query",
        "POST /checkpoints/{checkpoint_id}/plan", "POST /checkpoints/{checkpoint_id}/validate"}
WRITE = {"POST /checkpoints/{checkpoint_id}/apply", "POST /blobs"}
# Routes outside the project prefix, deliberately not weight-bearing.
NON_PROJECT = {"GET /api/v1/health", "POST /api/v1/session", "GET /openapi.json"}  # schema only, no data

WORKER = """
import hashlib, json, sys
from pathlib import Path
request = json.loads(Path(sys.argv[sys.argv.index('--request')+1]).read_bytes())
output = Path(sys.argv[sys.argv.index('--output')+1])
output.mkdir()
PREVIEW, AUDIT = json.loads(%r)
kind = request['kind']
if kind == 'preview':
    (output/'report.json').write_text(json.dumps(PREVIEW))
else:
    files = {'predictions.json': [], 'comparison.json': {'primary_slice_id': 'p', 'guardrail_slice_id': 'g',
             'slices': [], 'changes': [{'sample_id': 's', 'before': 1, 'after': 2}]}}
    if kind == 'intervention':
        files['recipe.json'] = {}
    artifacts = {}
    for name, value in files.items():
        data = json.dumps(value).encode()
        (output/name).write_bytes(data)
        artifacts[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    report = {'baseline_id': 'b', 'recipe_hash': request['payload'].get('recipe', {}) and request['payload']['recipe']['recipe_hash'],
              'evaluation_fingerprint': 'e', 'runtime': {'actual_device': 'cpu'}, 'artifacts': artifacts,
              'intervention_audit': AUDIT if kind == 'intervention' else {}}
    (output/'report.json').write_text(json.dumps(report))
"""


def real_weight_evidence():
    """Kernel statistics and raw values computed by the real engine, plus the set of numbers that must not leak."""
    torch.manual_seed(0)
    model = ReferenceCNN().eval()
    result = preview(model, torch.rand(1, 28, 28), TARGET, 0, "cpu")
    audit = patch_weight(copy.deepcopy(model), {"target_id": TARGET, "mode": "weight", "channel_index": 0,
                                                "operation": "scale", "scale": 0.5})
    before = result["kernel"]
    # A stat added to the preview after this was written must stay hidden too (allowlist, not blocklist).
    result["future_weight_stat"] = 1.2345678
    secret = {before["min"], before["max"], before["mean"], before["zero_fraction"], 1.2345678,
              audit["after"]["mean"], audit["after"]["max"], audit["after"]["min"], audit["after"]["zero_fraction"],
              *model.features.block1.conv.weight[0].flatten().tolist()}
    secret -= {0.0, 1.0}  # zero_fraction can equal ordinary 0/1 values elsewhere; the key scan covers those
    return result, audit, secret


def leaks(node, secret, where="$"):
    """Yield every place a kernel key, before/after audit statistic or secret number appears."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "kernel":
                yield f"{where}.kernel"
            if k == "intervention_audit" and isinstance(v, dict) and ({"before", "after"} & v.keys()):
                yield f"{where}.intervention_audit"
            yield from leaks(v, secret, f"{where}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from leaks(v, secret, f"{where}[{i}]")
    elif isinstance(node, float) and node in secret:
        yield f"{where}={node}"


def wait(client, path, headers):
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        state = client.get(path, headers=headers).json()
        if state["status"] in ("failed", "cancelled", "succeeded"):
            assert state["status"] == "succeeded", state
            return state
        time.sleep(.025)
    pytest.fail("job did not finish")


@pytest.fixture(params=[False, True], ids=["agent-without-capability", "agent-with-capability"])
def http(request, tmp_path, monkeypatch):
    preview_result, audit, secret = real_weight_evidence()
    fixture = {
        "artifacts": {"model.safetensors": "a" * 64, "dataset.json": "b" * 64, "exploration.json": "c" * 64},
        "manifest_sha256": "d" * 64, "training": {"seed": 1}, "model": describe_model(ReferenceCNN()),
    }
    manifest = {"dataset_revision_id": "mnist-v1", "split_sha256": {"holdout": "e" * 64}, "samples": [
        {"sample_id": f"{split}/{c}", "source_sample_id": split, "split": split, "condition": c, "label": 2}
        for split in ("holdout", "exploration") for c in ("normal", "blurred")]}
    (tmp_path / "reference").mkdir()
    save_file(ReferenceCNN().state_dict(), tmp_path / "reference" / "model.safetensors")  # a real table for the v2 routes
    monkeypatch.setattr(api, "load_fixture", lambda _: (fixture, manifest, {}))
    worker = tmp_path / "worker.py"
    worker.write_text(WORKER % json.dumps([preview_result, audit]), encoding="utf-8")
    monkeypatch.setattr(api, "Store", lambda path, **kw: Store(path, child_command=[sys.executable, str(worker)], **kw))
    creds = api.Credentials("demo", weights_read=request.param)
    app = api.create_app(tmp_path, creds, quota=10)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        agent = {"Authorization": "Bearer " + creds.agent}
        csrf = client.post("/api/v1/session", json={"ticket": creds.ticket}).json()["csrf"]
        human = {"X-CSRF-Token": csrf}
        yield app, client, agent, human, request.param, preview_result, audit, secret


def seed(client, agent):
    """One preview, one recipe, one weight run: every id the route sweep needs."""
    key = lambda n: {**agent, "Idempotency-Key": n}
    pv = client.post(BASE + "/previews", headers=key("pv"),
                     json={"sample_id": "exploration/normal", "target_id": TARGET, "channel_index": 0}).json()["run_id"]
    wait(client, f"{BASE}/previews/{pv}", agent)
    recipe = client.post(BASE + "/recipes", headers=agent, json={
        "hypothesis": "h", "intervention": {"target_id": TARGET, "mode": "weight", "channel_index": 0,
                                            "operation": "scale", "scale": 0.5}}).json()
    run = client.post(BASE + "/runs", headers=key("run"), json={
        "kind": "intervention", "recipe_id": recipe["recipe_id"]}).json()["run_id"]
    wait(client, f"{BASE}/runs/{run}", agent)
    return pv, run, recipe


def sweep(app, client, headers, pv, run, recipe):
    """Call every route of the app once. Returns {'METHOD /path/template': response}."""
    key = {"Idempotency-Key": "sweep-" + str(id(headers))}
    edit_body = {"program": [{"op": "scale", "args": {"k": 2.0}, "select": {"tensor": "classifier.bias"}}], "hypothesis": "h"}
    confirm = {"confirmed": True, "recipe_id": recipe["recipe_id"], "recipe_hash": recipe["recipe_hash"]}
    plan = {
        "GET /catalog": ("GET", "/catalog", {}),
        "GET /samples": ("GET", "/samples", {}),
        "POST /checkpoints": ("POST", "/checkpoints", {"files": {"file": ("x", b"x")}}),
        "POST /recipes": ("POST", "/recipes", {"json": {"hypothesis": "h2", "intervention": {
            "target_id": TARGET, "mode": "weight", "channel_index": 1, "operation": "zero"}}}),
        "POST /recipes/import": ("POST", "/recipes/import", {"json": {k: v for k, v in recipe.items() if k != "recipe_id"}}),
        "GET /recipes/{recipe_id}": ("GET", "/recipes/" + recipe["recipe_id"], {}),
        "GET /checkpoints": ("GET", "/checkpoints", {}),
        "GET /checkpoints/{checkpoint_id}": ("GET", "/checkpoints/" + "a" * 64, {}),
        "POST /checkpoint-imports": ("POST", "/checkpoint-imports", {"json": {"path": "C:/x.safetensors"}}),
        "POST /checkpoint-imports/upload": ("POST", "/checkpoint-imports/upload?filename=x.safetensors", {"content": b"x"}),
        "GET /checkpoint-imports/{import_id}": ("GET", "/checkpoint-imports/imp_x", {}),
        "GET /checkpoints/{checkpoint_id}/describe": ("GET", "/checkpoints/" + "a" * 64 + "/describe", {}),
        "GET /checkpoints/{checkpoint_id}/graph": ("GET", "/checkpoints/" + "a" * 64 + "/graph", {}),
        "GET /checkpoints/{checkpoint_id}/graph/nodes/{node_id}": ("GET", "/checkpoints/" + "a" * 64 + "/graph/nodes/n0", {}),
        "GET /checkpoints/{checkpoint_id}/groups": ("GET", "/checkpoints/" + "a" * 64 + "/groups", {}),
        "POST /checkpoints/{checkpoint_id}/graphs": ("POST", "/checkpoints/" + "a" * 64 + "/graphs",
                                                     {"json": {"provider": "builtin", "model": "reference_cnn"}}),
        "GET /graph-attachments/{attach_id}": ("GET", "/graph-attachments/att_x", {}),
        "POST /dataset-imports": ("POST", "/dataset-imports", {"json": {"path": "C:/x", "format": "image-text", "name": "x1"}}),
        "GET /dataset-imports/{import_id}": ("GET", "/dataset-imports/dsi_x", {}),
        "GET /datasets": ("GET", "/datasets", {}),
        "GET /datasets/{dataset_id}": ("GET", "/datasets/mnist-v1", {}),
        "POST /checkpoints/{checkpoint_id}/read-tensor": ("POST", "/checkpoints/" + "a" * 64 + "/read-tensor",
                                                          {"json": {"selector": {"tensor": "classifier.bias"}}}),
        "POST /checkpoints/{checkpoint_id}/stats": ("POST", "/checkpoints/" + "a" * 64 + "/stats",
                                                    {"json": {"selector": {"tensor": "classifier.bias"}}}),
        "POST /checkpoints/{checkpoint_id}/diff": ("POST", "/checkpoints/" + "a" * 64 + "/diff", {"json": {"other": "a" * 64}}),
        "POST /checkpoints/{checkpoint_id}/query": ("POST", "/checkpoints/" + "a" * 64 + "/query", {"json": {}}),
        "POST /checkpoints/{checkpoint_id}/plan": ("POST", "/checkpoints/" + "a" * 64 + "/plan", {"json": edit_body}),
        "POST /checkpoints/{checkpoint_id}/validate": ("POST", "/checkpoints/" + "a" * 64 + "/validate", {"json": edit_body}),
        "POST /checkpoints/{checkpoint_id}/apply": ("POST", "/checkpoints/" + "a" * 64 + "/apply",
                                                    {"headers": key, "json": edit_body}),
        "POST /blobs": ("POST", "/blobs", {"content": bytes(8)}),
        "GET /trace": ("GET", "/trace", {}),
        "GET /whoami": ("GET", "/whoami", {}),
        "GET /trace/sessions": ("GET", "/trace/sessions", {}),
        "POST /agent-sessions/{session}/revoke": ("POST", "/agent-sessions/agent-x/revoke", {}),
        "POST /agent-sessions/{session}/budget": ("POST", "/agent-sessions/agent-x/budget", {"json": {"dimension": "scan", "limit": 1}}),
        "GET /proposals": ("GET", "/proposals", {}),
        "POST /proposals/{proposal_id}/approve": ("POST", "/proposals/p_0/approve", {}),
        "POST /proposals/{proposal_id}/reject": ("POST", "/proposals/p_0/reject", {}),
        "POST /selections/resolve": ("POST", "/selections/resolve", {"json": {"ids": []}}),
        "GET /recipes/by-hash/{recipe_hash}": ("GET", "/recipes/by-hash/" + "0" * 64, {}),
        "POST /checkpoint-imports/{import_id}/cancel": ("POST", "/checkpoint-imports/imp_x/cancel", {}),
        "GET /holdout-audit": ("GET", "/holdout-audit", {}),
        "GET /agent-session": ("GET", "/agent-session", {}),
        "POST /previews": ("POST", "/previews", {"headers": key, "json": {
            "sample_id": "exploration/blurred", "target_id": TARGET, "channel_index": 0}}),
        "POST /runs": ("POST", "/runs", {"headers": key, "json": {"kind": "baseline"}}),
        "GET /runs": ("GET", "/runs", {}),
        "GET /runs/{run_id}": ("GET", f"/runs/{run}", {}),
        "GET /previews/{run_id}": ("GET", f"/previews/{pv}", {}),
        "POST /runs/{run_id}/cancel": ("POST", f"/runs/{run}/cancel", {}),
        "GET /runs/{run_id}/report": ("GET", f"/runs/{run}/report", {}),
        "GET /runs/{run_id}/comparison": ("GET", f"/runs/{run}/comparison", {}),
        "GET /runs/{run_id}/changes": ("GET", f"/runs/{run}/changes", {}),
        "POST /final-evaluations": ("POST", "/final-evaluations", {"headers": key, "json": confirm}),
        "POST /checkpoint-exports": ("POST", "/checkpoint-exports", {"headers": key, "json": {
            "confirmed": True, "source_run_id": run}}),
        "GET /checkpoint-exports/{run_id}": ("GET", f"/checkpoint-exports/{run}", {}),
        "GET /checkpoint-exports/{run_id}/file": ("GET", f"/checkpoint-exports/{run}/file", {}),
    }
    declared = {f"{m} {r.path.removeprefix(BASE.replace('demo', '{project_id}'))}"
                for r in app.routes if hasattr(r, "methods") for m in r.methods - {"HEAD", "OPTIONS"}}
    assert declared - NON_PROJECT == set(plan), "new route: add it to the sweep and decide who may call it"
    out = {}
    for name, (method, path, extra) in plan.items():
        extra = dict(extra)
        extra["headers"] = {**headers, **extra.get("headers", {})}
        out[name] = client.request(method, BASE + path, **extra)
    return out


def test_every_route_for_agent_token(http):
    app, client, agent, human, granted, preview_result, audit, secret = http
    ids = seed(client, agent)
    # Fresh cookie-less client for agent calls, so the human session cannot be mixed in.
    with TestClient(app, base_url="http://127.0.0.1") as anonymous:
        responses = sweep(app, anonymous, agent, *ids)
    found = {}
    for name, response in responses.items():
        if name in HUMAN_ONLY or name in WRITE or (name in READ and not granted):
            assert response.status_code == 403, (name, response.text)
            assert response.json()["error"]["code"] == "forbidden"
            continue
        if name in NEEDS_GRAPH:  # nothing attached in this fixture: a clear no_graph, never an empty success
            assert response.status_code == 422 and response.json()["error"]["code"] == "no_graph", (name, response.text)
            continue
        assert response.status_code in (200, 201, 202), (name, response.status_code, response.text)
        found[name] = list(leaks(response.json(), secret))
    # every route an agent token can reach (403s included) left an id-only event in its own session
    traced = client.get(BASE + "/trace?limit=500", headers=agent).json()["result"]["events"]
    assert set(responses) <= {e["route"] for e in traced}, set(responses) - {e["route"] for e in traced}
    assert all(e["session"] == traced[0]["session"] for e in traced)
    if granted:
        body = responses["GET /previews/{run_id}"].json()["result"]
        assert body["kernel"] == preview_result["kernel"] and body["activation"] == preview_result["activation"]
        assert responses["GET /runs/{run_id}/report"].json()["intervention_audit"] == audit
        assert responses["GET /agent-session"].json()["capabilities"] == ["model:read", "weights:read", "weights:values"]
        assert any(found.values())
        assert responses["POST /checkpoints/{checkpoint_id}/read-tensor"].json()["result"]["values"]  # weights, with the capability
    else:
        assert {k: v for k, v in found.items() if v} == {}
        body = responses["GET /previews/{run_id}"].json()["result"]
        assert "kernel" not in body and "future_weight_stat" not in body
        assert "activation" not in body and "image" not in body  # a full activation map plus the image reveals weights
        assert body["target_id"] == preview_result["target_id"]
        mapping = responses["GET /runs/{run_id}/report"].json()["intervention_audit"]
        assert mapping == {k: audit[k] for k in ("parameter_name", "channel_axis", "channel_index")}
        assert responses["GET /agent-session"].json()["capabilities"] == ["model:read"]
        assert responses["GET /checkpoints/{checkpoint_id}/describe"].json()["result"]["tensors"]  # structure, no values
        assert "before" in responses["GET /runs/{run_id}/changes"].json()["items"][0]  # prediction labels are not weights


def test_human_session_unchanged(http):
    app, client, agent, human, granted, preview_result, audit, secret = http
    pv, run, recipe = seed(client, agent)
    result = client.get(f"{BASE}/previews/{pv}", headers=human).json()["result"]
    assert result["kernel"] == preview_result["kernel"] and result == preview_result
    assert client.get(f"{BASE}/runs/{run}/report", headers=human).json()["intervention_audit"] == audit
    assert client.get(f"{BASE}/runs/{run}/comparison", headers=human).status_code == 200
    assert client.get(BASE + "/agent-session", headers=human).status_code == 403  # agent-only route, as before


def test_capability_defaults_off():
    assert api.Credentials("p").weights_read is False
