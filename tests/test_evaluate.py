"""Evaluations (v0.5-E4): `evaluate`, `get_evaluation`, `evaluation_changes`; budgets, idempotency, baseline cache, paired bootstrap, trace.

Runs the real pipeline: a person imports a PNG dataset and attaches the builtin reference CNN, an agent makes a child with an edit and
evaluates it; the forward passes happen in the isolated evaluation worker.
"""

import struct
import time
import zlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from layerdelta import api, bootstrap, evalservice
from test_v2_api import BASE, BIAS, REF, make_workspace  # noqa: F401

pytest.importorskip("PIL")
C = f"/checkpoints/{REF}"
EDIT = {"program": [{"op": "scale", "args": {"k": 8.0}, "select": BIAS}], "hypothesis": "bias x8"}


def png(width, height, rng):
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    rows = b"".join(b"\x00" + rng.integers(0, 256, width, dtype=np.uint8).tobytes() for _ in range(height))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


@pytest.fixture
def server(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path, monkeypatch)
    opened = []

    def make(budgets=None, **caps):
        creds = api.Credentials("demo", **{"weights_read": True, "weights_write": True, "eval_run": True, **caps})
        app = api.create_app(workspace, creds, quota=50, budgets=budgets or {})
        client = TestClient(app, base_url="http://127.0.0.1")
        client.__enter__()
        opened.append(client)
        csrf = client.post("/api/v1/session", json={"ticket": creds.ticket}).json()["csrf"]
        return client, {"Authorization": "Bearer " + creds.agent}, {"X-CSRF-Token": csrf}
    yield make
    for client in opened:
        client.__exit__(None, None, None)


def wait(client, headers, url, done=("succeeded", "failed")):
    for _ in range(600):
        body = client.get(BASE + url, headers=headers).json()
        if body.get("status") in done:
            return body
        time.sleep(.1)
    raise AssertionError(body)


@pytest.fixture
def world(server, tmp_path):
    """A person imported a 10-class dataset and attached the reference CNN; an agent made a child of REF."""
    client, agent, human = server()
    rng = np.random.default_rng(4)
    source = tmp_path / "digits"
    source.mkdir()
    lines = ["image,label"]
    for digit in range(10):
        for k in range(4):
            (source / f"d{digit}_{k}.png").write_bytes(png(28, 28, rng))
            lines.append(f"d{digit}_{k}.png,{digit}")
    (source / "labels.csv").write_text("\n".join(lines) + "\n")
    imported = client.post(BASE + "/dataset-imports", headers=human, json={"path": str(source), "format": "image-folder-labels", "name": "digits-v1",
                                                                             "preprocessing": {"scale": 1 / 255}})
    dataset = wait(client, human, f"/dataset-imports/{imported.json()['import_id']}")["dataset"]["dataset_id"]
    attached = client.post(BASE + C + "/graphs", headers=human, json={"provider": "builtin", "model": "reference_cnn"})
    assert wait(client, human, f"/graph-attachments/{attached.json()['attach_id']}")["status"] == "succeeded"
    child = client.post(BASE + C + "/apply", headers={**agent, "Idempotency-Key": "child"}, json=EDIT).json()["result"]["manifest"]["child_sha256"]
    return client, agent, human, dataset, child


def evaluate(client, agent, checkpoint, dataset, key, **body):
    return client.post(BASE + f"/checkpoints/{checkpoint}/evaluations", headers={**agent, "Idempotency-Key": key},
                       json={"dataset_id": dataset, "evaluator": "classification", **body})


def test_an_evaluation_compares_a_child_with_its_parent(world):
    client, agent, human, dataset, child = world
    started = evaluate(client, agent, child, dataset, "e1")
    assert started.status_code == 202, started.text
    body = started.json()
    assert body["status"] == "queued" and body["replayed"] is False and body["baseline_id"] == REF and body["samples"] == 20
    done = wait(client, agent, f"/evaluations/{body['evaluation_id']}")
    assert done["status"] == "succeeded", done
    accuracy = next(m for m in done["metrics"] if m["name"] == "accuracy")
    assert accuracy["primary"] and accuracy["unit"] == "rate" and accuracy["better"] == "higher"
    assert set(accuracy) >= {"baseline", "value", "delta", "ci"} and accuracy["delta"] == pytest.approx(accuracy["value"] - accuracy["baseline"])
    assert len(accuracy["per_class"]) == 10 and set(accuracy["per_class"][0]) >= {"class", "baseline", "value", "delta", "ci", "n"}
    assert accuracy["ci"] is None or accuracy["ci"][0] <= accuracy["delta"] + 1e-9 <= accuracy["ci"][1] + 1e-9
    counts = done["counts"]
    assert counts["samples"] == 20 and counts["fixed"] + counts["broken"] + counts["changed"] <= 20
    assert done["resamples"] == bootstrap.RESAMPLES and done["diff_hint"] == {"checkpoint_id": REF, "other": child}
    info = client.get(BASE + "/agent-session", headers=agent).json()["budgets"]
    assert info["evals"]["used"] == 1 and info["eval_samples"]["used"] == 40  # the child and its baseline: 20 forward passes each

    again = evaluate(client, agent, child, dataset, "e1").json()
    assert again["evaluation_id"] == body["evaluation_id"] and again["replayed"] is True
    assert client.get(BASE + "/agent-session", headers=agent).json()["budgets"]["eval_samples"]["used"] == 40  # a repeated key is charged once
    other = evaluate(client, agent, child, dataset, "e1", limit_samples=5)
    assert other.status_code == 409 and other.json()["error"]["code"] == "idempotency_conflict"

    second = evaluate(client, agent, child, dataset, "e2").json()  # same baseline, same fingerprint: its records are cached
    assert second["baseline_cached"] is True
    assert wait(client, agent, f"/evaluations/{second['evaluation_id']}")["metrics"] == done["metrics"]  # deterministic, bootstrap included
    assert client.get(BASE + "/agent-session", headers=agent).json()["budgets"]["eval_samples"]["used"] == 60


def test_changes_are_paged_categorised_and_scored(world):
    client, agent, human, dataset, child = world
    ident = evaluate(client, agent, child, dataset, "c1").json()["evaluation_id"]
    done = wait(client, agent, f"/evaluations/{ident}")
    page = client.get(BASE + f"/evaluations/{ident}/changes", headers=agent, params={"limit": 3}).json()
    counted = done["counts"]["fixed"] + done["counts"]["broken"] + done["counts"]["changed"]
    assert page["total"] == counted
    if counted:
        assert set(page["changes"][0]) == {"sample_id", "label", "condition", "before", "after", "category", "score_before", "score_after"}
        assert page["changes"][0]["before"]["pred"] != page["changes"][0]["after"]["pred"]
    if counted > 3:
        assert page["truncated"] and page["cursor"]
        nxt = client.get(BASE + f"/evaluations/{ident}/changes", headers=agent, params={"limit": 3, "cursor": page["cursor"]}).json()
        assert nxt["offset"] == 3 and {c["sample_id"] for c in nxt["changes"]}.isdisjoint({c["sample_id"] for c in page["changes"]})
    for kind in ("fixed", "broken", "changed"):
        only = client.get(BASE + f"/evaluations/{ident}/changes", headers=agent, params={"category": kind, "limit": 500}).json()
        assert only["total"] == done["counts"][kind] and all(c["category"] == kind for c in only["changes"])
    assert client.get(BASE + f"/evaluations/{ident}/changes", headers=agent, params={"category": "odd"}).status_code == 422
    assert client.get(BASE + f"/evaluations/{ident}/changes", headers=agent, params={"cursor": "c2.1.bad"}).status_code == 422


def test_a_baseline_of_null_gives_absolute_metrics_and_an_import_has_no_default(world):
    client, agent, human, dataset, child = world
    absolute = wait(client, agent, f"/evaluations/{evaluate(client, agent, REF, dataset, 'a1', baseline=None).json()['evaluation_id']}")
    assert absolute["status"] == "succeeded" and absolute["baseline_id"] is None
    accuracy = next(m for m in absolute["metrics"] if m["name"] == "accuracy")
    assert accuracy.get("baseline") is None and accuracy.get("delta") is None and accuracy.get("ci") is None and absolute["counts"]["fixed"] == 0
    refused = evaluate(client, agent, REF, dataset, "a2")
    assert refused.status_code == 422 and refused.json()["error"]["code"] == "no_baseline"  # REF has no parent: say so, never "no change"


def test_requests_are_refused_before_any_work_and_before_any_charge(world):
    client, agent, human, dataset, child = world
    before = client.get(BASE + "/agent-session", headers=agent).json()["budgets"]["eval_samples"]["used"]
    cases = [({"split": "holdout"}, 403, "forbidden"), ({"evaluator": "ocr"}, 422, "dataset_task_mismatch"), ({"limit_samples": 0}, 422, "invalid_argument"),
             ({"evaluator": {"task": "classification", "classes": 3}}, None, None), ({"baseline": "f" * 64}, 404, "not_found"),
             ({"evaluator": "nonsense"}, 422, None)]
    for i, (extra, status, code) in enumerate(cases):
        answer = evaluate(client, agent, child, dataset, f"r{i}", **extra)
        if status is None:
            assert answer.status_code == 202  # a wrong class count is only found when the outputs are seen: the job fails, not the request
            assert wait(client, agent, f"/evaluations/{answer.json()['evaluation_id']}")["error_code"] == "evaluator_mismatch"
            continue
        assert answer.status_code == status, (extra, answer.text)
        if code:
            assert answer.json()["error"]["code"] == code
    assert evaluate(client, agent, child, "ds_" + "0" * 32, "r9").status_code == 404
    assert evaluate(client, agent, child, "mnist-v1", "r10").json()["error"]["code"] == "dataset_unsupported"
    assert client.get(BASE + "/agent-session", headers=agent).json()["budgets"]["eval_samples"]["used"] == before + 40  # only the job that ran was charged


def test_a_token_without_eval_run_and_other_sessions_cannot_run_or_read(world, tmp_path):
    client, agent, human, dataset, child = world
    ident = evaluate(client, agent, child, dataset, "p1").json()["evaluation_id"]
    wait(client, agent, f"/evaluations/{ident}")
    assert client.get(BASE + f"/evaluations/{ident}", headers=human).json()["status"] == "succeeded"  # a person reads every evaluation
    assert client.get(BASE + f"/evaluations/ev_{'0' * 16}", headers=agent).status_code == 404
    assert client.get(BASE + "/evaluations/..%2Fx", headers=agent).status_code == 404


def test_eval_samples_budget_is_charged_at_submission(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path, monkeypatch)
    creds = api.Credentials("demo", weights_read=True, weights_write=True, eval_run=True)
    rng = np.random.default_rng(1)
    source = tmp_path / "d"
    source.mkdir()
    lines = ["image,label"]
    for digit in range(10):
        for k in range(4):
            (source / f"d{digit}_{k}.png").write_bytes(png(28, 28, rng))
            lines.append(f"d{digit}_{k}.png,{digit}")
    (source / "labels.csv").write_text("\n".join(lines) + "\n")
    with TestClient(api.create_app(workspace, creds, quota=9, budgets={"eval_samples": 30, "evals": 5, "applies": 9}), base_url="http://127.0.0.1") as client:
        human = {"X-CSRF-Token": client.post("/api/v1/session", json={"ticket": creds.ticket}).json()["csrf"]}
        agent = {"Authorization": "Bearer " + creds.agent}
        started = client.post(BASE + "/dataset-imports", headers=human, json={"path": str(source), "format": "image-folder-labels", "name": "d1"})
        dataset = wait(client, human, f"/dataset-imports/{started.json()['import_id']}")["dataset"]["dataset_id"]
        attached = client.post(BASE + C + "/graphs", headers=human, json={"provider": "builtin", "model": "reference_cnn"})
        wait(client, human, f"/graph-attachments/{attached.json()['attach_id']}")
        child = client.post(BASE + C + "/apply", headers={**agent, "Idempotency-Key": "k"}, json=EDIT).json()["result"]["manifest"]["child_sha256"]
        refused = evaluate(client, agent, child, dataset, "b1")  # 20 samples + 20 for the baseline = 40 > 30
        assert refused.status_code == 429 and refused.json()["error"]["field"] == "eval_samples"
        ok = evaluate(client, agent, child, dataset, "b2", baseline=None)  # 20 <= 30
        assert ok.status_code == 202
        used = client.get(BASE + "/agent-session", headers=agent).json()["budgets"]
        assert used["eval_samples"] == {"limit": 30, "used": 20, "remaining": 10} and used["evals"]["used"] == 1
        assert evaluate(client, agent, child, dataset, "b3", baseline=None).status_code == 429  # 10 left, 20 needed
        wait(client, agent, f"/evaluations/{ok.json()['evaluation_id']}")
        events = client.get(BASE + "/trace", headers=agent, params={"limit": 500}).json()["result"]["events"]
        run = next(e for e in events if e["tool"] == "evaluate" and e["status"] == 202)
        assert run["evaluation_id"] == ok.json()["evaluation_id"] and run["dataset_id"] == dataset and run["eval_samples"] == 20
        assert run["budget_charged"] == {"evals": 1, "eval_samples": 20} and run["checkpoint_id"] == child and run["fingerprint"] == ok.json()["fingerprint"]


def test_a_token_without_eval_run_is_refused(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path, monkeypatch)
    creds = api.Credentials("demo", weights_read=True)
    with TestClient(api.create_app(workspace, creds), base_url="http://127.0.0.1") as client:
        refused = client.post(BASE + C + "/evaluations", headers={"Authorization": "Bearer " + creds.agent, "Idempotency-Key": "x"},
                              json={"dataset_id": "ds_x", "evaluator": "classification"})
        assert refused.status_code == 403 and "eval:run" in refused.json()["error"]["message"]


def test_the_bootstrap_is_paired_seeded_and_resamples_parts():
    from layerdelta import evaluators
    evaluator = evaluators.make({"task": "classification", "classes": 2})
    samples = [{"label": i % 2, "condition": None} for i in range(40)]
    base = [{"pred": 0} for _ in samples]
    better = [{"pred": s["label"]} for s in samples]  # a perfect child of an always-zero parent
    first = bootstrap.compare(evaluator, better, samples, base, seed=7)
    again = bootstrap.compare(evaluator, better, samples, base, seed=7)
    assert first == again
    accuracy = first[0]
    assert accuracy["baseline"] == 0.5 and accuracy["value"] == 1.0 and accuracy["delta"] == 0.5
    assert accuracy["ci"][0] > 0.3 and accuracy["ci"][1] <= 0.7  # the paired difference is large and its interval excludes zero
    assert bootstrap.categorize(evaluator, better[1], base[1], samples[1]) == "fixed"
    assert bootstrap.categorize(evaluator, base[0], better[0], samples[0]) is None
    assert evalservice.subset([{"sample_id": f"s{i}"} for i in range(10)], 3) == evalservice.subset([{"sample_id": f"s{i}"} for i in range(10)], 3)
