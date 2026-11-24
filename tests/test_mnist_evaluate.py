"""mnist-v1 in evaluate, probe and the final evaluation (v0.5-E4e). Its samples are rows of the verified idx archives (normal and blurred), loaded by the
server and handed to the worker as arrays; nothing is decoded. The archives here are small ones the test writes, with their digests patched in."""

import gzip
import hashlib
import struct

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import load_file

from layerdelta import api, data as mnist
from layerdelta.model import ReferenceCNN
from test_evaluate import EDIT, evaluate, final, wait
from test_v2_api import BASE, REF, make_workspace


def idx_images(pixels):
    return gzip.compress(struct.pack(">IIII", 2051, len(pixels), 28, 28) + pixels.tobytes())


def idx_labels(labels):
    return gzip.compress(struct.pack(">II", 2049, len(labels)) + bytes(labels))


@pytest.fixture
def world(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path, monkeypatch)
    cache = workspace / "mnist"
    cache.mkdir()
    rng = np.random.default_rng(8)
    files = {"train-images-idx3-ubyte.gz": idx_images(rng.integers(0, 256, (20, 28, 28), dtype=np.uint8)),
             "train-labels-idx1-ubyte.gz": idx_labels([i % 10 for i in range(20)]),
             "t10k-images-idx3-ubyte.gz": idx_images(rng.integers(0, 256, (40, 28, 28), dtype=np.uint8)),
             "t10k-labels-idx1-ubyte.gz": idx_labels([i % 10 for i in range(40)])}
    for name, body in files.items():
        (cache / name).write_bytes(body)
        monkeypatch.setitem(mnist.SOURCE_SHA256, name, hashlib.sha256(body).hexdigest())
    manifest = mnist.build_manifest(cache)
    fixture = {"artifacts": {"model.safetensors": REF, "dataset.json": "b" * 64, "exploration.json": "c" * 64}, "manifest_sha256": "d" * 64,
               "training": {"seed": 1}, "model": api.load_fixture(None)[0]["model"]}
    monkeypatch.setattr(api, "load_fixture", lambda _: (fixture, manifest, {}))
    creds = api.Credentials("demo", weights_read=True, weights_write=True, eval_run=True)
    with TestClient(api.create_app(workspace, creds, quota=50), base_url="http://127.0.0.1") as client:
        human = {"X-CSRF-Token": client.post("/api/v1/session", json={"ticket": creds.ticket}).json()["csrf"]}
        agent = {"Authorization": "Bearer " + creds.agent}
        attached = client.post(BASE + f"/checkpoints/{REF}/graphs", headers=human, json={"provider": "builtin", "model": "reference_cnn"})
        assert wait(client, human, f"/graph-attachments/{attached.json()['attach_id']}")["status"] == "succeeded"
        child = client.post(BASE + f"/checkpoints/{REF}/apply", headers={**agent, "Idempotency-Key": "child"}, json=EDIT).json()["result"]["manifest"]["child_sha256"]
        yield client, agent, human, child, manifest, workspace


def eager_accuracy(workspace, manifest, condition, checkpoint_state):
    images, labels, rows = mnist.load_split(workspace / "mnist", manifest, "exploration")
    model = ReferenceCNN().eval()
    model.load_state_dict(checkpoint_state)
    with torch.no_grad():
        predicted = model(images).argmax(dim=1)
    keep = torch.tensor([r["condition"] == condition for r in rows])
    return float((predicted[keep] == labels[keep]).float().mean())


def test_an_evaluation_on_mnist_matches_the_eager_model_per_condition(world):
    client, agent, human, child, manifest, workspace = world
    absolute = wait(client, agent, f"/evaluations/{evaluate(client, agent, REF, 'mnist-v1', 'm1', baseline=None).json()['evaluation_id']}")
    assert absolute["status"] == "succeeded", absolute
    assert absolute["dataset_id"] == "mnist-v1" and absolute["counts"]["samples"] == 40 and absolute["evaluator"]["classes"] == 10
    metrics = {m["name"]: m for m in absolute["metrics"]}
    state = load_file(workspace / "reference" / "model.safetensors")
    for condition in ("normal", "blurred"):
        assert metrics[f"accuracy:{condition}"]["value"] == pytest.approx(eager_accuracy(workspace, manifest, condition, state))
    assert len(metrics["accuracy"]["per_class"]) == 10
    compared = wait(client, agent, f"/evaluations/{evaluate(client, agent, child, 'mnist-v1', 'm2').json()['evaluation_id']}")  # a child against its parent
    assert compared["status"] == "succeeded" and compared["baseline_id"] == REF and compared["counts"]["samples"] == 40
    page = client.get(BASE + f"/evaluations/{compared['evaluation_id']}/changes", headers=agent, params={"limit": 3})
    assert page.status_code == 200 and page.json()["total"] == sum(compared["counts"][k] for k in ("fixed", "broken", "changed"))
    events = client.get(BASE + "/trace", headers=agent, params={"limit": 500}).json()["result"]["events"]
    assert next(e for e in events if e.get("evaluation_id") == compared["evaluation_id"] and e["tool"] == "evaluate")["dataset_id"] == "mnist-v1"


def test_a_person_can_run_the_final_evaluation_of_a_root_and_a_child_on_mnist_holdout(world):
    client, agent, human, child, manifest, workspace = world
    root = final(client, human, REF, "mnist-v1", "f1", baseline=None)  # a root: confirmed with a null recipe hash
    assert root.status_code == 202 and root.json()["recipe_hash"] is None and root.json()["split"] == "holdout"
    done = wait(client, human, f"/evaluations/{root.json()['evaluation_id']}")
    assert done["status"] == "succeeded" and done["counts"]["samples"] == 40
    recipe = client.get(BASE + f"/checkpoints/{child}", headers=human).json()["recipe_hash"]
    again = final(client, human, child, "mnist-v1", "f2", recipe_hash=recipe)
    assert wait(client, human, f"/evaluations/{again.json()['evaluation_id']}")["status"] == "succeeded"
    assert evaluate(client, agent, child, "mnist-v1", "f3", split="holdout").status_code == 403  # an agent still cannot


def test_a_probe_on_mnist_aggregates_the_same_images(world):
    client, agent, human, child, manifest, workspace = world
    started = client.post(BASE + f"/checkpoints/{REF}/probes", headers={**agent, "Idempotency-Key": "p1"}, json={"dataset_id": "mnist-v1", "nodes": ["n0"],
                                                                                                                    "split_by": "condition"})
    done = wait(client, agent, f"/probes/{started.json()['probe_id']}")
    assert done["status"] == "succeeded" and set(done["nodes"][0]["groups"]) == {"normal", "blurred"} and done["nodes"][0]["units"] == 8
    assert done["nodes"][0]["groups"]["normal"]["n"] == 20 * 28 * 28
