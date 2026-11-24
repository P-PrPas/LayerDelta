"""Harness symmetry, the scoring half (v0.5-E7a, design §5.5 and V10): LayerDelta's person-only final evaluation on mnist-v1 holdout
gives the same numbers as the harness's in-process scorer (tools/bench/mnist_bench.py) for the task's metric and the guardrail. Both arms
are scored by the final evaluation, so this pins the in-process numbers used at generation and in the controls to what the run reports."""

import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.torch import load_file

from test_evaluate import final, wait
from test_mnist_evaluate import world  # noqa: F401  the E4e fixture: tiny verified idx archives, reference_cnn attached, one child
from test_v2_api import REF

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bench"))
import mnist_bench as mb  # noqa: E402


def test_the_final_evaluation_and_the_harness_score_a_checkpoint_alike(world):  # noqa: F811
    client, agent, human, child, manifest, workspace = world
    holdout = mb.load_exploration(workspace / "mnist", manifest, "holdout")
    parent, made = load_file(str(workspace / "reference" / "model.safetensors")), load_file(str(workspace / "checkpoints" / f"{child}.safetensors"))
    pred = mb.predict(made, holdout["images"])
    labels = holdout["labels"].numpy()
    blurred = np.array([c == "blurred" for c in holdout["conditions"]])
    errors = [(int(t), int(p)) for t, p in zip(labels[blurred], pred[blurred]) if t != p]
    truth, confused = max(set(errors), key=errors.count) if errors else (0, 1)  # a pair with errors, so the rate is not trivially 0
    recipe = client.get(f"/api/v1/projects/demo/checkpoints/{child}", headers=human).json()["recipe_hash"]
    started = final(client, human, child, "mnist-v1", "score-1", recipe_hash=recipe,
                    evaluator={"task": "classification", "confusion": {"truth": truth, "confused_with": confused, "condition": "blurred"}})
    assert started.status_code == 202, started.text
    done = wait(client, human, f"/evaluations/{started.json()['evaluation_id']}")
    assert done["status"] == "succeeded" and done["baseline_id"] == REF, done
    metrics = {m["name"]: m for m in done["metrics"]}
    rate = metrics[f"confusion:{truth}->{confused}:blurred"]
    assert rate["value"] == pytest.approx(mb.confusion_rate(pred, holdout, truth, confused))
    assert rate["baseline"] == pytest.approx(mb.confusion_rate(mb.predict(parent, holdout["images"]), holdout, truth, confused))
    assert metrics["accuracy:normal"]["value"] == pytest.approx(mb.accuracy(pred, holdout, "normal"))  # the guardrail
    assert metrics["accuracy"]["value"] == pytest.approx(mb.accuracy(pred, holdout))  # the planted faults' metric
