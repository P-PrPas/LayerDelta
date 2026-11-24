"""Scripted controls (v0.5-E7a, design §5.5): harness symmetry over HTTP, the policies, and the positive control's detectability."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import load_file

from layerdelta import api
from test_v2_api import BASE, REF, make_workspace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bench"))
import analysis  # noqa: E402
import controls  # noqa: E402
import mnist_bench as mb  # noqa: E402


@pytest.fixture
def service(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path, monkeypatch)
    creds = api.Credentials("demo", weights_read=True, weights_write=True)
    with TestClient(api.create_app(workspace, creds, quota=50), base_url="http://127.0.0.1") as client:
        client.post("/api/v1/session", json={"ticket": creds.ticket})
        yield client, {"Authorization": "Bearer " + creds.agent}, workspace


@pytest.mark.parametrize("seed", range(6))
def test_harness_symmetry_arm_l_and_arm_r_give_the_same_bits(service, seed):
    """The same program through LayerDelta (plan, apply: arm L) and on the file (arm R) gives a bit-identical checkpoint."""
    client, agent, workspace = service
    start = load_file(str(workspace / "reference" / "model.safetensors"))
    program = controls.random_policy({}, start, np.random.default_rng(seed))
    planned = client.post(BASE + f"/checkpoints/{REF}/plan", headers=agent, json={"program": program, "hypothesis": "symmetry"})
    assert planned.status_code == 200, planned.text
    applied = client.post(BASE + f"/checkpoints/{REF}/apply", headers={**agent, "Idempotency-Key": f"sym-{seed}"},
                          json={"recipe": planned.json()["result"]["recipe"]})
    assert applied.status_code == 200, applied.text
    child = applied.json()["result"]["diff_hint"]["other"]
    through_l = load_file(str(workspace / "checkpoints" / f"{child}.safetensors"))
    assert controls.same_bits(through_l, mb.apply_program(start, program))


@pytest.fixture(scope="module")
def world():
    torch.manual_seed(0)
    from layerdelta.model import ReferenceCNN
    base = {k: v.clone() for k, v in ReferenceCNN().state_dict().items()}
    rng = np.random.default_rng(0)
    n = 300
    data = {"images": torch.from_numpy(rng.uniform(0, 1, size=(n, 1, 28, 28)).astype(np.float32)),
            "labels": torch.from_numpy(rng.integers(0, 10, size=n)), "conditions": ["blurred" if i % 2 else "normal" for i in range(n)],
            "sample_ids": [f"s{i}" for i in range(n)]}
    return base, data


def test_the_stats_policy_undoes_a_planted_scale(world):
    base, _ = world
    for k in range(12):
        start, fault = mb.plant(base, "S-vis", np.random.default_rng([0, 1, k]))
        if fault["kind"] != "scale":
            continue
        program = controls.stats_policy({}, start)
        touched = {(s["select"]["tensor"].rsplit(".", 1)[0], u) for s in program for u in s["select"]["index"]["0"]}
        assert touched == {(fault["layer"], u) for u in fault["units"]}, (fault, touched)
        repaired = mb.apply_program(start, program)
        assert np.allclose(controls.unit_l2(repaired, fault["layer"])[fault["units"]], np.median(controls.unit_l2(start, fault["layer"])), rtol=1e-4)


def test_the_stats_policy_declines_a_clean_model_and_the_random_policy_is_seeded(world):
    base, _ = world
    assert controls.stats_policy({}, base) == []
    one, two = (controls.random_policy({}, base, np.random.default_rng(5)) for _ in range(2))
    assert one == two and one and {s["op"] for s in one} <= {"zero", "scale"}


def test_positive_control_logic_detects_an_effect_built_in(world):
    """On planted scale faults, the stats policy must beat the random policy by an interval that excludes 0 (here on fake data: the
    pipeline policy -> outcome -> analysis, not MNIST's magnitudes)."""
    base, data = world
    pairs = []
    for k in range(10):
        start, fault = mb.plant(base, "S-vis", np.random.default_rng([0, 1, k]))
        task = {"stratum": "S-vis", "target": {"metric": "accuracy", "condition": None, "better": "higher"}, "reference_edit": fault["undo"]}
        gap = mb.measure_gap(task, start, base, data)["gap"]
        if gap < 0.02:
            continue
        strong = controls.outcome(task, start, base, controls.stats_policy(task, start), data, guard_margin=1.0)["R"]
        for s in range(3):
            weak = controls.outcome(task, start, base, controls.random_policy(task, start, np.random.default_rng([k, s])), data, guard_margin=1.0)["R"]
            pairs.append({"instance": f"S-vis-{k}", "R_L": strong, "R_R": weak})
    assert pairs, "no planted fault moved the fake model enough"
    result = analysis.primary(pairs, resamples=1000, permutations=1000)
    assert result["ci"][0] > 0, result
