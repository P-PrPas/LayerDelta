"""The MNIST benchmark harness parts that need no host and no service (v0.5-E7a, design §5): tasks, arm R's kit, the edit rule, the audit,
the score. A seeded random ReferenceCNN and fake data stand in for the trained model and MNIST, so CI needs no download."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from layerdelta.model import ReferenceCNN

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bench"))
import mnist_bench as mb  # noqa: E402


@pytest.fixture(scope="module")
def world():
    torch.manual_seed(0)
    state = {k: v.clone() for k, v in ReferenceCNN().state_dict().items()}
    rng = np.random.default_rng(0)
    n = 400
    data = {"images": torch.from_numpy(rng.uniform(0, 1, size=(n, 1, 28, 28)).astype(np.float32)),
            "labels": torch.from_numpy(rng.integers(0, 10, size=n)), "conditions": ["blurred" if i % 2 else "normal" for i in range(n)],
            "sample_ids": [f"s{i}" for i in range(n)]}
    return state, data


def test_metrics_are_the_v01_definitions(world):
    _, data = world
    pred = data["labels"].numpy().copy()
    blurred_threes = np.nonzero((pred == 3) & np.array([c == "blurred" for c in data["conditions"]]))[0]
    pred[blurred_threes[:2]] = 5  # two blurred 3s read as 5
    assert mb.confusion_rate(pred, data, 3, 5) == pytest.approx(2 / len(blurred_threes))
    assert mb.accuracy(pred, data, "normal") == 1.0 and mb.accuracy(pred, data) == pytest.approx(1 - 2 / len(pred))
    task = {"target": {"metric": "confusion", "truth": 3, "confused_with": 5}}
    assert mb.metric(task, pred, data) == mb.confusion_rate(pred, data, 3, 5)
    values = mb.per_sample(task, pred, data)
    assert len(values) == len(blurred_threes) and values.mean() == pytest.approx(mb.confusion_rate(pred, data, 3, 5))


def test_planted_faults_and_their_undo(world):
    state, _ = world
    for stratum in ("S-vis", "S-inv"):
        for k in range(6):
            out, fault = mb.plant(state, stratum, np.random.default_rng([0, k]))
            layer, units = fault["layer"], fault["units"]
            changed = {name for name in state if not torch.equal(state[name], out[name])}
            assert changed <= {f"{layer}.weight", f"{layer}.bias"} and changed
            if fault["kind"] == "sign":  # invisible to norms: every per-unit L2 is unchanged
                w = f"{layer}.weight"
                assert torch.allclose(out[w].flatten(1).norm(dim=1), state[w].flatten(1).norm(dim=1)) and 2 <= len(units) <= 4
            if fault["undo"] is not None:  # the undo is a valid unit program on exactly the planted units
                assert [s["select"] for s in fault["undo"]] == [{"tensor": f"{layer}.{x}", "index": {"0": units}} for x in ("weight", "bias")]
                restored = {k: v.clone() for k, v in out.items()}
                for step in fault["undo"]:
                    restored[step["select"]["tensor"]][units] *= step["args"]["k"]
                assert all(torch.allclose(restored[n], state[n], rtol=1e-5, atol=1e-6) for n in state)


def test_generate_keeps_by_the_noise_scaled_gap_and_records_drops(world):
    state, data = world
    tasks = mb.generate(state, data, s_nat=3, planted_per_stratum=2)
    assert {t["stratum"] for t in tasks} >= {"S-vis", "S-inv"}
    for t in tasks:
        e = t["exploration"]
        assert t["kept"] == (e["gap"] >= e["min_gap"]) and e["min_gap"] >= mb.GAP_FLOOR[t["target"]["metric"]]
        assert e["min_gap"] >= mb.GAP_SE * e["se"] and (t["reason"] is None) == t["kept"]
        assert e["measured_on"] == ("half B" if t["stratum"] == "S-nat" else "all")
        if t["stratum"] == "S-nat" and t["reference_edit"] is not None:  # the search picks a single-unit zero on both weight and bias
            assert [s["op"] for s in t["reference_edit"]] == ["zero", "zero"]
    for stratum in ("S-vis", "S-inv"):
        assert sum(t["kept"] for t in tasks if t["stratum"] == stratum) <= 2


def test_the_gap_is_paired_and_its_se_is_of_the_difference(monkeypatch):
    # 400 blurred 3s: the start reads 40 of them as 5, the reference fixes 30 of those and breaks 10 others -> gap (40 - 20) / 400 = 0.05
    data = {"labels": torch.tensor([3] * 400), "conditions": ["blurred"] * 400, "images": torch.zeros(400, 1, 28, 28)}
    start_pred = np.array([5] * 40 + [3] * 360)
    ref_pred = np.array([5] * 10 + [3] * 30 + [5] * 10 + [3] * 350)
    monkeypatch.setattr(mb, "predict", lambda state, images: start_pred if state == "start" else ref_pred)
    task = {"target": {"metric": "confusion", "truth": 3, "confused_with": 5, "better": "lower"}}
    m = mb.measure_gap(task, "start", "ref", data)
    diff = -1 * ((ref_pred == 5).astype(float) - (start_pred == 5).astype(float))
    assert m["gap"] == pytest.approx(0.05) and m["se"] == pytest.approx(diff.std(ddof=1) / 20) and m["n"] == 400
    assert mb.min_gap(task, m) == pytest.approx(max(0.02, 3 * m["se"]))


def test_s_nat_is_chosen_on_one_half_and_measured_on_the_other(world):
    _, data = world
    a, b = mb.halves(data)
    assert not set(a["sample_ids"]) & set(b["sample_ids"]) and len(a["sample_ids"]) + len(b["sample_ids"]) == len(data["sample_ids"])
    assert mb.halves(data)[0]["sample_ids"] == a["sample_ids"]  # fixed by the sample ids, not by a random draw
    assert 0.35 < len(a["sample_ids"]) / len(data["sample_ids"]) < 0.65


def test_the_holdout_check_drops_a_wrong_sign_gap(world, monkeypatch):
    state, data = world
    task = {"stratum": "S-vis", "target": {"metric": "accuracy", "condition": None, "better": "higher"}, "reference_edit": None}
    monkeypatch.setattr(mb, "measure_gap", lambda task, start, ref, holdout: {"m_start": 0.9, "m_ref": 0.88, "gap": -0.02, "se": 0.01, "n": 100})
    checked = mb.holdout_check(task, state, state, data)
    assert not checked["kept"] and "holdout gap" in checked["reason"]
    monkeypatch.setattr(mb, "measure_gap", lambda task, start, ref, holdout: {"m_start": 0.8, "m_ref": 0.94, "gap": 0.14, "se": 0.01, "n": 100})
    assert mb.holdout_check(task, state, state, data)["kept"]


def test_apply_program_matches_the_ops(world):
    state, _ = world
    out, fault = mb.plant(state, "S-vis", np.random.default_rng([0, 1]))
    if fault["undo"] is not None:
        restored = mb.apply_program(out, fault["undo"])
        assert all(torch.allclose(restored[n], state[n], rtol=1e-5, atol=1e-6) for n in state)
    zeroed = mb.apply_program(state, [{"op": "zero", "args": {}, "select": {"tensor": "features.block1.conv.weight", "index": {"0": [3]}}}])
    assert zeroed["features.block1.conv.weight"][3].abs().sum() == 0 and torch.equal(zeroed["features.block1.conv.weight"][2], state["features.block1.conv.weight"][2])


def test_arm_r_kit_is_complete_and_self_contained(tmp_path, world):
    state, data = world
    task = {"stratum": "S-nat", "target": {"metric": "confusion", "truth": 9, "confused_with": 4, "better": "lower"}}
    hashes = mb.write_r_kit(tmp_path / "s1", state, data, task, "60 minutes and 80 steps")
    assert set(hashes) == {"start.safetensors", "model.py", "exploration.npz", "TASK.md"}
    text = (tmp_path / "s1" / "TASK.md").read_text(encoding="utf-8")
    assert "No gradient training" in text and "reads the digit **9** as **4**" in text and "final.safetensors" in text
    assert "LayerDelta" not in text  # arm R never hears of the tools
    source = (tmp_path / "s1" / "model.py").read_text(encoding="utf-8")
    assert "layerdelta" not in source.lower()
    namespace: dict = {}
    exec(compile(source, "model.py", "exec"), namespace)  # noqa: S102  our own generated source
    from safetensors.torch import load_file
    model = namespace["ReferenceCNN"]().eval()
    model.load_state_dict(load_file(str(tmp_path / "s1" / "start.safetensors")))
    kit = np.load(tmp_path / "s1" / "exploration.npz")
    with torch.no_grad():
        assert np.array_equal(model(torch.from_numpy(kit["images"][:16])).argmax(1).numpy(), mb.predict(state, data["images"][:16]))
    assert kit["conditions"].tolist() == data["conditions"] and np.array_equal(kit["labels"], data["labels"].numpy())


def test_arm_l_text_names_the_tools_and_the_final_line():
    task = {"stratum": "S-nat", "target": {"metric": "confusion", "truth": 9, "confused_with": 4, "better": "lower"}}
    text = mb.task_text(task, "L", "60 minutes", checkpoint_id="abc", dataset_id="mnist-v1")
    assert "FINAL: <child checkpoint id>" in text and "`abc`" in text and '"truth": 9' in text and "No gradient training" in text


def test_check_final_the_edit_rule(world):
    state, _ = world
    edited = {k: v.clone() for k, v in state.items()}
    edited["features.block1.conv.weight"][2] = 0
    ok = mb.check_final(state, edited)
    assert ok["valid"] and not ok["over_cap"] and ok["changed_elements"] == 9  # one 1x3x3 kernel
    trained = {k: v + 1e-3 for k, v in state.items()}  # what training looks like: everything moves a little
    assert mb.check_final(state, trained)["over_cap"]
    renamed = dict(state)
    renamed["extra"] = torch.zeros(1)
    assert not mb.check_final(state, renamed)["valid"]
    retyped = {k: v.double() for k, v in state.items()}
    assert not mb.check_final(state, retyped)["valid"]


@pytest.mark.parametrize("code, kind", [
    ("loss.backward()", "training"), ("opt = torch.optim.Adam(model.parameters())", "training"), ("model.fit(x, y)", "training"),
    ("w.requires_grad_(True)", "training"), ("g = torch.autograd.grad(loss, w)", "training"), ("optimizer.zero_grad()", "training"),
    ("curl -O https://example.org/mnist.gz", "download"), ("pip install torchvision", "download"),
    ("from torchvision.datasets import MNIST", "download"), ("fetch_openml('mnist_784')", "download"),
])
def test_the_audit_flags_training_and_downloads(code, kind):
    found = mb.audit_text(code)
    assert found[kind], code
    assert not found["training" if kind == "download" else "download"] or "torchvision" in code or "https" in code


def test_the_audit_lets_plain_editing_through():
    edit = "w = load_file('start.safetensors')\nw['features.block2.conv.weight'][5] *= -1\nsave_file(w, 'final.safetensors')\npred = model(x).argmax(1)"
    assert mb.audit_text(edit) == {"training": [], "download": []}


def test_score_is_unclipped_with_intention_to_treat_and_the_guardrail():
    assert mb.score({}, 0.02, 0.09, 0.01, 0.96, 0.96, 0.01)["R"] == pytest.approx(0.875)  # confusion 0.09 -> 0.02 of a possible 0.08
    assert mb.score({}, 0.12, 0.09, 0.01, 0.96, 0.96, 0.01)["R"] == pytest.approx(-0.375)  # worse than the start: below 0
    assert mb.score({}, None, 0.09, 0.01, None, 0.96, 0.01) == {"R": 0.0, "R_guarded": 0.0, "valid_final": False}
    broke = mb.score({}, 0.02, 0.09, 0.01, 0.90, 0.96, 0.01)
    assert broke["R"] > 0 and broke["R_guarded"] == 0.0 and broke["guardrail_broken"]
    assert json.loads(json.dumps(broke)) == broke
