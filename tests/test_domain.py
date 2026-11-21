from copy import deepcopy
from dataclasses import asdict

import pytest

from layerdelta.domain import DependencyMismatch, canonical_hash, freeze_recipe, normalize_intervention, verify_recipe
from layerdelta.model import ReferenceCNN, target_catalog


@pytest.fixture
def fixture():
    return {
        "model": {"architecture_id": "mnist-two-conv", "architecture_revision": 1, "adapter_version": 1,
                  "targets": [asdict(t) for t in target_catalog(ReferenceCNN()).values()]},
        "artifacts": {"model.safetensors": "a" * 64, "dataset.json": "b" * 64, "exploration.json": "d" * 64},
        "training": {"seed": 20261122},
        "manifest_sha256": "c" * 64,
    }


SPEC = {"target_id": "features.block2.relu.output", "mode": "activation", "channel_index": 17,
        "operation": "scale", "scale": 1}


def test_golden_recipe_replays_across_workspaces_and_devices(fixture):
    recipe = freeze_recipe(fixture, SPEC, "Reduce channel 17 to test digit ๙.")
    assert recipe["recipe_hash"] == "bbfce16650a76ab3e9aa81ff740fafb7586706e8f254b79d18b7e8b8d406e822"
    clone = {**deepcopy(fixture), "workspace": "another workspace", "runtime": {"device": "cuda:0"}}
    assert verify_recipe(clone, recipe) == recipe
    assert freeze_recipe(clone, {**SPEC, "scale": 1.0}, recipe["hypothesis"]) == recipe
    assert freeze_recipe(clone, SPEC, "Another hypothesis")["recipe_hash"] != recipe["recipe_hash"]
    assert normalize_intervention({**SPEC, "scale": -0.0}) == {**SPEC, "scale": 0.0}


@pytest.mark.parametrize("changes", [
    {"scale": float("inf")}, {"scale": float("nan")}, {"scale": -0.1}, {"scale": True},
    {"scale": 2.01}, {"scale": "1"}, {"channel_index": 24}, {"channel_index": -1},
    {"channel_index": False}, {"operation": "zero"}, {"mode": "weight"}, {"mode": None},
    {"target_id": "features.block2.relu"}, {"target_id": {}}, {"unsupported": 1},
])
def test_intervention_is_strict(changes):
    with pytest.raises(ValueError):
        normalize_intervention({**SPEC, **changes})


def test_import_checks_digest_then_reports_each_unresolved_dependency(fixture):
    recipe = freeze_recipe(fixture, SPEC, "preserve baseline")
    recipe["mapping"]["channel_axis"] = 0
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_recipe(fixture, recipe)
    recipe["recipe_hash"] = canonical_hash({k: v for k, v in recipe.items() if k != "recipe_hash"})
    fixture["artifacts"]["model.safetensors"] = "e" * 64
    fixture["manifest_sha256"] = "f" * 64
    with pytest.raises(DependencyMismatch) as error:
        verify_recipe(fixture, recipe)
    assert set(error.value.fields) == {"checkpoint_sha256", "manifest_sha256", "mapping"}
