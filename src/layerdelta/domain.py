"""Portable, versioned recipes for single-channel MNIST experiments."""

import hashlib
import json
import re
from dataclasses import asdict
from functools import lru_cache

import torch

from .data import BLUR_KERNEL
from .model import ADAPTER_VERSION, ARCHITECTURE_ID, ARCHITECTURE_REVISION, ReferenceCNN, target_catalog


EVALUATOR_REVISION = "paired-classification-v1"


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DependencyMismatch(ValueError):
    def __init__(self, fields: list[str]):
        self.fields = fields
        super().__init__("Recipe dependency mismatch: " + ", ".join(fields))


@lru_cache(maxsize=1)
def _targets():
    # Resolve the installed adapter's shapes without allocating or randomizing weights.
    with torch.device("meta"):
        return target_catalog(ReferenceCNN())


def normalize_intervention(spec: dict) -> dict:
    if not isinstance(spec, dict):
        raise ValueError("intervention must be an object")
    operation = spec.get("operation")
    if operation not in ("zero", "scale"):
        raise ValueError("operation must be zero or scale")
    fields = {"target_id", "mode", "channel_index", "operation"}
    if operation == "scale":
        fields.add("scale")
    if fields != spec.keys():
        raise ValueError(f"Invalid intervention fields: {sorted(fields ^ spec.keys())}")
    target_id = spec["target_id"]
    target = _targets().get(target_id) if isinstance(target_id, str) else None
    if target is None or spec["mode"] != target.mode:
        raise ValueError("Invalid target_id for intervention mode")
    target.check_channel(spec["channel_index"])
    normalized = dict(spec)
    if operation == "scale":
        scale = spec["scale"]
        if type(scale) not in (float, int) or not 0 <= scale <= 2:
            raise ValueError("scale must be finite and within [0, 2]")
        normalized["scale"] = 0.0 if scale == 0 else float(scale)
    return normalized


def _dependencies(fixture: dict) -> dict:
    model = fixture["model"]
    dependencies = {
        "architecture_id": model["architecture_id"],
        "architecture_revision": model["architecture_revision"],
        "adapter_version": model["adapter_version"],
        "checkpoint_sha256": fixture["artifacts"]["model.safetensors"],
        "manifest_sha256": fixture["manifest_sha256"],
        "dataset_file_sha256": fixture["artifacts"]["dataset.json"],
        "demo_evidence_sha256": fixture["artifacts"]["exploration.json"],
        "training_seed": fixture["training"]["seed"],
        "dataset_revision_id": "mnist-v1",
        "preprocessing": {"revision": "box-blur-v1", "kernel_width": BLUR_KERNEL,
                          "stride": 1, "padding": "zeros", "normalization": "float32 pixel / 255"},
        "evaluator_revision": EVALUATOR_REVISION,
    }
    for key, value in dependencies.items():
        if key.endswith("_sha256") and (not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value)):
            raise ValueError(f"Invalid fixture digest at {key}")
    expected = {"architecture_id": ARCHITECTURE_ID, "architecture_revision": ARCHITECTURE_REVISION,
                "adapter_version": ADAPTER_VERSION}
    unsupported = [k for k, v in expected.items() if type(dependencies[k]) != type(v) or dependencies[k] != v]
    if unsupported:
        raise DependencyMismatch(unsupported)
    return dependencies


def freeze_recipe(fixture: dict, intervention: dict, hypothesis: str) -> dict:
    if type(hypothesis) is not str or not hypothesis.strip() or len(hypothesis) > 4096:
        raise ValueError("hypothesis must contain 1 to 4096 characters")
    spec = normalize_intervention(intervention)
    target = _targets()[spec["target_id"]]
    mapping = asdict(target)
    fixture_target = next((t for t in fixture["model"]["targets"] if t["target_id"] == target.target_id), None)
    if canonical_hash(fixture_target) != canonical_hash(mapping):
        raise DependencyMismatch(["mapping"])
    recipe = {
        "schema_version": 1,
        "hypothesis": hypothesis,
        "dependencies": _dependencies(fixture),
        "evaluation": {
            "split": "exploration",
            "condition_ids": ["normal", "blurred"],
            "metric_ids": ["accuracy", "confusion_matrix"],
            "slice_ids": ["overall", "normal", "blurred", "primary", "baseline_error"],
        },
        "intervention": spec,
        "mapping": mapping,
    }
    return {**recipe, "recipe_hash": canonical_hash(recipe)}


def verify_recipe(fixture: dict, recipe: dict) -> dict:
    if not isinstance(recipe, dict) or type(recipe.get("schema_version")) is not int or recipe["schema_version"] != 1:
        raise ValueError("Unsupported recipe schema_version")
    fields = {"schema_version", "hypothesis", "dependencies", "evaluation", "intervention", "mapping", "recipe_hash"}
    if fields != recipe.keys() or not isinstance(recipe["dependencies"], dict):
        raise ValueError("Invalid frozen recipe fields")
    payload = {key: value for key, value in recipe.items() if key != "recipe_hash"}
    if canonical_hash(payload) != recipe["recipe_hash"]:
        raise ValueError("Recipe hash mismatch")
    expected = freeze_recipe(fixture, recipe["intervention"], recipe["hypothesis"])
    supplied, required = recipe["dependencies"], expected["dependencies"]
    different = [key for key in sorted(supplied.keys() | required.keys())
                 if key not in supplied or key not in required or canonical_hash(supplied[key]) != canonical_hash(required[key])]
    if canonical_hash(recipe["mapping"]) != canonical_hash(expected["mapping"]):
        different.append("mapping")
    if different:
        raise DependencyMismatch(different)
    if expected["recipe_hash"] != recipe["recipe_hash"]:
        raise ValueError("Noncanonical intervention or unsupported evaluation config")
    return expected
