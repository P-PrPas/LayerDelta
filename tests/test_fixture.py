import json

import pytest
import torch
from safetensors.torch import load_file

from layerdelta import fixture
from layerdelta.model import ReferenceCNN, describe_model, target_catalog


def _variant(source, condition, label):
    return {
        "sample_id": f"{source}/{condition}",
        "source_sample_id": source,
        "condition": condition,
        "label": label,
        "split": "exploration",
    }


def test_adapter_describes_actual_shapes_and_targets():
    model = ReferenceCNN()
    description = describe_model(model)
    layers = {layer["layer_id"]: layer for layer in description["layers"]}
    assert layers["features.block1.relu"]["output_shape"] == [8, 28, 28]
    assert layers["features.block2.relu"]["output_shape"] == [24, 14, 14]
    assert layers["classifier"]["read_only"]
    for target in target_catalog(model).values():
        for valid in (0, target.channel_count - 1):
            target.check_channel(valid)
        for invalid in (-1, target.channel_count, True):
            with pytest.raises(ValueError):
                target.check_channel(invalid)
        assert target.target_id in layers[target.module_path]["target_ids"]
        if target.mode == "weight":
            assert model.get_parameter(target.parameter_name).shape[0] == target.channel_count
    assert all(not mod._forward_hooks for mod in model.modules())


def test_demo_counts_use_all_true_class_rows_and_stable_pairing():
    rows = [_variant(src, condition, label)
            for condition in ("blurred", "normal")
            for src, label in (("a", 8), ("b", 8), ("c", 8), ("d", 3))]
    result = fixture.exploration_evidence(rows, [3, 3, 8, 8, 3, 8, 8, 3])
    demo = result["demo"]
    assert (demo["truth"], demo["baseline_prediction"], demo["pair_count"]) == (8, 3, 2)
    assert demo["primary_slice"] == {"condition": "blurred", "truth": 8, "N": 3}
    assert demo["guardrail_slice"] == {"condition": "normal", "N": 4}
    assert [pair["source_sample_id"] for pair in demo["example_pairs"]] == ["b", "a"]
    assert result["conditions"]["normal"]["correct"] == 3
    rows[0]["split"] = "holdout"
    with pytest.raises(ValueError, match="exploration samples"):
        fixture.exploration_evidence(rows, [3] * len(rows))


def test_build_repeats_and_loads_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(fixture, "download_mnist", lambda _: None)
    monkeypatch.setattr(fixture, "build_manifest", lambda _: {"manifest_sha256": "abc"})
    images = torch.zeros((20, 1, 28, 28))
    truth = torch.arange(10).repeat_interleave(2)
    rows = [_variant(str(label), condition, label)
            for label in range(10) for condition in ("normal", "blurred")]
    requested = []

    def load_mock_split(cache, manifest, split):
        requested.append(split)
        return images, truth, rows

    monkeypatch.setattr(fixture, "load_split", load_mock_split)
    first = tmp_path / "first"
    actual = fixture.build_fixture(first, tmp_path / "cache")
    second = fixture.build_fixture(tmp_path / "second", tmp_path / "cache")
    assert requested == ["train", "exploration"] * 2
    assert actual == second == json.loads((first / "fixture.json").read_bytes())
    model = ReferenceCNN()
    model.load_state_dict(load_file(first / "model.safetensors"), strict=True)
    with torch.inference_mode():
        predictions = model(images).argmax(1).tolist()
    assert fixture.exploration_evidence(rows, predictions) == json.loads((first / "exploration.json").read_bytes())
    with pytest.raises(FileExistsError):
        fixture.build_fixture(first, tmp_path / "cache")
