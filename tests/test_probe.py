import json

import pytest
import torch

from layerdelta.probe import probe


def test_cuda_request_on_cpu_only_runtime_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        probe(tmp_path, tmp_path, "cuda")


def test_modified_weights_rejected_before_inference(tmp_path):
    fixture = {"artifacts": {"model.safetensors": "0" * 64}}
    (tmp_path / "fixture.json").write_text(json.dumps(fixture), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"modified")
    with pytest.raises(ValueError, match="digest mismatch: model.safetensors"):
        probe(tmp_path, tmp_path, "cpu")
