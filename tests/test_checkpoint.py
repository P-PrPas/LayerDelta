import hashlib
import json
import struct

import pytest
import torch
from safetensors.torch import save

from layerdelta.checkpoint import MAX_FILE_BYTES, MAX_HEADER_BYTES, load_checkpoint
from layerdelta.model import ReferenceCNN


def test_snapshot_exact_hash_finite_strict_and_independent(tmp_path):
    model = ReferenceCNN().eval()
    path = tmp_path / "model.safetensors"
    snapshot = save(model.state_dict(), metadata={"format": "pt"})
    path.write_bytes(snapshot)
    digest = hashlib.sha256(snapshot).hexdigest()
    first = load_checkpoint(path, digest)
    assert not first.training
    with torch.no_grad():
        first.classifier.bias.add_(1)
    second = load_checkpoint(path)
    assert torch.equal(second.classifier.bias, model.classifier.bias)
    assert not torch.equal(first.classifier.bias, second.classifier.bias)
    with pytest.raises(ValueError, match="SHA-256"):
        load_checkpoint(path, "0" * 64)
    model.classifier.bias.data[0] = float("nan")
    path.write_bytes(save(model.state_dict()))
    with pytest.raises(ValueError, match=r"Non-finite.*classifier.bias"):
        load_checkpoint(path)


@pytest.mark.parametrize("failure", ["keys", "dtype", "shape", "extent", "overlap", "trailing", "header_limit", "file_limit"])
def test_loader_validates_envelope_before_deserialization(tmp_path, monkeypatch, failure):
    snapshot = save(ReferenceCNN().state_dict())
    length, = struct.unpack_from("<Q", snapshot)
    header = json.loads(snapshot[8:8+length])
    payload = snapshot[8+length:]
    record = header["classifier.weight"]
    if failure == "keys":
        del header["classifier.bias"]
    elif failure == "dtype":
        record["dtype"] = "I32"
    elif failure == "shape":
        record["shape"].reverse()
    elif failure == "extent":
        record["data_offsets"][1] -= 4
    elif failure == "overlap":
        record["data_offsets"] = [v-4 for v in record["data_offsets"]]
    elif failure == "trailing":
        payload += bytes(4)
    encoded = json.dumps(header).encode()
    if failure == "header_limit":
        encoded = encoded.ljust(MAX_HEADER_BYTES+1)
    elif failure == "file_limit":
        payload += bytes(MAX_FILE_BYTES)
    path = tmp_path / "input"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    monkeypatch.setattr("layerdelta.checkpoint.load", lambda _: pytest.fail("loaded before validation"))
    with pytest.raises(ValueError):
        load_checkpoint(path)
