"""Load a tensor-only checkpoint for the registered MNIST architecture."""

import hashlib
import json
import struct
from pathlib import Path

import torch
from safetensors import SafetensorError
from safetensors.torch import load

from .model import ReferenceCNN


# The reference architecture needs under 55 KB of parameters plus string metadata.
MAX_FILE_BYTES = 1 << 20
MAX_HEADER_BYTES = 16 << 10


def _unique_fields(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate checkpoint field: {name}")
        result[name] = value
    return result


def load_checkpoint(path: Path, expected_sha256: str | None = None) -> ReferenceCNN:
    """Validate and load the same bounded snapshot, even if the file is replaced."""
    with Path(path).open("rb") as stream:
        contents = stream.read(MAX_FILE_BYTES + 1)
    if len(contents) > MAX_FILE_BYTES or len(contents) < 8:
        raise ValueError("Invalid checkpoint file size")
    if expected_sha256 is not None and hashlib.sha256(contents).hexdigest() != expected_sha256:
        raise ValueError("Checkpoint SHA-256 mismatch")
    size = struct.unpack_from("<Q", contents)[0]
    if not 2 <= size <= min(MAX_HEADER_BYTES, len(contents) - 8):
        raise ValueError("Invalid checkpoint header size")
    try:
        header = json.loads(contents[8:8 + size].decode("utf-8"), object_pairs_hook=_unique_fields)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Invalid checkpoint header JSON") from exc
    if not isinstance(header, dict):
        raise ValueError("Invalid checkpoint header")
    metadata = header.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(type(v) is not str for v in metadata.values()):
        raise ValueError("Invalid checkpoint string metadata")

    model = ReferenceCNN().float().cpu()
    schema = model.state_dict()
    if set(header) != set(schema):
        raise ValueError(f"Checkpoint key set mismatch: {sorted(set(header) ^ set(schema))}")
    intervals = []
    for name, tensor in schema.items():
        record = header[name]
        if not isinstance(record, dict) or set(record) != {"dtype", "shape", "data_offsets"}:
            raise ValueError(f"Invalid tensor header for {name}")
        shape = record["shape"]
        if (not isinstance(shape, list) or any(type(n) is not int for n in shape)
                or shape != list(tensor.shape) or record["dtype"] != "F32"):
            raise ValueError(f"Checkpoint dtype/shape mismatch for {name}")
        offsets = record["data_offsets"]
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(i) is not int for i in offsets)
                or offsets[1] - offsets[0] != tensor.numel() * tensor.element_size()):
            raise ValueError(f"Checkpoint byte extent mismatch for {name}")
        intervals.append((*offsets, name))
    end = 0
    for start, stop, name in sorted(intervals):
        if start != end:
            raise ValueError(f"Checkpoint has overlapping or missing bytes at {name}")
        end = stop
    if end != len(contents) - 8 - size:
        raise ValueError("Checkpoint data size mismatch")

    try:
        state = load(contents)
    except SafetensorError as exc:
        raise ValueError("Invalid Safetensors encoding") from exc
    for name, tensor in state.items():
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite tensor in checkpoint: {name}")
    model.load_state_dict(state, strict=True)
    return model.eval()
