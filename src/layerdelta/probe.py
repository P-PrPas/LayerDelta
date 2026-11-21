"""Run one frozen exploration pair on a requested device in a fresh CLI process."""

import argparse
import json
import os
import platform
from pathlib import Path

import torch
from safetensors.torch import load_file

from .data import load_split
from .fixture import file_sha256
from .model import ReferenceCNN


def probe(fixture_dir: Path, data_dir: Path, device: str) -> dict:
    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in this worker")

    fixture = json.loads((fixture_dir / "fixture.json").read_bytes())
    for name in ("model.safetensors", "exploration.json", "dataset.json"):
        if file_sha256(fixture_dir / name) != fixture["artifacts"][name]:
            raise ValueError(f"Fixture digest mismatch: {name}")
    dataset = json.loads((fixture_dir / "dataset.json").read_bytes())
    if fixture["manifest_sha256"] != dataset["manifest_sha256"]:
        raise ValueError("Dataset manifest does not match the reference fixture")
    exploration = json.loads((fixture_dir / "exploration.json").read_bytes())
    example = exploration["demo"]["example_pairs"][0]
    torch.set_num_threads(1)
    images, _, rows = load_split(data_dir, dataset, "exploration")
    index_by_id = {row["sample_id"]: index for index, row in enumerate(rows)}
    sample_ids = [example[condition]["sample_id"] for condition in ("normal", "blurred")]
    pair = images[[index_by_id[sample_id] for sample_id in sample_ids]].to(device)

    model = ReferenceCNN()
    model.load_state_dict(load_file(str(fixture_dir / "model.safetensors")), strict=True)
    model.to(device).eval()
    with torch.inference_mode():
        logits = model(pair)
    if logits.shape != (2, 10) or not torch.isfinite(logits).all().item():
        raise ValueError("Expected ten finite logits for each exploration image")
    actual = logits.device
    result = {
        "requested_device": device,
        "actual_device": str(actual),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
        "parent_sha256": fixture["artifacts"]["model.safetensors"],
        "manifest_sha256": fixture["manifest_sha256"],
        "sample_ids": sample_ids,
        "predictions": logits.argmax(1).tolist(),
        "logits": logits.cpu().tolist(),
        "worker_pid": os.getpid(),
    }
    if actual.type == "cuda":
        properties = torch.cuda.get_device_properties(actual)
        result.update(gpu=properties.name, vram_bytes=properties.total_memory)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    try:
        result = probe(args.fixture, args.data_dir, args.device)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Reference probe failed: {error}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
