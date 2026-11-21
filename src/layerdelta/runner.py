"""Run an isolated baseline or single-channel experiment on the built-in fixture."""

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import uuid


def write_json(path, document):
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(document, stream, separators=(",", ":"), allow_nan=False)


def read_json(path):
    return json.loads(path.read_bytes())


def runtime(device, batch_size):
    import torch

    if not 1 <= batch_size <= 1024:
        raise ValueError("batch_size must be between 1 and 1024")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA device is unavailable")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    actual = torch.empty(0, device=device).device
    metadata = {
        "requested_device": device, "actual_device": str(actual),
        "torch": str(torch.__version__), "torch_build": torch.__config__.show(),
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu": platform.processor(), "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
        "batch_size": batch_size, "dtype": "float32", "deterministic": True,
        "mkldnn": torch.backends.mkldnn.enabled, "tf32": False,
    }
    if device == "cuda":
        properties = torch.cuda.get_device_properties(actual)
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, timeout=10,
        ).splitlines()[0].strip()
        metadata.update(gpu=properties.name, uuid=str(properties.uuid),
                        vram_bytes=properties.total_memory, driver=driver,
                        cuda_build=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    return metadata


def load_fixture(directory):
    from .fixture import file_sha256

    fixture = read_json(directory / "fixture.json")
    for name in ("model.safetensors", "dataset.json", "exploration.json"):
        if file_sha256(directory / name) != fixture["artifacts"][name]:
            raise ValueError(f"Fixture artifact digest mismatch: {name}")
    manifest = read_json(directory / "dataset.json")
    if manifest["manifest_sha256"] != fixture["manifest_sha256"]:
        raise ValueError("Fixture and dataset manifest disagree")
    return fixture, manifest, read_json(directory / "exploration.json")["demo"]


def baseline(cache, fingerprint, calculate):
    from .domain import canonical_hash

    cache.mkdir(exist_ok=True, parents=True)
    path = cache / (fingerprint + ".json")
    try:
        record = read_json(path)
        if record["fingerprint"] == fingerprint and canonical_hash(record["rows"]) == record["rows_digest"]:
            return record["rows"], "warm"
    except (OSError, ValueError, TypeError, KeyError):
        pass
    rows = calculate()
    record = {"rows": rows, "fingerprint": fingerprint, "rows_digest": canonical_hash(rows)}
    with tempfile.TemporaryDirectory(dir=cache) as stage:
        staged = Path(stage) / "baseline.json"
        write_json(staged, record)
        os.replace(staged, path)
    return rows, "cold"


def execute(fixture_dir, data_dir, cache, output, device, batch_size=128,
            recipe=None, split="exploration", checkpoint=None):
    """Worker entry point. The caller owns the fresh process and its output directory."""
    import psutil
    import torch
    from .checkpoint import load_checkpoint
    from .data import load_split
    from .domain import canonical_hash, verify_recipe
    from .engine import infer
    from .evaluation import compare
    from .fixture import file_sha256

    start = time.perf_counter()
    if split not in ("exploration", "holdout"):
        raise ValueError("Unsupported evaluation split")
    meta = runtime(device, batch_size)
    fixture, manifest, demo = load_fixture(fixture_dir)
    checkpoint = checkpoint or fixture_dir / "model.safetensors"
    parent_hash = file_sha256(checkpoint)
    if checkpoint == fixture_dir / "model.safetensors" and parent_hash != fixture["artifacts"]["model.safetensors"]:
        raise ValueError("Parent checkpoint changed during setup")
    if parent_hash != fixture["artifacts"]["model.safetensors"]:
        fixture["training"]["seed"] = None
        fixture["artifacts"]["model.safetensors"] = parent_hash
    if recipe is not None:
        recipe = verify_recipe(fixture, recipe)
    model = load_checkpoint(checkpoint, parent_hash)
    images, _, rows = load_split(data_dir, manifest, split)
    fingerprint = canonical_hash({
        "version": 1, "parent": parent_hash, "model": fixture["model"],
        "manifest": fixture["manifest_sha256"], "split_sha256": manifest["split_sha256"][split],
        "evaluator_revision": 1, "runtime": meta,
    })
    inference_seconds = 0.
    audit = {}

    def predict(spec=None):
        nonlocal inference_seconds
        tick = time.perf_counter()
        predictions = infer(model, images, rows, device, batch_size, spec, audit=audit)
        inference_seconds += time.perf_counter() - tick
        return predictions

    before, cache_state = baseline(cache, fingerprint, predict)
    after = predict(recipe["intervention"]) if recipe else before
    result = compare(rows, before, after, demo["truth"], demo["baseline_prediction"])
    output.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        write_json(stage / "predictions.json", after)
        write_json(stage / "comparison.json", result)
        if recipe:
            write_json(stage / "recipe.json", recipe)
        process_memory = psutil.Process().memory_info()
        report = {
            "run_id": uuid.uuid4().hex, "baseline_id": fingerprint,
            "evaluation_fingerprint": fingerprint, "recipe_hash": recipe["recipe_hash"] if recipe else None,
            "parent_sha256": parent_hash, "manifest_sha256": fixture["manifest_sha256"],
            "split": split, "runtime": meta, "worker_pid": os.getpid(), "cache": cache_state,
            "intervention_audit": audit,
            "slices": result["slices"], "sample_count": len(rows),
            "inference_seconds": inference_seconds, "worker_seconds": time.perf_counter() - start,
            "peak_ram_bytes": getattr(process_memory, "peak_wset", None),
            "peak_vram_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
            "artifacts": {f.name: {"sha256": file_sha256(f), "bytes": f.stat().st_size} for f in stage.iterdir()},
        }
        write_json(stage / "report.json", report)
        if output.exists():
            raise FileExistsError(f"Choose a new output directory: {output}")
        stage.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", default=128, type=int)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--recipe", type=Path, help="Replay frozen recipe JSON")
    group.add_argument("--target", help="Catalog target ID for a new intervention")
    parser.add_argument("--channel", type=int)
    parser.add_argument("--operation", choices=("scale", "zero"))
    parser.add_argument("--scale", type=float)
    parser.add_argument("--hypothesis")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker:
        from .process import popen

        child = popen([sys.executable, "-m", "layerdelta.runner", "--worker", *sys.argv[1:]])
        try:
            raise SystemExit(child.wait())
        except KeyboardInterrupt:
            child.kill()
            child.wait()
            raise SystemExit(130)
    try:
        recipe = read_json(args.recipe) if args.recipe else None
        if args.target:
            from .domain import freeze_recipe

            fixture, _, _ = load_fixture(args.fixture)
            targets = {t["target_id"]: t for t in fixture["model"]["targets"]}
            spec = {"target_id": args.target, "mode": targets[args.target]["mode"],
                    "channel_index": args.channel, "operation": args.operation}
            if args.scale is not None:
                spec["scale"] = args.scale
            recipe = freeze_recipe(fixture, spec, args.hypothesis)
        elif any(val is not None for val in (args.channel, args.operation, args.scale, args.hypothesis)):
            parser.error("Intervention options require --target")
        result = execute(args.fixture, args.data_dir, args.cache, args.output, args.device, args.batch_size, recipe)
        print(json.dumps({k: result[k] for k in ("run_id", "baseline_id", "recipe_hash", "cache")}))
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.exit(1, f"Run failed: {error}\n")


if __name__ == "__main__":
    main()
