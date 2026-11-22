"""Isolated supervisor job process. Workspace paths are resolved from catalog digests."""

import argparse
import json
from pathlib import Path
import re
import sys
import tempfile

from .runner import execute, load_fixture, read_json, runtime, write_json


def checkpoint_path(workspace, digest, fixture):
    if digest == fixture["artifacts"]["model.safetensors"]:
        return workspace / "reference" / "model.safetensors"
    if not isinstance(digest, str) or not re.fullmatch("[a-f0-9]{64}", digest):
        raise ValueError("Invalid checkpoint digest")
    from .checkpoints import lookup_location

    found = lookup_location(workspace, digest)  # the catalog is the source of truth for where a checkpoint lives
    if found is not None:
        path, sharded = found
        if sharded:
            raise ValueError("Sharded checkpoints cannot run on the MNIST adapter")
        return path
    return workspace / "checkpoints" / (digest + ".safetensors")  # pre-catalog workspace


def run(workspace, request, output):
    from .checkpoint import load_checkpoint
    from .data import load_split
    from .domain import canonical_hash, verify_recipe
    from .engine import patch_weight, preview
    from .fixture import file_sha256

    p = request["payload"]
    kind = request["kind"]
    directory = workspace / "reference"
    fixture, manifest, _ = load_fixture(directory)
    digest = p["checkpoint_id"]
    checkpoint = checkpoint_path(workspace, digest, fixture)
    if file_sha256(checkpoint) != digest:
        raise ValueError("Checkpoint digest mismatch")
    if digest != fixture["artifacts"]["model.safetensors"]:
        fixture["training"]["seed"] = None
        fixture["artifacts"]["model.safetensors"] = digest
    recipe = p.get("recipe")
    if recipe:
        verify_recipe(fixture, recipe)
    if kind in ("baseline", "intervention", "final_evaluation"):
        execute(directory, workspace / "mnist", workspace / "cache" / "baseline",
                output, p["device"], recipe=recipe, split=request["split"], checkpoint=checkpoint)
        return
    output.mkdir(parents=True)
    model = load_checkpoint(checkpoint, digest)
    if kind == "preview":
        meta = runtime(p["device"], 1)
        cache_key = canonical_hash([1, digest, manifest["manifest_sha256"], p["sample_id"],
                                    p["target_id"], p["channel_index"], meta])
        cache_dir = workspace / "cache" / "preview"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / (cache_key + ".json")
        try:
            record = read_json(cached)
            if record["digest"] != canonical_hash([cache_key, record["preview"]]):
                raise ValueError("Preview cache digest mismatch")
            report = record["preview"]
        except (OSError, ValueError, TypeError, KeyError):
            images, _, rows = load_split(workspace / "mnist", manifest, "exploration")
            index = next((i for i, r in enumerate(rows) if r["sample_id"] == p["sample_id"]), None)
            if index is None:
                raise ValueError("Unknown exploration sample")
            report = preview(model, images[index], p["target_id"], p["channel_index"], p["device"])
            report.update(rows[index], runtime=meta, cache_key=cache_key, checkpoint_id=digest)
            with tempfile.TemporaryDirectory(dir=cache_dir) as stage:
                path = Path(stage) / "preview.json"
                write_json(path, {"preview": report, "digest": canonical_hash([cache_key, report])})
                path.replace(cached)
    elif kind == "checkpoint_export":
        from safetensors.torch import save_file

        audit = patch_weight(model, recipe["intervention"])
        child = output / "child.safetensors"
        save_file(model.state_dict(), str(child))
        report = {"child_sha256": file_sha256(child), "parent_sha256": digest,
                  "recipe_hash": recipe["recipe_hash"], "source_run_id": p["source_run_id"],
                  "intervention_audit": audit}
    else:
        raise ValueError("Unknown job kind")
    write_json(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser()
    for flag in ("workspace", "request", "output"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.workspace, read_json(args.request), args.output)
    except Exception as exc:
        args.output.mkdir(exist_ok=True, parents=True)
        code = "gpu_oom" if type(exc).__name__ == "OutOfMemoryError" else "worker_failed"
        write_json(args.output / "error.json", {"code": code, "message": str(exc)[:512]})
        sys.exit(1)


if __name__ == "__main__":
    main()
