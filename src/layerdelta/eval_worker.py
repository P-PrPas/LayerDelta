"""The isolated worker of an evaluation (v0.5-E4): decode and preprocess a dataset's samples, run an executable with a checkpoint's
weights, decode the outputs into per-sample records.

Run as `python -m layerdelta.eval_worker --request request.json --output DIR` by evalrun.py (the same isolation as forward_worker.py).
It decodes untrusted images (preprocess.py) and loads untrusted executables, so it is the one process that does both. Result:
`DIR/records.json` {"records": [...], "sample_ids": [...]} or `DIR/error.json` {code, message}. The parent computes metrics, intervals and
changes from the records; no pixel and no weight leaves this process.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from . import datasets, evaluators, forward_worker, preprocess


def _sample(task: str, row: dict, record: dict, files: Path, spec: dict) -> tuple[np.ndarray, dict]:
    """(preprocessed image [C, H, W] as float32, the sample dict an evaluator reads)."""
    image = preprocess.decode((files / row["image"]).read_bytes())
    mask = preprocess.decode((files / row["mask"]).read_bytes(), mask=True) if task == "segmentation" else None
    out = preprocess.apply(spec, image, mask=mask, boxes=row.get("boxes") if task == "detection" else None, ignore_index=record.get("ignore_index", 255))
    sample = {"sample_id": row["sample_id"], "condition": row.get("condition"), "geometry": out["geometry"]}
    if task == "classification":
        sample["label"] = record["classes"].index(row["label"])
    elif task == "segmentation":
        sample["mask"] = out["mask"].numpy()
    elif task == "detection":
        sample["boxes"] = row.get("boxes", [])
    elif task == "ocr":
        sample["text"] = row["text"]
    return out["image"].numpy().astype(np.float32), sample


def _interpreted(request: dict):
    """A runner that changes some units of one node's output while the program runs (an activation edit; no checkpoint is made)."""
    from . import graph_run, probe_worker
    program = probe_worker._program(request)
    weights = forward_worker._weights(request)
    iv = request["intervention"]
    axis, units = iv["unit_axis"], torch.tensor(iv["units"], dtype=torch.long)

    def change(output: torch.Tensor) -> torch.Tensor:
        out = output.clone()
        if iv["op"] == "zero":
            return out.index_fill_(axis, units, 0.0)
        return out.index_copy_(axis, units, output.index_select(axis, units) * iv["factor"])

    def run(batch):
        outputs, _ = graph_run.run(program, weights, request["prefix"], [torch.from_numpy(b) for b in batch], change={iv["node_id"]: change},
                                      expect=request["expect"])
        return [o.detach().cpu().numpy() for o in outputs]
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        torch.set_num_threads(request["threads"])
        run = _interpreted(request) if request.get("intervention") else forward_worker._runner(request)
        record, rows, files = request["dataset"], request["rows"], Path(request["files"])
        evaluator = evaluators.make(request["evaluator"])
        batch, records, ids = request["batch"], [], []
        for start in range(0, len(rows), batch):
            chunk = rows[start:start + batch]
            built = [_sample(record["task"], row, record, files, request["preprocessing"]) for row in chunk]
            shapes = {tuple(image.shape) for image, _ in built}
            if len(shapes) != 1:
                raise forward_worker.Refused("forward_failed", f"preprocessing gave samples of different shapes {sorted(shapes)}: add resize or pad_to")
            images = np.stack([image for image, _ in built])
            if images.shape[0] < batch:
                images = np.concatenate([images, np.zeros((batch - images.shape[0], *images.shape[1:]), images.dtype)])
            outputs = run([images])
            if len(outputs) != request["outputs"] or any(o.ndim == 0 or o.shape[0] != batch for o in outputs):
                raise forward_worker.Refused("forward_failed", "every output must have the batch size as its first dimension")
            samples = [s for _, s in built]
            records += evaluator.records([o[:len(chunk)] for o in outputs], samples)
            ids += [s["sample_id"] for s in samples]
        (args.output / "records.json").write_text(json.dumps({"records": records, "sample_ids": ids}), encoding="utf-8")
    except BaseException as exc:  # a failure is data: the parent maps it to a stable code
        if isinstance(exc, forward_worker.Refused):
            code, message = exc.code, exc.message
        elif isinstance(exc, (evaluators.EvaluatorError, datasets.DatasetError)):
            code, message = exc.code, exc.message
        else:
            code, message = "forward_failed", f"{type(exc).__name__}: {exc}"[:512]
        (args.output / "error.json").write_text(json.dumps({"code": code, "message": message[:512]}), encoding="utf-8")
        sys.exit(1)


if __name__ == "__main__":
    main()
