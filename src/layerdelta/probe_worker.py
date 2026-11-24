"""The isolated worker of a probe (v0.5-E5): run a checkpoint on dataset samples and aggregate the output of graph nodes per unit.

Run as `python -m layerdelta.probe_worker --request request.json --output DIR` (the same isolation as eval_worker). It loads the executable's
`ExportedProgram` (builtin: exported from the registry model; torch_export: the saved `.pt2`, re-hashed first), substitutes the checkpoint's weights
by name and runs it through graph_run.py, which taps the nodes. Result `DIR/probe.json` or `DIR/error.json`.

Aggregates are per unit over every sample and every other position: n, mean, std (population), l2 (the norm of the unit's values), the fraction of
positive values and, if asked, a histogram on a shared range. They can be split by `label` or `condition`. `values` returns raw activations of the first
samples (only a session with weights:values is ever given them: the route decides, this worker computes what it is told).
"""

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

from . import datasets, eval_worker, forward_worker, graph_run

VALUES_MAX = 65536


def _program(request: dict):
    if request["kind"] == "builtin":
        from . import providers
        entry = providers._REGISTRY.get(request["model"])
        if entry is None:
            raise forward_worker.Refused("executable_missing", f"no builtin model named {request['model']!r} is registered in this process")
        inputs = tuple(torch.zeros(shape, dtype=getattr(torch, dtype)) for shape, dtype in entry.example_inputs)
        return torch.export.export(entry.factory().eval(), inputs)
    if request["kind"] != "torch_export":
        raise forward_worker.Refused("probe_unsupported", f"a {request['kind']} executable cannot be probed")
    data = forward_worker._hashed_bytes(request["artifact"], request["artifact_sha256"])
    if request["torch_version"] != "torch-" + torch.__version__.split("+")[0]:
        raise forward_worker.Refused("executable_changed", f"the program was exported with {request['torch_version']}, this is torch {torch.__version__}")
    return torch.export.load(io.BytesIO(data))


def _group_of(split_by, sample: dict, row: dict, classes):
    if split_by == "label":
        return str(row.get("label")) if classes is not None else "all"
    if split_by == "condition":
        return row.get("condition") or "none"
    return "all"


class _Sums:
    def __init__(self, units: int):
        self.n = 0
        self.total = np.zeros(units)
        self.squares = np.zeros(units)
        self.positive = np.zeros(units)
        self.counts = None

    def add(self, flat: np.ndarray, edges=None):  # flat: [units, elements]
        self.n += flat.shape[1]
        self.total += flat.sum(axis=1)
        self.squares += (flat ** 2).sum(axis=1)
        self.positive += (flat > 0).sum(axis=1)
        if edges is not None:
            hist = np.stack([np.histogram(row, bins=edges)[0] for row in flat])
            self.counts = hist if self.counts is None else self.counts + hist

    def out(self, edges=None) -> dict:
        mean = self.total / self.n
        body = {"n": int(self.n), "mean": mean.tolist(), "std": np.sqrt(np.maximum(self.squares / self.n - mean ** 2, 0)).tolist(),
                "l2": np.sqrt(self.squares).tolist(), "frac_positive": (self.positive / self.n).tolist()}
        if edges is not None:
            body["histogram"] = {"edges": edges.tolist(), "counts": self.counts.tolist()}
        return body


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        torch.set_num_threads(request["threads"])
        program = _program(request)
        weights = forward_worker._weights(request)
        record, rows, files = request["dataset"], request["rows"], Path(request["files"])
        nodes = request["nodes"]  # [{"node_id", "unit_axis", "units"}]
        watch = [n["node_id"] for n in nodes]
        batch, split_by, bins, want = request["batch"], request["split_by"], request["bins"], request["values"]
        edges = {}
        passes = 2 if bins else 1
        sums, raw = {}, {}
        for step in range(passes):
            sums = {n["node_id"]: {} for n in nodes}
            low = {w: np.inf for w in watch}
            high = {w: -np.inf for w in watch}
            for start in range(0, len(rows), batch):
                chunk = rows[start:start + batch]
                built = [eval_worker._sample(record["task"], row, record, files, request["preprocessing"]) for row in chunk]
                shapes = {tuple(image.shape) for image, _ in built}
                if len(shapes) != 1:
                    raise forward_worker.Refused("forward_failed", f"preprocessing gave samples of different shapes {sorted(shapes)}: add resize or pad_to")
                images = np.stack([image for image, _ in built])
                if images.shape[0] < batch:
                    images = np.concatenate([images, np.zeros((batch - images.shape[0], *images.shape[1:]), images.dtype)])
                _, seen = graph_run.run(program, weights, request["prefix"], [torch.from_numpy(images)], watch=watch)
                for spec in nodes:
                    tensor = seen[spec["node_id"]][:len(chunk)].to(torch.float64).numpy()
                    axis = spec["unit_axis"]
                    if step == 0 and want and spec["node_id"] not in raw and start == 0:
                        raw[spec["node_id"]] = {"shape": list(tensor.shape), "data": tensor.reshape(-1)[:VALUES_MAX].tolist()}
                    if bins and step == 0:
                        low[spec["node_id"]] = min(low[spec["node_id"]], float(tensor.min()))
                        high[spec["node_id"]] = max(high[spec["node_id"]], float(tensor.max()))
                        continue
                    for k, (_, sample) in enumerate(built):
                        group = _group_of(split_by, sample, chunk[k], record.get("classes"))
                        unit_major = np.moveaxis(tensor[k:k + 1], axis, 0).reshape(tensor.shape[axis], -1)  # [units, positions of this sample]
                        bucket = sums[spec["node_id"]].setdefault(group, _Sums(tensor.shape[axis]))
                        bucket.add(unit_major, edges.get(spec["node_id"]))
            if bins and step == 0:
                edges = {w: np.linspace(low[w], high[w] if high[w] > low[w] else low[w] + 1.0, bins + 1) for w in watch}
        result = {"nodes": [{"node_id": spec["node_id"], "unit_axis": spec["unit_axis"], "units": spec["units"],
                             "groups": {g: s.out(edges.get(spec["node_id"])) for g, s in sorted(sums[spec["node_id"]].items())}} for spec in nodes]}
        if want:
            result["values"] = raw
        (args.output / "probe.json").write_text(json.dumps(result), encoding="utf-8")
    except BaseException as exc:  # a failure is data: the parent maps it to a stable code
        if isinstance(exc, forward_worker.Refused):
            code, message = exc.code, exc.message
        elif isinstance(exc, (datasets.DatasetError, graph_run.NodeUnsupported)):
            code, message = getattr(exc, "code", "probe_unsupported"), getattr(exc, "message", str(exc))
        else:
            code, message = "forward_failed", f"{type(exc).__name__}: {exc}"[:512]
        (args.output / "error.json").write_text(json.dumps({"code": code, "message": message[:512]}), encoding="utf-8")
        sys.exit(1)


if __name__ == "__main__":
    main()
