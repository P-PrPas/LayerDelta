"""The isolated worker of a forward pass (v0.5-E1): loads an executable, substitutes a checkpoint's weights by name, runs samples.

Run as `python -m layerdelta.forward_worker --request request.json --output DIR` by executable.py (scrubbed environment, own directory,
timeout, no stdin). Result: `DIR/outputs.npz` (arrays `o0..`, samples first) or `DIR/error.json` {code, message}. This is the one place
a saved `.pt2` is loaded: `torch.export.load` unpickles the program's state dict and constants (torch 2.6 `weights_only=False`), so the
file is re-hashed against the record first and only a file the export worker itself saved is ever loaded (docs/reference/executables.md).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

_NP = {"F32": np.float32, "F16": np.float16, "F64": np.float64, "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
       "U8": np.uint8, "BOOL": np.bool_}


class Refused(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def _rehash(path: str, sha256: str) -> None:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        raise Refused("executable_changed", "the stored executable does not match its recorded SHA-256")


def _weights(request: dict) -> dict:
    """{graph parameter name: array} from the checkpoint, by name (a tied name may be stored under another member)."""
    from safetensors import safe_open
    tie_of = {name: group for group in request["ties"] for name in group}
    found, problems = {}, []
    with safe_open(request["checkpoint"], framework="pt") as handle:
        stored = set(handle.keys())
        for param in request["params"]:
            name = param["name"]
            source = name if name in stored else next((m for m in tie_of.get(name, []) if m in stored), None)
            if source is None:
                problems.append(f"{name}: missing from the checkpoint")
                continue
            tensor = handle.get_tensor(source)
            if list(tensor.shape) != param["shape"]:
                problems.append(f"{name}: shape {list(tensor.shape)} in the checkpoint, {param['shape']} in the graph")
                continue
            found[name] = tensor
    if problems:
        raise Refused("executable_mismatch", "; ".join(problems)[:512])
    return found


def _torch_module(request: dict):
    if request["kind"] == "builtin":
        from . import providers
        entry = providers._REGISTRY.get(request["model"])
        if entry is None:
            raise Refused("executable_missing", f"no builtin model named {request['model']!r} is registered in this process")
        module = entry.factory().eval()
    else:
        _rehash(request["artifact"], request["artifact_sha256"])
        if request["torch_version"] != "torch-" + torch.__version__.split("+")[0]:
            raise Refused("executable_changed", f"the program was exported with {request['torch_version']}, this is torch {torch.__version__}")
        module = torch.export.load(request["artifact"]).module()  # exported from an eval-mode module: no .eval() (unsupported on it)
    state = module.state_dict()
    prefix, update = request["prefix"], {}
    for name, tensor in _weights(request).items():
        key = name[len(prefix):] if prefix and name.startswith(prefix) else name
        if key not in state:
            raise Refused("executable_mismatch", f"{name}: the executable has no state tensor {key!r}")
        update[key] = tensor.to(state[key].dtype)
    module.load_state_dict(update, strict=False)
    return module


def _flatten(value) -> list:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [t for item in value for t in _flatten(item)]
    if isinstance(value, dict):
        return [t for key in sorted(value) for t in _flatten(value[key])]
    raise Refused("forward_failed", f"the program returned a {type(value).__name__}, not tensors")


def _runner(request: dict):
    """A function: list of numpy batches (exactly `batch` samples each) -> list of numpy outputs."""
    if request["kind"] != "onnx":
        module = _torch_module(request)

        def run(batch):
            with torch.no_grad():
                return [t.detach().cpu().numpy() for t in _flatten(module(*[torch.from_numpy(a) for a in batch]))]
        return run
    try:
        import onnx
        import onnxruntime
        from onnx import numpy_helper
    except ImportError as exc:
        raise Refused("provider_disabled", 'evaluating an ONNX graph needs: pip install "layerdelta[eval-onnx]"') from exc
    _rehash(request["artifact"], request["artifact_sha256"])
    model = onnx.load(request["artifact"], load_external_data=False)
    names = {p["name"] for p in request["params"]}
    weights = _weights(request)
    replaced = 0
    for index, initializer in enumerate(model.graph.initializer):
        if initializer.name in names:
            want = numpy_helper.to_array(initializer).dtype
            array = weights[initializer.name].float().numpy() if weights[initializer.name].dtype == torch.bfloat16 else weights[initializer.name].numpy()
            model.graph.initializer[index].CopyFrom(numpy_helper.from_array(array.astype(want), initializer.name))
            replaced += 1
    if replaced != len(names):
        raise Refused("executable_mismatch", f"the ONNX file holds {replaced} of the graph's {len(names)} weights as initializers")
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = request["threads"]
    session = onnxruntime.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    feeds = [i.name for i in session.get_inputs() if i.name not in {t.name for t in model.graph.initializer}]

    def run(batch):
        return list(session.run(None, dict(zip(feeds, batch))))
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        torch.set_num_threads(request["threads"])
        run = _runner(request)
        with np.load(request["inputs"], allow_pickle=False) as data:
            arrays = [data[f"i{k}"] for k in range(len(data.files))]
        batch, samples = request["batch"], arrays[0].shape[0]
        collected: list[list] = []
        for start in range(0, samples, batch):
            part = [a[start:start + batch] for a in arrays]
            count = part[0].shape[0]
            padded = [np.concatenate([p, np.zeros((batch - count, *p.shape[1:]), p.dtype)]) if count < batch else p for p in part]
            outputs = run(padded)
            if len(outputs) != request["outputs"] or any(o.ndim == 0 or o.shape[0] != batch for o in outputs):
                raise Refused("forward_failed", "every output must have the batch size as its first dimension")
            collected.append([o[:count] for o in outputs])
        np.savez(args.output / "outputs.npz", **{f"o{k}": np.concatenate([c[k] for c in collected]) for k in range(request["outputs"])})
    except BaseException as exc:  # a failure is data: the parent maps it to a stable code
        code = exc.code if isinstance(exc, Refused) else "forward_failed"
        message = exc.message if isinstance(exc, Refused) else f"{type(exc).__name__}: {exc}"[:512]
        (args.output / "error.json").write_text(json.dumps({"code": code, "message": message}), encoding="utf-8")
        sys.exit(1)


if __name__ == "__main__":
    main()
