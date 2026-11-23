"""The isolated worker of the `torch_export` provider (v0.3-G6): imports USER code, exports one module, writes JSON.

Run as `python -m layerdelta.export_worker --request request.json --output DIR` by `export_provider.py` in a separate process
with a scrubbed environment, a working directory of its own, a timeout and no stdin. Its only result is `DIR/ir.json` (a
GraphIR as canonical JSON); a handled failure writes `DIR/error.json` instead. The parent never imports the user's code,
never unpickles anything and re-verifies the IR it reads, so this module is the one place user code runs.

Everything layerdelta needs is imported BEFORE the user's directory is put on sys.path, so user code cannot shadow it by name.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch

from .graph import ir_json
from .providers import GraphBuildError, export_to_ir

_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16, "F64": torch.float64, "I64": torch.int64,
           "I32": torch.int32, "I16": torch.int16, "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}


def build(request: dict) -> dict:
    sys.path.insert(0, request["code_dir"])
    module_name, _, attribute = request["entry"].partition(":")
    target = importlib.import_module(module_name)
    for part in attribute.split("."):
        target = getattr(target, part)
    model = target()
    if not isinstance(model, torch.nn.Module):
        raise GraphBuildError("export_failed", f"{request['entry']} returned a {type(model).__name__}, not an nn.Module")
    root = request.get("root")
    prefix = ""
    if root:
        modules = dict(model.named_modules())
        if root not in modules:
            raise GraphBuildError("export_failed", f"the model has no submodule {root!r}")
        model, prefix = modules[root], root + "."
    inputs = tuple(torch.zeros(item["shape"], dtype=_DTYPES[item["dtype"]]) for item in request["example_inputs"])
    provider = {"name": "torch_export", "version": "torch-" + torch.__version__.split("+")[0],
                "spec_sha256": request["spec_sha256"]}
    with torch.no_grad():
        return export_to_ir(model, inputs, provider, prefix=prefix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        ir = build(request)
        (args.output / "ir.json").write_text(ir_json(ir), encoding="utf-8")
    except BaseException as exc:  # a failure is data: the parent maps it to a stable code
        code = exc.code if isinstance(exc, GraphBuildError) else "export_failed"
        message = f"{type(exc).__name__}: {exc}"[:512]
        (args.output / "error.json").write_text(json.dumps({"code": code, "message": message}), encoding="utf-8")
        sys.exit(1)


if __name__ == "__main__":
    main()
