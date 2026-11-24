"""Executable artifacts (v0.5-E1): what a forward pass of an attached graph runs, and how to run it on a checkpoint's weights.

A graph (graph.py) describes a model; an evaluation needs to RUN it with the weights of a CHILD checkpoint. Each provider leaves one
executable behind at attach time (a human action), recorded per graph in the catalog (`executables`):

- `builtin`: the registered module of this code base (nothing to store; the record names the registry key);
- `torch_export`: the `ExportedProgram` saved as `executables/<sha256>.pt2` by the export worker during the attach;
- `onnx`: the attached `.onnx` file, copied to `executables/<sha256>.onnx` and proven to be the very bytes the graph was built from.

`forward` runs it in an isolated worker process (forward_worker.py: scrubbed environment, timeout and tree kill, size caps; the API
process never runs a model, and never loads a `.pt2`, which unpickles: docs/reference/executables.md). Weights are substituted BY
CHECKPOINT NAME from the graph's `params`; a parameter the checkpoint lacks or reshapes is refused as `executable_mismatch` BEFORE any
process starts. The batch size is the exported one: samples are padded with zeros to it and the outputs trimmed, so one program
serves any number of samples (a model whose output mixes samples in eval mode is out of scope).
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import graph as graph_mod
from .export_provider import (LOG_MAX, _kill, _one_export, _scrubbed_env, _tree_size)
from .process import popen
from .selector import InspectError, digest
from .tensors import open_tensors

DIR = "executables"
OUTPUT_MAX = 256 << 20          # bytes of outputs one forward call may return
INPUT_MAX = 256 << 20           # bytes of inputs one forward call may be given
SAMPLES_MAX = 1 << 20


class ForwardError(InspectError):
    def __init__(self, code: str, message: str, field: str = "executable"):
        super().__init__(code, message, field)


def _sha256_file(path: Path) -> str:
    digest_ = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest_.update(chunk)
    return digest_.hexdigest()


def _place(workspace: Path, source: Path, suffix: str) -> tuple[str, str]:
    """Copy `source` into the content-addressed executables directory (atomically); return (sha256, path relative to the workspace)."""
    sha = _sha256_file(source)
    folder = workspace / DIR
    folder.mkdir(exist_ok=True)
    target = folder / (sha + suffix)
    if not target.exists():
        partial = folder / f".{sha}{suffix}.partial-{os.getpid()}"
        shutil.copyfile(source, partial)
        os.replace(partial, target)
    return sha, f"{DIR}/{target.name}"


def capture(store, ir: dict, spec: dict, table, workspace: Path, artifact_dir: Path | None = None) -> dict | None:
    """Record the executable of a graph that was just built from `spec` (by the dispatcher, for a person). None when the spec reuses a
    stored graph: its executable is already recorded. `artifact_dir` is where the export worker left `program.pt2`."""
    kind = spec.get("provider")
    base = {"kind": kind, "batch": (ir["inputs"][0]["shape"][0] if ir["inputs"] and ir["inputs"][0]["shape"] else None),
            "torch_version": ir["provider"]["version"] if kind != "onnx" else None}
    if kind == "builtin":
        record = {**base, "model": spec["model"], "artifact": None}
    elif kind == "torch_export":
        program = None if artifact_dir is None else artifact_dir / "program.pt2"
        if program is not None and program.is_file():
            sha, relative = _place(Path(workspace), program, ".pt2")
            record = {**base, "artifact": relative, "sha256": sha, "bytes": program.stat().st_size, "root_prefix": _prefix(ir, spec)}
        else:
            why = "the export worker saved no program"
            if artifact_dir is not None and (artifact_dir / "program_error.txt").is_file():
                why = (artifact_dir / "program_error.txt").read_text(encoding="utf-8", errors="replace")[:512]
            record = {**base, "artifact": None, "reason": why}
    elif kind == "onnx":
        from . import onnx_provider
        checked = onnx_provider.check_spec(spec)
        source = Path(checked["path"])
        sha = _sha256_file(source)
        wanted = digest({"spec": checked, "model_sha256": sha, "table_signature": graph_mod.table_signature(table)})
        if wanted != ir["provider"]["spec_sha256"]:  # the file changed between the parse and now: it is not the file the graph describes
            record = {**base, "artifact": None, "reason": "the .onnx file changed while it was being attached"}
        else:
            sha, relative = _place(Path(workspace), source, ".onnx")
            record = {**base, "artifact": relative, "sha256": sha, "bytes": source.stat().st_size, "onnx_version": ir["provider"]["version"]}
    else:
        return None
    store.put_executable(ir["ir_sha256"], record)
    return record


def _prefix(ir: dict, spec: dict) -> str:
    root = spec.get("root")
    return (root + ".") if root else ""


def fingerprint(record: dict) -> str:
    """The part of an evaluation fingerprint that names the executable (artifact hash, framework version)."""
    return digest({k: record.get(k) for k in ("kind", "model", "sha256", "torch_version", "onnx_version", "batch")})


def check_ready(ir: dict, record: dict | None, table, workspace: Path) -> dict:
    """Everything that can be refused before a process starts. Returns the record."""
    if record is None:
        raise ForwardError("executable_missing", "this graph was attached before evaluations existed: attach it again")
    if record["kind"] != "builtin" and record.get("artifact") is None:
        raise ForwardError("executable_missing", f"this graph has no executable: {record.get('reason', 'none was saved')}")
    if record.get("batch") is None:
        raise ForwardError("executable_mismatch", "the graph's first input has no batch dimension to pad", "inputs")
    try:
        bound = graph_mod.bind(ir, table)
    except graph_mod.GraphMismatch as exc:
        raise ForwardError("executable_mismatch", f"the checkpoint does not fit the graph: {exc}"[:512]) from exc
    if record["kind"] == "onnx":  # an unmatched float weight is read from the file, so an edit to it would not change the output
        loose = [n for n in bound.names("unbound") if table.tensors[n].dtype in ("F32", "F16", "BF16", "F64")]
        if loose:
            raise ForwardError("executable_mismatch", f"{len(loose)} float tensors of the checkpoint are not bound to the ONNX graph "
                               f"(for example {loose[0]}): export with do_constant_folding=False and training=PRESERVE so that every "
                               "weight keeps its checkpoint name")
    path = None if record.get("artifact") is None else Path(workspace) / record["artifact"]
    if path is not None and not path.is_file():
        raise ForwardError("executable_missing", "the stored executable file is gone: attach the graph again")
    return record


def forward(workspace: Path, store, ir: dict, checkpoint_path: Path, inputs: list, *, timeout: float = 120.0, threads: int = 1) -> dict:
    """Run the graph's executable on `inputs` (arrays, first axis = samples) with the weights of the checkpoint at `checkpoint_path`.

    Returns {"outputs": [array per graph output, samples first], "executable": {...}}. Raises ForwardError."""
    workspace = Path(workspace)
    table = open_tensors(checkpoint_path)
    record = check_ready(ir, store.executable_for(ir["ir_sha256"]), table, workspace)
    arrays = [np.ascontiguousarray(a) for a in inputs]
    if len(arrays) != len(ir["inputs"]):
        raise ForwardError("invalid_argument", f"the graph takes {len(ir['inputs'])} inputs, {len(arrays)} were given", "inputs")
    samples = arrays[0].shape[0] if arrays and arrays[0].ndim else 0
    for array, spec in zip(arrays, ir["inputs"]):
        if array.ndim != len(spec["shape"]) or list(array.shape[1:]) != spec["shape"][1:] or array.shape[0] != samples:
            raise ForwardError("invalid_argument", f"an input has shape {list(array.shape)}; the graph expects [N, *{spec['shape'][1:]}]", "inputs")
    if not 1 <= samples <= SAMPLES_MAX or sum(a.nbytes for a in arrays) > INPUT_MAX:
        raise ForwardError("invalid_argument", f"give between 1 and {SAMPLES_MAX} samples and at most {INPUT_MAX} bytes", "inputs")
    request = {"kind": record["kind"], "model": record.get("model"), "checkpoint": str(Path(checkpoint_path).resolve()),
               "artifact": None if record.get("artifact") is None else str((workspace / record["artifact"]).resolve()),
               "artifact_sha256": record.get("sha256"), "torch_version": record.get("torch_version"),
               "prefix": record.get("root_prefix", ""), "batch": record["batch"], "threads": threads,
               "params": [{"name": p["name"], "shape": p["shape"], "dtype": p["dtype"]} for p in ir["params"]], "ties": ir["ties"],
               "outputs": len(ir["outputs"])}
    with _one_export, tempfile.TemporaryDirectory(prefix="ld-forward-", ignore_cleanup_errors=True) as scratch:
        work = Path(scratch)
        out = work / "out"
        out.mkdir()
        np.savez(work / "inputs.npz", **{f"i{k}": a for k, a in enumerate(arrays)})
        request["inputs"] = str(work / "inputs.npz")
        (work / "request.json").write_text(json.dumps(request), encoding="utf-8")
        with (work / "stdout.txt").open("wb") as stdout, (work / "stderr.txt").open("wb") as stderr:
            flags = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {"start_new_session": True}
            child = popen([sys.executable, "-m", "layerdelta.forward_worker", "--request", str(work / "request.json"), "--output", str(out)],
                          cwd=work, env=_scrubbed_env(), stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, **flags)
            deadline = time.monotonic() + timeout
            while child.poll() is None:
                if time.monotonic() > deadline:
                    _kill(child)
                    raise ForwardError("worker_timeout", f"the forward pass did not finish within {timeout:g} seconds")
                if max((work / n).stat().st_size for n in ("stdout.txt", "stderr.txt")) > LOG_MAX or _tree_size(out) > OUTPUT_MAX + 4096:
                    _kill(child)
                    raise ForwardError("worker_output", "the forward worker wrote more than its allowance")
                time.sleep(0.02)
        _kill(child)
        return _read(child.returncode, out, work / "stderr.txt", record, ir)


def _read(returncode: int, out: Path, stderr: Path, record: dict, ir: dict) -> dict:
    error = out / "error.json"
    if error.is_file():
        try:
            body = json.loads(error.read_bytes()[:4096].decode("utf-8", errors="replace"))
            code, message = str(body["code"]), str(body["message"])[:512]
        except (ValueError, KeyError, TypeError):
            code, message = "forward_failed", "the worker reported an unreadable error"
        raise ForwardError(code if code in CODES else "forward_failed", message)
    result = out / "outputs.npz"
    if returncode != 0 or not result.is_file():
        tail = stderr.read_text(encoding="utf-8", errors="replace")[-300:] if stderr.is_file() else ""
        raise ForwardError("forward_failed", f"the worker exited with status {returncode} and no result: {tail.strip()}"[:512])
    with np.load(result, allow_pickle=False) as data:
        outputs = [data[f"o{k}"] for k in range(len(data.files))]
    if len(outputs) != len(ir["outputs"]):
        raise ForwardError("forward_failed", f"the program returned {len(outputs)} outputs, the graph has {len(ir['outputs'])}")
    return {"outputs": outputs, "executable": {"kind": record["kind"], "fingerprint": fingerprint(record), "sha256": record.get("sha256")}}


CODES = {"executable_mismatch", "executable_changed", "executable_missing", "forward_failed", "provider_disabled", "invalid_argument"}
