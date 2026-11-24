"""The `torch_export` graph provider (v0.3-G6): export a USER model in an isolated worker process, take back JSON only.

Owner decision Q1 (option 3): user code may run, but only here, only for a human, only when the project opted in
(`create_app(allow_user_code=True)`, `layerdelta-server --allow-user-code`). What the isolation does and does not give:

- it does: a separate process with its own working directory, a scrubbed environment (no layerdelta credentials, tokens or
  paths), no stdin, a hard timeout (the process is killed), a cap on the size of its result, and a result that is only
  canonical JSON, re-verified here (`verify_ir`) and re-labelled by the dispatcher. A crash, hang, exception or malformed
  result leaves no graph. Nothing the user's code returns is unpickled, imported or executed by this process.
- it does NOT: sandbox a malicious program. The worker runs as the same operating-system user and can read the files that
  user can read. That is why attaching is human-only and opt-in: it is for code the person already trusts to run.

The spec: `{"provider": "torch_export", "entry": "pkg.mod:factory", "path": "<absolute directory put on sys.path>",
"example_inputs": [{"shape": [1, 3, 224, 224], "dtype": "F32"}], "root": null | "backbone"}`. `factory()` takes no arguments and
returns an `nn.Module`; `root` exports one submodule (the workaround for post-processing the exporter cannot trace); the
parameter names in the graph then carry the `root.` prefix so they bind to the full checkpoint.
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from .graph import verify_ir
from .process import popen
from .providers import GraphBuildError
from .selector import InspectError, digest

TIMEOUT_DEFAULT = 120.0
RESULT_MAX = 64 << 20
LOG_MAX = 8 << 20       # stdout and stderr of the worker, each
INPUTS_MAX, DIMS_MAX, ELEMS_MAX = 8, 8, 1 << 28
DTYPES = ("F32", "F16", "BF16", "F64", "I64", "I32", "I16", "I8", "U8", "BOOL")
_ENTRY = re.compile(r"[A-Za-z_]\w*(\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(\.[A-Za-z_]\w*)*")
_ROOT = re.compile(r"[A-Za-z_]\w*(\.\w+)*")
_ENV_KEEP = ("PATH", "SYSTEMROOT", "SystemDrive", "WINDIR", "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL", "COMSPEC", "PATHEXT",
             "USERNAME", "USER", "LOGNAME", "USERPROFILE", "HOME", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "APPDATA",
             "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE")  # torch/getpass need an identity; nothing secret-bearing


def check_spec(spec) -> dict:
    """The canonical form of a torch_export spec; every key and value is checked before anything runs."""
    bad = lambda message, field: GraphBuildError("invalid_spec", f"{message} ({field})")  # noqa: E731
    keys = {"provider", "entry", "path", "example_inputs", "root"}
    if not isinstance(spec, dict) or spec.get("provider") != "torch_export" or not {"provider", "entry", "path", "example_inputs"} <= set(spec) <= keys:
        raise GraphBuildError("invalid_spec", "a torch_export spec is {provider, entry, path, example_inputs, root?}")
    entry, path, root = spec["entry"], spec["path"], spec.get("root")
    if not isinstance(entry, str) or not _ENTRY.fullmatch(entry):
        raise bad("entry must be 'package.module:callable'", "entry")
    if not isinstance(path, str) or "\0" in path or not Path(path).is_absolute() or not Path(path).is_dir():
        raise bad("path must be an absolute directory that exists", "path")
    if root is not None and (not isinstance(root, str) or not _ROOT.fullmatch(root)):
        raise bad("root must be a dotted submodule name or null", "root")
    inputs = spec["example_inputs"]
    if not isinstance(inputs, list) or not 1 <= len(inputs) <= INPUTS_MAX:
        raise bad(f"example_inputs must be a list of 1 to {INPUTS_MAX} inputs", "example_inputs")
    clean, total = [], 0
    for i, item in enumerate(inputs):
        field = f"example_inputs[{i}]"
        shape = item.get("shape") if isinstance(item, dict) else None
        if (not isinstance(item, dict) or set(item) != {"shape", "dtype"} or item["dtype"] not in DTYPES
                or not isinstance(shape, list) or not 0 <= len(shape) <= DIMS_MAX
                or any(type(n) is not int or n < 1 for n in shape)):
            raise bad(f"each input is {{shape: [positive ints], dtype: one of {list(DTYPES)}}}", field)
        elems = 1
        for n in shape:
            elems *= n
        total += elems
        if elems > ELEMS_MAX or total > ELEMS_MAX:
            raise bad("the inputs are too large", field)
        clean.append({"shape": list(shape), "dtype": item["dtype"]})
    return {"provider": "torch_export", "entry": entry, "path": str(Path(path)), "example_inputs": clean, "root": root}


def _scrubbed_env() -> dict:
    """No credential, token or layerdelta setting reaches user code; it gets what Python and torch need to start."""
    env = {k: v for k, v in os.environ.items() if k in _ENV_KEEP}
    package_root = str(Path(__file__).resolve().parent.parent)
    env.update(PYTHONPATH=package_root, PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS=env.get("OMP_NUM_THREADS", "2"))
    return env


def _kill(child) -> None:
    """Kill the worker and everything it started. The tree goes first: once the parent is dead its children can no longer be found."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(child.pid)], capture_output=True, check=False)
    else:
        try:
            os.killpg(child.pid, signal.SIGKILL)  # the worker leads its own session (start_new_session)
        except ProcessLookupError:
            pass
    child.kill()
    child.wait(10)


_one_export = threading.Lock()  # ponytail: one worker at a time per process; a queue if exports ever need to overlap


def _tree_size(out: Path) -> int:
    return sum(f.stat().st_size for f in out.rglob("*") if f.is_file())


class TorchExportProvider:
    name = "torch_export"

    def __init__(self, *, enabled: bool = False, timeout: float = TIMEOUT_DEFAULT, result_max: int = RESULT_MAX):
        self.enabled, self.timeout, self.result_max = enabled, timeout, result_max

    def build(self, spec: dict, artifact_dir: Path | None = None) -> dict:
        """The GraphIR. With `artifact_dir`, the worker also saves the exported program there as `program.pt2` (or `program_error.txt`)."""
        if not self.enabled:
            raise GraphBuildError("provider_disabled", "Running user model code is not enabled for this project")
        spec = check_spec(spec)
        request = {**spec, "code_dir": spec["path"], "spec_sha256": digest(spec), "save_program": artifact_dir is not None}
        request.pop("provider"), request.pop("path")
        return run_worker("layerdelta.export_worker", request, timeout=self.timeout, result_max=self.result_max, provider=self.name,
                          artifact_dir=artifact_dir)


ARTIFACT_MAX = 1 << 30  # bytes of the saved program (it carries the exported model's weights)


def run_worker(module: str, request: dict, *, timeout: float, result_max: int, provider: str, artifact_dir: Path | None = None) -> dict:
    """Run `python -m <module>` as an isolated worker (scrubbed environment, own directory, no stdin, timeout and tree kill, size
    caps) and return the GraphIR it wrote, re-verified and checked to carry `provider`'s name. Shared by `torch_export` and `onnx`."""
    with _one_export, tempfile.TemporaryDirectory(prefix="ld-export-", ignore_cleanup_errors=True) as scratch:
        work = Path(scratch)
        out = work / "out"
        out.mkdir()
        (work / "request.json").write_text(json.dumps(request), encoding="utf-8")
        with (work / "stdout.txt").open("wb") as stdout, (work / "stderr.txt").open("wb") as stderr:
            flags = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {"start_new_session": True}
            child = popen([sys.executable, "-m", module, "--request", str(work / "request.json"),
                           "--output", str(out)], cwd=work, env=_scrubbed_env(), stdin=subprocess.DEVNULL,
                          stdout=stdout, stderr=stderr, **flags)
            deadline = time.monotonic() + timeout
            while child.poll() is None:
                if time.monotonic() > deadline:
                    _kill(child)
                    raise GraphBuildError("worker_timeout", f"the worker did not finish within {timeout:g} seconds")
                if max((work / name).stat().st_size for name in ("stdout.txt", "stderr.txt")) > LOG_MAX:
                    _kill(child)
                    raise GraphBuildError("worker_output", f"the worker wrote more than {LOG_MAX} bytes of output")
                if _tree_size(out) > result_max + 4096 + (ARTIFACT_MAX if artifact_dir else 0):  # the result, or error.json, is still being written
                    _kill(child)
                    raise GraphBuildError("result_too_large", f"the worker's output is larger than {result_max} bytes")
                time.sleep(0.05)
        _kill(child)  # a clean exit may still leave a background grandchild holding files in the scratch directory
        ir = _read_result(child.returncode, out, work / "stderr.txt", result_max, provider)
        if artifact_dir is not None:
            for name in ("program.pt2", "program_error.txt"):
                if (out / name).is_file():
                    shutil.move(str(out / name), artifact_dir / name)
        return ir


def _read_result(returncode: int, out: Path, stderr: Path, result_max: int, provider: str) -> dict:
    error, result = out / "error.json", out / "ir.json"
    if error.is_file():
        try:
            with error.open("rb") as handle:
                body = json.loads(handle.read(4096).decode("utf-8", errors="replace"))
            code, message = str(body["code"]), str(body["message"])[:512]
        except (ValueError, KeyError, TypeError):
            code, message = "export_failed", "the worker reported an unreadable error"
        raise GraphBuildError(code if re.fullmatch(r"[a-z_]{1,40}", code) else "export_failed", message)
    if returncode != 0 or not result.is_file():
        tail = stderr.read_text(encoding="utf-8", errors="replace")[-300:] if stderr.is_file() else ""
        raise GraphBuildError("worker_crashed", f"the worker exited with status {returncode} and no result: {tail.strip()}"[:512])
    if result.stat().st_size > result_max:
        raise GraphBuildError("result_too_large", f"the worker's result is larger than {result_max} bytes")
    try:
        ir = json.loads(result.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except (ValueError, UnicodeDecodeError) as exc:
        raise GraphBuildError("invalid_graph", f"the worker's result is not JSON: {exc}"[:512]) from exc
    try:
        verify_ir(ir)  # schema, references, hash: the parent trusts nothing it was handed
    except InspectError as exc:
        raise GraphBuildError("invalid_graph", exc.message) from exc
    if ir["provider"]["name"] != provider:  # the dispatcher decides the provider, not the worker
        raise GraphBuildError("provider_mismatch", f"the worker's graph says it was built by {ir['provider']['name']!r}")
    return ir


def _reject_constant(name):
    raise ValueError(f"{name} is not allowed in JSON")
