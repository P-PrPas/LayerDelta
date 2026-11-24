"""One benchmark session in Antigravity (`agy`), headless, with a transcript audit (v0.5-E7b, design §5.2; evidence: docs/research/v0.5-e7s-antigravity-spike.md).

This is the part of the session driver that both isolation options need. What it does:
- **launches** `agy -p <task>` in the session directory with a model, an effort and `--output-format stream-json`, one session at a time;
- **enforces the budgets** the design pre-registers: wall time, steps (the tool steps of the stream) and completion tokens (output + thinking), by killing the
  whole process tree; total tokens (prompt, cache reads included) are reported, not limited;
- **keeps the conversation store clean**: the CLI's store (`~/.gemini/antigravity-cli/conversations`) is shared by every session of the OS account, so session k
  could read the other arm's transcripts. The store is moved aside for the session and put back after, and a backup left by a crashed run stops the next one;
- **manages arm L's MCP bridge**, which the CLI configures globally (`agy mcp add|remove`): added before, removed after; arm R refuses to start while it is configured;
- **audits** every tool call of the transcript: web and browser tools; file reads or writes outside the session directory; anything under the agent's own store,
  the LayerDelta workspace, the MNIST archives or the holdout (the `protected` roots); downloads and training signatures in commands and in code written.

What it does not do: prevent anything. A violation is reported after it happened; prevention (a dedicated OS account and an outbound firewall rule) is the owner's
decision (Q5) and is not here. `--dangerously-skip-permissions` is only passed when the caller says so explicitly.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = "gemini-4-hydrogen-high"
EFFORT = "high"
STORE = Path.home() / ".gemini" / "antigravity-cli"
MCP_NAME = "layerdelta"

WEB_TOOLS = {"search_web", "read_url_content", "open_browser_url", "execute_browser_javascript", "read_browser_page", "list_browser_pages",
             "capture_browser_screenshot", "capture_browser_console_logs", "click_browser_pixel"}
READ_TOOLS = {"read_file", "view_file", "list_dir", "grep_search", "find_by_name", "sed_file", "read_resource", "list_resources"}
WRITE_TOOLS = {"write_to_file", "replace_file_content", "multi_replace_file_content", "notebook_edit"}
COMMAND_TOOLS = {"run_command", "send_command_input"}
_PATH_KEYS = re.compile(r"path|file|directory|dir|folder|target|cwd", re.IGNORECASE)
_ABSOLUTE = re.compile(r"(?<![\w.])([A-Za-z]:[\\/][^\s\"'`|;&<>]*|/[A-Za-z]/[^\s\"'`|;&<>]+|~[\\/][^\s\"'`|;&<>]*|%USERPROFILE%[^\s\"'`|;&<>]*|\$HOME[^\s\"'`|;&<>]*|\$env:USERPROFILE[^\s\"'`|;&<>]*)")


class StoreBusy(RuntimeError):
    """A conversation-store backup from an earlier session is still there (the run crashed): put it back by hand before another session."""


class McpConflict(RuntimeError):
    """Arm R would start with the LayerDelta bridge configured, or the bridge could not be added or removed."""


@dataclass
class Limits:
    wall_seconds: float = 1800.0
    steps: int = 200
    completion_tokens: int = 200_000


@dataclass
class Stream:
    """What the stream-json transcript says, as it arrives."""
    steps: int = 0                  # DONE tool steps
    completion_tokens: int = 0      # output + thinking tokens of every agent response
    total_tokens: int = 0           # as the host counts them (prompt, cache reads included in `total`)
    input_tokens: int = 0
    cache_read_tokens: int = 0
    calls: list = field(default_factory=list)
    conversation_id: str | None = None
    tools: list = field(default_factory=list)
    result: dict | None = None
    lines: int = 0

    def feed(self, event: dict) -> None:
        self.lines += 1
        kind = event.get("event")
        if kind == "init":
            self.conversation_id = event["init"].get("conversation_id") or event.get("conversation_id")
            self.tools = event["init"].get("tools", [])
        elif kind == "step_update":
            step = event["step_update"]
            if step.get("state") != "DONE":
                return
            if step.get("step_type") == "tool":
                self.steps += 1
                info = step.get("tool_info") or {}
                self.calls.append({"step": step.get("step_index"), "name": step.get("tool_name") or info.get("name"), "parameters": info.get("parameters") or {},
                                   "output": str(info.get("output", ""))[:2000], "seconds": step.get("duration_seconds")})
            elif step.get("usage"):
                u = step["usage"]
                self.completion_tokens += int(u.get("output_tokens", 0)) + int(u.get("thinking_tokens", 0))
                self.total_tokens += int(u.get("total_tokens", 0))
                self.input_tokens += int(u.get("input_tokens", 0))
                self.cache_read_tokens += int(u.get("cache_read_tokens", 0))
        elif kind == "result":
            self.result = event["result"]


# ---------------------------------------------------------------- the audit

def _inside(path: Path, root: Path) -> bool:
    try:
        Path(os.path.normcase(os.path.abspath(path))).relative_to(Path(os.path.normcase(os.path.abspath(root))))
        return True
    except ValueError:
        return False


def _expand(text: str, session: Path) -> Path:
    home = str(Path.home())
    text = text.replace("%USERPROFILE%", home).replace("$env:USERPROFILE", home).replace("$HOME", home)
    if text.startswith("~"):
        text = home + text[1:]
    m = re.match(r"^/([A-Za-z])/(.*)$", text)  # Git-bash style /c/Users/...
    if m:
        text = f"{m.group(1).upper()}:/{m.group(2)}"
    p = Path(text.rstrip(".,;:)]}'\""))
    return p if p.is_absolute() else Path(os.path.abspath(session / p))


def _string_values(value, key=""):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _string_values(v, str(k))
    elif isinstance(value, list):
        for v in value:
            yield from _string_values(v, key)
    elif isinstance(value, str):
        yield key, value


def audit(calls: list, session: Path, *, protected=(), store: Path = STORE) -> list[dict]:
    """The violations (and reviews) of a transcript's tool calls. `protected`: roots an agent must never touch (the LayerDelta workspace, the MNIST archives,
    the holdout); the agent's own store is always protected. A `violation` is what the pre-registered rule counts; a `review` is reported, not counted."""
    from importlib import import_module
    sys.path.insert(0, str(HERE))
    try:
        mnist_bench = import_module("mnist_bench")
    finally:
        sys.path.pop(0)
    session = Path(session)
    store = Path(store)
    roots = [store, *([store.parent] if store.parent.name == ".gemini" else []), *map(Path, protected)]  # ~/.gemini holds the conversations of every session
    out = []

    def add(kind, call, detail, severity="violation"):
        out.append({"kind": kind, "severity": severity, "step": call["step"], "tool": call["name"], "detail": str(detail)[:300]})

    def protected_hit(p: Path):
        return next((r for r in roots if _inside(p, r)), None)

    for call in calls:
        name, params = call["name"], call["parameters"]
        if name in WEB_TOOLS or (name or "").startswith("browser_"):
            add("web_tool", call, name)
            continue
        if name in READ_TOOLS | WRITE_TOOLS:
            for key, value in _string_values(params):
                if not _PATH_KEYS.search(key) or "\n" in value or len(value) > 400:
                    continue
                p = _expand(value, session)
                if protected_hit(p):
                    add("protected_path", call, f"{name} {key}={value}")
                elif not _inside(p, session):
                    add("write_outside" if name in WRITE_TOOLS else "read_outside", call, f"{key}={value}")
        if name in WRITE_TOOLS | COMMAND_TOOLS:  # code the agent wrote or ran
            text = "\n".join(v for _, v in _string_values(params))
            found = mnist_bench.audit_text(text)
            if found["training"]:
                add("training", call, found["training"])
            if found["download"]:
                add("download", call, found["download"])
        if name in COMMAND_TOOLS:
            text = " ".join(v for k, v in _string_values(params) if k.lower() in ("commandline", "command", "input"))
            for m in _ABSOLUTE.finditer(text):
                p = _expand(m.group(1), session)
                if protected_hit(p):
                    add("protected_path", call, m.group(1))
                elif not _inside(p, session):
                    add("path_outside", call, m.group(1), severity="review")
            if ".." in re.findall(r"[^\s\"']+", text) or re.search(r"(?:^|[\s\"'\\/])\.\.[\\/]", text):
                add("path_outside", call, "a parent directory", severity="review")
    return out


# ---------------------------------------------------------------- the store and the bridge

class Quarantine:
    """Move the conversation store aside for one session and put it back (also when the run fails). A backup that is already there is never overwritten."""

    def __init__(self, store: Path = STORE):
        self.store, self.backup = Path(store) / "conversations", Path(store) / "conversations.bench-backup"

    def __enter__(self):
        if self.backup.exists():
            raise StoreBusy(f"{self.backup} exists: a session crashed before it could restore the conversation store; restore it by hand")
        if self.store.exists():
            self.store.rename(self.backup)
        self.store.mkdir(parents=True, exist_ok=True)
        return self

    def collect(self, destination: Path) -> None:
        """Copy what this session wrote into the store (its conversation) next to the transcript."""
        if self.store.exists():
            shutil.copytree(self.store, destination, dirs_exist_ok=True)

    def __exit__(self, *exc):
        shutil.rmtree(self.store, ignore_errors=True)
        if self.backup.exists():
            self.backup.rename(self.store)
        return False


def _agy(agy: list, *args, timeout=60) -> subprocess.CompletedProcess:
    return subprocess.run([*agy, *args], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)


def mcp_names(agy: list) -> set:
    out = _agy(agy, "mcp", "list").stdout
    return {line.split()[0] for line in out.splitlines() if line.strip() and not line.startswith(("No MCP", " "))}


def _kill_tree(process: subprocess.Popen) -> None:
    if process.poll() is None:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], capture_output=True, check=False)
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.kill()
    process.wait(15)


# ---------------------------------------------------------------- the session

def run_session(arm: str, session: Path, task: str, *, results: Path, agy: list | None = None, model: str = MODEL, effort: str = EFFORT,
                limits: Limits = Limits(), store: Path = STORE, mcp: list | None = None, protected=(), skip_permissions: bool = False,
                env: dict | None = None) -> dict:
    """Run one session of arm "L" (with the LayerDelta bridge, `mcp` = the command to add) or "R" (without) and return its record.
    `session` is the fresh directory (its kit is already in it); `results` receives `transcript.ndjson`, `stderr.txt` and the session's conversation."""
    if arm not in ("L", "R"):
        raise ValueError("arm is L or R")
    agy = agy or ["agy"]
    session, results = Path(session), Path(results)
    results.mkdir(parents=True, exist_ok=True)
    configured = mcp_names(agy)
    if arm == "R" and MCP_NAME in configured:
        raise McpConflict(f"the {MCP_NAME} MCP server is configured: remove it (agy mcp remove {MCP_NAME}) before an arm R session")
    if arm == "L" and (not mcp or MCP_NAME in configured):
        raise McpConflict("arm L needs `mcp` (the bridge command) and an unconfigured host")
    argv = [*agy, "-p", task, "--model", model, "--effort", effort, "--output-format", "stream-json", "--print-timeout", f"{int(limits.wall_seconds)}s"]
    if skip_permissions:
        argv.append("--dangerously-skip-permissions")
    stream, stopped, started = Stream(), None, time.monotonic()
    with Quarantine(store) as quarantine:
        added = False
        try:
            if arm == "L":
                done = _agy(agy, "mcp", "add", MCP_NAME, "--", *mcp)
                if done.returncode != 0:
                    raise McpConflict(f"agy mcp add failed: {done.stderr.strip()[:200]}")
                added = True
            flags = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)} if os.name == "nt" else {"start_new_session": True}
            with (results / "transcript.ndjson").open("w", encoding="utf-8") as transcript, (results / "stderr.txt").open("wb") as errors:
                process = subprocess.Popen(argv, cwd=session, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=errors, env=env, text=True,
                                           encoding="utf-8", errors="replace", **flags)
                lock = threading.Lock()

                def read():
                    for line in process.stdout:
                        transcript.write(line)
                        transcript.flush()
                        try:
                            event = json.loads(line)
                        except ValueError:
                            continue
                        with lock:
                            stream.feed(event)

                reader = threading.Thread(target=read, daemon=True)
                reader.start()
                try:
                    while process.poll() is None:
                        with lock:
                            over = ("steps" if stream.steps > limits.steps else "tokens" if stream.completion_tokens > limits.completion_tokens else None)
                        if over is None and time.monotonic() - started > limits.wall_seconds + 5:
                            over = "wall"
                        if over:
                            stopped = over
                            _kill_tree(process)
                            break
                        time.sleep(0.05)
                finally:
                    _kill_tree(process)  # a clean exit may still leave a shell the agent started
                    reader.join(10)
            quarantine.collect(results / "conversations")
        finally:
            if added:
                _agy(agy, "mcp", "remove", MCP_NAME)
    violations = audit(stream.calls, session, protected=protected, store=store)  # the reader has finished
    record = {"arm": arm, "model": model, "effort": effort, "stopped": stopped, "exit_code": process.returncode, "seconds": round(time.monotonic() - started, 2),
              "steps": stream.steps, "completion_tokens": stream.completion_tokens, "total_tokens": stream.total_tokens, "input_tokens": stream.input_tokens,
              "cache_read_tokens": stream.cache_read_tokens, "status": (stream.result or {}).get("status"), "conversation_id": stream.conversation_id,
              "tool_calls": len(stream.calls), "tools_offered": len(stream.tools), "violations": [v for v in violations if v["severity"] == "violation"],
              "reviews": [v for v in violations if v["severity"] == "review"], "skip_permissions": skip_permissions}
    (results / "session.json").write_text(json.dumps({**record, "calls": stream.calls}, indent=1), encoding="utf-8")
    return record


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run one benchmark session (a smoke test of the driver; the benchmark harness calls run_session).")
    parser.add_argument("--arm", choices=("L", "R"), required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--wall", type=float, default=600)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--completion-tokens", type=int, default=100_000)
    parser.add_argument("--allow-skip-permissions", action="store_true", help="pass --dangerously-skip-permissions (the owner's decision)")
    parser.add_argument("--protect", type=Path, action="append", default=[])
    parser.add_argument("--mcp", nargs="+", help="arm L: the bridge command, e.g. layerdelta-mcp --connection FILE")
    args = parser.parse_args(argv)
    record = run_session(args.arm, args.session, args.task_file.read_text(encoding="utf-8"), results=args.results, mcp=args.mcp, protected=args.protect,
                         limits=Limits(args.wall, args.steps, args.completion_tokens), skip_permissions=args.allow_skip_permissions)
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
