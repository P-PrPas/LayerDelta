"""A stand-in for the `agy` CLI, for tests of tools/bench/agy_session.py (no host, no cost). It speaks just enough: `mcp list|add|remove` on a JSON file and
`-p <task> ...` printing the stream-json events of a script (FAKE_AGY_SCRIPT), recording its argv and what it saw in FAKE_AGY_RECORD."""

import json
import os
import sys
import time
from pathlib import Path

args = sys.argv[1:]
if args[:1] == ["mcp"]:
    path = Path(os.environ["FAKE_AGY_MCP"])
    state = json.loads(path.read_text()) if path.exists() else {}
    if args[1] == "list":
        print("\n".join(f"{name}  stdio  {' '.join(command)}" for name, command in state.items()) or "No MCP servers configured.")
    elif args[1] == "add":
        rest = args[2:]
        state[rest[0]] = [a for a in rest[1:] if a != "--"]
        path.write_text(json.dumps(state))
    elif args[1] == "remove":
        state.pop(args[2], None)
        path.write_text(json.dumps(state))
    sys.exit(0)

store = Path(os.environ["FAKE_AGY_STORE"]) / "conversations"
Path(os.environ["FAKE_AGY_RECORD"]).write_text(json.dumps({"argv": args, "cwd": os.getcwd(), "store_entries": sorted(p.name for p in store.glob("*")) if store.exists() else None,
                                                           "mcp": json.loads(Path(os.environ["FAKE_AGY_MCP"]).read_text()) if Path(os.environ["FAKE_AGY_MCP"]).exists() else {}}))
(store / "this-session.json").write_text("{}")
for event in json.loads(Path(os.environ["FAKE_AGY_SCRIPT"]).read_text()):
    pause = event.pop("_sleep", 0)
    print(json.dumps(event), flush=True)
    time.sleep(pause)
