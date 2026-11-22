"""Start the LayerDelta project server on an ephemeral loopback port."""

import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import threading
import time

import uvicorn

from .api import Credentials, create_app
from .checkpoints import ImportLimits
from .fixture import build_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--assets", type=Path, help="Built desktop renderer directory")
    parser.add_argument("--desktop", action="store_true")
    parser.add_argument("--quota", type=int, default=10)
    parser.add_argument("--agent-weights-write", action="store_true",
                        help="Grant the agent token weights:write (apply edits, upload blobs); needs --agent-weights-read")
    parser.add_argument("--max-import-bytes", type=int, default=ImportLimits.max_file_bytes,
                        help="Largest single checkpoint file an import or upload may bring")
    parser.add_argument("--agent-token-file", type=Path, help="Create a new file with local agent connection details")
    parser.add_argument("--agent-weights-read", action="store_true",
                        help="Grant the agent token the weights:read capability (kernel statistics). Off by default.")
    args = parser.parse_args()
    if args.quota < 0:
        parser.error("quota must be nonnegative")
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    if not (workspace / "reference").exists():
        build_fixture(workspace / "reference", workspace / "mnist")
    project = hashlib.sha256(str(workspace).encode()).hexdigest()[:24]
    if args.agent_weights_write and not args.agent_weights_read:
        parser.error("--agent-weights-write needs --agent-weights-read")
    creds = Credentials(project, weights_read=args.agent_weights_read, weights_write=args.agent_weights_write)
    limits = ImportLimits(max_file_bytes=args.max_import_bytes, max_total_bytes=4 * args.max_import_bytes)
    app = create_app(workspace, creds, args.assets, quota=args.quota, import_limits=limits)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        agent = {"url": url, "project_id": project, "token": creds.agent}
        desktop = {"url": url + "/#" + creds.ticket, "project_id": project}
        server = uvicorn.Server(uvicorn.Config(app, access_log=False, log_level="warning"))
        if args.agent_token_file:
            import os

            fd = os.open(args.agent_token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(agent, f)

        def announce():
            while not server.started and not server.should_exit:
                time.sleep(.025)
            if server.started:
                print(json.dumps(desktop if args.desktop else agent), flush=True)
                if args.desktop:
                    sys.stdin.buffer.read(1)
                    server.should_exit = True

        threading.Thread(target=announce, daemon=True).start()
        try:
            server.run(sockets=[sock])
        finally:
            if args.agent_token_file:
                args.agent_token_file.unlink(missing_ok=True)
        if not server.started:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
