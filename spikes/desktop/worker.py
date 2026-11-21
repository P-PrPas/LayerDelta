"""Read-only loopback server used to exercise the renderer/worker boundary."""

import argparse
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
from threading import Thread


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, assets, workspace, **kwargs):
        self.assets = assets
        self.workspace = workspace
        super().__init__(*args, **kwargs)

    def do_GET(self):
        host, port = self.server.server_address
        if self.headers.get("Host") != f"{host}:{port}":
            self.send_error(403)
            return
        if self.path == "/workspace":
            content_type = "application/json"
            payload = json.dumps({"name": self.workspace.name}).encode()
        elif self.path in ("/", "/app.js"):
            name, content_type = (
                ("index.html", "text/html") if self.path == "/"
                else ("app.js", "application/javascript")
            )
            payload = (self.assets / name).read_bytes()
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Content-Security-Policy", "default-src 'self'; object-src 'none'")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    if not args.workspace.is_dir():
        parser.error("workspace must be an existing directory")
    handler = partial(Handler, assets=args.assets, workspace=args.workspace)
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        host, port = server.server_address
        print(json.dumps({"url": f"http://{host}:{port}/", "pid": os.getpid()}), flush=True)

        def wait_for_shell():
            sys.stdin.buffer.read(1)
            server.shutdown()

        Thread(target=wait_for_shell, daemon=True).start()
        server.serve_forever(poll_interval=0.1)


if __name__ == "__main__":
    main()
