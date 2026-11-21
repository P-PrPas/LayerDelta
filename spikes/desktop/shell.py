"""Windows launcher prototype. --smoke exercises the bundled React/WebView2 path."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def pick_workspace():
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    try:
        selection = filedialog.askdirectory(parent=root, title="Open LayerDelta workspace")
        return Path(selection) if selection else None
    finally:
        root.destroy()


def main():
    import webview

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--runtime", type=Path, help="External Python executable for the worker")
    parser.add_argument("--smoke", action="store_true", help="Render, assert the workspace, then exit")
    args = parser.parse_args()
    if args.smoke and args.workspace is None:
        parser.error("--smoke requires --workspace")
    workspace = args.workspace or pick_workspace()
    if workspace is None:
        return
    if not workspace.is_dir():
        parser.error("workspace must be an existing directory")
    workspace = workspace.resolve()
    base = Path(__file__).resolve().parent
    if args.runtime is not None:
        command = [args.runtime.resolve(strict=True), "-I", base / "worker.py"]
    elif getattr(sys, "frozen", False):
        command = [sys.executable, "--worker"]
    else:
        command = [sys.executable, "-I", base / "worker.py"]
    command += ["--workspace", workspace, "--assets", base / "static"]
    start = time.perf_counter()
    worker = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    evidence = {}
    try:
        ready = json.loads(worker.stdout.readline())
        window = webview.create_window(
            "LayerDelta", ready["url"], hidden=args.smoke, width=960, height=680,
        )

        def verify_renderer():
            try:
                for _ in range(100):
                    value = window.evaluate_js(
                        """document.querySelector('[data-state="ready"] output')?.textContent"""
                    )
                    if value == workspace.name:
                        evidence.update(
                            workspace=value, worker_pid=ready["pid"],
                            startup_seconds=round(time.perf_counter() - start, 3),
                        )
                        break
                    time.sleep(0.1)
            finally:
                window.destroy()

        webview.start(verify_renderer if args.smoke else None, gui="edgechromium")
    finally:
        worker.stdin.close()
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()
            raise RuntimeError("worker failed to exit on parent pipe close")
        finally:
            worker.stdout.close()
    if args.smoke:
        if not evidence or worker.returncode != 0:
            raise SystemExit("renderer or worker shutdown check failed")
        print(json.dumps({**evidence, "worker_exit": worker.returncode}), flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--worker"]:
        from worker import main as worker_main

        del sys.argv[1]
        worker_main()
    else:
        main()
