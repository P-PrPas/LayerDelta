"""Launch the local LayerDelta server and display its React workspace in WebView2."""

import argparse
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Thread
from urllib.parse import urlsplit


def choose_workspace():
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    try:
        selected = filedialog.askdirectory(parent=root, title="Open LayerDelta workspace")
        return Path(selected) if selected else None
    finally:
        root.destroy()


def select_runtime(explicit):
    if explicit:
        return explicit.resolve(strict=True)
    if getattr(sys, "frozen", False):
        candidate = Path(sys.executable).parent / "runtime" / "Scripts" / "python.exe"
        if not candidate.is_file():
            raise ValueError("Install the CPU or CUDA runtime next to LayerDelta.exe, or pass --runtime.")
        return candidate
    return Path(sys.executable)


def launch(args):
    import webview

    workspace = args.workspace or choose_workspace()
    if workspace is None:
        return
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    assets = Path(__file__).resolve().parent / "dist"
    if not (assets / "index.html").is_file():
        raise ValueError("Build the desktop renderer with desktop/build.ps1")
    command = [select_runtime(args.runtime), "-I", "-m", "layerdelta.server",
               "--workspace", workspace, "--desktop", "--assets", assets]
    logfile = workspace / "desktop-server.log"
    with logfile.open("a", encoding="utf-8") as log, subprocess.Popen(
        command, cwd=workspace, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
        text=True, encoding="utf-8", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ) as server:
        ready = Queue()
        Thread(target=lambda: ready.put(server.stdout.readline()), daemon=True).start()
        try:
            try:
                line = ready.get(timeout=600)
            except Empty:
                raise RuntimeError(f"Workspace startup timed out. See {logfile}") from None
            if not line:
                raise RuntimeError(f"Server exited during startup. See {logfile}")
            url = json.loads(line)["url"]
            parsed = urlsplit(url)
            if parsed.hostname != "127.0.0.1" or parsed.scheme != "http" or not parsed.fragment:
                raise ValueError("Invalid desktop session URL")
            webview.settings["ALLOW_DOWNLOADS"] = True
            window = webview.create_window(f"LayerDelta · {workspace.name}", url, hidden=bool(args.smoke),
                                           width=1280, height=850, min_size=(390, 490))
            result = {}

            def test():
                from smoke import exercise
                try:
                    result.update(exercise(window, args.smoke))
                except Exception as error:
                    result["error"] = str(error)
                finally:
                    window.destroy()

            webview.start(test if args.smoke else None, gui="edgechromium", private_mode=True)
            if args.smoke:
                if "error" in result:
                    raise RuntimeError(result["error"])
                print(json.dumps(result), flush=True)
        finally:
            server.stdin.close()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--runtime", type=Path, help="Python executable with the selected LayerDelta profile")
    parser.add_argument("--smoke", choices=("cpu", "cuda"), help="Exercise the installed workbench in a hidden window")
    args = parser.parse_args()
    if args.smoke and not args.workspace:
        parser.error("--smoke requires a test workspace")
    try:
        launch(args)
    except Exception as error:
        if not args.smoke and os.name == "nt":
            import ctypes
            ctypes.windll.user32.MessageBoxW(0, str(error), "LayerDelta startup", 0x10)
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
