"""Run the packaged spike outside the checkout with developer tools off PATH."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, help="Also test an external Python runtime")
    args = parser.parse_args()
    package = Path(__file__).resolve().parent / "dist" / "LayerDeltaSpike"
    with TemporaryDirectory(prefix="layerdelta-desktop-") as temp:
        stage = Path(temp)
        app = Path(shutil.copytree(package, stage / "app"))
        package_bytes = sum(path.stat().st_size for path in app.rglob("*") if path.is_file())
        workspace = stage / "ทดลอง workspace"
        workspace.mkdir()
        env = dict(os.environ, PATH=os.pathsep.join([os.environ["SystemRoot"] + r"\System32", os.environ["SystemRoot"]]))
        cases = [("bundled", [])]
        if args.runtime is not None:
            cases.append(("external", ["--runtime", args.runtime.resolve(strict=True)]))
        for profile, extra in cases:
            result = subprocess.run(
                [app / "LayerDeltaSpike.exe", "--smoke", "--workspace", workspace, *extra],
                cwd=workspace, env=env, stdout=subprocess.PIPE, timeout=60, check=True,
            )
            evidence = json.loads(result.stdout)
            if evidence["workspace"] != workspace.name or evidence["worker_exit"] != 0:
                raise RuntimeError(f"Workspace or worker mismatch: {evidence}")
            print(json.dumps(dict(profile=profile, package_bytes=package_bytes, **evidence)))


if __name__ == "__main__":
    main()
