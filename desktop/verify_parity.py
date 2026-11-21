"""Replay a published desktop run through the typed agent in a fresh workspace."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from uuid import uuid4

from layerdelta.client import AgentClient, wait_for
from layerdelta.schemas import Run


def replay(workspace, run_id):
    source = workspace / "runs" / run_id
    frozen = json.loads((source / "recipe.json").read_bytes())
    expected = json.loads((source / "comparison.json").read_bytes())
    report = json.loads((source / "report.json").read_bytes())
    device = report["runtime"]["requested_device"]
    with TemporaryDirectory(prefix="parity-", dir=workspace.parent) as name:
        target = Path(name)
        for item in ("reference", "mnist"):
            shutil.copytree(workspace / item, target / item)
        config = target / "agent.json"
        command = [sys.executable, "-I", "-m", "layerdelta.server", "--workspace", str(target),
                   "--desktop", "--agent-token-file", str(config)]
        with subprocess.Popen(command, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as server:
            try:
                if not server.stdout.readline():
                    raise RuntimeError("Agent server failed to start")
                with AgentClient(**json.loads(config.read_bytes())) as client:
                    imported = client.import_recipe(frozen)
                    assert imported["recipe_hash"] == frozen["recipe_hash"]
                    job = client.submit(Run(kind="intervention", device=device, recipe_id=imported["recipe_id"]),
                                        key=uuid4().hex)
                    job_id = job["run_id"]
                    wait_for(lambda: client.status(job_id))
                    result = client.comparison(job_id)
                    changes = list(client.changes(job_id))
                    assert result["slices"] == expected["slices"]
                    assert changes == expected["changes"]
                    assert result["runtime"]["actual_device"] == report["runtime"]["actual_device"]
                    return {
                        "desktop_run": run_id, "agent_run": job_id, "recipe_hash": imported["recipe_hash"],
                        "actual_device": result["runtime"]["actual_device"], "changes": len(changes),
                        "slices": len(result["slices"]),
                        "changed_ids_sha256": hashlib.sha256(json.dumps(
                            [row["sample_id"] for row in changes]).encode()).hexdigest(),
                    }
            finally:
                server.stdin.close()
                try:
                    server.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("run_id")
    options = parser.parse_args()
    print(json.dumps(replay(options.workspace.resolve(), options.run_id)))
