"""Exercise Agent/human parity, workspace replay and verified child export over loopback HTTP."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from queue import Queue
import shutil
import subprocess
import sys
from threading import Thread
from uuid import uuid4

import httpx
import torch

from layerdelta.agent_demo import experiment
from layerdelta.checkpoint import load_checkpoint
from layerdelta.client import AgentClient, wait_for
from layerdelta.schemas import Run


@contextmanager
def project(root, source):
    root.mkdir()
    for name in ("reference", "mnist"):
        shutil.copytree(source / name, root / name)
    token_file = root / "agent-session.json"
    with subprocess.Popen([sys.executable, "-m", "layerdelta.server", "--workspace", str(root),
                           "--desktop", "--agent-token-file", str(token_file)], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True) as proc:
        try:
            lines = Queue()
            Thread(target=lambda: lines.put(proc.stdout.readline()), daemon=True).start()
            origin, ticket = json.loads(lines.get(timeout=90))["url"].split("/#")
            connection = json.loads(token_file.read_bytes())
            project_path = f"/api/v1/projects/{connection['project_id']}/"
            with httpx.Client(base_url=origin, trust_env=False) as session, AgentClient(**connection) as agent:
                auth = session.post("/api/v1/session", json={"ticket": ticket}).raise_for_status().json()
                session.headers.update({"X-CSRF-Token": auth["csrf"]})

                def human(method, path, **kwargs):
                    if method == "POST":
                        kwargs["headers"] = {"Idempotency-Key": uuid4().hex}
                    return session.request(method, project_path + path, **kwargs).raise_for_status()

                yield agent, human
        finally:
            proc.stdin.close()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def human_job(human, route, payload):
    accepted = human("POST", route, json=payload).json()
    poll = "runs" if route == "final-evaluations" else route
    return wait_for(lambda: human("GET", f"{poll}/{accepted['run_id']}").json())


def equivalent(left, right):
    for field in ("slices", "changes", "recipe_hash", "evaluation_fingerprint"):
        if left[field] != right[field]:
            raise ValueError(f"Comparison differs at {field}")


def assert_child(parent, child, spec, mapping):
    a, b = load_checkpoint(parent).state_dict(), load_checkpoint(child).state_dict()
    parameter, channel = mapping["parameter_name"], spec["channel_index"]
    for name, tensor in a.items():
        expected = tensor.clone()
        if name == parameter:
            expected[channel].zero_()
            if torch.equal(expected, tensor):
                raise ValueError("Acceptance channel already zero before edit")
        if not torch.equal(expected, b[name]):
            raise ValueError(f"Unexpected child tensor diff in {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(".layerdelta"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    evidence = {}
    with project(args.output / "author", args.source) as (agent, human):
        for device in args.devices:
            print(f"{device}: previews, deterministic Agent and human session", flush=True)
            payloads = []

            def stub(payload):
                from layerdelta.agent_demo import hypothesis
                payloads.append(payload)
                return hypothesis(payload)

            trial = experiment(agent, device=device, mode="weight", propose_hypothesis=stub)
            assert len(payloads) == 1 and trial["submissions"] == 1
            assert set(payloads[0]) == {"target_id", "truth", "channel_index", "blurred", "normal"}
            for condition in ("blurred", "normal"):
                assert set(payloads[0][condition]) == {"sample_id", "mean", "max", "zero_fraction"}
            recipe = agent.get_recipe(trial["recipe_id"])
            run = human_job(human, "runs", {"kind": "intervention", "device": device, "recipe_id": trial["recipe_id"]})
            human_id = run["run_id"]
            source = agent.comparison(trial["run_id"])
            equivalent(source, human("GET", f"runs/{human_id}/comparison").json())
            assert trial["changes"] == source["changes"]
            confirmed = human_job(human, "final-evaluations", {
                "device": device, "recipe_id": trial["recipe_id"], "recipe_hash": trial["recipe_hash"], "confirmed": True,
            })
            final_report = human("GET", f"runs/{confirmed['run_id']}/report").json()
            assert final_report["split"] == "holdout" and final_report["sample_count"] == 10002
            exported = human_job(human, "checkpoint-exports", {"source_run_id": human_id, "confirmed": True})
            child = human("GET", f"checkpoint-exports/{exported['run_id']}/file").content
            assert exported["result"]["child_sha256"] == hashlib.sha256(child).hexdigest()
            child_path = args.output / f"{device}-child.safetensors"
            child_path.write_bytes(child)
            assert_child(args.source / "reference/model.safetensors", child_path, recipe["intervention"], recipe["mapping"])
            report = human("GET", f"runs/{human_id}/report").json()
            evidence[device] = {
                "agent_run": trial["run_id"], "human_run": human_id, "holdout_run": confirmed["run_id"],
                "recipe": recipe, "source_comparison": source, "child_digest": exported["result"]["child_sha256"],
                "predictions_digest": report["artifacts"]["predictions.json"]["sha256"],
                "source_samples": report["sample_count"], "changed_samples": len(trial["changes"]),
            }
    with project(args.output / "replay", args.source) as (agent, human):
        for device, result in evidence.items():
            print(f"{device}: import, replay and child-only baseline", flush=True)
            recipe = agent.import_recipe(result.pop("recipe"))
            replay = agent.submit(Run(kind="intervention", recipe_id=recipe["recipe_id"], device=device), key=uuid4().hex)
            replay_id = replay["run_id"]
            wait_for(lambda: agent.status(replay_id))
            equivalent(result.pop("source_comparison"), agent.comparison(replay_id))
            child = (args.output / f"{device}-child.safetensors").read_bytes()
            imported = human("POST", "checkpoints", files={"file": ("child.safetensors", child)}).json()
            assert imported["checkpoint_id"] == result["child_digest"]
            baseline = human_job(human, "runs", {"kind": "baseline", "checkpoint_id": imported["checkpoint_id"],
                                                "device": device})
            report = human("GET", f"runs/{baseline['run_id']}/report").json()
            assert report["recipe_hash"] is None
            assert report["artifacts"]["predictions.json"]["sha256"] == result["predictions_digest"]
            result.update(replay_run=replay_id, child_run=baseline["run_id"], recipe_hash=recipe["recipe_hash"],
                          actual_device=report["runtime"]["actual_device"])
    output = args.output / "acceptance.json"
    output.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
