"""Run the two scripted pilot arms and frozen holdout on an isolated local service."""

import argparse
from contextlib import contextmanager
import hashlib
import http.cookiejar
import json
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from urllib.error import HTTPError
from urllib.request import HTTPCookieProcessor, Request, build_opener
import uuid


class Session:
    def __init__(self, url, project, headers):
        self.url = url
        self.base = "/api/v1/projects/" + project
        self.headers = headers
        self.opener = build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def request(self, path, data=None, **extra):
        url = self.url + (path if path.startswith("/api/") else self.base + path)
        body = None if data is None else json.dumps(data).encode()
        req = Request(url, body, {"Content-Type": "application/json", **self.headers, **extra})
        with self.opener.open(req, timeout=30) as response:
            return json.load(response)

    def job(self, path, data, poll_path):
        job = self.request(path, data, **{"Idempotency-Key": uuid.uuid4().hex})
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            job = self.request(f"{poll_path}/{job['run_id']}")
            if job["status"] == "succeeded":
                return job
            if job["status"] in ("failed", "cancelled"):
                raise RuntimeError(f"{job['run_id']}: {job['error']}")
            time.sleep(.25)
        raise TimeoutError(job["run_id"])


@contextmanager
def service(workspace):
    # Desktop and agent receive independent credentials through private local channels.
    ticket_file = workspace / "scripted-agent.json"
    proc = subprocess.Popen([sys.executable, "-m", "layerdelta.server", "--workspace", str(workspace),
                             "--desktop", "--agent-token-file", str(ticket_file), "--quota", "1"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        ready = queue.Queue()
        threading.Thread(target=lambda: ready.put(proc.stdout.readline()), daemon=True).start()
        boot = json.loads(ready.get(timeout=120))
        url, ticket = boot["url"].split("/#")
        project = boot["project_id"]
        human = Session(url, project, {})
        human.headers["X-CSRF-Token"] = human.request("/api/v1/session", {"ticket": ticket})["csrf"]
        token = json.loads(ticket_file.read_bytes())["token"]
        yield Session(url, project, {"Authorization": "Bearer " + token}), human
    finally:
        proc.stdin.close()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        proc.stdout.close()


def arm(agent, informed, device):
    catalog = agent.request("/catalog")
    sample = catalog["demo"]["example_pairs"][0]["blurred"]["sample_id"]
    target = next(t for t in catalog["targets"] if t["target_id"] == "features.block2.relu.output")
    channels = (0, target["channel_count"] - 1)
    # One deterministic policy. Missing channel evidence has the same zero prior for each candidate.
    scores = dict.fromkeys(channels, 0.0)
    if informed:
        for channel in channels:
            preview = agent.job("/previews", {"sample_id": sample, "target_id": target["target_id"],
                                "channel_index": channel, "device": device}, "/previews")
            scores[channel] = preview["result"]["activation"]["mean"]
    selected = max(channels, key=scores.get)
    hypothesis = f"Halving post-ReLU channel {selected} will improve blurred true-{catalog['demo']['truth']} accuracy."
    recipe = agent.request("/recipes", {
        "hypothesis": hypothesis,
        "intervention": {"target_id": target["target_id"], "channel_index": selected,
                         "mode": "activation", "operation": "scale", "scale": .5},
    })
    budget_before = agent.request("/agent-session")
    assert budget_before["remaining"] == 1
    start = time.perf_counter()
    job = agent.job("/runs", {"kind": "intervention", "device": device, "recipe_id": recipe["recipe_id"]}, "/runs")
    comparison = agent.request(f"/runs/{job['run_id']}/comparison")
    comparison_seconds = time.perf_counter() - start
    report = agent.request(f"/runs/{job['run_id']}/report")
    budget_after = agent.request("/agent-session")
    assert budget_after["remaining"] == 0 and budget_after["charged"] == 1
    metrics = slice_metrics(comparison)
    return {
        "arm": "channel" if informed else "aggregate", "policy": "highest mean, stable candidate order",
        "sample_id": sample, "channel_scores": scores, "observed_channel_statistics": informed,
        "selected_channel": selected, "recipe_hash": recipe["recipe_hash"], "run_id": job["run_id"],
        "valid_recipes": 1, "submissions": 1, "server_remaining": budget_after["remaining"],
        "predicted_primary_direction": "increase", "direction_correct": metrics["primary"]["delta_pp"] > 0,
        "metrics": metrics, "runtime_device": comparison["runtime"]["actual_device"],
        "comparison_seconds": comparison_seconds, "worker_seconds": report["worker_seconds"],
        "peak_ram_bytes": report["peak_ram_bytes"], "peak_vram_bytes": report["peak_vram_bytes"],
        "artifact_bytes": sum(a["bytes"] for a in report["artifacts"].values()),
        "cache": report["cache"],
    }, recipe


def slice_metrics(comparison):
    slices = {s["slice_id"]: s for s in comparison["slices"]}
    return {name: {k: slices[comparison[key]][k] for k in
            ("n", "correct_before", "correct_after", "delta_pp", "fixed", "regressed", "changed_wrong")}
            for name, key in (("primary", "primary_slice_id"), ("normal", "guardrail_slice_id"))}


def run(source, output, device):
    if output.exists():
        raise FileExistsError(f"Choose a new result file: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="m3-pilot-", dir=output.parent) as tmp:
        workspace = Path(tmp)
        for sub in ("reference", "mnist"):
            shutil.copytree(source / sub, workspace / sub)
        with service(workspace) as (agent, _):
            aggregate, _recipe = arm(agent, False, device)
        with service(workspace) as (agent, human):
            informed, frozen = arm(agent, True, device)
            # The channel-informed arm is designated for confirmation before looking at outcomes.
            request = {"recipe_id": frozen["recipe_id"], "recipe_hash": frozen["recipe_hash"],
                       "confirmed": True, "device": device}
            job = human.job("/final-evaluations", request, "/runs")
            run_id = job["run_id"]
            comparison = human.request(f"/runs/{run_id}/comparison")
            assert comparison["recipe_hash"] == frozen["recipe_hash"]
            audit = human.request("/holdout-audit")["items"]
            assert len(audit) == 1 and audit[0]["run_id"] == run_id and audit[0]["recipe_hash"] == frozen["recipe_hash"]
            try:
                agent.request(f"/runs/{run_id}/report")
            except HTTPError as exc:
                if exc.code != 404:
                    raise
            else:
                raise AssertionError("Agent could read the holdout report")
            result = {
                "schema_version": 1, "pilot_type": "paired deterministic stub, scripted confirmation",
                "human_participants": 0, "repetitions_per_arm": 1, "arms": [aggregate, informed],
                "checkpoint_sha256": frozen["dependencies"]["checkpoint_sha256"],
                "manifest_sha256": frozen["dependencies"]["manifest_sha256"],
                "holdout": {"run_id": run_id, "recipe_hash": frozen["recipe_hash"],
                            "metrics": slice_metrics(comparison), "agent_report_status": 404,
                            "audit": audit[0]},
            }
    content = (json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    with output.open("xb") as stream:
        stream.write(content)
    print(f"{output} sha256={hashlib.sha256(content).hexdigest()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Prepared reference/ and mnist/ directories")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--confirm-scripted-holdout", required=True, action="store_true",
                        help="Confirm a frozen recipe in the disposable pilot workspace")
    args = parser.parse_args()
    run(args.source, args.output, args.device)
