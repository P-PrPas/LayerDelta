"""Run an exploration-only experiment with a deterministic local Agent."""

import argparse
from collections.abc import Callable
import json
from pathlib import Path
from typing import Literal
from uuid import uuid4

from .client import AgentClient, ClientError, wait_for
from .schemas import Edit, Preview, Run


def hypothesis(evidence: dict) -> str:
    normal, blurred = evidence["normal"], evidence["blurred"]
    return (f"Channel {evidence['channel_index']} at {evidence['target_id']} had mean activation "
            f"{normal['mean']:.8g} on {normal['sample_id']} and {blurred['mean']:.8g} "
            f"on {blurred['sample_id']}. Test whether zeroing it improves blurred true "
            f"{evidence['truth']} accuracy while preserving normal-image accuracy.")


def experiment(client: AgentClient, *, device: Literal["cpu", "cuda"] = "cpu",
               budget: int = 1, mode: Literal["activation", "weight"] = "activation",
               propose_hypothesis: Callable[[dict], str] = hypothesis) -> dict:
    if type(budget) is not int or budget < 0 or mode not in ("activation", "weight"):
        raise ValueError("A nonnegative integer budget and a supported intervention mode are required")
    quota = client.quota()
    if budget == 0 or quota["remaining"] == 0:
        return {"submissions": 0, "stop_reason": "user_budget" if budget == 0 else "quota_exhausted"}
    catalog = client.catalog()
    example = catalog["demo"]["example_pairs"][0]
    pair = list(client.samples(source_sample_id=example["source_sample_id"]))
    if len(pair) != 2 or {row["condition"] for row in pair} != {"normal", "blurred"}:
        raise ValueError("Expected one normal/blurred exploration pair")
    target = [t for t in catalog["targets"] if t["mode"] == mode][-1]
    channel = min(17, target["channel_count"] - 1)
    evidence = {"target_id": target["target_id"], "channel_index": channel, "truth": catalog["demo"]["truth"]}
    for sample in pair:
        job = client.preview(Preview(sample_id=sample["sample_id"], target_id=target["target_id"],
                                     channel_index=channel, device=device))
        result = wait_for(lambda: client.preview_status(job["run_id"]))["result"]
        if "activation" not in result:  # a token without weights:read gets no activation map or image
            raise ClientError("forbidden", "Activation previews need the weights:read capability: start the service "
                              "with --agent-weights-read", 403)
        values = result["activation"]
        # Only these statistics and IDs enter the provider callback.
        evidence[sample["condition"]] = {"sample_id": sample["sample_id"], "mean": values["mean"],
                                         "max": values["max"], "zero_fraction": values["zero_fraction"]}
    recipe = client.propose(Edit(
        hypothesis=propose_hypothesis(evidence), checkpoint_id=catalog["checkpoints"][0]["checkpoint_id"],
        dataset_revision_id=catalog["dataset_revision_id"],
        intervention={"mode": mode, "target_id": target["target_id"], "channel_index": channel, "operation": "zero"},
    ))
    if client.quota()["remaining"] <= 0:
        return {"submissions": 0, "stop_reason": "quota_exhausted"}
    key = uuid4().hex
    try:
        job = client.submit(Run(kind="intervention", recipe_id=recipe["recipe_id"], device=device), key=key)
    except ClientError as exc:
        if exc.code == "quota_exhausted":
            return {"submissions": 0, "stop_reason": "quota_exhausted"}
        raise
    run_id = job["run_id"]
    wait_for(lambda: client.status(run_id))
    comparison = client.comparison(run_id)
    selected = {"overall", comparison["primary_slice_id"], comparison["guardrail_slice_id"]}
    return {
        "submissions": 1, "stop_reason": "user_budget" if budget == 1 else "question_answered",
        "recipe_id": recipe["recipe_id"], "recipe_hash": recipe["recipe_hash"], "run_id": run_id,
        "hypothesis": recipe["hypothesis"], "submission_key": key, "intervention": recipe["intervention"],
        "baseline_id": comparison["baseline_run_id"], "device": comparison["runtime"]["actual_device"],
        "manifest_sha256": catalog["manifest_sha256"], "checkpoint_sha256": recipe["dependencies"]["checkpoint_sha256"],
        "slices": [s for s in comparison["slices"] if s["slice_id"] in selected],
        "changes": list(client.changes(run_id)), "quota": client.quota(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connection", required=True, type=Path, help="Local server connection JSON file")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--mode", choices=["activation", "weight"], default="activation")
    parser.add_argument("--budget", type=int, default=1)
    args = parser.parse_args()
    with AgentClient(**{k: v for k, v in json.loads(args.connection.read_bytes()).items() if k in ("url", "project_id", "token")}) as client:
        print(json.dumps(experiment(client, device=args.device, budget=args.budget, mode=args.mode), indent=2))


if __name__ == "__main__":
    main()
