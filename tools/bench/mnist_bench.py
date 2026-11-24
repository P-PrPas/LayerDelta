"""The v0.5 benchmark on the owner's scope (design §5): one plain CNN (`ReferenceCNN`), MNIST, the same agent with and without LayerDelta.

This module is the part of the harness (task E7a) that needs no host and no service:
- **tasks**: S-nat confusion pairs on blurred images (primary), planted S-vis and S-inv faults; the minimum gap at generation (the larger
  of a floor and 3 paired standard errors; S-nat chosen on one half of exploration and measured on the other) and the holdout check;
- **arm R's session kit**: `start.safetensors`, a standalone `model.py`, `exploration.npz`, `TASK.md`;
- **the edit rule and the isolation audit**: a final checkpoint's shape check, its changed-elements fraction, and transcript signatures of
  training and downloads;
- **the score**: R and R_guarded from holdout metrics (which the harness gets from LayerDelta's human-only final evaluation, E4e).

Metrics here are the v0.1 definitions (design V1): the confusion rate `truth -> confused_with` among BLURRED samples of label `truth`
(lower is better) and accuracy (overall for planted faults; on NORMAL samples for the guardrail). `layerdelta.data` and `layerdelta.model`
are reused as they are: the same split and the same blur as every other MNIST run.
"""

import hashlib
import inspect
import json
import re
from pathlib import Path

import numpy as np
import torch

VERSION = 1
# The minimum gap is measured against the noise of the gap itself: at least FLOOR, and at least 3 PAIRED standard errors of
# (m_ref - m_start) on the same samples. A confusion rate on blurred images of one digit is ~0.03 to 0.09 over ~250 samples per half,
# so an absolute 0.05 would drop real tasks while accepting noise elsewhere. For S-nat the reference is CHOSEN on half A of exploration
# and the gap MEASURED on half B (no winner's curse); the harness then checks the holdout gap before any session. Pre-registered (§5.3).
GAP_FLOOR = {"confusion": 0.02, "accuracy": 0.05}
GAP_SE = 3.0
FAULT_LAYERS = ("features.block1.conv", "features.block2.conv")
CHANGED_CAP = 0.10  # planned; fixed in the pre-registration before the dry run

# ---------------------------------------------------------------- data and model


def load_exploration(cache_dir: Path, manifest: dict, split: str = "exploration") -> dict:
    """{"images": float32 [N, 1, 28, 28] (model input), "labels": int64 [N], "conditions": [N] of normal|blurred, "sample_ids": [N]}."""
    from layerdelta.data import load_split
    images, labels, rows = load_split(Path(cache_dir), manifest, split)
    return {"images": images, "labels": labels, "conditions": [r["condition"] for r in rows], "sample_ids": [r["sample_id"] for r in rows]}


def predict(state: dict, images: torch.Tensor) -> np.ndarray:
    from layerdelta.model import ReferenceCNN
    model = ReferenceCNN().eval()
    model.load_state_dict(state)
    with torch.no_grad():
        return torch.cat([model(images[i:i + 2000]).argmax(1) for i in range(0, len(images), 2000)]).numpy()


def confusion_rate(pred: np.ndarray, data: dict, truth: int, confused_with: int) -> float:
    member = (data["labels"].numpy() == truth) & np.array([c == "blurred" for c in data["conditions"]])
    return float((pred[member] == confused_with).mean()) if member.any() else float("nan")


def accuracy(pred: np.ndarray, data: dict, condition: str | None = None) -> float:
    member = np.ones(len(pred), bool) if condition is None else np.array([c == condition for c in data["conditions"]])
    return float((pred[member] == data["labels"].numpy()[member]).mean())


def metric(task: dict, pred: np.ndarray, data: dict) -> float:
    target = task["target"]
    if target["metric"] == "confusion":
        return confusion_rate(pred, data, target["truth"], target["confused_with"])
    return accuracy(pred, data, target.get("condition"))


# ---------------------------------------------------------------- tasks

def _zero_unit(state: dict, layer: str, unit: int) -> dict:
    out = dict(state)
    for suffix in ("weight", "bias"):
        out[f"{layer}.{suffix}"] = state[f"{layer}.{suffix}"].clone()
        out[f"{layer}.{suffix}"][unit] = 0
    return out


def _unit_program(op: str, layer: str, units: list, k: float | None = None) -> list:
    """The LayerDelta edit program for an op on whole units (weight rows and bias entries) of one layer: what the reference edit is."""
    args = {} if op == "zero" else {"k": k}
    return [{"op": op, "args": args, "select": {"tensor": f"{layer}.{suffix}", "index": {"0": sorted(units)}}} for suffix in ("weight", "bias")]


def confusion_pairs(state: dict, data: dict, top: int = 6) -> list[tuple[int, int, int]]:
    """(truth, predicted, count) of the most frequent errors on blurred exploration samples."""
    pred = predict(state, data["images"])
    blurred = np.array([c == "blurred" for c in data["conditions"]])
    labels = data["labels"].numpy()
    counts = np.zeros((10, 10), int)
    np.add.at(counts, (labels[blurred], pred[blurred]), 1)
    np.fill_diagonal(counts, 0)
    order = np.argsort(-counts, axis=None, kind="stable")[:top]
    return [(int(i // 10), int(i % 10), int(counts.flat[i])) for i in order if counts.flat[i] > 0]


def search_reference(state: dict, data: dict, task: dict) -> tuple[list, float]:
    """S-nat's m_ref: the best single-unit zero edit of the fixed search (every unit of FAULT_LAYERS), CHOSEN on this (exploration) data."""
    best_program, best = None, metric(task, predict(state, data["images"]), data)
    lower = task["target"]["better"] == "lower"
    for layer in FAULT_LAYERS:
        for unit in range(state[f"{layer}.weight"].shape[0]):
            value = metric(task, predict(_zero_unit(state, layer, unit), data["images"]), data)
            if (value < best) if lower else (value > best):
                best, best_program = value, _unit_program("zero", layer, [unit])
    return best_program, best


def plant(state: dict, stratum: str, rng: np.random.Generator) -> tuple[dict, dict]:
    """(faulted state, fault record with its exact undo as an edit program)."""
    layer = str(rng.choice(FAULT_LAYERS))
    weight, bias = f"{layer}.weight", f"{layer}.bias"
    units_total = state[weight].shape[0]
    out = {k: v.clone() for k, v in state.items()}
    if stratum == "S-inv":  # sign flip: per-unit norms unchanged; repair = scale by -1 (an op both arms have)
        units = sorted(int(u) for u in rng.choice(units_total, size=int(rng.integers(2, 5)), replace=False))
        out[weight][units] *= -1
        out[bias][units] *= -1
        return out, {"kind": "sign", "layer": layer, "units": units, "undo": _unit_program("scale", layer, units, -1.0)}
    units = sorted(int(u) for u in rng.choice(units_total, size=int(rng.integers(2, 4)), replace=False))
    if rng.random() < 0.5:
        alpha = float(np.round(rng.uniform(8, 30), 3))
        out[weight][units] *= alpha
        out[bias][units] *= alpha
        return out, {"kind": "scale", "layer": layer, "units": units, "alpha": alpha, "undo": _unit_program("scale", layer, units, 1 / alpha)}
    beta = float(np.round(rng.uniform(3, 6), 3))
    noise = torch.from_numpy(rng.normal(0, beta * float(state[weight].std()), size=tuple(out[weight][units].shape))).float()
    out[weight][units] += noise
    return out, {"kind": "noise", "layer": layer, "units": units, "beta": beta, "undo": None}  # undone only by the base weights


def halves(data: dict) -> tuple[dict, dict]:
    """Exploration split into two fixed halves by a hash of the sample id: half A to CHOOSE (confusion pairs, the S-nat reference edit),
    half B to MEASURE the gap. Choosing and measuring on the same samples would make the gap a winner's curse."""
    import hashlib as _h
    in_a = np.array([_h.sha256(f"layerdelta-bench/{s}".encode()).digest()[0] % 2 == 0 for s in data["sample_ids"]])

    def take(mask):
        index = np.nonzero(mask)[0]
        return {"images": data["images"][torch.from_numpy(index)], "labels": data["labels"][torch.from_numpy(index)],
                "conditions": [data["conditions"][i] for i in index], "sample_ids": [data["sample_ids"][i] for i in index]}
    return take(in_a), take(~in_a)


def per_sample(task: dict, pred: np.ndarray, data: dict) -> np.ndarray:
    """The metric's per-sample values on the samples it is computed on (their mean is the metric): 1 if a member blurred `truth` was
    read as `confused_with` (confusion), or 1 if a member sample is right (accuracy)."""
    target, labels = task["target"], data["labels"].numpy()
    if target["metric"] == "confusion":
        member = (labels == target["truth"]) & np.array([c == "blurred" for c in data["conditions"]])
        return (pred[member] == target["confused_with"]).astype(np.float64)
    member = np.ones(len(pred), bool) if target.get("condition") is None else np.array([c == target["condition"] for c in data["conditions"]])
    return (pred[member] == labels[member]).astype(np.float64)


def apply_program(state: dict, program: list | None) -> dict:
    """A reference edit (zero or scale on whole-unit index lists) applied in-process, exactly as LayerDelta's ops define them."""
    out = {k: v.clone() for k, v in state.items()}
    for step in program or ():
        rows = step["select"]["index"]["0"]
        tensor = out[step["select"]["tensor"]]
        tensor[rows] = 0 if step["op"] == "zero" else tensor[rows] * step["args"]["k"]
    return out


def measure_gap(task: dict, start: dict, ref: dict, data: dict) -> dict:
    """m_start, m_ref, the gap in the better direction and its PAIRED standard error (the same samples under both models)."""
    a, b = per_sample(task, predict(start, data["images"]), data), per_sample(task, predict(ref, data["images"]), data)
    sign = -1.0 if task["target"]["better"] == "lower" else 1.0
    diff = sign * (b - a)
    n = len(diff)
    se = float(diff.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    return {"m_start": float(a.mean()) if n else float("nan"), "m_ref": float(b.mean()) if n else float("nan"),
            "gap": float(diff.mean()) if n else float("nan"), "se": se, "n": n}


def min_gap(task: dict, measured: dict) -> float:
    return max(GAP_FLOOR[task["target"]["metric"]], GAP_SE * measured["se"])


def generate(base: dict, data: dict, *, s_nat: int = 6, planted_per_stratum: int = 12, seed: int = 0) -> list[dict]:
    """Task instances, from EXPLORATION only, before any session. Kept when the gap passes min_gap on the measuring half (S-nat) or on
    all of exploration (planted faults: their reference is the base model, chosen by nothing). Dropped instances stay in the list with
    `kept: false` and the reason. Planted instances carry `start_state` (not JSON: the caller saves it)."""
    choose, measure = halves(data)
    tasks = []
    for truth, confused, count in confusion_pairs(base, choose, top=s_nat):
        task = {"stratum": "S-nat", "seed": None, "target": {"metric": "confusion", "truth": truth, "confused_with": confused, "better": "lower"},
                "guardrail": {"metric": "accuracy", "condition": "normal"}, "fault": None, "errors_on_choosing_half": count}
        task["reference_edit"], _ = search_reference(base, choose, task)
        task["exploration"] = {**measure_gap(task, base, apply_program(base, task["reference_edit"]), measure), "measured_on": "half B",
                               "chosen_on": "half A"}
        tasks.append(task)
    for stratum in ("S-vis", "S-inv"):
        kept, k, faults = 0, 0, {}
        while kept < planted_per_stratum and k < 20 * planted_per_stratum:
            rng = np.random.default_rng([seed, {"S-vis": 1, "S-inv": 2}[stratum], k])
            state, fault = plant(base, stratum, rng)
            same = (fault["kind"], fault["layer"], tuple(fault["units"]), fault.get("alpha"), fault.get("beta"))
            if fault["kind"] != "noise" and same in faults:  # an 8-unit layer repeats small faults; a copy would weigh one fault twice
                tasks.append({"stratum": stratum, "seed": k, "fault": fault, "kept": False, "duplicate_of": faults[same],
                              "reason": f"the same fault as seed {faults[same]}", "task_id": f"mnist-{stratum}-{k}"})
                k += 1
                continue
            faults[same] = k
            task = {"stratum": stratum, "seed": k, "target": {"metric": "accuracy", "condition": None, "better": "higher"},
                    "guardrail": {"metric": "accuracy", "condition": "normal"}, "fault": fault, "reference_edit": fault["undo"], "start_state": state}
            task["exploration"] = {**measure_gap(task, state, base, data), "measured_on": "all"}
            tasks.append(task)
            k += 1
            kept += int(task["exploration"]["gap"] >= min_gap(task, task["exploration"]))
    for task in tasks:
        if "duplicate_of" in task:
            continue
        e = task["exploration"]
        e["min_gap"] = min_gap(task, e)
        task["kept"] = bool(e["gap"] >= e["min_gap"])
        task["reason"] = None if task["kept"] else f"gap {e['gap']:.4f} < {e['min_gap']:.4f} on exploration ({e['measured_on']})"
        target = task["target"]
        name = f"{target['truth']}to{target['confused_with']}" if task["stratum"] == "S-nat" else f"{task['seed']}"
        task["task_id"] = f"mnist-{task['stratum']}-{name}"
    return tasks


def holdout_check(task: dict, start: dict, base: dict, holdout: dict) -> dict:
    """The person-level check the harness runs on holdout BEFORE any session (the same for both arms, independent of any agent): R's
    denominator m_ref - m_start is measured on holdout, so an instance whose holdout gap has the wrong sign or is under the floor is
    dropped by this pre-registered rule. Records the holdout values the score will use."""
    ref = base if task["stratum"] != "S-nat" else apply_program(base, task["reference_edit"])
    measured = measure_gap(task, start, ref, holdout)
    ok = measured["gap"] >= GAP_FLOOR[task["target"]["metric"]]
    return {**measured, "kept": bool(ok), "reason": None if ok else f"holdout gap {measured['gap']:.4f} < floor {GAP_FLOOR[task['target']['metric']]}"}


# ---------------------------------------------------------------- arm R's session kit

TASK_TEXT = """# Task: repair a CNN by editing its weights

You are given a trained MNIST classifier (`ReferenceCNN`: two convolution blocks and a linear classifier) that has a problem:
{problem}

Make the model better at this by **editing its weights directly**. Rules:
- **No gradient training.** Do not use an optimiser, `backward`, autograd gradients or any training or fine-tuning loop. Change values
  directly (set, scale, zero, add, copy or permute values) where you decide. A final model that looks trained is a rule violation.
- **Do not use any data other than what you are given.** Do not download MNIST or any other dataset or model.
- Keep accuracy on normal images close to what it is now.
- You have {budget}.

{arm}
"""

ARM_TEXT = {
    "R": """You have these files in this directory:
- `start.safetensors`: the model's weights;
- `model.py`: the model's code (`ReferenceCNN`);
- `exploration.npz`: images (`images`, float32 [N, 1, 28, 28], already the model's input), labels (`labels`) and conditions
  (`conditions`: normal or blurred) to measure your changes on.

When you are done, write your final weights to `final.safetensors` in this directory, with exactly the same tensor names, shapes and
dtypes as `start.safetensors`.""",
    "L": """Use the LayerDelta tools: the model is checkpoint `{checkpoint_id}`, and its data is dataset `{dataset_id}` (exploration split).
Measure with `evaluate` (the evaluator `{{"task": "classification", "confusion": {confusion}}}`), change weights with `plan` and `apply`,
and inspect them with the read tools. When you are done, end your final message with the line `FINAL: <child checkpoint id>`.""",
}


def problem_text(task: dict) -> str:
    t = task["target"]
    if task["stratum"] == "S-nat":
        return (f"on **blurred** images it often reads the digit **{t['truth']}** as **{t['confused_with']}**. Lower that confusion rate "
                f"(the share of blurred {t['truth']}s predicted as {t['confused_with']}).")
    return "its accuracy has dropped (something in its weights is wrong). Raise its accuracy back."


def task_text(task: dict, arm: str, budget: str, **ids) -> str:
    confusion = json.dumps({"truth": task["target"].get("truth"), "confused_with": task["target"].get("confused_with"), "condition": "blurred"}) \
        if task["stratum"] == "S-nat" else "null"
    return TASK_TEXT.format(problem=problem_text(task), budget=budget, arm=ARM_TEXT[arm].format(confusion=confusion, **ids)).strip() + "\n"


def model_source() -> str:
    """A standalone `model.py`: ReferenceCNN with no LayerDelta import."""
    from layerdelta import model
    return ('"""ReferenceCNN, the model of this task (standalone copy)."""\n\nfrom collections import OrderedDict\n\nimport torch\nfrom torch import Tensor, nn\n\n\n'
            + inspect.getsource(model._block) + "\n\n" + inspect.getsource(model.ReferenceCNN))


def write_r_kit(directory: Path, start: dict, data: dict, task: dict, budget: str) -> dict:
    """Arm R's session directory. Returns {file: sha256} for the audit."""
    from safetensors.torch import save_file
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in start.items()}, str(directory / "start.safetensors"))
    (directory / "model.py").write_text(model_source(), encoding="utf-8")
    np.savez_compressed(directory / "exploration.npz", images=data["images"].numpy().astype(np.float32), labels=data["labels"].numpy(),
                        conditions=np.array(data["conditions"]))
    (directory / "TASK.md").write_text(task_text(task, "R", budget), encoding="utf-8")
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir()) if p.is_file()}


# ---------------------------------------------------------------- the edit rule and the audit

def check_final(start: dict, final: dict) -> dict:
    """Same tensor names, shapes and dtypes as the start (else not a valid answer), and the changed-elements fraction (the edit rule)."""
    if set(start) != set(final) or any(start[k].shape != final[k].shape or start[k].dtype != final[k].dtype for k in start):
        return {"valid": False, "reason": "tensor names, shapes or dtypes differ from the start checkpoint", "changed_fraction": None}
    total = sum(t.numel() for t in start.values())
    changed = sum(int((start[k] != final[k]).sum()) for k in start)
    return {"valid": True, "reason": None, "changed_fraction": changed / total, "changed_elements": changed,
            "over_cap": changed / total > CHANGED_CAP}


TRAINING = re.compile(r"\.backward\s*\(|torch\.optim|\boptim\.(SGD|Adam\w*|RMSprop)|autograd\.grad|\.fit\s*\(|requires_grad_?\s*\(\s*True"
                      r"|\.step\s*\(\s*\)\s*$|zero_grad\s*\(", re.MULTILINE)
DOWNLOAD = re.compile(r"https?://|\b(wget|curl|Invoke-WebRequest|iwr)\b|pip\s+install|torchvision\.datasets|tensorflow_datasets|keras\.datasets"
                      r"|fetch_openml|load_digits|huggingface_hub|datasets\.load_dataset", re.IGNORECASE)


def audit_text(text: str) -> dict:
    """Signatures of training and downloads in code or commands an agent wrote or ran (the transcript's tool inputs)."""
    return {"training": sorted({m.group(0).strip() for m in TRAINING.finditer(text)}),
            "download": sorted({m.group(0).strip() for m in DOWNLOAD.finditer(text)})}


# ---------------------------------------------------------------- the score

def score(task: dict, m_final: float | None, m_start: float, m_ref: float, guard_final: float | None, guard_start: float,
          guard_margin: float) -> dict:
    """R = (m_final - m_start) / (m_ref - m_start), not clipped; no valid final checkpoint scores R = 0 (intention to treat).
    R_guarded = min(R, 0) when the guardrail fell by more than `guard_margin`."""
    if m_final is None:
        return {"R": 0.0, "R_guarded": 0.0, "valid_final": False}
    r = (m_final - m_start) / (m_ref - m_start)
    broke = guard_final is not None and (guard_start - guard_final) > guard_margin
    return {"R": r, "R_guarded": min(r, 0.0) if broke else r, "valid_final": True, "guardrail_broken": broke}


# ---------------------------------------------------------------- command line

def main(argv=None) -> None:
    """Generate the task set from a workspace's verified MNIST archives and v0.1 reference checkpoint:
    python tools/bench/mnist_bench.py --workspace WS --out DIR   (WS holds mnist/ and reference/{model.safetensors,dataset.json})."""
    import argparse
    from safetensors.torch import load_file, save_file
    parser = argparse.ArgumentParser(description="Generate the v0.5 MNIST benchmark tasks (design §5.3)")
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--s-nat", type=int, default=6)
    parser.add_argument("--planted", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    manifest = json.loads((args.workspace / "reference" / "dataset.json").read_text(encoding="utf-8"))
    base_path = args.workspace / "reference" / "model.safetensors"
    base = load_file(str(base_path))
    data = load_exploration(args.workspace / "mnist", manifest)
    tasks = generate(base, data, s_nat=args.s_nat, planted_per_stratum=args.planted, seed=args.seed)
    holdout = load_exploration(args.workspace / "mnist", manifest, "holdout")  # the harness acts as the person here, before any session
    for task in tasks:
        if task["kept"]:
            task["holdout"] = holdout_check(task, task.get("start_state") or base, base, holdout)
            if not task["holdout"]["kept"]:
                task["kept"], task["reason"] = False, task["holdout"]["reason"]
    args.out.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        state = task.pop("start_state", None)
        if task["kept"] and state is not None:
            path = args.out / f"{task['task_id']}.safetensors"
            save_file({k: v.contiguous() for k, v in state.items()}, str(path))
            task["start_checkpoint"] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        elif task["kept"]:
            task["start_checkpoint"] = {"file": "base", "sha256": hashlib.sha256(base_path.read_bytes()).hexdigest()}
    document = {"version": VERSION, "base_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
                "manifest_sha256": manifest["manifest_sha256"], "gap_floor": GAP_FLOOR, "gap_se": GAP_SE, "gap_rule": "chosen on exploration half A, measured on half B (S-nat); paired SE; holdout gap >= floor", "seed": args.seed, "tasks": tasks}
    (args.out / "tasks.json").write_text(json.dumps(document, indent=1), encoding="utf-8")
    kept = [t["task_id"] for t in tasks if t["kept"]]
    print(json.dumps({"kept": len(kept), "dropped": len(tasks) - len(kept), "by_stratum": {s: sum(t["kept"] for t in tasks if t["stratum"] == s)
                                                                                            for s in ("S-nat", "S-vis", "S-inv")}}))


if __name__ == "__main__":
    main()
