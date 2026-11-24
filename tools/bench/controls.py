"""Scripted controls of the benchmark (v0.5-E7a, design §5.5). No provider call.

- **Policies** that write an edit program from exploration data alone:
  - `random_policy`: a seeded random unit edit (the A/A arm and the positive control's weak arm);
  - `stats_policy`: per-unit L2 of each fault layer, robust z-scores, outlier units scaled back to the layer's median norm (the positive
    control's strong arm: what an agent with weight statistics could do).
- **Harness symmetry** is `apply_program` (arm R's path: the file edited directly) against LayerDelta's `plan`/`apply` (arm L's path).
  The two must give bit-identical checkpoints for the same program; the test does it over HTTP.
- **Outcome** of a policy on a task: its program applied to the start checkpoint, scored on holdout as the harness scores an arm.
"""

import numpy as np
import torch

import mnist_bench as mb


def random_policy(task: dict, start: dict, rng: np.random.Generator) -> list:
    layer = str(rng.choice(mb.FAULT_LAYERS))
    units = sorted(int(u) for u in rng.choice(start[f"{layer}.weight"].shape[0], size=int(rng.integers(1, 4)), replace=False))
    if rng.random() < 0.5:
        return mb._unit_program("zero", layer, units)
    return mb._unit_program("scale", layer, units, float(rng.choice([0.5, 2.0])))


def unit_l2(start: dict, layer: str) -> np.ndarray:
    """Per-unit L2 of a layer's weight rows: what the `stats` tool reports for it (reduce over every axis but 0)."""
    return start[f"{layer}.weight"].flatten(1).norm(dim=1).double().numpy()


def stats_policy(task: dict, start: dict, rng: np.random.Generator | None = None, z_min: float = 3.5) -> list:
    """Find the layer whose most extreme unit L2 has the largest robust z-score, and scale each outlier unit (z > z_min) back to the
    layer's median L2. No outlier: an empty program (the policy declines to edit)."""
    best = None
    for layer in mb.FAULT_LAYERS:
        l2 = unit_l2(start, layer)
        median = float(np.median(l2))
        mad = float(np.median(np.abs(l2 - median))) * 1.4826 or 1e-12
        z = (l2 - median) / mad
        if best is None or z.max() > best[0]:
            best = (float(z.max()), layer, l2, median, z)
    _, layer, l2, median, z = best
    program = []
    for unit in np.nonzero(z > z_min)[0]:
        program += mb._unit_program("scale", layer, [int(unit)], median / float(l2[unit]))
    return program


def scorer(task: dict, start: dict, base: dict, holdout: dict, guard_margin: float = 0.01):
    """`score(program) -> R and R_guarded` on holdout for one task. The start and reference are predicted ONCE here; each program then
    costs one forward pass (the controls score dozens of programs per task)."""
    ref = base if task["stratum"] != "S-nat" else mb.apply_program(base, task["reference_edit"])
    m = mb.measure_gap(task, start, ref, holdout)
    guard_start = mb.accuracy(mb.predict(start, holdout["images"]), holdout, "normal")

    def score(program: list) -> dict:
        pred = mb.predict(mb.apply_program(start, program), holdout["images"])
        return mb.score(task, mb.metric(task, pred, holdout), m["m_start"], m["m_ref"], mb.accuracy(pred, holdout, "normal"), guard_start,
                        guard_margin)
    return score


def outcome(task: dict, start: dict, base: dict, program: list, holdout: dict, guard_margin: float = 0.01) -> dict:
    """The arm's score on holdout: R and R_guarded, from the task's own metric and the normal-accuracy guardrail."""
    return scorer(task, start, base, holdout, guard_margin)(program)


def same_bits(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]) for k in a)


def main(argv=None) -> None:
    """The V10 gate on real data: python tools/bench/controls.py --workspace WS --tasks DIR/tasks.json --out controls.json
    (WS has mnist/ and reference/; DIR is what mnist_bench.py wrote). Scores on holdout as the harness does, acting as the person."""
    import argparse
    import json
    from pathlib import Path

    from safetensors.torch import load_file

    import analysis
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--random-draws", type=int, default=40)
    parser.add_argument("--aa-pairs", type=int, default=29, help="pairs per S-nat instance in each simulated study (87 / 3 instances)")
    args = parser.parse_args(argv)
    manifest = json.loads((args.workspace / "reference" / "dataset.json").read_text(encoding="utf-8"))
    base = load_file(str(args.workspace / "reference" / "model.safetensors"))
    holdout = mb.load_exploration(args.workspace / "mnist", manifest, "holdout")
    document = json.loads(args.tasks.read_text(encoding="utf-8"))
    kept = [t for t in document["tasks"] if t["kept"]]

    def start_of(task):
        f = task["start_checkpoint"]["file"]
        return base if f == "base" else load_file(str(args.tasks.parent / f))

    pairs, pool = [], {}
    for task in kept:
        start = start_of(task)
        score = scorer(task, start, base, holdout)
        weak = [score(random_policy(task, start, np.random.default_rng([i, 7])))["R"] for i in range(args.random_draws)]
        if task["stratum"] == "S-vis":
            strong = score(stats_policy(task, start))["R"]
            pairs += [{"instance": task["task_id"], "R_L": strong, "R_R": w} for w in weak]
        if task["stratum"] == "S-nat":
            pool[task["task_id"]] = np.array(weak)
    positive = analysis.primary(pairs)
    rate = analysis.aa_rate(pool, args.aa_pairs, studies=1000, resamples=1000)
    report = {"positive_control": {**positive, "passes": positive["ci"][0] > 0}, "aa": {"false_positive_rate": rate, "band": [0.03, 0.07],
              "passes": 0.03 <= rate <= 0.07, "pool_sizes": {k: len(v) for k, v in pool.items()}, "pool_means": {k: float(v.mean()) for k, v in pool.items()}}}
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({"positive_ci": positive["ci"], "positive_passes": report["positive_control"]["passes"], "aa_rate": rate,
                      "aa_passes": report["aa"]["passes"]}))



if __name__ == "__main__":
    main()
