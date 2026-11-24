"""The benchmark's primary analysis and its A/A calibration (v0.5-E7a, design §5.5).

The claim is about fixed instances (these confusions on this checkpoint), so instances are fixed effects and the pairs within an instance
are exchangeable replicates:
- ΔR per instance = mean of (R_L - R_R) over its pairs; the pooled ΔR weights instances equally;
- the 95% interval resamples pairs WITHIN each instance (never instances: with 3 to 10 of them that would measure the draw);
- the test flips the sign of each pair's difference (two-sided sign-flip permutation, within instances by construction).

`aa_rate` is the gate on the analysis itself: drawing both arms' R from one pool (the scripted random policy's outcomes) must give an
interval that excludes 0 in about 5% of simulated studies (pre-registered band [0.03, 0.07]).
"""

import numpy as np


def _pooled(diffs: dict) -> float:
    return float(np.mean([np.mean(d) for d in diffs.values()]))


def primary(pairs: list[dict], *, resamples: int = 10_000, permutations: int = 10_000, seed: int = 0) -> dict:
    """`pairs`: [{instance, R_L, R_R}]. Returns the per-instance table (the headline), the pooled ΔR with its 95% within-instance
    bootstrap interval, and the two-sided sign-flip permutation p."""
    diffs: dict = {}
    zeros: dict = {}  # pairs where both arms scored exactly 0 (no valid final from either: ITT zeros a mean would hide)
    for p in pairs:
        diffs.setdefault(p["instance"], []).append(p["R_L"] - p["R_R"])
        zeros[p["instance"]] = zeros.get(p["instance"], 0) + int(p["R_L"] == 0 and p["R_R"] == 0)
    diffs = {k: np.asarray(v, dtype=np.float64) for k, v in sorted(diffs.items())}
    rng = np.random.default_rng(seed)
    observed = _pooled(diffs)
    boot = np.empty(resamples)
    for b in range(resamples):
        boot[b] = np.mean([d[rng.integers(0, len(d), size=len(d))].mean() for d in diffs.values()])
    flips = np.empty(permutations)
    for b in range(permutations):
        flips[b] = np.mean([(d * rng.choice((-1.0, 1.0), size=len(d))).mean() for d in diffs.values()])
    lo, hi = np.percentile(boot, (2.5, 97.5))
    return {"per_instance": [{"instance": k, "pairs": int(len(d)), "both_zero": zeros[k], "delta_R": float(d.mean())} for k, d in diffs.items()],
            "delta_R": observed, "ci": [float(lo), float(hi)],
            "p_sign_flip": float((1 + np.sum(np.abs(flips) >= abs(observed) - 1e-12)) / (1 + permutations)),
            "resamples": resamples, "permutations": permutations}


def aa_rate(pool: dict, pairs_per_instance: int, *, studies: int = 1000, resamples: int = 1000, seed: int = 0) -> float:
    """Share of simulated A/A studies whose 95% interval excludes 0. `pool`: {instance: array of R outcomes of ONE policy}; each study
    draws both arms' R for every pair from that same pool. (`resamples` is lower than the primary's 10,000 to keep 1,000 studies fast;
    the interval's coverage, not its third decimal, is what is checked.)"""
    rng = np.random.default_rng(seed)
    hits = 0
    for s in range(studies):
        pairs = [{"instance": k, "R_L": float(rng.choice(v)), "R_R": float(rng.choice(v))} for k, v in pool.items() for _ in range(pairs_per_instance)]
        lo, hi = primary(pairs, resamples=resamples, permutations=1, seed=int(rng.integers(1 << 31)))["ci"]
        hits += int(lo > 0 or hi < 0)
    return hits / studies
