"""Paired comparison of two evaluations on the same samples (v0.5-E4): metrics with baseline, delta and a paired-bootstrap interval.

The interval is the 95% percentile interval of the paired difference over `RESAMPLES` resamples of the samples, with the SAME resampled
indices for both evaluations and a seed derived from the evaluation fingerprint, so a given fingerprint always gives the same numbers.
Every evaluator resamples through `Evaluator.resampler` (parts are summed; detection matches once and repeats per-image rows), so all tasks
get `RESAMPLES`. `ci` is null when a metric is
undefined on a resample (a class absent from it) in more than 5% of them, or when there is no baseline.
"""

import numpy as np

RESAMPLES = 2000
CONFIDENCE = (2.5, 97.5)
MIN_DEFINED = 0.95


def _flat(metrics: list) -> dict:
    """{(name, class or None): metric row} for every metric and every per-class row."""
    out = {}
    for m in metrics:
        out[(m["name"], None)] = m
        for row in m.get("per_class", ()):
            out[(m["name"], row["class"])] = row
    return out


def _interval(deltas: np.ndarray):
    defined = ~np.isnan(deltas)
    if defined.mean() < MIN_DEFINED:
        return None
    lo, hi = np.percentile(deltas[defined], CONFIDENCE)
    return [float(lo), float(hi)]


def compare(evaluator, records: list, samples: list, baseline_records: list | None, seed: int) -> list:
    """The metrics of `records` on `samples`, each with the baseline's value, the delta and its interval (None without a baseline)."""
    value = evaluator.metrics(records, samples)
    if baseline_records is None:
        return value
    base = evaluator.metrics(baseline_records, samples)
    rng = np.random.default_rng(seed)
    n = len(samples)
    run_child, run_base = evaluator.resampler(records, samples), evaluator.resampler(baseline_records, samples)  # per-sample work happens once, here
    deltas: dict = {}
    for index in rng.integers(0, n, size=(RESAMPLES, n)):  # the SAME indices for both: the difference is paired
        a, b = _flat(run_child(index)), _flat(run_base(index))
        for key in a.keys() & b.keys():
            deltas.setdefault(key, []).append(np.nan if a[key]["value"] is None or b[key]["value"] is None else a[key]["value"] - b[key]["value"])
    base_flat = _flat(base)

    def fill(row: dict, key: tuple) -> dict:
        was = base_flat.get(key, {}).get("value")
        change = None if row["value"] is None or was is None else row["value"] - was
        series = deltas.get(key)
        return {**{k: v for k, v in row.items() if k != "per_class"}, "baseline": was, "delta": change,
                "ci": None if series is None or change is None else _interval(np.array(series, dtype=np.float64))}

    out = []
    for m in value:
        row = fill(m, (m["name"], None))
        if "per_class" in m:
            row["per_class"] = [fill(c, (m["name"], c["class"])) for c in m["per_class"]]
        out.append(row)
    return out


def categorize(evaluator, record: dict, baseline: dict, sample: dict) -> str | None:
    """fixed (incorrect -> correct), broken (correct -> incorrect), changed (the record changed, correctness did not), None (same)."""
    if record == baseline:
        return None
    was, now = evaluator.correct(baseline, sample), evaluator.correct(record, sample)
    return "fixed" if now and not was else "broken" if was and not now else "changed"
