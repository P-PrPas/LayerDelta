"""Paired comparison of two evaluations on the same samples (v0.5-E4): metrics with baseline, delta and a paired-bootstrap interval.

The interval is the 95% percentile interval of the paired difference over `RESAMPLES` resamples of the samples, with the SAME resampled
indices for both evaluations and a seed derived from the evaluation fingerprint, so a given fingerprint always gives the same numbers.
An evaluator whose metric decomposes (`metric_parts`) is resampled by summing per-sample parts (a matrix product); one that does not
(detection's mAP) is resampled through `metrics`, which is slower, so it gets `RESAMPLES_NO_PARTS`. `ci` is null when a metric is
undefined on a resample (a class absent from it) in more than 5% of them, or when there is no baseline.
"""

import numpy as np

RESAMPLES = 2000
RESAMPLES_NO_PARTS = 200
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


def _sample_sums(parts: dict, counts: np.ndarray) -> dict:
    """Per-resample sums of every part: counts [R, N] @ part [N, ...]."""
    out = {}
    for key, array in parts.items():
        flat = np.asarray(array, dtype=np.float64).reshape(array.shape[0], -1)
        out[key] = (counts @ flat).reshape((counts.shape[0], *array.shape[1:]))
    return out


def compare(evaluator, records: list, samples: list, baseline_records: list | None, seed: int) -> list:
    """The metrics of `records` on `samples`, each with the baseline's value, the delta and its interval (None without a baseline)."""
    value = evaluator.metrics(records, samples)
    if baseline_records is None:
        return value
    base = evaluator.metrics(baseline_records, samples)
    rng = np.random.default_rng(seed)
    n = len(samples)
    child_parts, base_parts = evaluator.metric_parts(records, samples), evaluator.metric_parts(baseline_records, samples)
    deltas: dict = {}
    if child_parts is not None and base_parts is not None:
        counts = rng.multinomial(n, np.full(n, 1 / n), size=RESAMPLES).astype(np.float64)
        child_sums, base_sums = _sample_sums(child_parts, counts), _sample_sums(base_parts, counts)
        for r in range(RESAMPLES):
            a = _flat(evaluator.metrics_from_parts({k: v[r] for k, v in child_sums.items()}))
            b = _flat(evaluator.metrics_from_parts({k: v[r] for k, v in base_sums.items()}))
            for key in a.keys() & b.keys():
                deltas.setdefault(key, []).append(np.nan if a[key]["value"] is None or b[key]["value"] is None else a[key]["value"] - b[key]["value"])
        total = RESAMPLES
    else:
        total = RESAMPLES_NO_PARTS
        for _ in range(total):
            idx = rng.integers(0, n, size=n)
            pick = lambda xs: [xs[i] for i in idx]  # noqa: E731
            a = _flat(evaluator.metrics(pick(records), pick(samples)))
            b = _flat(evaluator.metrics(pick(baseline_records), pick(samples)))
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
