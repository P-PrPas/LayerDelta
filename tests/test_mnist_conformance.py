"""V1 (v0.5-E4d): the classification plugin is pinned to the v0.1 definitions. The legacy routes keep the v0.1 worker and `evaluation.compare`
untouched; this test says the plugin's accuracy, per-class recall and confusion-pair rate equal what v0.1's comparison reports, number for
number, and that `compare` over rows made from plugin records is the same document as `compare` over the original rows."""

import json

import numpy as np

from layerdelta import evaluators
from layerdelta.evaluation import compare

TRUTH, CONFUSED = 9, 4


def manifest(n=600, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append({"sample_id": f"mnist/t10k/{i:05d}/{'blurred' if i % 2 else 'normal'}", "source_sample_id": f"mnist/t10k/{i:05d}",
                     "split": "exploration", "condition": "blurred" if i % 2 else "normal", "label": int(rng.integers(0, 10))})
    return rows, rng


def logits(rows, rng, quality):
    """Small integer logits, so ties between classes happen: argmax must break them the way the v0.1 worker did (the first maximum)."""
    out = rng.integers(0, 3, size=(len(rows), 10)).astype(np.float32)
    for i, row in enumerate(rows):
        if rng.random() < quality:
            out[i, row["label"]] = 3.0
        if row["condition"] == "blurred" and row["label"] == TRUTH and rng.random() < .3:
            out[i, CONFUSED] = 4.0
    return out


def legacy_rows(rows, predictions):
    return [{**row, "prediction": int(p)} for row, p in zip(rows, predictions)]


def test_the_plugin_reports_the_numbers_v0_1_reports():
    rows, rng = manifest()
    before, after = logits(rows, rng, .55), logits(rows, rng, .75)
    evaluator = evaluators.make({"task": "classification", "classes": 10, "confusion": {"truth": TRUTH, "confused_with": CONFUSED, "condition": "blurred"}})
    samples = [{"label": r["label"], "condition": r["condition"]} for r in rows]
    golden = compare(rows, legacy_rows(rows, before.argmax(axis=1)), legacy_rows(rows, after.argmax(axis=1)), TRUTH, CONFUSED)
    by_slice = {s["slice_id"]: s for s in golden["slices"]}
    for side, array in (("before", before), ("after", after)):
        records = evaluator.records([array], samples)
        assert [r["pred"] for r in records] == array.argmax(axis=1).tolist()  # the same decoder: argmax, first maximum on a tie
        metrics = {m["name"]: m for m in evaluator.metrics(records, samples)}
        assert metrics["accuracy"]["value"] == by_slice["overall"][f"accuracy_{side}"]
        for condition in ("normal", "blurred"):
            assert metrics[f"accuracy:{condition}"]["value"] == by_slice[condition][f"accuracy_{side}"]
        for digit, recall in enumerate(metrics["accuracy"]["per_class"]):  # recall of a digit over both conditions
            n = sum(by_slice[f"{c}_truth_{digit}"]["n"] for c in ("normal", "blurred"))
            hit = sum(by_slice[f"{c}_truth_{digit}"][f"correct_{side}"] for c in ("normal", "blurred"))
            assert recall["n"] == n and recall["value"] == hit / n
        blurred = by_slice[f"blurred_truth_{TRUTH}"]
        rate = metrics[f"confusion:{TRUTH}->{CONFUSED}:blurred"]
        assert rate["value"] == blurred[f"confusion_{side}"][TRUTH][CONFUSED] / blurred["n"] and rate["better"] == "lower" and rate["primary"]


def test_compare_over_plugin_records_is_the_same_document():
    rows, rng = manifest(seed=12)
    before, after = logits(rows, rng, .6), logits(rows, rng, .8)
    evaluator = evaluators.make({"task": "classification", "classes": 10})
    samples = [{"label": r["label"], "condition": r["condition"]} for r in rows]
    from_records = lambda array: legacy_rows(rows, [r["pred"] for r in evaluator.records([array], samples)])  # noqa: E731
    direct = compare(rows, legacy_rows(rows, before.argmax(axis=1)), legacy_rows(rows, after.argmax(axis=1)), TRUTH, CONFUSED)
    through = compare(rows, from_records(before), from_records(after), TRUTH, CONFUSED)
    assert json.dumps(through, sort_keys=True).encode() == json.dumps(direct, sort_keys=True).encode()
    # and the plugin's fixed/broken agree with the legacy categories (regressed = broken)
    kinds = {"fixed": 0, "regressed": 0}
    for change in direct["changes"]:
        if change["category"] in kinds:
            kinds[change["category"]] += 1
    counted = {"fixed": 0, "broken": 0}
    from layerdelta import bootstrap
    b, a = evaluator.records([before], samples), evaluator.records([after], samples)
    for x, y, s in zip(b, a, samples):
        category = bootstrap.categorize(evaluator, y, x, s)
        if category in counted:
            counted[category] += 1
    assert counted["fixed"] == kinds["fixed"] and counted["broken"] == kinds["regressed"]
