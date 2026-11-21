import json
from pathlib import Path

import pytest

from layerdelta.evaluation import compare


def oracle():
    fixture = json.loads((Path(__file__).parent / "fixtures/paired-oracle.json").read_bytes())
    manifest, before, after = [], [], []
    for source, truth, condition, left, right in fixture["rows"]:
        row = dict(sample_id=f"{source}/{condition}", source_sample_id=source,
                   label=truth, condition=condition, split=fixture["split"])
        manifest.append(row)
        before.append(dict(row, prediction=left))
        after.append(dict(row, prediction=right))
    return fixture["expected"], manifest, before, after


def test_counts_and_change_identity_are_independent_of_prediction_order():
    expected, manifest, before, after = oracle()
    report = compare(manifest, reversed(before), after[2:] + after[:2], 9, 4)
    slices = {row["slice_id"]: row for row in report["slices"]}
    for slice_id, metrics in expected.items():
        assert {key: slices[slice_id][key] for key in metrics} == metrics
    overall = slices["overall"]
    assert overall["delta_pp"] == pytest.approx(100 / 6)
    assert overall["confusion_before"][9][4] == 2
    assert overall["confusion_after"][9][4] == 1
    assert sum(map(sum, overall["confusion_after"])) == 6
    assert {r["sample_id"]: r["category"] for r in report["changes"]} == {
        "a/normal": "regressed", "a/blurred": "fixed",
        "b/blurred": "changed_wrong", "c/blurred": "fixed",
    }


def test_primary_and_baseline_confusion_have_different_membership():
    _, manifest, before, after = oracle()
    before[1]["prediction"] = 9
    result = compare(manifest, before, after, 9, 4)
    slices = {row["slice_id"]: row for row in result["slices"]}
    assert slices[result["primary_slice_id"]]["n"] == 2
    confusion = slices[result["baseline_confusion_slice_id"]]
    assert confusion["n"] == 1 and confusion["changed_wrong"] == 1
    assert confusion["confusion_after"][9][7] == 1
    assert slices[result["guardrail_slice_id"]]["n"] == 3
    assert slices["blurred_truth_0"]["empty_reason"]


@pytest.mark.parametrize("side", ["before", "after", "manifest"])
@pytest.mark.parametrize("mutation", ["drop", "duplicate", "extra"])
def test_pairing_requires_exact_membership(side, mutation):
    _, manifest, before, after = oracle()
    rows = {"before": before, "after": after, "manifest": manifest}[side]
    if mutation == "drop":
        rows.pop()
    elif mutation == "extra":
        rows.append(dict(rows[0], sample_id="extra"))
    else:
        rows.append(rows[0])
    with pytest.raises(ValueError, match="sample ID"):
        compare(manifest, before, after, 9, 4)


@pytest.mark.parametrize(("field", "value"), [
    ("split", "holdout"), ("label", False), ("condition", "blurred"), ("source_sample_id", "other"),
    ("prediction", 10), ("prediction", -1), ("prediction", True),
])
def test_reject_mismatched_metadata_and_bad_predictions(field, value):
    _, manifest, before, after = oracle()
    after[0][field] = value
    with pytest.raises(ValueError):
        compare(manifest, before, after, 9, 4)
