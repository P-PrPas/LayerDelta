"""Paired prediction metrics for the ten-class reference adapter."""


def _digit(value):
    if type(value) is not int or not 0 <= value < 10:
        raise ValueError(f"Expected integer digit 0..9, got {value!r}")
    return value


def _index(rows, name):
    indexed = {}
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Invalid sample ID in {name}")
        if sample_id in indexed:
            raise ValueError(f"Duplicate sample ID in {name}: {sample_id}")
        indexed[sample_id] = row
    return indexed


def _predictions(rows, manifest, name):
    indexed = _index(rows, name)
    if indexed.keys() != manifest.keys():
        raise ValueError(f"{name} sample IDs differ from selected manifest")
    predictions = {}
    for sample_id, row in indexed.items():
        reference = manifest[sample_id]
        for key in ("label", "condition", "split", "source_sample_id"):
            if key not in row or row[key] != reference[key] or type(row[key]) is not type(reference[key]):
                raise ValueError(f"{name} {key} differs from manifest for {sample_id}")
        predictions[sample_id] = _digit(row.get("prediction"))
    return predictions


def _new_slice(slice_id):
    return {
        "slice_id": slice_id, "n": 0, "correct_before": 0, "correct_after": 0,
        "fixed": 0, "regressed": 0, "changed_wrong": 0,
        "confusion_before": [[0] * 10 for _ in range(10)],
        "confusion_after": [[0] * 10 for _ in range(10)],
    }


def compare(manifest_rows, before, after, truth: int, confused_with: int) -> dict:
    """Join by sample ID and compute all denominators on one selected manifest split."""
    _digit(truth)
    _digit(confused_with)
    if truth == confused_with:
        raise ValueError("Demo confusion must use two distinct digits")
    manifest = _index(manifest_rows, "manifest")
    split = None
    for row in manifest.values():
        _digit(row.get("label"))
        if row.get("split") not in ("exploration", "holdout") or row.get("condition") not in ("normal", "blurred"):
            raise ValueError("Invalid evaluation split or condition")
        if not isinstance(row.get("source_sample_id"), str) or not row["source_sample_id"]:
            raise ValueError("Missing source sample ID")
        if split is not None and split != row["split"]:
            raise ValueError("Cannot combine multiple evaluation splits")
        split = row["split"]

    base = _predictions(before, manifest, "baseline")
    candidate = _predictions(after, manifest, "intervention")
    confusion_id = f"baseline_{truth}_as_{confused_with}_blurred"
    slice_ids = ["overall", "normal", "blurred", confusion_id]
    slice_ids.extend(f"{condition}_truth_{digit}" for condition in ("normal", "blurred") for digit in range(10))
    slices = {slice_id: _new_slice(slice_id) for slice_id in slice_ids}
    changes = []
    for sample_id, row in manifest.items():
        label, condition = row["label"], row["condition"]
        left, right = base[sample_id], candidate[sample_id]
        category = None
        if left != right:
            if right == label:
                category = "fixed"
            elif left == label:
                category = "regressed"
            else:
                category = "changed_wrong"
            changes.append({
                "sample_id": sample_id, "source_sample_id": row["source_sample_id"],
                "condition": condition, "truth": label,
                "before": left, "after": right, "category": category,
            })
        memberships = ["overall", condition, f"{condition}_truth_{label}"]
        if condition == "blurred" and label == truth and left == confused_with:
            memberships.append(confusion_id)
        for slice_id in memberships:
            result = slices[slice_id]
            result["n"] += 1
            result["correct_before"] += int(left == label)
            result["correct_after"] += int(right == label)
            result["confusion_before"][label][left] += 1
            result["confusion_after"][label][right] += 1
            if category:
                result[category] += 1

    for result in slices.values():
        n = result["n"]
        n_before, n_after = result["correct_before"], result["correct_after"]
        result.update(
            accuracy_before=n_before / n if n else None,
            accuracy_after=n_after / n if n else None,
            delta_pp=100 * (n_after - n_before) / n if n else None,
            empty_reason=None if n else "No selected samples matched this slice.",
        )
    return {
        "primary_slice_id": f"blurred_truth_{truth}", "guardrail_slice_id": "normal",
        "baseline_confusion_slice_id": confusion_id, "slices": list(slices.values()),
        "changes": changes,
    }
