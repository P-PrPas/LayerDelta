"""Segmentation: logits [N, C, H, W] -> argmax per pixel, compared with the preprocessed mask (`sample["mask"]`, pad = ignore index).

A prediction at another resolution is resized to the mask with nearest-exact (pixel-centre) sampling, the rule preprocessing uses.
Records are per-class pixel counts, so dataset metrics are sums: mIoU and Dice are dataset-level (per-class intersection and union
summed over samples, then averaged over the classes present). A class absent from both truth and prediction is left out of a mean;
a sample with nothing to score (all ignore, or no class present) is correct with score 1.0.
"""

import numpy as np

from .base import Evaluator, EvaluatorError, metric, ratio


def nearest_exact(labels: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a [h, w] label map to [height, width] by pixel-centre sampling (torch `nearest-exact`)."""
    h, w = labels.shape
    rows = np.minimum(((np.arange(height) + 0.5) * h / height).astype(np.int64), h - 1)
    cols = np.minimum(((np.arange(width) + 0.5) * w / width).astype(np.int64), w - 1)
    return labels[rows[:, None], cols[None, :]]


class Segmentation(Evaluator):
    task = "segmentation"
    DEFAULTS = {"output": 0, "classes": None, "ignore_index": 255, "correct_miou": 0.5}
    REQUIRED = ("classes",)

    def records(self, outputs, samples):
        logits = self._output(outputs, samples, 4, "logits [N, C, H, W]")
        classes, ignore = self.spec["classes"], self.spec["ignore_index"]
        if logits.shape[1] != classes:
            raise EvaluatorError("evaluator_mismatch", f"{logits.shape[1]} class maps per sample, the spec says {classes}", "classes")
        out = []
        for scores, sample in zip(logits, samples):
            truth = np.asarray(sample["mask"], dtype=np.int64)
            pred = scores.argmax(axis=0)
            if pred.shape != truth.shape:
                pred = nearest_exact(pred, *truth.shape)
            valid = truth != ignore
            t, p = truth[valid], pred[valid]
            if t.size and (t.min() < 0 or t.max() >= classes):
                raise EvaluatorError("evaluator_mismatch", f"a mask value is outside 0..{classes - 1} and is not the ignore index", "mask")
            out.append({"inter": np.bincount(t[t == p], minlength=classes).tolist(),
                        "pred": np.bincount(p, minlength=classes).tolist(), "truth": np.bincount(t, minlength=classes).tolist()})
        return out

    @staticmethod
    def _sample_miou(record) -> float:
        inter, pred, truth = (np.asarray(record[k], dtype=np.float64) for k in ("inter", "pred", "truth"))
        union = pred + truth - inter
        present = union > 0
        return float((inter[present] / union[present]).mean()) if present.any() else 1.0

    def correct(self, record, sample):
        return self._sample_miou(record) >= self.spec["correct_miou"]

    def score(self, record, sample):
        return self._sample_miou(record)

    def metric_parts(self, records, samples):
        inter, pred, truth = (np.array([r[k] for r in records], dtype=np.int64).reshape(len(records), self.spec["classes"])
                              for k in ("inter", "pred", "truth"))
        return {"inter": inter, "union": pred + truth - inter, "pred_truth": pred + truth, "truth": truth}

    def metrics_from_parts(self, sums):
        inter, union, both = (np.asarray(sums[k], dtype=np.float64) for k in ("inter", "union", "pred_truth"))
        iou = [ratio(i, u) for i, u in zip(inter, union)]
        present = union > 0
        miou = float((inter[present] / union[present]).mean()) if present.any() else None
        dice_present = both > 0
        dice = float((2 * inter[dice_present] / both[dice_present]).mean()) if dice_present.any() else None
        return [metric("miou", miou, unit="score", better="higher", primary=True,
                       per_class=[{"class": c, "value": v, "n": int(t)} for c, (v, t) in enumerate(zip(iou, sums["truth"]))]),
                metric("dice", dice, unit="score", better="higher"),
                metric("pixel_accuracy", ratio(inter.sum(), np.asarray(sums["truth"]).sum()), unit="rate", better="higher")]
