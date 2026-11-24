"""Detection: boxes before NMS [N, K, 4 + 1 + C] -> score threshold, class-wise NMS, boxes in SOURCE-image coordinates.

Output contract (design §3.1): per candidate, 4 box values in `box_format` (`xyxy` or `cxcywh`; `absolute` model-input pixels or
`relative` to the model input size), an objectness probability and C class probabilities (sigmoid or softmax applied in the graph).
A candidate's class is the argmax, its score objectness × that probability. Truth is `sample["boxes"]`: COCO `bbox` [x, y, w, h] and
`category_id`, in source coordinates; `class_ids` maps class index -> category_id (default: identical).

Matching (one rule for AP, F1 and correctness): per image and class, detections in descending score order (ties: candidate order)
each take the unmatched truth box of highest IoU at or above the threshold, one to one. AP is COCO's 101-point interpolated precision;
a class with no truth box is left out of the mean. Per image, at `conf_threshold`: F1 = 2TP / (2TP + FP + FN), 1.0 when there is
neither truth nor detection; correct = no FP and no FN. mAP does not decompose into sums: resample it through `metrics`.
"""

import numpy as np

from .base import Evaluator, EvaluatorError, metric

THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)
RECALLS = np.linspace(0.0, 1.0, 101)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix of xyxy boxes a [n, 4] and b [m, 4] (continuous coordinates, no +1). An inverted box (x2 < x1) has zero area and
    matches nothing."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area = lambda x: np.clip(x[:, 2] - x[:, 0], 0, None) * np.clip(x[:, 3] - x[:, 1], 0, None)  # noqa: E731
    union = area(a)[:, None] + area(b)[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, threshold: float) -> np.ndarray:
    """Indices kept by greedy class-wise NMS, in descending score order (stable)."""
    order = np.argsort(-scores, kind="stable")
    kept = []
    # shortcut: O(K^2) per image; fine at the few hundred candidates of the synthetic and corpus models, vectorise if K grows past 10^4
    suppressed = np.zeros(len(order), dtype=bool)
    for rank, i in enumerate(order):
        if suppressed[rank]:
            continue
        kept.append(i)
        rest = order[rank + 1:]
        same = classes[rest] == classes[i]
        overlap = iou(boxes[i:i + 1], boxes[rest])[0] > threshold
        suppressed[rank + 1:] |= same & overlap
    return np.array(kept, dtype=np.int64)


def match(pred: np.ndarray, truth: np.ndarray, threshold: float) -> np.ndarray:
    """pred [n, 6] (xyxy, score, class); truth [m, 5] (xyxy, class). Returns the TP flag per prediction, in the order given; the
    matching itself visits predictions by descending score (ties: the order given)."""
    tp = np.zeros(len(pred), dtype=bool)
    taken = np.zeros(len(truth), dtype=bool)
    if not len(pred) or not len(truth):
        return tp
    overlaps = iou(pred[:, :4], truth[:, :4])
    for i in np.argsort(-pred[:, 4], kind="stable"):
        candidates = (~taken) & (truth[:, 4] == pred[i, 5]) & (overlaps[i] >= threshold)
        if candidates.any():
            j = int(np.argmax(np.where(candidates, overlaps[i], -1.0)))
            taken[j] = tp[i] = True
    return tp


def _interpolated(hits: np.ndarray, positives: int) -> np.ndarray:
    """101-point interpolated AP of each column of `hits` [rows, T], rows already in descending score order."""
    if not len(hits):
        return np.zeros(hits.shape[1])
    hits = hits.astype(np.float64)
    true_pos, false_pos = np.cumsum(hits, axis=0), np.cumsum(1 - hits, axis=0)
    recall = true_pos / positives
    precision = np.maximum.accumulate((true_pos / np.maximum(true_pos + false_pos, np.finfo(np.float64).eps))[::-1], axis=0)[::-1]
    out = np.empty(hits.shape[1])
    for k in range(hits.shape[1]):
        index = np.searchsorted(recall[:, k], RECALLS, side="left")
        out[k] = np.where(index < len(hits), precision[np.minimum(index, len(hits) - 1), k], 0.0).mean()
    return out


def average_precision(scores: np.ndarray, tp: np.ndarray, positives: int) -> float:
    if not len(scores):
        return 0.0
    order = np.argsort(-scores, kind="stable")
    hits = tp[order].astype(np.float64)
    true_pos, false_pos = np.cumsum(hits), np.cumsum(1 - hits)
    recall = true_pos / positives
    precision = true_pos / np.maximum(true_pos + false_pos, np.finfo(np.float64).eps)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # the envelope: precision never rises with recall
    index = np.searchsorted(recall, RECALLS, side="left")
    return float(np.where(index < len(precision), precision[np.minimum(index, len(precision) - 1)], 0.0).mean())


class Detection(Evaluator):
    task = "detection"
    DEFAULTS = {"output": 0, "classes": None, "class_ids": None, "box_format": "xyxy", "box_scale": "absolute",
                "score_threshold": 0.001, "nms_iou": 0.5, "max_detections": 100, "conf_threshold": 0.25, "match_iou": 0.5}
    REQUIRED = ("classes",)

    def _check(self):
        if self.spec["box_format"] not in ("xyxy", "cxcywh") or self.spec["box_scale"] not in ("absolute", "relative"):
            raise EvaluatorError("invalid_argument", "box_format is xyxy or cxcywh; box_scale is absolute or relative", "box_format")
        ids = self.spec["class_ids"]
        if ids is not None and (len(ids) != self.spec["classes"] or len(set(ids)) != len(ids)):
            raise EvaluatorError("invalid_argument", "class_ids lists one distinct category_id per class", "class_ids")
        self._index = {c: i for i, c in enumerate(ids if ids is not None else range(self.spec["classes"]))}

    def _decode(self, raw: np.ndarray, geometry: dict) -> np.ndarray:
        """[K, 5 + C] -> kept detections [k, 6] (xyxy in source coordinates, score, class), descending score."""
        boxes = raw[:, :4].astype(np.float64)
        if self.spec["box_format"] == "cxcywh":
            boxes = np.concatenate([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], axis=1)
        out_h, out_w = geometry["output"]
        if self.spec["box_scale"] == "relative":
            boxes = boxes * np.array([out_w, out_h, out_w, out_h])
        cls = raw[:, 5:].argmax(axis=1)
        scores = raw[:, 4] * raw[np.arange(len(raw)), 5 + cls]
        keep = scores >= self.spec["score_threshold"]
        boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
        kept = nms(boxes, scores, cls, self.spec["nms_iou"])[:self.spec["max_detections"]]
        (sx, sy), (ox, oy), (h, w) = geometry["scale"], geometry["offset"], geometry["source"]
        source = (boxes[kept] - np.array([ox, oy, ox, oy])) / np.array([sx, sy, sx, sy])
        source = np.clip(source, 0, np.array([w, h, w, h]))
        return np.concatenate([source, scores[kept, None], cls[kept, None].astype(np.float64)], axis=1)

    def records(self, outputs, samples):
        raw = self._output(outputs, samples, 3, "candidates [N, K, 4 + 1 + C]")
        if raw.shape[2] != 5 + self.spec["classes"]:
            raise EvaluatorError("evaluator_mismatch", f"{raw.shape[2]} values per candidate, expected 5 + {self.spec['classes']}", "classes")
        return [{"boxes": self._decode(r, s["geometry"]).tolist()} for r, s in zip(raw, samples)]

    def _truth(self, sample) -> np.ndarray:
        rows = []
        for box in sample.get("boxes") or []:
            if box["category_id"] not in self._index:
                raise EvaluatorError("evaluator_mismatch", f"category_id {box['category_id']} is not in class_ids", "boxes")
            x, y, w, h = box["bbox"]
            rows.append([x, y, x + w, y + h, self._index[box["category_id"]]])
        return np.array(rows, dtype=np.float64).reshape(-1, 5)

    def _counts(self, record, sample) -> tuple[int, int, int]:
        pred = np.array(record["boxes"], dtype=np.float64).reshape(-1, 6)
        pred = pred[pred[:, 4] >= self.spec["conf_threshold"]]
        truth = self._truth(sample)
        tp = int(match(pred, truth, self.spec["match_iou"]).sum())
        return tp, len(pred) - tp, len(truth) - tp

    def correct(self, record, sample):
        _, fp, fn = self._counts(record, sample)
        return fp == 0 and fn == 0

    def score(self, record, sample):
        tp, fp, fn = self._counts(record, sample)
        return 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)

    def resampler(self, records, samples):
        """Matching depends on one image only, so it is done once here; a resample only repeats each image's (score, hit) rows by its
        count and recomputes AP. `metrics` is this with every image once, so the two can never disagree."""
        preds = [np.array(r["boxes"], dtype=np.float64).reshape(-1, 6) for r in records]
        truths = [self._truth(s) for s in samples]
        n, classes = len(samples), self.spec["classes"]
        positives = np.array([[int((t[:, 4] == c).sum()) for c in range(classes)] for t in truths], dtype=np.int64).reshape(n, classes)
        rows = {}  # class -> (scores, image of each row, hits [rows, thresholds])
        for c in range(classes):
            scores, image, hits = [], [], []
            for i, (p, t) in enumerate(zip(preds, truths)):
                mine, theirs = p[p[:, 5] == c], t[t[:, 4] == c]
                scores.append(mine[:, 4])
                image.append(np.full(len(mine), i, dtype=np.int64))
                hits.append(np.stack([match(mine, theirs, threshold) for threshold in THRESHOLDS], axis=1).reshape(len(mine), len(THRESHOLDS)))
            scores, image, hits = np.concatenate(scores), np.concatenate(image), np.concatenate(hits)
            order = np.argsort(-scores, kind="stable")  # sorted once: repeating rows keeps the order, so a resample never sorts
            rows[c] = (image[order], hits[order])

        def run(index) -> list:
            counts = np.bincount(np.asarray(index, dtype=np.int64), minlength=n)
            per_class, by_threshold = [], {}
            for c in range(classes):
                total = int(counts @ positives[:, c])
                if total == 0:
                    per_class.append({"class": c, "value": None, "n": 0})
                    continue
                image, hits = rows[c]
                aps = _interpolated(np.repeat(hits, counts[image], axis=0), total).tolist()
                by_threshold[c] = aps
                per_class.append({"class": c, "value": aps[0], "n": total})
            if not by_threshold:
                return [metric("map50", None, unit="score", better="higher", primary=True, per_class=per_class),
                        metric("map50_95", None, unit="score", better="higher")]
            table = np.array(list(by_threshold.values()))
            return [metric("map50", table[:, 0].mean(), unit="score", better="higher", primary=True, per_class=per_class),
                    metric("map50_95", table.mean(), unit="score", better="higher")]
        return run

    def metrics(self, records, samples):
        return self.resampler(records, samples)(np.arange(len(samples)))
