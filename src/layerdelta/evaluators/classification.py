"""Classification: logits [N, C] -> argmax. Metrics: accuracy, accuracy per condition, per-class recall, and the confusion-pair rate
of v0.1 (`truth -> confused_with`, optionally within one condition: the 8 read as 3 on blurred images)."""

import numpy as np

from .base import Evaluator, EvaluatorError, metric, ratio


class Classification(Evaluator):
    task = "classification"
    DEFAULTS = {"output": 0, "classes": None, "confusion": None}

    def _check(self):
        confusion = self.spec["confusion"]
        if confusion is not None:
            if not isinstance(confusion, dict) or set(confusion) - {"truth", "confused_with", "condition"} or \
                    not all(isinstance(confusion.get(k), int) for k in ("truth", "confused_with")) or \
                    confusion["truth"] == confusion["confused_with"]:
                raise EvaluatorError("invalid_argument", "confusion is {truth, confused_with, condition?} with two distinct class ids",
                                     "confusion")

    def records(self, outputs, samples):
        logits = self._output(outputs, samples, 2, "logits [N, C]")
        if self.spec["classes"] is not None and logits.shape[1] != self.spec["classes"]:
            raise EvaluatorError("evaluator_mismatch", f"{logits.shape[1]} logits per sample, the spec says {self.spec['classes']}", "classes")
        return [{"pred": int(p)} for p in logits.argmax(axis=1)]

    def correct(self, record, sample):
        return record["pred"] == sample["label"]

    def score(self, record, sample):
        return float(self.correct(record, sample))

    def metric_parts(self, records, samples):
        pred = np.array([r["pred"] for r in records], dtype=np.int64)
        label = np.array([s["label"] for s in samples], dtype=np.int64)
        classes = self.spec["classes"] or int(max(pred.max(initial=0), label.max(initial=0)) + 1)
        if label.size and (label.min() < 0 or label.max() >= classes):
            raise EvaluatorError("evaluator_mismatch", f"a label is outside 0..{classes - 1}", "label")
        hit = (pred == label).astype(np.int64)
        onehot = np.zeros((len(label), classes), dtype=np.int64)
        onehot[np.arange(len(label)), label] = 1
        parts = {"n": np.ones_like(hit), "correct": hit, "class_n": onehot, "class_correct": onehot * hit[:, None]}
        for condition in sorted({s.get("condition") for s in samples} - {None}):
            member = np.array([s.get("condition") == condition for s in samples], dtype=np.int64)
            parts[f"cond_n:{condition}"], parts[f"cond_correct:{condition}"] = member, member * hit
        confusion = self.spec["confusion"]
        if confusion is not None:
            within = np.array([confusion.get("condition") in (None, s.get("condition")) for s in samples])
            base = (label == confusion["truth"]) & within
            parts["confusion_n"] = base.astype(np.int64)
            parts["confusion_hit"] = (base & (pred == confusion["confused_with"])).astype(np.int64)
        return parts

    def metrics_from_parts(self, sums):
        confusion = self.spec["confusion"]
        recall = [{"class": c, "value": ratio(sums["class_correct"][c], n), "n": int(n)} for c, n in enumerate(sums["class_n"])]
        out = [metric("accuracy", ratio(sums["correct"], sums["n"]), unit="rate", better="higher", primary=confusion is None,
                      per_class=recall)]
        for key in sorted(k for k in sums if k.startswith("cond_n:")):
            condition = key.split(":", 1)[1]
            out.append(metric(f"accuracy:{condition}", ratio(sums[f"cond_correct:{condition}"], sums[key]), unit="rate", better="higher"))
        if confusion is not None:
            where = f":{confusion['condition']}" if confusion.get("condition") else ""
            out.insert(0, metric(f"confusion:{confusion['truth']}->{confusion['confused_with']}{where}",
                                 ratio(sums["confusion_hit"], sums["confusion_n"]), unit="rate", better="lower", primary=True))
        return out
