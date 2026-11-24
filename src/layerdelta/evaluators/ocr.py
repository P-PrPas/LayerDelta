"""OCR: per-step logits [N, T, A] (batch-first, design §3.1) -> CTC greedy decode -> text; truth is `sample["text"]`.

`alphabet` lists the A - 1 non-blank symbols in index order, with the blank at index `blank`. Metrics: CER and WER (edit distance summed
over samples / reference length summed; they can exceed 1) and the exact-match rate. Per sample: correct = exact match; score = CER
(lower is better), with an empty reference counted as length 1.
"""

import numpy as np

from .base import Evaluator, EvaluatorError, metric, ratio


def edit_distance(a, b) -> int:
    """Levenshtein distance of two sequences (insert, delete, substitute: cost 1)."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


class OCR(Evaluator):
    task = "ocr"
    score_better = "lower"
    DEFAULTS = {"output": 0, "alphabet": None, "blank": 0}
    REQUIRED = ("alphabet",)

    def _check(self):
        alphabet, blank = self.spec["alphabet"], self.spec["blank"]
        if not isinstance(alphabet, str) or not alphabet or len(set(alphabet)) != len(alphabet):
            raise EvaluatorError("invalid_argument", "alphabet is a non-empty string of distinct symbols", "alphabet")
        if not isinstance(blank, int) or not 0 <= blank <= len(alphabet):
            raise EvaluatorError("invalid_argument", f"blank is an index in 0..{len(alphabet)}", "blank")
        self._symbols = list(alphabet[:blank]) + [""] + list(alphabet[blank:])

    def decode(self, ids) -> str:
        out, last = [], None
        for i in ids:
            if i != last and i != self.spec["blank"]:
                out.append(self._symbols[i])
            last = i
        return "".join(out)

    def records(self, outputs, samples):
        logits = self._output(outputs, samples, 3, "logits [N, T, A] (batch-first)")
        if logits.shape[2] != len(self._symbols):
            raise EvaluatorError("evaluator_mismatch", f"{logits.shape[2]} symbols per step, the alphabet and blank make "
                                 f"{len(self._symbols)}", "alphabet")
        return [{"text": self.decode(row.tolist())} for row in logits.argmax(axis=2)]

    def correct(self, record, sample):
        return record["text"] == sample["text"]

    def score(self, record, sample):
        return edit_distance(sample["text"], record["text"]) / max(len(sample["text"]), 1)

    def metric_parts(self, records, samples):
        rows = []
        for record, sample in zip(records, samples):
            ref, hyp = sample["text"], record["text"]
            rows.append((edit_distance(ref, hyp), len(ref), edit_distance(ref.split(), hyp.split()), len(ref.split()), int(ref == hyp)))
        table = np.array(rows, dtype=np.int64).reshape(-1, 5)
        return {"char_edits": table[:, 0], "chars": table[:, 1], "word_edits": table[:, 2], "words": table[:, 3],
                "exact": table[:, 4], "n": np.ones(len(rows), dtype=np.int64)}

    def metrics_from_parts(self, sums):
        return [metric("cer", ratio(sums["char_edits"], sums["chars"]), unit="rate", better="lower", primary=True),
                metric("wer", ratio(sums["word_edits"], sums["words"]), unit="rate", better="lower"),
                metric("exact_match", ratio(sums["exact"], sums["n"]), unit="rate", better="higher")]
