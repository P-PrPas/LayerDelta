"""Task evaluators (v0.5-E3): classification, segmentation, detection, OCR. See base.py for the contract and
docs/reference/evaluators.md for each task's output contract and metric definitions."""

from .base import Evaluator, EvaluatorError
from .classification import Classification
from .detection import Detection
from .ocr import OCR
from .segmentation import Segmentation

EVALUATORS = {e.task: e for e in (Classification, Segmentation, Detection, OCR)}


def make(spec: dict) -> Evaluator:
    """An evaluator from a spec `{task, version?, ...settings}`."""
    task = (spec or {}).get("task")
    if task not in EVALUATORS:
        raise EvaluatorError("invalid_argument", f"task must be one of {sorted(EVALUATORS)}", "task")
    return EVALUATORS[task](spec)


__all__ = ["EVALUATORS", "Evaluator", "EvaluatorError", "make"]
