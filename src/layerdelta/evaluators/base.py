"""The evaluator contract (v0.5-E3). numpy only: it runs in the evaluation worker and imports nothing with side effects.

An evaluator turns a model's raw outputs into per-sample records (the decoder), says whether each record is correct and how good it is,
and computes the task's metrics. E4 runs the forward pass and the preprocessing; it calls:

- `records(outputs, samples)`: `outputs` are the graph's outputs, batch-first, N samples; `samples` are N dicts (manifest row plus what
  preprocessing produced: `label`, `mask`, `boxes`, `text`, `geometry`). Returns N JSON-able records.
- `correct(record, sample)` and `score(record, sample)` (direction `score_better`): the per-sample category and severity (design §3.2).
- `metrics(records, samples)`: a list of `{name, primary, unit, better, value, per_class?}` on the whole set.
- `metric_parts(records, samples)` -> {key: array [N, ...]} of per-sample sufficient statistics that SUM, and
  `metrics_from_parts(sums)`; `metrics(r, s) == metrics_from_parts(sum of parts over s)`. The paired bootstrap sums parts over resampled
  indices. An evaluator whose metric does not decompose (mAP) returns None from `metric_parts`.
- `resampler(records, samples)` -> `run(index)`: the metrics of a resample, built once; detection does its matching once there, so
  2000 resamples of mAP cost about as much as one evaluation.
- `canonical_spec()`: the spec with every default filled in, hashed into the evaluation fingerprint (design §1 rule 7).
"""

import numpy as np


class EvaluatorError(ValueError):
    """`code` is an error-catalog code; `field` names the spec key or input at fault."""

    def __init__(self, code: str, message: str, field: str = "evaluator"):
        super().__init__(message)
        self.code, self.message, self.field = code, message, field


def metric(name: str, value, *, unit: str, better: str, primary: bool = False, per_class: list | None = None) -> dict:
    out = {"name": name, "primary": primary, "unit": unit, "better": better, "value": None if value is None else float(value)}
    if per_class is not None:
        out["per_class"] = per_class
    return out


def ratio(numerator, denominator):
    return None if denominator == 0 else float(numerator) / float(denominator)


class Evaluator:
    task: str = ""
    version: int = 1
    score_better = "higher"
    DEFAULTS: dict = {}
    REQUIRED: tuple = ()

    def __init__(self, spec: dict | None = None):
        spec = dict(spec or {})
        if spec.pop("task", self.task) != self.task:
            raise EvaluatorError("invalid_argument", f"this is the {self.task} evaluator", "task")
        if spec.pop("version", self.version) != self.version:
            raise EvaluatorError("invalid_argument", f"the {self.task} evaluator is version {self.version}", "version")
        unknown = sorted(set(spec) - set(self.DEFAULTS))
        if unknown:
            raise EvaluatorError("invalid_argument", f"unknown {self.task} evaluator settings: {unknown}", unknown[0])
        self.spec = {**self.DEFAULTS, **spec}
        for key in self.REQUIRED:
            if self.spec.get(key) is None:
                raise EvaluatorError("invalid_argument", f"the {self.task} evaluator needs `{key}`", key)
        self._check()

    def _check(self) -> None:
        pass

    def canonical_spec(self) -> dict:
        return {"task": self.task, "version": self.version, **dict(sorted(self.spec.items()))}

    def _output(self, outputs: list, samples: list, ndim: int, layout: str) -> np.ndarray:
        index = self.spec["output"]
        if not isinstance(index, int) or not 0 <= index < len(outputs):
            raise EvaluatorError("invalid_argument", f"output {index!r} does not exist: the graph has {len(outputs)}", "output")
        array = np.asarray(outputs[index])
        if array.ndim != ndim or array.shape[0] != len(samples):
            raise EvaluatorError("evaluator_mismatch", f"the {self.task} evaluator expects output {index} as {layout} with N = "
                                 f"{len(samples)}; it has shape {list(array.shape)}", "output")
        return array

    def correct(self, record: dict, sample: dict) -> bool:
        raise NotImplementedError

    def score(self, record: dict, sample: dict) -> float:
        raise NotImplementedError

    def metric_parts(self, records: list, samples: list) -> dict | None:
        return None

    def metrics_from_parts(self, sums: dict) -> list:
        raise NotImplementedError

    def metrics(self, records: list, samples: list) -> list:
        parts = self.metric_parts(records, samples)
        return self.metrics_from_parts({key: value.sum(axis=0) for key, value in parts.items()})

    def resampler(self, records: list, samples: list):
        """`run(index) -> metrics` of the resample that takes sample `index[k]` k-th time (indices may repeat). Built once per evaluation:
        an evaluator without parts (detection) overrides it to do its per-sample work once, not once per resample."""
        parts = self.metric_parts(records, samples)
        if parts is None:
            return lambda index: self.metrics([records[i] for i in index], [samples[i] for i in index])
        return lambda index: self.metrics_from_parts({key: value[np.asarray(index)].sum(axis=0) for key, value in parts.items()})
