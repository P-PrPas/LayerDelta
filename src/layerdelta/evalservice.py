"""Evaluations (v0.5-E4): run a checkpoint on a registered dataset with an evaluator, against a baseline, as a keyed, budgeted job.

`prepare` resolves a request into a `Plan` (checkpoint, dataset rows, evaluator spec, executable, fingerprint) and refuses what can be
refused before any work. `Ledger` keeps one row per (actor, idempotency key), so a repeated key is the same evaluation and is charged
once. `execute` is the job: forward passes in the evaluation worker (eval_worker.py), then metrics and the paired bootstrap here.
Baselines are cached by (checkpoint sha, fingerprint), as v0.1 did, so comparing a second child with the same parent runs one forward pass.

Holdout is not reachable through `evaluate`: its only split is `exploration`. The one road to holdout is `prepare(final=True)`, called by the
human-only final-evaluation route after a person confirmed the child's frozen recipe hash; those rows are the holdout audit.
"""

import hashlib
import importlib.metadata
import json
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from . import bootstrap, datasets, evaluators, executable, graph as graph_mod
from .selector import InspectError, canonical_json, digest
from .tensors import open_tensors

DIR = "evaluations"
CACHE = "evalcache"
SAMPLES_MAX = 100_000
EVAL_TIMEOUT = 900.0
_ID = re.compile(r"ev_[0-9a-f]{16}")


class EvalError(InspectError):
    def __init__(self, code: str, message: str, field: str = "evaluator"):
        super().__init__(code, message, field)


@dataclass
class Plan:
    checkpoint_id: str
    checkpoint_path: Path
    baseline_id: str | None
    baseline_path: Path | None
    dataset: dict
    files: Path
    rows: list
    split: str
    evaluator: dict          # canonical spec
    preprocessing: dict
    ir: dict
    record: dict             # the executable record
    fingerprint: str
    baseline_cached: bool
    request_digest: str
    recipe_hash: str | None = None   # a final evaluation: the frozen recipe hash the person confirmed
    intervention: dict | None = None  # E5b: an activation edit applied to the evaluated checkpoint's run only
    cache_key: str = ""              # the fingerprint WITHOUT the intervention: what a baseline run is cached under


def _pillow() -> str:
    try:
        return importlib.metadata.version("Pillow")
    except importlib.metadata.PackageNotFoundError as exc:
        raise EvalError("provider_disabled", 'evaluating needs the optional extra: pip install "layerdelta[eval]"', "dataset") from exc


def implementation_version() -> str:
    """The preprocessing implementation (preprocess.py) as a string, without importing Pillow into the server."""
    import torch
    return f"preprocess-v{_preprocess_version()}/torch-{torch.__version__.split('+')[0]}/pillow-{_pillow()}"


def _preprocess_version() -> int:
    from . import preprocess
    return preprocess.VERSION


def fill_evaluator(spec, record: dict) -> dict:
    """The evaluator spec a request means: a registered name (the task) or a spec, with the dataset's own classes, ignore index, categories
    and alphabet filled in where the spec leaves them out."""
    if isinstance(spec, str):
        spec = {"task": spec}
    if not isinstance(spec, dict):
        raise EvalError("invalid_argument", "evaluator is a task name or a spec object", "evaluator")
    spec = dict(spec)
    if spec.get("task", record["task"]) != record["task"]:
        raise EvalError("dataset_task_mismatch", f"the dataset is a {record['task']} dataset, the evaluator is {spec['task']}", "evaluator")
    spec["task"] = record["task"]
    if record["task"] == "classification":
        spec.setdefault("classes", len(record["classes"]))
    elif record["task"] == "segmentation":
        spec.setdefault("classes", len(record["classes"]))
        spec.setdefault("ignore_index", record["ignore_index"])
    elif record["task"] == "detection":
        spec.setdefault("classes", len(record["categories"]))
        spec.setdefault("class_ids", [c["id"] for c in record["categories"]])
    elif record["task"] == "ocr":
        spec.setdefault("alphabet", record["alphabet"])
    try:
        return evaluators.make(spec).canonical_spec()
    except evaluators.EvaluatorError as exc:
        raise EvalError(exc.code, exc.message, exc.field) from exc


def subset(rows: list, limit: int | None) -> list:
    """The first `limit` samples in sample-id-hash order (deterministic; the subset's hash is in the fingerprint)."""
    rows = sorted(rows, key=lambda r: r["sample_id"])
    if limit is None or limit >= len(rows):
        return rows
    return sorted(sorted(rows, key=lambda r: hashlib.sha256(f"layerdelta-eval/{r['sample_id']}".encode()).digest())[:limit], key=lambda r: r["sample_id"])


def prepare(workspace: Path, ckpts, gstore, *, checkpoint_id: str, dataset_id: str, evaluator, split: str, baseline, baseline_given: bool,
            limit_samples: int | None, graph: str | None = None, final: bool = False, recipe_hash: str | None = None,
            intervention: dict | None = None, view_for=None) -> Plan:
    if split != ("holdout" if final else "exploration"):
        raise EvalError("forbidden", "Only the exploration split can be evaluated here; holdout is reached through a final evaluation", "split")
    if limit_samples is not None and not 1 <= limit_samples <= SAMPLES_MAX:
        raise EvalError("invalid_argument", f"limit_samples is 1 to {SAMPLES_MAX}", "limit_samples")
    entry = ckpts.get(checkpoint_id)
    found = ckpts.location(checkpoint_id) if entry else None
    if entry is None or found is None:
        raise EvalError("not_found", "Unknown checkpoint", "checkpoint_id")
    cid = entry["checkpoint_id"]
    if dataset_id == datasets.MNIST_ID:
        raise EvalError("dataset_unsupported", "mnist-v1 is evaluated through the v0.1 run routes; its samples are not image files", "dataset_id")
    try:
        data = datasets.open_dataset(workspace, dataset_id)
    except datasets.DatasetError as exc:
        raise EvalError("not_found", "Unknown dataset", "dataset_id") from exc
    spec = fill_evaluator(evaluator, data.record)
    base_id = base_path = None
    if baseline_given:
        if baseline is not None:
            b = ckpts.get(baseline)
            bfound = ckpts.location(baseline) if b else None
            if b is None or bfound is None:
                raise EvalError("not_found", "Unknown baseline checkpoint", "baseline")
            base_id, base_path = b["checkpoint_id"], Path(bfound[0])
    else:
        parent = entry.get("parent_id")
        if parent is None:
            raise EvalError("no_baseline", "This checkpoint has no parent to compare with: give `baseline` (a checkpoint id) or null", "baseline")
        pfound = ckpts.location(parent)
        if pfound is None:
            raise EvalError("no_baseline", "The parent checkpoint is no longer in the catalog: give `baseline`", "baseline")
        base_id, base_path = parent, Path(pfound[0])
    table = open_tensors(Path(found[0]))
    if base_path is not None and graph_mod.table_signature(open_tensors(base_path)) != graph_mod.table_signature(table):
        raise EvalError("executable_mismatch", "The baseline has another layout than the checkpoint: they cannot run through one graph", "baseline")
    records = gstore.for_table(table)
    if graph is not None:
        records = [r for r in records if r["ir_sha256"] == graph]
    if not records:
        raise EvalError("no_graph", "This checkpoint has no graph attached: a person attaches one before it can be evaluated", "checkpoint_id")
    ir = gstore.load(records[-1]["ir_sha256"], records[-1]["table_signature"])
    exe = gstore.executable_for(ir["ir_sha256"])
    try:
        exe = executable.check_ready(ir, exe, table, workspace)
    except executable.ForwardError as exc:
        raise EvalError(exc.code, exc.message, exc.field) from exc
    rows = subset(data.samples(split), limit_samples)
    if not rows:
        raise EvalError("dataset_too_small", f"The {split} split has no sample", "dataset_id")
    iv = check_intervention(intervention, ir, exe, view_for(table) if view_for and isinstance(intervention, dict) and "group" in intervention else None) \
        if intervention is not None else None
    cache_key = digest({"evaluator": spec, "split": data.record["split_sha256"][split], "samples": digest([r["sample_id"] for r in rows]),
                        "preprocessing": data.record["preprocessing"], "preprocess": implementation_version(),
                        "executable": executable.fingerprint(exe), "device": "cpu", "dtype": "float32"})
    fingerprint = cache_key if iv is None else digest([cache_key, iv])
    cached = base_id is not None and (Path(workspace) / CACHE / f"{base_id}-{cache_key}.json").is_file()
    request_digest = digest([cid, dataset_id, split, spec, base_id, limit_samples, ir["ir_sha256"], recipe_hash, iv])
    return Plan(cid, Path(found[0]), base_id, base_path, data.record, data.files, rows, split, spec, data.record["preprocessing"], ir, exe,
                fingerprint, cached, request_digest, recipe_hash, iv, cache_key)


UNITS_MAX = 4096


def check_intervention(spec, ir: dict, exe: dict, view=None) -> dict:
    """The canonical activation edit `{node_id, unit_axis, units, op, factor}` or an error. The node is one floating-point tensor of the graph with a
    batch axis, in a builtin or torch_export executable (an exported program the interpreter runs)."""
    bad = lambda message, field="intervention": EvalError("invalid_argument", message, field)  # noqa: E731
    if isinstance(spec, dict) and "group" in spec:  # a coupling group instead of a node: resolve its producing node and unit axis
        if set(spec) - {"group", "units", "op", "factor"} or not {"group", "units", "op"} <= set(spec):
            raise bad("a group intervention is {group, units, op: zero|scale, factor?}")
        from . import groupnode
        try:
            node_id, axis, _ = groupnode.resolve(view, ir, spec["group"])
        except InspectError as exc:
            raise EvalError(exc.code, exc.message, "intervention.group") from exc
        spec = {"node_id": node_id, "unit_axis": axis, **{k: v for k, v in spec.items() if k != "group"}}
    if not isinstance(spec, dict) or not {"node_id", "units", "op"} <= set(spec) <= {"node_id", "units", "op", "factor", "unit_axis"}:
        raise bad("intervention is {node_id, units, op: zero|scale, factor?, unit_axis?} or {group, units, op, factor?}")
    if exe["kind"] not in ("builtin", "torch_export"):
        raise EvalError("probe_unsupported", f"an activation edit cannot run in a {exe['kind']} graph in this version", "intervention")
    node = next((n for n in ir["nodes"] if n["id"] == spec["node_id"]), None) if isinstance(spec["node_id"], str) else None
    if node is None:
        raise EvalError("probe_unsupported", f"the graph has no node {spec['node_id']!r}", "intervention.node_id")
    out = node["outputs"][0] if len(node["outputs"]) == 1 else None
    if out is None or out["kind"] != "tensor" or len(out["shape"]) < 2 or out["dtype"] not in ("F32", "F16", "BF16", "F64"):
        raise EvalError("probe_unsupported", f"node {spec['node_id']} does not give one floating-point tensor with a batch and a unit axis", "intervention.node_id")
    axis = spec.get("unit_axis", 1)
    if type(axis) is not int or not 1 <= axis < len(out["shape"]):
        raise bad(f"unit_axis is 1 to {len(out['shape']) - 1} for this node", "intervention.unit_axis")
    units = spec["units"]
    if (not isinstance(units, list) or not 1 <= len(units) <= UNITS_MAX or len(set(units)) != len(units)
            or not all(type(u) is int and 0 <= u < out["shape"][axis] for u in units)):
        raise bad(f"units is 1 to {UNITS_MAX} distinct indices below {out['shape'][axis]}", "intervention.units")
    op, factor = spec["op"], spec.get("factor")
    if op not in ("zero", "scale") or (op == "scale") != (factor is not None) or (factor is not None and (isinstance(factor, bool) or not isinstance(factor, (int, float))
                                                                                                          or factor != factor or abs(factor) == float("inf"))):
        raise bad("op is zero (no factor) or scale (a finite factor)", "intervention.op")
    return {"node_id": spec["node_id"], "unit_axis": axis, "units": sorted(units), "op": op, "factor": None if factor is None else float(factor)}


def cost(plan: Plan) -> int:
    """`eval_samples`: forward passes, the baseline's included unless it is a cache hit."""
    return len(plan.rows) * (1 + (0 if plan.baseline_id is None or plan.baseline_cached else 1))


# ---------------------------------------------------------------- the ledger

class Ledger:
    """Rows of the `evaluations` table (catalog schema). One per (actor, key)."""

    def __init__(self, catalog):
        self.catalog = catalog

    def find(self, actor: str, key: str):
        with self.catalog._db() as db:
            row = db.execute("SELECT * FROM evaluations WHERE actor=? AND key=?", (actor, key)).fetchone()
        return None if row is None else dict(row)

    def get(self, evaluation_id: str):
        with self.catalog._db() as db:
            row = db.execute("SELECT * FROM evaluations WHERE evaluation_id=?", (evaluation_id,)).fetchone()
        return None if row is None else dict(row)

    def delete(self, evaluation_id: str) -> None:
        with self.catalog._db(True) as db:
            db.execute("DELETE FROM evaluations WHERE evaluation_id=?", (evaluation_id,))

    def create(self, actor: str, key: str, plan: Plan) -> dict | None:
        """The new row, or None when (actor, key) already has one (a concurrent request with the same key won the race)."""
        ident = "ev_" + uuid.uuid4().hex[:16]
        try:
            self._insert(actor, key, plan, ident)
        except sqlite3.IntegrityError:
            return None
        return self.get(ident)

    def _insert(self, actor: str, key: str, plan: Plan, ident: str) -> None:
        with self.catalog._db(True) as db:
            db.execute("INSERT INTO evaluations(evaluation_id,actor,key,request_digest,checkpoint_id,dataset_id,split,baseline_id,fingerprint,status,"
                       "samples,baseline_cached,recipe_hash,intervention,created_at) VALUES (?,?,?,?,?,?,?,?,?,'queued',?,?,?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
                       (ident, actor, key, plan.request_digest, plan.checkpoint_id, plan.dataset["dataset_id"], plan.split, plan.baseline_id,
                        plan.fingerprint, len(plan.rows), int(plan.baseline_cached), plan.recipe_hash,
                        None if plan.intervention is None else canonical_json(plan.intervention)))

    def update(self, evaluation_id: str, **fields) -> None:
        names = ", ".join(f"{k}=?" for k in fields)
        with self.catalog._db(True) as db:
            db.execute(f"UPDATE evaluations SET {names} WHERE evaluation_id=?", (*fields.values(), evaluation_id))

    def holdout(self) -> list:
        """The final evaluations, newest first: what the holdout audit lists next to the v0.1 final evaluations."""
        with self.catalog._db() as db:
            return [dict(r) for r in db.execute("SELECT evaluation_id, recipe_hash, created_at, dataset_id, checkpoint_id, fingerprint, status FROM evaluations "
                                                "WHERE split='holdout' ORDER BY created_at DESC, evaluation_id DESC")]

    def interrupt(self) -> None:
        """A restart ends what was running."""
        with self.catalog._db(True) as db:
            db.execute("UPDATE evaluations SET status='failed', error=?, finished_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') "
                       "WHERE status IN ('queued','running')", (canonical_json({"code": "interrupted", "message": "the service restarted"}),))


# ---------------------------------------------------------------- the job

def _light_samples(task: str, rows: list, classes: list | None) -> list:
    """What an evaluator needs to judge records, without decoding any image (a mask is not needed to correct or score)."""
    out = []
    for row in rows:
        sample = {"sample_id": row["sample_id"], "condition": row.get("condition")}
        if task == "classification":
            sample["label"] = classes.index(row["label"])
        elif task == "detection":
            sample["boxes"] = row.get("boxes", [])
        elif task == "ocr":
            sample["text"] = row["text"]
        out.append(sample)
    return out


def _run_worker(workspace: Path, plan: Plan, checkpoint_path: Path, timeout: float, intervention: dict | None = None) -> list:
    request = {**executable.worker_request(workspace, plan.record, plan.ir, checkpoint_path), "dataset": plan.dataset, "rows": plan.rows,
               "files": str(plan.files), "preprocessing": plan.preprocessing, "evaluator": plan.evaluator, "intervention": intervention,
               "expect": {n["id"]: n["op"] for n in plan.ir["nodes"] if intervention is not None and n["id"] == intervention["node_id"]}}  # graph_run.check_ops

    def collect(returncode, out, stderr):
        error = out / "error.json"
        if error.is_file():
            try:
                body = json.loads(error.read_bytes()[:4096].decode("utf-8", errors="replace"))
                code, message = str(body["code"]), str(body["message"])[:512]
            except (ValueError, KeyError, TypeError):
                code, message = "forward_failed", "the worker reported an unreadable error"
            raise EvalError(code if re.fullmatch(r"[a-z_]{1,40}", code) else "forward_failed", message)
        result = out / "records.json"
        if returncode != 0 or not result.is_file():
            tail = stderr.read_text(encoding="utf-8", errors="replace")[-300:] if stderr.is_file() else ""
            raise EvalError("forward_failed", f"the worker exited with status {returncode} and no result: {tail.strip()}"[:512])
        body = json.loads(result.read_text(encoding="utf-8"))
        if body["sample_ids"] != [r["sample_id"] for r in plan.rows] or len(body["records"]) != len(plan.rows):
            raise EvalError("forward_failed", "the worker returned records for other samples")
        return body["records"]

    try:
        return executable.run_isolated("layerdelta.eval_worker", request, timeout=timeout, collect=collect, out_max=256 << 20)
    except executable.ForwardError as exc:
        raise EvalError(exc.code, exc.message, exc.field) from exc


def _baseline_records(workspace: Path, plan: Plan, timeout: float) -> list:
    cache = Path(workspace) / CACHE / f"{plan.baseline_id}-{plan.cache_key}.json"
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    records = _run_worker(workspace, plan, plan.baseline_path, timeout)
    cache.parent.mkdir(exist_ok=True)
    partial = cache.with_name(cache.name + f".partial-{uuid.uuid4().hex[:8]}")
    partial.write_text(json.dumps(records), encoding="utf-8")
    partial.replace(cache)
    return records


def execute(workspace: Path, ledger: Ledger, evaluation_id: str, plan: Plan, timeout: float = EVAL_TIMEOUT) -> None:
    """The job: forward passes, then metrics, changes and the paired bootstrap. Never raises; the ledger row is the outcome."""
    workspace = Path(workspace)
    try:
        ledger.update(evaluation_id, status="running", started_at=_now())
        baseline = _baseline_records(workspace, plan, timeout) if plan.baseline_id else None
        records = _run_worker(workspace, plan, plan.checkpoint_path, timeout, plan.intervention)
        folder = workspace / DIR / evaluation_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "records.json").write_text(json.dumps({"records": records, "baseline": baseline, "sample_ids": [r["sample_id"] for r in plan.rows],
                                                          "evaluator": plan.evaluator}), encoding="utf-8")
        evaluator = evaluators.make(plan.evaluator)
        samples = _light_samples(plan.dataset["task"], plan.rows, plan.dataset.get("classes"))
        seed = int(plan.fingerprint[:16], 16)
        metrics = bootstrap.compare(evaluator, records, samples, baseline, seed)
        counts = {"samples": len(records), "fixed": 0, "broken": 0, "changed": 0}
        if baseline is not None:
            for record, before, sample in zip(records, baseline, samples):
                category = bootstrap.categorize(evaluator, record, before, sample)
                if category:
                    counts[category] += 1
        result = {"metrics": metrics, "counts": counts, "resamples": None if baseline is None else bootstrap.RESAMPLES}
        ledger.update(evaluation_id, status="succeeded", result=canonical_json(result), finished_at=_now())
    except InspectError as exc:
        ledger.update(evaluation_id, status="failed", error=canonical_json({"code": exc.code, "message": exc.message}), finished_at=_now())
    except Exception as exc:  # noqa: BLE001 the job's own failure never takes the server down
        ledger.update(evaluation_id, status="failed", error=canonical_json({"code": "forward_failed", "message": type(exc).__name__}),
                      finished_at=_now())


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def view(row: dict) -> dict:
    """The public body of an evaluation row."""
    error = json.loads(row["error"]) if row["error"] else None
    result = json.loads(row["result"]) if row["result"] else None
    body = {"evaluation_id": row["evaluation_id"], "status": row["status"], "created_at": row["created_at"], "started_at": row["started_at"],
            "finished_at": row["finished_at"], "error_code": error and error["code"], "message": error and error["message"],
            "checkpoint_id": row["checkpoint_id"], "baseline_id": row["baseline_id"], "dataset_id": row["dataset_id"], "split": row["split"],
            "fingerprint": row["fingerprint"], "samples": row["samples"], "baseline_cached": bool(row["baseline_cached"]),
            "diff_hint": {"checkpoint_id": row["baseline_id"], "other": row["checkpoint_id"]} if row["baseline_id"] else None}
    if row.get("intervention"):
        body["intervention"] = json.loads(row["intervention"])
    if row["split"] == "holdout":  # only a person ever has one of these
        body["recipe_hash"] = row["recipe_hash"]
    if result:
        body.update(metrics=result["metrics"], counts=result["counts"], resamples=result["resamples"])
    return body


def changes(workspace: Path, row: dict, data, *, category: str | None, sort: str, limit: int, offset: int) -> dict:
    """A page of the samples whose record changed between the baseline and the checkpoint (no pixels)."""
    body = json.loads((Path(workspace) / DIR / row["evaluation_id"] / "records.json").read_text(encoding="utf-8"))
    if body["baseline"] is None:
        return {"changes": [], "total": 0, "offset": offset, "truncated": False}
    by_id = {r["sample_id"]: r for r in data.manifest["samples"]}
    rows = [by_id[i] for i in body["sample_ids"]]
    evaluator = evaluators.make(body["evaluator"])
    samples = _light_samples(data.record["task"], rows, data.record.get("classes"))
    found = []
    for record, before_record, sample, source in zip(body["records"], body["baseline"], samples, rows):
        kind = bootstrap.categorize(evaluator, record, before_record, sample)
        if kind is None or (category and kind != category):
            continue
        found.append({"sample_id": sample["sample_id"], "label": source.get("label", source.get("text")), "condition": sample["condition"],
                      "before": before_record, "after": record, "category": kind,
                      "score_before": evaluator.score(before_record, sample), "score_after": evaluator.score(record, sample)})
    if sort == "score_change":  # the worst change first: the most a score fell (or, for a lower-is-better score, rose)
        better = 1 if evaluator.score_better == "higher" else -1
        found.sort(key=lambda c: (better * (c["score_after"] - c["score_before"]), c["sample_id"]))
    page = found[offset:offset + limit]
    return {"changes": page, "total": len(found), "offset": offset, "truncated": offset + limit < len(found)}
