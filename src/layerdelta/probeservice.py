"""Probes (v0.5-E5): per-unit aggregates of graph-node outputs on exploration samples, as a keyed, budgeted job.

A probe needs weights:read (activations on known inputs recover weights algebraically, Roadmap R2), so the K rule of egress class 1 applies: a
unit's statistics must aggregate at least K elements (samples x positions); a split group below K is skipped (`skipped_small`), and a probe where
every group is below K is refused (`aggregate_too_small`). Raw activations need weights:values. Only the `builtin` and `torch_export` executables
can be probed; a node the program does not give as one tensor is `probe_unsupported`, never a partial answer.
"""

import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path

from . import datasets, evalservice, executable, graph as graph_mod, groupnode
from .selector import InspectError, canonical_json, digest
from .tensors import open_tensors

PROBE_TIMEOUT = 900.0
NODES_MAX = 8
BINS_MAX = 64
_ID = re.compile(r"pr_[0-9a-f]{16}")
_NODE = re.compile(r"n[0-9]{1,6}")


class ProbeError(InspectError):
    def __init__(self, code: str, message: str, field: str = "nodes"):
        super().__init__(code, message, field)


@dataclass
class ProbePlan:
    checkpoint_id: str
    checkpoint_path: Path
    dataset: dict
    files: Path
    rows: list
    ir: dict
    record: dict
    preprocessing: dict
    nodes: list           # [{node_id, unit_axis, units, positions}]
    split_by: str | None
    bins: int
    values: bool
    skipped: list         # groups below K: [{group, n}]
    request_digest: str
    mnist: dict | None = None


def prepare(workspace: Path, ckpts, gstore, *, checkpoint_id: str, dataset_id: str, nodes, unit_axis: int, split_by: str | None, bins: int,
            values: bool, limit_samples: int | None, min_aggregate: int, graph: str | None = None, group: str | None = None,
            view_for=None, mnist_manifest: dict | None = None) -> ProbePlan:
    if (nodes is None) == (group is None):
        raise ProbeError("invalid_argument", "give either nodes (node ids of the graph) or group (a coupling group id)", "nodes")
    if group is None and not isinstance(nodes, list):
        raise ProbeError("invalid_argument", f"nodes is 1 to {NODES_MAX} distinct node ids of the graph (n0, n1, ...)")
    if group is not None:
        nodes = ["n0"]  # replaced below, once the graph is known
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= NODES_MAX or not all(isinstance(n, str) and _NODE.fullmatch(n) for n in nodes) or len(set(nodes)) != len(nodes):
        raise ProbeError("invalid_argument", f"nodes is 1 to {NODES_MAX} distinct node ids of the graph (n0, n1, ...)")
    if split_by not in (None, "label", "condition"):
        raise ProbeError("invalid_argument", "split_by is label, condition or null", "split_by")
    if not isinstance(bins, int) or not 0 <= bins <= BINS_MAX:
        raise ProbeError("invalid_argument", f"bins is 0 to {BINS_MAX}", "bins")
    if limit_samples is not None and not 1 <= limit_samples <= evalservice.SAMPLES_MAX:
        raise ProbeError("invalid_argument", f"limit_samples is 1 to {evalservice.SAMPLES_MAX}", "limit_samples")
    entry = ckpts.get(checkpoint_id)
    found = ckpts.location(checkpoint_id) if entry else None
    if entry is None or found is None:
        raise ProbeError("not_found", "Unknown checkpoint", "checkpoint_id")
    try:
        data = datasets.open_any(workspace, dataset_id, mnist_manifest)
    except datasets.DatasetError as exc:
        raise ProbeError(exc.code if exc.code == "dataset_unsupported" else "not_found", exc.message if exc.code == "dataset_unsupported" else "Unknown dataset",
                         "dataset_id") from exc
    table = open_tensors(Path(found[0]))
    records = gstore.for_table(table)
    if graph is not None:
        records = [r for r in records if r["ir_sha256"] == graph]
    if not records:
        raise ProbeError("no_graph", "This checkpoint has no graph attached: a person attaches one before it can be probed", "checkpoint_id")
    ir = gstore.load(records[-1]["ir_sha256"], records[-1]["table_signature"])
    try:
        exe = executable.check_ready(ir, gstore.executable_for(ir["ir_sha256"]), table, workspace)
    except executable.ForwardError as exc:
        raise ProbeError(exc.code, exc.message, exc.field) from exc
    if exe["kind"] not in ("builtin", "torch_export"):
        raise ProbeError("probe_unsupported", f"a {exe['kind']} graph cannot be probed in this version (builtin and torch_export can)", "checkpoint_id")
    if group is not None:  # the units of a coupling group, where they show up as activations
        try:
            node_id, unit_axis, _ = groupnode.resolve(view_for(table) if view_for else None, ir, group)
        except InspectError as exc:
            raise ProbeError(exc.code, exc.message, exc.field) from exc
        nodes = [node_id]
    by_id = {n["id"]: n for n in ir["nodes"]}
    specs = []
    for node_id in nodes:
        node = by_id.get(node_id)
        if node is None:
            raise ProbeError("probe_unsupported", f"the graph has no node {node_id}", "nodes")
        out = node["outputs"][0] if len(node["outputs"]) == 1 else None
        if out is None or out["kind"] != "tensor" or len(out["shape"]) < 2 or out["dtype"] not in ("F32", "F16", "BF16", "F64"):
            raise ProbeError("probe_unsupported", f"node {node_id} ({node['op']}) does not give one floating-point tensor with a batch and a unit axis", "nodes")
        if not 1 <= unit_axis < len(out["shape"]):
            raise ProbeError("invalid_argument", f"unit_axis is 1 to {len(out['shape']) - 1} for node {node_id}", "unit_axis")
        positions = 1
        for axis, size in enumerate(out["shape"]):
            if axis not in (0, unit_axis):
                positions *= size
        specs.append({"node_id": node_id, "unit_axis": unit_axis, "units": out["shape"][unit_axis], "positions": positions})
    rows = evalservice.subset(data.samples("exploration"), limit_samples)
    if not rows:
        raise ProbeError("dataset_too_small", "The exploration split has no sample", "dataset_id")
    groups: dict = {}
    for row in rows:
        key = str(row.get("label")) if split_by == "label" and data.record["task"] == "classification" else (row.get("condition") or "none") if split_by == "condition" else "all"
        groups[key] = groups.get(key, 0) + 1
    skipped = []
    if min_aggregate:  # egress class 1: a unit aggregates at least K elements
        smallest = min(spec["positions"] for spec in specs)
        need = min_aggregate * max(1, bins)  # a histogram of `bins` buckets needs K elements per bucket (the v0.4 S4a.2 rule), not just K
        for group, n in sorted(groups.items()):
            if n * smallest < need:
                skipped.append({"group": group, "n": n})
        if len(skipped) == len(groups):
            from . import budget
            raise budget.AggregateTooSmall(max(groups.values()) * smallest, need, "limit_samples" if not bins else "bins")
    request_digest = digest([exe.get("sha256") or exe.get("model"), checkpoint_id, dataset_id, specs, split_by, bins, values, limit_samples, ir["ir_sha256"]])
    return ProbePlan(entry["checkpoint_id"], Path(found[0]), data.record, data.files, rows, ir, exe, data.record["preprocessing"], specs, split_by, bins,
                     values, skipped, request_digest, mnist_manifest if dataset_id == datasets.MNIST_ID else None)


class ProbeLedger:
    def __init__(self, catalog):
        self.catalog = catalog

    def find(self, actor: str, key: str):
        with self.catalog._db() as db:
            row = db.execute("SELECT * FROM probes WHERE actor=? AND key=?", (actor, key)).fetchone()
        return None if row is None else dict(row)

    def get(self, probe_id: str):
        with self.catalog._db() as db:
            row = db.execute("SELECT * FROM probes WHERE probe_id=?", (probe_id,)).fetchone()
        return None if row is None else dict(row)

    def create(self, actor: str, key: str, plan: ProbePlan) -> dict | None:
        ident = "pr_" + uuid.uuid4().hex[:16]
        try:
            with self.catalog._db(True) as db:
                db.execute("INSERT INTO probes(probe_id,actor,key,request_digest,checkpoint_id,dataset_id,status,samples,created_at) "
                           "VALUES (?,?,?,?,?,?,'queued',?,strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
                           (ident, actor, key, plan.request_digest, plan.checkpoint_id, plan.dataset["dataset_id"], len(plan.rows)))
        except sqlite3.IntegrityError:
            return None
        return self.get(ident)

    def delete(self, probe_id: str) -> None:
        with self.catalog._db(True) as db:
            db.execute("DELETE FROM probes WHERE probe_id=?", (probe_id,))

    def update(self, probe_id: str, **fields) -> None:
        names = ", ".join(f"{k}=?" for k in fields)
        with self.catalog._db(True) as db:
            db.execute(f"UPDATE probes SET {names} WHERE probe_id=?", (*fields.values(), probe_id))

    def interrupt(self) -> None:
        with self.catalog._db(True) as db:
            db.execute("UPDATE probes SET status='failed', error=?, finished_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE status IN ('queued','running')",
                       (canonical_json({"code": "interrupted", "message": "the service restarted"}),))


def execute(workspace: Path, ledger: ProbeLedger, probe_id: str, plan: ProbePlan, timeout: float = PROBE_TIMEOUT) -> None:
    """The job. Never raises; the ledger row is the outcome."""
    try:
        ledger.update(probe_id, status="running", started_at=evalservice._now())
        skip = {s["group"] for s in plan.skipped}
        rows = [r for r in plan.rows if _group(plan, r) not in skip]
        request = {**executable.worker_request(workspace, plan.record, plan.ir, plan.checkpoint_path), "dataset": plan.dataset, "rows": rows,
                   "files": str(plan.files), "preprocessing": plan.preprocessing, "nodes": plan.nodes, "split_by": plan.split_by, "bins": plan.bins,
                   "values": plan.values, "expect": {n["id"]: n["op"] for n in plan.ir["nodes"] if n["id"] in {s["node_id"] for s in plan.nodes}}}

        def collect(returncode, out, stderr):
            error = out / "error.json"
            if error.is_file():
                try:
                    body = json.loads(error.read_bytes()[:4096].decode("utf-8", errors="replace"))
                    code, message = str(body["code"]), str(body["message"])[:512]
                except (ValueError, KeyError, TypeError):
                    code, message = "forward_failed", "the worker reported an unreadable error"
                raise ProbeError(code if re.fullmatch(r"[a-z_]{1,40}", code) else "forward_failed", message)
            result = out / "probe.json"
            if returncode != 0 or not result.is_file():
                tail = stderr.read_text(encoding="utf-8", errors="replace")[-300:] if stderr.is_file() else ""
                raise ProbeError("forward_failed", f"the worker exited with status {returncode} and no result: {tail.strip()}"[:512])
            return json.loads(result.read_text(encoding="utf-8"))

        arrays = None
        if plan.mnist is not None:
            from . import data as mnist
            images, _, loaded = mnist.load_split(Path(plan.files), plan.mnist, "exploration")
            at = {r["sample_id"]: i for i, r in enumerate(loaded)}
            arrays = [images.numpy()[[at[r["sample_id"]] for r in rows]]]
        request["mnist"] = plan.mnist is not None
        result = executable.run_isolated("layerdelta.probe_worker", request, arrays=arrays, timeout=timeout, collect=collect, out_max=64 << 20)
        for node in result["nodes"]:  # the rows of a skipped group are never sent to the worker; this keeps it true if that ever changes
            for group in skip:
                node["groups"].pop(group, None)
        result["skipped_small"] = plan.skipped
        ledger.update(probe_id, status="succeeded", result=canonical_json(result), finished_at=evalservice._now())
    except executable.ForwardError as exc:
        ledger.update(probe_id, status="failed", error=canonical_json({"code": exc.code, "message": exc.message}), finished_at=evalservice._now())
    except InspectError as exc:
        ledger.update(probe_id, status="failed", error=canonical_json({"code": exc.code, "message": exc.message}), finished_at=evalservice._now())
    except Exception as exc:  # noqa: BLE001 the job's own failure never takes the server down
        ledger.update(probe_id, status="failed", error=canonical_json({"code": "forward_failed", "message": type(exc).__name__}), finished_at=evalservice._now())


def _group(plan: ProbePlan, row: dict) -> str:
    if plan.split_by == "label" and plan.dataset["task"] == "classification":
        return str(row.get("label"))
    if plan.split_by == "condition":
        return row.get("condition") or "none"
    return "all"


def view(row: dict, *, values: bool) -> dict:
    """The body of a probe row. Raw activations are in it only when the reader has weights:values."""
    error = json.loads(row["error"]) if row["error"] else None
    body = {"probe_id": row["probe_id"], "status": row["status"], "created_at": row["created_at"], "started_at": row["started_at"],
            "finished_at": row["finished_at"], "error_code": error and error["code"], "message": error and error["message"],
            "checkpoint_id": row["checkpoint_id"], "dataset_id": row["dataset_id"], "samples": row["samples"]}
    if row["result"]:
        result = json.loads(row["result"])
        if not values:
            result.pop("values", None)
        body.update(result)
    return body
