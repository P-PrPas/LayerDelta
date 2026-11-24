"""v2 HTTP routes: tensor-space inspection, edit programs and blobs (v0.2-T4c).

Every route sits under /api/v1/projects/{p}/checkpoints/{checkpoint_id}/... (the id is the T4b
catalog's content-addressed id, resolved through the catalog: the URL text is never used as an
identity) except POST /blobs. Each one wraps one library call and returns its v1 envelope
(schema_version, resolved, result, fingerprints, side_effects, truncated, cursor). Capabilities:
model:read (describe), weights:read (stats, diff, query, plan, validate), weights:values (read-tensor, diff values),
weights:write (apply, blobs). The human session has all of them. Pure routes always return
side_effects []. Cursors are HMAC-signed here, so a client cannot forge or edit one.
"""

import hashlib
import hmac
import os
import queue
import re
import secrets
import shutil
import sys
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, Request
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from . import budget, coupling, datasets, edit, evalservice, executable, graph_ops, group_ops, inspect_ops, onnx_provider, probeservice, providers, structure, trace
from .checkpoints import ImportFailure, register_child
from .graph import GraphMismatch, GraphStore, table_signature
from .export_provider import TorchExportProvider
from .graphview import GraphView, provider_list
from .registry import HUMAN_CAPABILITIES
from .schemas import (ApplyRequest, AttachGraph, BudgetChange, DatasetImport, DiffRequest, EditRequest, QueryRequest, ReadTensorRequest, ResolveInterventions, ResolveSelections, EvaluateRequest, FinalEvaluateRequest, ProbeRequest,
                      StatsRequest)
from .selector import InspectError, digest
from .tensors import TensorFileError, open_tensors

BLOB_TOTAL_MAX = 1 << 30
ATTACH_KEEP = 200       # finished attach jobs remembered
PROPOSALS_PENDING_MAX = 50   # proposals one session may have waiting for a person
DIFF_HINT_MAX = 20      # selectors an apply result echoes for its diff
DERIVED_KEEP = 16       # derivations (roles and groups) kept per process
APPLY_WAIT_SECONDS = 60


@dataclass
class Supervision:
    """What a person can change about the running agent session (mutable: the actor dependency and the routes share it)."""
    revoked: bool = False
    approval: bool = False   # an agent apply waits for a person (--agent-apply-requires-approval)
    allow_noop: bool = False  # an agent's apply never answers no_op (--agent-allow-noop): a refusal would say "this changed nothing"


@dataclass
class V2Context:
    base: str
    workspace: Path
    creds: object
    ckpts: object
    store: object
    limits: object
    actor: object
    key: object
    has: object          # has(user, capability) -> bool
    fail: object         # fail(code, message, status) raises
    blob_max: int
    user_code: bool = False         # the project opted in to running user model code (torch_export)
    export_timeout: float = 120.0
    budgets: object = None          # the session's Budgets (limits and live use), for the supervision summary
    supervision: object = None      # Supervision: revoked flag and the approval gate
    manifest: dict = None           # the v0.1 reference manifest: the `mnist-v1` dataset
    eval_timeout: float = 900.0     # seconds one evaluation worker may run


def install_v2(app, ctx: V2Context) -> None:
    base, fail, ckpts, workspace = ctx.base, ctx.fail, ctx.ckpts, ctx.workspace
    secret = secrets.token_bytes(32)  # dedicated cursor key: never derived from a credential a browser holds
    gstore = GraphStore(ckpts)
    derived_cache: OrderedDict = OrderedDict()
    attachments: OrderedDict = OrderedDict()
    attach_lock = threading.Lock()
    derived_lock = threading.Lock()  # handler threads and attach threads share derived_cache

    def needs(*capabilities):
        def dependency(user=Depends(ctx.actor)):
            for capability in capabilities:
                if not ctx.has(user, capability):
                    fail("forbidden", f"The {capability} capability is required", 403)
            return user
        return dependency

    def structure_session(user):
        """Only the structure profile may hold weights:write without weights:read (a token that somehow does otherwise is refused)."""
        return user != "human" and ctx.creds.profile == "structure"

    def needs_edit(user=Depends(ctx.actor)):
        """plan, validate and diff: weights:read, or weights:write for a session that sees the structure view only (v0.5-E6)."""
        if not (ctx.has(user, "weights:read") or (ctx.has(user, "weights:write") and structure_session(user))):
            fail("forbidden", "The weights:read capability is required", 403)
        return user

    def needs_write(user=Depends(ctx.actor)):
        if not (ctx.has(user, "weights:write") and (ctx.has(user, "weights:read") or structure_session(user))):
            fail("forbidden", "The weights:write capability is required", 403)
        return user

    def blind(user):
        """A session without weights:read: its edit responses are rebuilt from an allow-list (structure.py)."""
        return not ctx.has(user, "weights:read")

    def place_new(source: Path, target: Path) -> bool:
        """Make `source` visible as `target` without ever replacing a file; False if `target` already exists."""
        try:
            os.link(source, target)
        except FileExistsError:
            return False
        except OSError:  # a filesystem without hard links
            if target.exists():
                return False
            os.replace(source, target)
            return True
        source.unlink(missing_ok=True)
        return True

    def sign(cursor):
        if cursor is None:
            return None
        return f"c2.{cursor[3:]}.{hmac.digest(secret, cursor.encode(), 'sha256')[:16].hex()}"

    def unsign(cursor):
        if cursor is None:
            return None
        try:
            tag, body, mac = cursor.split(".")
            original = "c1." + body
            if tag != "c2" or not hmac.compare_digest(mac, hmac.digest(secret, original.encode(), "sha256")[:16].hex()):
                raise ValueError
            return original
        except (ValueError, AttributeError):
            raise InspectError("invalid_cursor", "cursor was not issued for this session", "cursor") from None

    def finish(envelope):
        trace.note(selection_id=envelope["fingerprints"].get("selection"),
                   other_checkpoint_id=envelope["fingerprints"].get("checkpoint_b"))
        prints, resolved = envelope["fingerprints"], envelope.get("resolved") or {}
        if prints.get("selection") and isinstance(resolved.get("selector"), dict):  # so a human can resolve the id the trace holds
            trace.remember_selection(ckpts, prints.get("checkpoint") or prints.get("checkpoint_a"), prints["selection"], resolved["selector"])
        envelope["cursor"] = sign(envelope["cursor"])
        return envelope

    def open_path(path):
        try:
            return open_tensors(path, max_file_bytes=ctx.limits.max_file_bytes, max_header_bytes=ctx.limits.max_header_bytes)
        except TensorFileError as exc:
            fail(exc.code, str(exc), 409)

    def open_checkpoint(checkpoint_id):
        """(catalog id, table). The id returned is the catalog's own, never the raw URL text."""
        entry = ckpts.get(checkpoint_id)
        found = ckpts.location(checkpoint_id) if entry else None
        if entry is None or found is None:
            fail("not_found", "Unknown checkpoint", 404)
        return entry["checkpoint_id"], open_path(found[0])

    def given(model, *names):
        return {n: getattr(model, n) for n in names}

    exporter = TorchExportProvider(enabled=ctx.user_code, timeout=ctx.export_timeout)

    def offered(table):
        return {"providers": provider_list(providers.registered_models(), ctx.user_code), "reusable": gstore.offers(table)}

    def view_for(table):
        """The checkpoint's graph (the newest one its table signature carries) with roles and groups, or None."""
        records = gstore.for_table(table)
        if not records:
            return None
        record = records[-1]
        ir = gstore.load(record["ir_sha256"], record["table_signature"])
        key = (record["ir_sha256"], record["table_signature"])
        with derived_lock:
            derived = derived_cache.get(key)
            if derived is not None:
                derived_cache.move_to_end(key)
        if derived is None:  # deriving can take a while: never hold the lock for it
            derived = coupling.derive(ir, table)
            with derived_lock:
                derived_cache[key] = derived
                while len(derived_cache) > DERIVED_KEEP:
                    derived_cache.popitem(last=False)
        view = GraphView.build(ir, table, record, derived)
        view.offered = offered(table)
        return view

    def view_if(table, selector):
        """Only a group selector needs the graph; every other request never derives anything."""
        return view_for(table) if group_ops.is_group(selector) else None

    def human_only(user=Depends(ctx.actor)):
        if user != "human":
            fail("forbidden", "Attaching a graph is a human action", 403)
        return user

    # ------------------------------------------------------------ pure inspection
    @app.get(base + "/checkpoints/{checkpoint_id}/describe")
    def describe(checkpoint_id: str, user=Depends(needs("model:read")), name_glob: str | None = None,
                 limit: int | None = None, cursor: str | None = None):
        cid, table = open_checkpoint(checkpoint_id)
        result = finish(inspect_ops.describe(table, name_glob=name_glob, limit=limit, cursor=unsign(cursor), checkpoint_id=cid,
                                             graph=view_for(table)))
        if blind(user):  # tensor hashes and identical-bytes groups pin content: a session without weights:read never sees them filled
            result["result"]["identical_bytes"] = None
            for row in result["result"]["tensors"]:
                row["tensor_sha256"] = row["identical_bytes_group"] = None
        return result

    @app.post(base + "/checkpoints/{checkpoint_id}/read-tensor")
    def read_tensor(checkpoint_id: str, data: ReadTensorRequest, user=Depends(needs("weights:read", "weights:values"))):
        cid, table = open_checkpoint(checkpoint_id)
        return finish(inspect_ops.read_tensor(table, data.selector, max_elems=data.max_elems, cursor=unsign(data.cursor),
                                              include_range=data.include_range, checkpoint_id=cid,
                                              graph=view_if(table, data.selector)))

    @app.post(base + "/checkpoints/{checkpoint_id}/stats")
    def stats(checkpoint_id: str, data: StatsRequest, user=Depends(needs("weights:read"))):
        cid, table = open_checkpoint(checkpoint_id)
        return finish(inspect_ops.stats(table, data.selector, reduce_over_axes=data.reduce_over_axes, bins=data.bins,
                                        limit=data.limit, cursor=unsign(data.cursor), checkpoint_id=cid,
                                        graph=view_if(table, data.selector)))

    @app.post(base + "/checkpoints/{checkpoint_id}/diff")
    def diff(checkpoint_id: str, data: DiffRequest, user=Depends(needs_edit)):
        if data.values and not ctx.has(user, "weights:values"):  # element-wise deltas are values; summaries are aggregates
            fail("forbidden", "The weights:values capability is required", 403)
        if blind(user):  # whole tensors, a status each: a selector or a unit breakdown says which units an edit moved
            if data.selector is not None or data.values or data.include_range:
                fail("forbidden", "A session without weights:read compares whole tensors only", 403)
            data = data.model_copy(update={"unit_axis": None})
        cid, table = open_checkpoint(checkpoint_id)
        other_id, other = open_checkpoint(data.other)
        result = finish(inspect_ops.diff(
            table, other, selector=data.selector, name_glob=data.name_glob, unit_axis=data.unit_axis,
            values=data.values, include_same=data.include_same, limit=data.limit, max_elems=data.max_elems,
            unit_axis_implicit="unit_axis" not in data.model_fields_set,
            include_range=data.include_range, cursor=unsign(data.cursor), checkpoint_a=cid, checkpoint_b=other_id,
            graph=view_if(table, data.selector)))
        return structure.view("diff", result) if blind(user) else result

    @app.post(base + "/checkpoints/{checkpoint_id}/query")
    def query(checkpoint_id: str, data: QueryRequest, user=Depends(needs("weights:read"))):
        cid, table = open_checkpoint(checkpoint_id)
        fields = given(data, "name_glob", "dtype", "min_ndim", "max_ndim", "editable", "min_bytes", "max_bytes", "where",
                       "limit", "max_scan_elems")
        graph = view_for(table) if isinstance(data.where, dict) and "group" in data.where else None
        return finish(inspect_ops.query(table, cursor=unsign(data.cursor), checkpoint_id=cid, graph=graph, **fields))

    # ------------------------------------------------------------ graph, groups, nodes (read), attach (human)
    @app.get(base + "/checkpoints/{checkpoint_id}/graph")
    def graph(checkpoint_id: str, user=Depends(needs("model:read")), view: str | None = None, module_prefix: str | None = None,
              op: str | None = None, limit: int | None = None, cursor: str | None = None):
        cid, table = open_checkpoint(checkpoint_id)
        return finish(graph_ops.graph_view(table, view_for(table), offered=offered(table), view_name=view,
                                           module_prefix=module_prefix, op=op, limit=limit, cursor=unsign(cursor),
                                           checkpoint_id=cid))

    @app.get(base + "/checkpoints/{checkpoint_id}/graph/nodes/{node_id}")
    def graph_node(checkpoint_id: str, node_id: str, user=Depends(needs("model:read"))):
        cid, table = open_checkpoint(checkpoint_id)
        return finish(graph_ops.node(table, view_for(table), node_id, checkpoint_id=cid))

    @app.get(base + "/checkpoints/{checkpoint_id}/groups")
    def groups(checkpoint_id: str, user=Depends(needs("model:read")), tensor: str | None = None, node: str | None = None,
               module_prefix: str | None = None, closed: bool | None = None, kind: str | None = None,
               limit: int | None = None, cursor: str | None = None):
        cid, table = open_checkpoint(checkpoint_id)
        return finish(graph_ops.groups(table, view_for(table), tensor=tensor, node=node, module_prefix=module_prefix,
                                       closed=closed, kind=kind, limit=limit, cursor=unsign(cursor), checkpoint_id=cid))

    def run_attach(record, path, spec):
        """Build or reuse a graph, prove it binds, store it, derive it. Runs in a thread; the record is its status."""
        def stage(name, done):
            with attach_lock:
                record.update(stage=name, done=done)
        keep = None
        try:
            table = open_path(path)
            with attach_lock:
                record["status"] = "running"
            stage("building", 0)
            keep = (workspace / "staging" / f"attach-{uuid.uuid4().hex}")  # what the export worker saves besides the graph
            keep.mkdir(parents=True)
            if "reuse" in spec:
                ir, provider = gstore.load(spec["reuse"]), gstore.provider_of(spec["reuse"])
            elif spec["provider"] == "torch_export":
                ir, provider = exporter.build(spec, artifact_dir=keep), "torch_export"
            elif spec["provider"] == "onnx":
                ir, provider = onnx_provider.build(spec, table, timeout=ctx.export_timeout), "onnx"
            else:
                ir, provider = providers.build_graph(spec), spec["provider"]
            stage("binding", 1)
            derived = coupling.derive(ir, table)  # a graph the rules cannot handle is refused here, never stored
            gstore.put(ir, table, provider=provider, checkpoint_id=record["checkpoint_id"])
            if "reuse" not in spec:  # the executable an evaluation runs (v0.5-E1); a failure here never fails the attach
                try:
                    executable.capture(gstore, ir, spec, table, workspace, keep)
                except Exception as exc:  # noqa: BLE001
                    print(f"attach: could not record the executable: {type(exc).__name__}", file=sys.stderr)
            stage("deriving", 2)
            with derived_lock:
                derived_cache[(ir["ir_sha256"], table_signature(table))] = derived
                while len(derived_cache) > DERIVED_KEEP:
                    derived_cache.popitem(last=False)
            view = view_for(table)
            with attach_lock:
                record.update(status="succeeded", stage="done", done=3, graph=view.meta(), error=None)
        except GraphMismatch as exc:
            fail_attach(record, "graph_mismatch", str(exc), list(exc.names))
        except InspectError as exc:
            fail_attach(record, exc.code, exc.message, [])
        except providers.GraphBuildError as exc:
            fail_attach(record, exc.code, str(exc), [])
        except Exception as exc:  # the job's own failure never takes the server down
            fail_attach(record, "attach_failed", f"{type(exc).__name__}", [])
        finally:
            if keep is not None:
                shutil.rmtree(keep, ignore_errors=True)

    def fail_attach(record, code, message, names):
        with attach_lock:
            record.update(status="failed", error={"code": code, "message": message, "names": names})

    @app.post(base + "/checkpoints/{checkpoint_id}/graphs", status_code=202)
    def attach(checkpoint_id: str, data: AttachGraph, user=Depends(human_only)):
        entry = ckpts.get(checkpoint_id)
        found = ckpts.location(checkpoint_id) if entry else None
        if entry is None or found is None:
            fail("not_found", "Unknown checkpoint", 404)
        cid = entry["checkpoint_id"]
        if (data.reuse is None) == (data.provider is None):
            raise InspectError("invalid_argument", "give a provider (with its model) or a graph to reuse", "provider")
        if data.reuse is not None:
            table = open_path(found[0])
            if data.reuse not in {o["ir_sha256"] for o in gstore.offers(table)} | {r["ir_sha256"] for r in gstore.for_table(table)}:
                raise InspectError("not_offered", "that graph is not offered for this checkpoint's layout", "reuse")
            spec = {"reuse": data.reuse}
        else:
            enabled = {p["name"]: p for p in provider_list([], ctx.user_code)}
            if data.provider not in enabled or not enabled[data.provider]["enabled"]:
                raise InspectError("provider_disabled", enabled.get(data.provider, {}).get("reason") or "unknown provider", "provider")
            if data.provider == "builtin":
                if not isinstance(data.model, str) or any(v is not None for v in (data.entry, data.path, data.example_inputs, data.root, data.dynamic_dim)):
                    raise InspectError("invalid_argument", "the builtin provider takes a model name and nothing else", "model")
                spec = {"provider": "builtin", "model": data.model}
            elif data.provider == "onnx":
                if any(v is not None for v in (data.model, data.entry, data.example_inputs, data.root)):
                    raise InspectError("invalid_argument", "onnx takes a path (and dynamic_dim), nothing else", "provider")
                spec = {"provider": "onnx", "path": data.path, **({"dynamic_dim": data.dynamic_dim} if data.dynamic_dim is not None else {})}
                try:
                    spec = onnx_provider.check_spec(spec)
                except providers.GraphBuildError as exc:
                    raise InspectError(exc.code, exc.message, "spec") from exc
            else:
                if data.model is not None or data.dynamic_dim is not None:
                    raise InspectError("invalid_argument", "torch_export takes entry, path, example_inputs and root, not a model", "model")
                spec = {"provider": "torch_export", "entry": data.entry, "path": data.path, "example_inputs": data.example_inputs,
                        **({"root": data.root} if data.root is not None else {})}
                try:
                    from .export_provider import check_spec
                    spec = check_spec(spec)  # a bad spec is a request error, before any process starts
                except providers.GraphBuildError as exc:
                    raise InspectError(exc.code, exc.message, "spec") from exc
        record = {"attach_id": "att_" + uuid.uuid4().hex[:16], "checkpoint_id": cid, "status": "queued", "stage": "queued",
                  "done": 0, "total": 3, "unit": "stages", "graph": None, "error": None}
        with attach_lock:
            attachments[record["attach_id"]] = record
            while len(attachments) > ATTACH_KEEP:
                attachments.popitem(last=False)
        threading.Thread(target=run_attach, args=(record, found[0], spec), daemon=True, name="attach").start()
        return dict(record)

    @app.get(base + "/graph-attachments/{attach_id}")
    def attach_status(attach_id: str, user=Depends(human_only)):
        with attach_lock:
            record = attachments.get(attach_id)
            if record is None:
                fail("not_found", "Unknown graph attachment", 404)
            return dict(record)

    # ------------------------------------------------------------ datasets (v0.5-E2): a person imports, a token lists and reads
    imports: OrderedDict = OrderedDict()
    import_lock = threading.Lock()

    def run_import(record, data):
        def finish(**fields):
            with import_lock:
                record.update(fields)
        try:
            finish(status="running")
            source = Path(data.path)
            if not source.is_absolute() or not source.is_dir():
                raise datasets.DatasetError("invalid_path", "path must be an absolute directory that exists", "path")
            result = datasets.import_dataset(workspace, source, fmt=data.format, name=data.name, preprocessing=data.preprocessing)
            finish(status="succeeded", dataset=result, error=None)
        except InspectError as exc:
            finish(status="failed", error={"code": exc.code, "message": exc.message, "field": exc.field})
        except Exception as exc:  # noqa: BLE001 the job's own failure never takes the server down
            finish(status="failed", error={"code": "import_failed", "message": type(exc).__name__, "field": None})

    @app.post(base + "/dataset-imports", status_code=202)
    def import_dataset_route(data: DatasetImport, user=Depends(human_only)):
        with import_lock:
            if sum(1 for r in imports.values() if r["status"] in ("queued", "running")) >= 2:
                fail("busy", "Two dataset imports are already running", 429)
            record = {"import_id": "dsi_" + uuid.uuid4().hex[:16], "status": "queued", "dataset": None, "error": None}
            imports[record["import_id"]] = record
            while len(imports) > ATTACH_KEEP:
                imports.popitem(last=False)
        threading.Thread(target=run_import, args=(record, data), daemon=True, name="dataset-import").start()
        return dict(record)

    @app.get(base + "/dataset-imports/{import_id}")
    def import_status(import_id: str, user=Depends(human_only)):
        with import_lock:
            record = imports.get(import_id)
            if record is None:
                fail("not_found", "Unknown dataset import", 404)
            return dict(record)

    def visible(user, record):
        return record if user == "human" else datasets.agent_view(record)

    @app.get(base + "/datasets")
    def list_datasets(user=Depends(needs("model:read"))):
        records = [datasets.mnist_record(ctx.manifest), *datasets.list_records(workspace)]
        return {"datasets": [visible(user, r) for r in records]}

    @app.get(base + "/datasets/{dataset_id}")
    def get_dataset(dataset_id: str, user=Depends(needs("model:read"))):
        if dataset_id == datasets.MNIST_ID:
            return visible(user, datasets.mnist_record(ctx.manifest))
        try:
            return visible(user, datasets.open_dataset(workspace, dataset_id).record)
        except datasets.DatasetError:
            fail("not_found", "Unknown dataset", 404)

    # ------------------------------------------------------------ evaluations (v0.5-E4)
    ledger = evalservice.Ledger(ckpts)
    jobs: queue.Queue = queue.Queue()
    started = []

    def worker():
        while True:
            evaluation_id, plan = jobs.get()
            evalservice.execute(workspace, ledger, evaluation_id, plan, ctx.eval_timeout)

    def enqueue(evaluation_id, plan):
        if not started:  # one evaluation at a time (one worker process at a time anyway); a daemon thread, so shutdown never waits for it
            started.append(threading.Thread(target=worker, daemon=True, name="evaluate"))
            started[0].start()
        jobs.put((evaluation_id, plan))

    STATUS = {"not_found": 404, "forbidden": 403}

    def own(user, evaluation_id):
        row = ledger.get(evaluation_id) if isinstance(evaluation_id, str) and evalservice._ID.fullmatch(evaluation_id) else None
        if row is None or (user != "human" and row["actor"] != user):  # a token reads only its own session's evaluations
            fail("not_found", "Unknown evaluation", 404)
        return row

    @app.post(base + "/checkpoints/{checkpoint_id}/evaluations", status_code=202)
    def evaluate(checkpoint_id: str, data: EvaluateRequest, user=Depends(needs("eval:run")), idem=Depends(ctx.key)):
        idem = trace.key_digest(idem) or hashlib.sha256(idem.encode("utf-8")).hexdigest()[:32]
        if data.intervention is not None and not ctx.has(user, "weights:read"):  # ablating a unit and watching the output is a probe by another name
            fail("forbidden", "The weights:read capability is required for an activation edit", 403)
        try:
            plan = evalservice.prepare(workspace, ckpts, gstore, checkpoint_id=checkpoint_id, dataset_id=data.dataset_id, evaluator=data.evaluator,
                                       split=data.split, baseline=data.baseline, baseline_given="baseline" in data.model_fields_set,
                                       limit_samples=data.limit_samples, graph=data.graph, intervention=data.intervention, view_for=view_for)
        except evalservice.EvalError as exc:
            fail(exc.code, exc.message, STATUS.get(exc.code, 422))
        def replay(found):
            if found["request_digest"] != plan.request_digest:
                fail("idempotency_conflict", "This key was used for a different request", 409)
            trace.note(evaluation_id=found["evaluation_id"], replayed=True)
            return {**evalservice.view(found), "replayed": True}

        found = ledger.find(user, idem)
        if found is not None:
            return replay(found)
        row = ledger.create(user, idem, plan)  # the row first: two requests with one key cannot both be charged
        if row is None:
            return replay(ledger.find(user, idem))
        try:
            budget.charge_many([("evals", 1), ("eval_samples", evalservice.cost(plan))])  # before any work, both or neither
        except BaseException:
            ledger.delete(row["evaluation_id"])  # a refused evaluation leaves no row, so its key is free again
            raise
        trace.note(evaluation_id=row["evaluation_id"], dataset_id=plan.dataset["dataset_id"], fingerprint=plan.fingerprint, checkpoint_id=plan.checkpoint_id,
                   split=plan.split, split_sha256=plan.dataset["split_sha256"][plan.split],
                   baseline_id=plan.baseline_id, baseline_cached=plan.baseline_cached, eval_samples=evalservice.cost(plan), replayed=False,
                   intervention_id=None if plan.intervention is None else evalservice.intervention_id(plan.intervention),  # the edit itself stays in the row
                   units=None if plan.intervention is None else len(plan.intervention["units"]))
        enqueue(row["evaluation_id"], plan)
        return {**evalservice.view(row), "replayed": False}

    @app.post(base + "/checkpoints/{checkpoint_id}/final-evaluations", status_code=202)
    def final_evaluate(checkpoint_id: str, data: FinalEvaluateRequest, user=Depends(human_only), idem=Depends(ctx.key)):
        """The one road to holdout: a person, the dataset, and the frozen recipe hash of the child they confirmed (never charged, never an agent)."""
        idem = trace.key_digest(idem) or hashlib.sha256(idem.encode("utf-8")).hexdigest()[:32]
        entry = ckpts.get(checkpoint_id)
        if entry is None:
            fail("not_found", "Unknown checkpoint", 404)
        if data.confirmed is not True:
            fail("confirmation_required", "Confirm the frozen recipe hash before holdout", 422)
        if data.recipe_hash != entry["recipe_hash"]:  # a root (imported) checkpoint has none: confirm it with null, never with a made-up hash
            fail("dependency_mismatch", "The confirmed recipe hash is not the one this checkpoint was made by", 409)
        try:
            plan = evalservice.prepare(workspace, ckpts, gstore, checkpoint_id=checkpoint_id, dataset_id=data.dataset_id, evaluator=data.evaluator,
                                       split="holdout", baseline=data.baseline, baseline_given="baseline" in data.model_fields_set,
                                       limit_samples=None, graph=data.graph, final=True, recipe_hash=data.recipe_hash)
        except evalservice.EvalError as exc:
            fail(exc.code, exc.message, STATUS.get(exc.code, 422))
        found = ledger.find(user, idem)
        row = None if found is not None else ledger.create(user, idem, plan)
        if row is None:
            found = found or ledger.find(user, idem)
            if found["request_digest"] != plan.request_digest:
                fail("idempotency_conflict", "This key was used for a different request", 409)
            trace.note(evaluation_id=found["evaluation_id"], replayed=True)
            return {**evalservice.view(found), "replayed": True}
        trace.note(evaluation_id=row["evaluation_id"], dataset_id=plan.dataset["dataset_id"], fingerprint=plan.fingerprint, checkpoint_id=plan.checkpoint_id,
                   split="holdout", split_sha256=plan.dataset["split_sha256"]["holdout"],
                   baseline_id=plan.baseline_id, baseline_cached=plan.baseline_cached, replayed=False)
        enqueue(row["evaluation_id"], plan)
        return {**evalservice.view(row), "replayed": False}

    # ------------------------------------------------------------ probes (v0.5-E5): activations derive from weights, so weights:read and the K rule
    probe_ledger = probeservice.ProbeLedger(ckpts)
    probe_jobs: queue.Queue = queue.Queue()
    probe_started = []

    def probe_worker():
        while True:
            probe_id, plan = probe_jobs.get()
            probeservice.execute(workspace, probe_ledger, probe_id, plan, ctx.eval_timeout)

    @app.post(base + "/checkpoints/{checkpoint_id}/probes", status_code=202)
    def probe(checkpoint_id: str, data: ProbeRequest, user=Depends(needs("eval:run", "weights:read")), idem=Depends(ctx.key)):
        idem = trace.key_digest(idem) or hashlib.sha256(idem.encode("utf-8")).hexdigest()[:32]
        if data.values and not ctx.has(user, "weights:values"):  # raw activations leave the machine only with weights:values (class 2)
            fail("forbidden", "The weights:values capability is required", 403)
        try:
            plan = probeservice.prepare(workspace, ckpts, gstore, checkpoint_id=checkpoint_id, dataset_id=data.dataset_id, nodes=data.nodes,
                                        unit_axis=data.unit_axis, split_by=data.split_by, bins=data.bins, values=data.values,
                                        limit_samples=data.limit_samples, min_aggregate=budget.min_aggregate(), graph=data.graph, group=data.group,
                                        view_for=view_for)
        except probeservice.ProbeError as exc:
            fail(exc.code, exc.message, STATUS.get(exc.code, 422))

        def replay(found):
            if found["request_digest"] != plan.request_digest:
                fail("idempotency_conflict", "This key was used for a different request", 409)
            trace.note(probe_id=found["probe_id"], replayed=True)
            return {**probeservice.view(found, values=ctx.has(user, "weights:values")), "replayed": True}

        found = probe_ledger.find(user, idem)
        if found is not None:
            return replay(found)
        row = probe_ledger.create(user, idem, plan)
        if row is None:
            return replay(probe_ledger.find(user, idem))
        try:
            budget.charge_many([("evals", 1), ("eval_samples", len(plan.rows))])
        except BaseException:
            probe_ledger.delete(row["probe_id"])
            raise
        trace.note(probe_id=row["probe_id"], dataset_id=plan.dataset["dataset_id"], checkpoint_id=plan.checkpoint_id, eval_samples=len(plan.rows),
                   split="exploration", split_sha256=plan.dataset["split_sha256"]["exploration"],
                   ops=len(plan.nodes), replayed=False)
        if not probe_started:
            probe_started.append(threading.Thread(target=probe_worker, daemon=True, name="probe"))
            probe_started[0].start()
        probe_jobs.put((row["probe_id"], plan))
        return {**probeservice.view(row, values=ctx.has(user, "weights:values")), "replayed": False}

    @app.get(base + "/probes/{probe_id}")
    def get_probe(probe_id: str, user=Depends(needs("weights:read"))):
        row = probe_ledger.get(probe_id) if isinstance(probe_id, str) and probeservice._ID.fullmatch(probe_id) else None
        if row is None or (user != "human" and row["actor"] != user):  # a token reads only its own session's probes
            fail("not_found", "Unknown probe", 404)
        trace.note(probe_id=row["probe_id"], dataset_id=row["dataset_id"])
        return probeservice.view(row, values=ctx.has(user, "weights:values"))

    @app.get(base + "/evaluations/{evaluation_id}")
    def get_evaluation(evaluation_id: str, user=Depends(needs("model:read"))):
        row = own(user, evaluation_id)
        trace.note(evaluation_id=row["evaluation_id"], dataset_id=row["dataset_id"], fingerprint=row["fingerprint"])
        return evalservice.view(row)

    @app.get(base + "/evaluations/{evaluation_id}/changes")
    def evaluation_changes(evaluation_id: str, user=Depends(needs("model:read")), category: str | None = None, sort: str = "sample_id",
                           limit: int = 50, cursor: str | None = None):
        row = own(user, evaluation_id)
        if row["status"] != "succeeded":
            fail("evaluation_not_ready", f"The evaluation is {row['status']}", 409)
        if category not in (None, "fixed", "broken", "changed") or sort not in ("sample_id", "score_change") or not 1 <= limit <= 500:
            fail("invalid_argument", "category is fixed, broken or changed; sort is sample_id or score_change; limit is 1 to 500", 422)
        try:
            offset = int(unsign(cursor)[3:]) if cursor else 0
        except ValueError:
            fail("invalid_cursor", "cursor was not issued for this session", 422)
        page = evalservice.changes(workspace, row, datasets.open_dataset(workspace, row["dataset_id"]), category=category, sort=sort, limit=limit,
                                   offset=offset)
        trace.note(evaluation_id=row["evaluation_id"], offset=offset, page_limit=limit)
        page["cursor"] = sign(f"c1.{offset + limit}") if page["truncated"] else None
        return page

    # A person looks at a sample (E9). The file is the one the manifest names, re-hashed; its Content-Type comes from the manifest, never
    # from the bytes; the server never decodes it. Holdout samples are visible to a person (the gate is on agents), and to no token.
    def mnist_sample(sample_id):
        """A row of the v0.1 reference manifest (`mnist/t10k/00012/blurred`); the manifest is the only thing a sample id is looked up in."""
        row = next((r for r in ctx.manifest.get("samples", []) if r["sample_id"] == sample_id), None) if isinstance(sample_id, str) else None
        if row is None:
            fail("not_found", "Unknown sample", 404)
        return row

    def sample_row(dataset_id, sample_id):
        if not isinstance(sample_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", sample_id):
            fail("not_found", "Unknown sample", 404)
        try:
            data = datasets.open_dataset(workspace, dataset_id)
        except datasets.DatasetError:
            fail("not_found", "Unknown dataset", 404)
        row = next((r for r in data.manifest["samples"] if r["sample_id"] == sample_id), None)
        if row is None:
            fail("not_found", "Unknown sample", 404)
        return data, row

    def stored(data, sha, kind):
        if kind not in datasets.CONTENT_TYPES:  # a dataset imported before the manifest recorded file types: its type is not guessed
            fail("dataset_changed", "This dataset was imported by an older version and has no recorded file type: import it again", 409)
        try:
            body = data.path(sha).read_bytes()
        except OSError:
            body = b""
        if hashlib.sha256(body).hexdigest() != sha:  # a file that changed on disk is never served
            fail("dataset_changed", "The stored file does not match the manifest", 409)
        return Response(body, media_type=datasets.CONTENT_TYPES[kind], headers={
            "X-Content-Type-Options": "nosniff", "Content-Disposition": f'inline; filename="sample.{kind}"', "Cache-Control": "no-store"})

    @app.get(base + "/datasets/{dataset_id}/samples/{sample_id:path}/image")
    def sample_image(dataset_id: str, sample_id: str, user=Depends(human_only)):
        if dataset_id == datasets.MNIST_ID:  # made from the verified idx archive (the blurred variant is the v0.1 box blur), encoded here, never decoded
            row = mnist_sample(sample_id)
            body = datasets.mnist_png(workspace / "mnist", row)
            return Response(body, media_type="image/png", headers={"X-Content-Type-Options": "nosniff", "Content-Disposition": 'inline; filename="sample.png"',
                                                                   "Cache-Control": "no-store"})
        data, row = sample_row(dataset_id, sample_id)
        return stored(data, row["image"], row.get("image_type"))

    @app.get(base + "/datasets/{dataset_id}/samples/{sample_id:path}/label")
    def sample_label(dataset_id: str, sample_id: str, user=Depends(human_only)):
        if dataset_id == datasets.MNIST_ID:
            row = mnist_sample(sample_id)
            return JSONResponse({"label": str(row["label"]), "label_index": row["label"]}, headers={"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"})
        data, row = sample_row(dataset_id, sample_id)
        task = data.record["task"]
        if task == "segmentation":
            return stored(data, row["mask"], row.get("mask_type"))
        headers = {"X-Content-Type-Options": "nosniff", "Cache-Control": "no-store"}
        if task == "ocr":  # text/plain: the person's UI renders it as text, never as markup
            return Response(row["text"], media_type="text/plain; charset=utf-8", headers=headers)
        if task == "detection":
            return JSONResponse({"boxes": row["boxes"], "categories": data.record["categories"], "width": row["width"], "height": row["height"]}, headers=headers)
        return JSONResponse({"label": row["label"], "label_index": data.record["classes"].index(row["label"])}, headers=headers)

    def remember_edit(cid, recipe, ops, applied=False):
        """The recipe (with its stated hypothesis) and the selector of each op, for the supervision views. Humans read them; the trace
        itself keeps ids only."""
        trace.remember_recipe(ckpts, recipe, applied=applied)  # an applied recipe is always kept: the apply quota bounds them
        for entry, op in zip(recipe["program"], ops):
            if isinstance(entry.get("select"), dict):
                trace.remember_selection(ckpts, cid, op["selection_id"], entry["select"])

    def diff_hint(parent, child, recipe):
        """What to copy into `diff` to check an apply: the pair, and the resolved selector of every op (deduplicated, in order).
        `diff` takes one selector per call; with none it compares the whole table."""
        selectors, seen = [], set()
        for entry in recipe["program"]:
            key = digest(entry["select"])
            if key not in seen:
                seen.add(key)
                selectors.append(entry["select"])
        return {"checkpoint_id": parent, "other": child, "selectors": selectors[:DIFF_HINT_MAX],
                "selectors_truncated": len(selectors) > DIFF_HINT_MAX}

    # ------------------------------------------------------------ edit programs
    def build_recipe(table, cid, data):
        if (data.recipe is None) == (data.program is None):
            raise InspectError("invalid_argument", "give exactly one of recipe or program (with hypothesis)",
                               "recipe" if data.recipe is not None else "program")
        if data.recipe is not None:
            return edit.verify_recipe(data.recipe)
        if data.hypothesis is None:
            raise InspectError("invalid_argument", "hypothesis is required with program", "hypothesis")
        grouped = any(isinstance(e, dict) and group_ops.is_group(e.get("select")) for e in data.program) if isinstance(data.program, list) else False
        return edit.freeze_recipe(table, cid, data.program, data.hypothesis, evaluation=data.evaluation,
                                  graph=view_for(table) if grouped else None)

    def plan_graph(table, recipe):
        """A graph only to REPORT on the groups a recipe was expanded from; the plan itself never needs it."""
        return view_for(table) if any("via" in e for e in recipe["program"]) else None

    def gather(recipe):
        """Tables and blob bytes the recipe names; anything unknown is left out so the library reports it."""
        sources, blobs = {}, {}
        for step in recipe["program"]:
            args = step["args"]
            source = args.get("source")
            if source is not None and source not in sources and (found := ckpts.location(source)) is not None:
                sources[source] = open_path(found[0])
            blob = args.get("blob")
            if blob is not None and blob not in blobs and (path := workspace / "blobs" / blob[7:]).is_file():
                blobs[blob] = path.read_bytes()
        return sources, blobs

    def envelope(cid, recipe_hash, resolved, result, side_effects=(), revision=edit.OPS_REVISION):
        return {"schema_version": 1, "resolved": resolved, "result": result,
                "fingerprints": {"checkpoint": cid, "recipe": recipe_hash, "selection": None, "op_revision": revision},
                "side_effects": list(side_effects), "truncated": False, "cursor": None}

    @app.post(base + "/checkpoints/{checkpoint_id}/plan")
    def plan(checkpoint_id: str, data: EditRequest, user=Depends(needs_edit)):
        cid, table = open_checkpoint(checkpoint_id)
        recipe = build_recipe(table, cid, data)
        sources, blobs = gather(recipe)
        result = edit.plan(table, recipe, base_id=cid, sources=sources, blobs=blobs, graph=plan_graph(table, recipe))
        result["result"]["recipe"] = recipe
        trace.note(recipe_hash=recipe["recipe_hash"], ops=len(recipe["program"]),
                   selections=[o["selection_id"] for o in result["result"]["ops"]])
        remember_edit(cid, recipe, result["result"]["ops"])
        return structure.view("plan", result) if blind(user) else result

    @app.post(base + "/checkpoints/{checkpoint_id}/validate")
    def validate(checkpoint_id: str, data: EditRequest, user=Depends(needs_edit)):
        cid, table = open_checkpoint(checkpoint_id)
        if (data.recipe is None) == (data.program is None):
            raise InspectError("invalid_argument", "give exactly one of recipe or program (with hypothesis)", "program")
        try:
            recipe = build_recipe(table, cid, data)
        except InspectError as exc:  # a bad program is a validation result, not a request error
            return envelope(cid, None, {"base_checkpoint": cid},
                            {"valid": False, "checks": list(edit.CHECKS), "error": exc.to_dict(), "recipe": None})
        sources, blobs = gather(recipe)
        trace.note(recipe_hash=recipe["recipe_hash"], ops=len(recipe["program"]))
        report = edit.validate(table, recipe, base_id=cid, sources=sources, blobs=blobs, graph=plan_graph(table, recipe))
        done = envelope(cid, recipe["recipe_hash"], {"recipe_hash": recipe["recipe_hash"], "base_checkpoint": cid,
                                                     "ops": len(recipe["program"])}, {**report, "recipe": recipe},
                        revision=edit.revision_of(recipe))
        return structure.view("validate", done) if blind(user) else done

    def begin(user, key, request_digest, cid, recipe_hash, capabilities, charge):
        for _ in range(3):
            try:
                state, stored = ckpts.begin_edit(user, key, request_digest, cid, recipe_hash, capabilities, charge=charge,
                                                 quota=ctx.store.quota)
            except ImportFailure as exc:
                fail(exc.code, exc.message, 429 if exc.code == "quota_exhausted" else 409)
            if state != "running":
                return state, stored
            deadline = time.monotonic() + APPLY_WAIT_SECONDS  # a concurrent request with the same key is applying
            while time.monotonic() < deadline:
                time.sleep(.05)
                current = ckpts.get_edit(user, key)
                if current is None:
                    break  # it failed and released the key: take over
                if current[0] == "succeeded":
                    return "replay", current[1]
            else:
                fail("in_progress", "The same Idempotency-Key is still being applied", 409)
        fail("in_progress", "The same Idempotency-Key is still being applied", 409)

    @app.post(base + "/checkpoints/{checkpoint_id}/apply")
    def apply(checkpoint_id: str, data: ApplyRequest, user=Depends(needs_write), idem=Depends(ctx.key)):
        idem = trace.key_digest(idem) or hashlib.sha256(idem.encode("utf-8")).hexdigest()[:32]  # the ledger, like the trace, keeps only a digest of the client's key
        envelope_ = run_apply(user, checkpoint_id, data, idem, gate=True)
        return structure.view("apply", envelope_) if blind(user) and isinstance(envelope_, dict) and "manifest" in envelope_["result"] else envelope_

    def propose(user, idem, request_digest, cid, recipe, ops, data):
        """The approval gate: an agent's apply is recorded as a proposal and writes nothing. Keyed, so a retry is the same proposal."""
        found = ckpts.find_proposal(user, idem)
        if found is not None and found["request_digest"] != request_digest:
            fail("idempotency_conflict", "This key was used for a different request", 409)
        if found is not None and found["status"] == "rejected":
            fail("proposal_rejected", "A person rejected this proposal", 409)
        new = found is None
        if new:
            if ctx.store.quota_info(user)["remaining"] <= 0:  # it could never be approved, and a person's queue is the scarce thing
                fail("quota_exhausted", "The apply quota is used up: a proposal could not be approved", 429)
            if len(ckpts.list_proposals(user, "pending")) >= PROPOSALS_PENDING_MAX:
                fail("too_many_proposals", f"{PROPOSALS_PENDING_MAX} proposals are already waiting for a person", 429)
            # the FROZEN recipe is what a person reviews and what approval runs: nothing is re-frozen against a later graph
            found = ckpts.create_proposal(user, idem, request_digest, cid, recipe["recipe_hash"],
                                          [o["selection_id"] for o in ops], {"recipe": recipe, "allow_noop": data.allow_noop})
        budget.charge_deferred()  # it made no child, but the planning reads were reads
        trace.note(recipe_hash=recipe["recipe_hash"], ops=len(recipe["program"]), selections=[o["selection_id"] for o in ops],
                   proposal_id=found["proposal_id"])
        remember_edit(cid, recipe, ops)
        body = envelope(cid, recipe["recipe_hash"], {"base_checkpoint": cid},
                        {"proposal_id": found["proposal_id"], "status": found["status"], "recipe_hash": recipe["recipe_hash"]},
                        [{"kind": "proposal_created", "id": found["proposal_id"]}] if new else [])
        return JSONResponse(body, status_code=202)

    def run_apply(user, checkpoint_id, data, idem, *, gate):
        if user != "human" and (ctx.supervision.allow_noop or blind(user)) and not data.allow_noop:
            data = data.model_copy(update={"allow_noop": True})  # an agent is never told "no_op": that refusal is an oracle on zeros
        cid, table = open_checkpoint(checkpoint_id)
        budget.defer_scan()  # an apply costs one `applies`; its planning reads count against `scan` only if it makes no child
        recipe = build_recipe(table, cid, data)
        report_graph = plan_graph(table, recipe)
        request_digest = digest([cid, recipe["recipe_hash"], data.allow_noop])
        prior = ckpts.get_edit(user, idem)  # a replay or a conflict is answered before anything is re-planned
        if prior is not None:
            if prior[2] != request_digest:
                fail("idempotency_conflict", "This key was used for a different request", 409)
            if prior[0] == "succeeded":
                trace.note(recipe_hash=recipe["recipe_hash"], replayed=True, ops=len(recipe["program"]),
                           child_id=prior[1]["result"]["manifest"]["child_sha256"],
                           selections=[o["selection_id"] for o in prior[1]["result"]["ops"]])
                return {**prior[1], "result": {**prior[1]["result"], "replayed": True}, "side_effects": []}
        sources, blobs = gather(recipe)
        planned = edit.plan(table, recipe, base_id=cid, sources=sources, blobs=blobs, graph=report_graph)  # invalid programs cost nothing
        if planned["result"]["noop"] and not data.allow_noop:
            raise InspectError("no_op", "the program changes no stored value (e.g. scale by 1)", "program")
        if gate and ctx.supervision.approval and user != "human":
            return propose(user, idem, request_digest, cid, recipe, planned["result"]["ops"], data)
        budget.free_scan()  # a child will be written: copying the file is part of the one `applies` charge
        charge = user != "human"
        capabilities = sorted(ctx.creds.capabilities) if charge else HUMAN_CAPABILITIES
        state, stored = begin(user, idem, request_digest, cid, recipe["recipe_hash"], capabilities, charge)
        if state == "replay":
            trace.note(recipe_hash=recipe["recipe_hash"], replayed=True, ops=len(recipe["program"]),
                       child_id=stored["result"]["manifest"]["child_sha256"],
                       selections=[o["selection_id"] for o in stored["result"]["ops"]])
            return {**stored, "result": {**stored["result"], "replayed": True}, "side_effects": []}
        staging = workspace / "staging" / f"apply-{uuid.uuid4().hex}.safetensors"
        try:
            envelope_ = edit.apply(table, recipe, staging, base_id=cid, sources=sources, blobs=blobs,
                                   allow_noop=data.allow_noop, graph=report_graph)
            child = envelope_["result"]["manifest"]["child_sha256"]
            final = workspace / "checkpoints" / (child + ".safetensors")
            known = ckpts.get(child)
            created = False
            if known is None:  # identical bytes already in the catalog (same recipe, same result, a no-op) place nothing
                created = place_new(staging, final)
                if final.exists():
                    parent = ckpts.get(cid)
                    register_child(ckpts, final, child, f"{parent['display_name']} edit {recipe['recipe_hash'][:8]}",
                                   cid, recipe["recipe_hash"])
                    known = ckpts.get(child)
            staging.unlink(missing_ok=True)
            # An existing entry carries ITS lineage; only show it as ours when it really is.
            ours = known is not None and known["parent_id"] == cid and known["recipe_hash"] == recipe["recipe_hash"]
            effects = [({"kind": "checkpoint_created", "id": child, "bytes": final.stat().st_size} if created
                        else {"kind": "checkpoint_existing", "id": child})]
            if charge:
                effects.append({"kind": "quota_charged", "runs": 1})
            envelope_["side_effects"] = effects
            envelope_["result"].update(checkpoint=known if ours else {"checkpoint_id": child}, recipe=recipe,
                                       replayed=False, existing=not created, diff_hint=diff_hint(cid, child, recipe))
            remember_edit(cid, recipe, envelope_["result"]["ops"], applied=True)
            ckpts.finish_edit(user, idem, child, envelope_)
            trace.note(recipe_hash=recipe["recipe_hash"], child_id=child, replayed=False, existing=not created,
                       charged=int(charge), ops=len(recipe["program"]),
                       selections=[o["selection_id"] for o in envelope_["result"]["ops"]])
        except BaseException:
            staging.unlink(missing_ok=True)
            ckpts.abort_edit(user, idem)  # not charged, key reusable
            raise
        return envelope_

    # ------------------------------------------------------------ trace
    @app.get(base + "/trace")
    def session_trace(user=Depends(needs("model:read")), session: str | None = None, after: int = 0, limit: int = 100):
        """The event timeline. An agent sees only its own session; the human session sees all (or one)."""
        if user != "human" and session not in (None, user):
            fail("forbidden", "An agent may only read its own session", 403)
        if not 1 <= limit <= 500 or after < 0:
            fail("invalid_limit", "limit must be 1..500 and after non-negative", 422)
        target = user if user != "human" else session
        live = None
        if target == ctx.creds.principal and ctx.budgets is not None:  # only the running session's limits are known
            live = {**ctx.budgets.limits, "applies": ctx.store.quota}
        data = trace.timeline(ckpts, target, after=after, limit=limit, limits=live)
        events = [{("event_id" if k == "id" else k): v for k, v in e.items()} for e in data["events"]]
        return {"schema_version": 1, "resolved": {"session": target, "after": after, "limit": limit},
                "result": {"session": target, "events": events, "summary": data["summary"], "next_after": data["next_after"]},
                "fingerprints": {"checkpoint": None, "selection": None, "op_revision": 1},
                "side_effects": [], "truncated": data["next_after"] is not None, "cursor": None}

    # ------------------------------------------------------------ supervision controls (human only)
    def agent_session(session):
        if session == "human":
            fail("invalid_argument", "only an agent session can be supervised", 422)
        if session != ctx.creds.principal:
            fail("not_found", "Unknown or finished agent session: only the running one can be changed", 404)
        return session

    @app.post(base + "/agent-sessions/{session}/revoke")
    def revoke(session: str, user=Depends(human_only)):
        """The kill switch: every later request of the session answers 401 session_revoked. Children it made stay."""
        agent_session(session)
        if ctx.supervision.revoked:
            fail("already_revoked", "This session is already revoked", 409)
        ctx.supervision.revoked = True
        ckpts.set_revoked(session)
        trace.record_control(ckpts, session, "revoke")
        return {"schema_version": 1, "result": {"session": session, "revoked": True}}

    @app.post(base + "/agent-sessions/{session}/budget")
    def change_budget(session: str, data: BudgetChange, user=Depends(human_only)):
        """Set one limit of the running agent session (null removes it; applies must stay a number). Lowering below use is allowed."""
        agent_session(session)
        if data.dimension == "applies":
            if data.limit is None:
                fail("invalid_argument", "the apply quota needs a number", 422)
            before, ctx.store.quota = ctx.store.quota, data.limit
        else:
            before = ctx.budgets.set_limit(data.dimension, data.limit)
        trace.record_control(ckpts, session, "budget_change", dimension=data.dimension, **{"from": before, "to": data.limit})
        return {"schema_version": 1, "result": {"session": session, "dimension": data.dimension, "from": before, "to": data.limit}}

    def public_proposal(p):
        return {"proposal_id": p["proposal_id"], "session": p["session"], "seq": trace.proposal_seq(ckpts, p["session"], p["proposal_id"]),
                "checkpoint_id": p["checkpoint_id"], "recipe_hash": p["recipe_hash"], "selections": p["selections"],
                "status": p["status"], "created_at": p["created_at"], "decided_at": p["decided_at"], "child_id": p["child_id"]}

    @app.get(base + "/proposals")
    def proposals(user=Depends(human_only), session: str | None = None, status: str | None = None):
        if status not in (None, "pending", "approved", "rejected"):
            fail("invalid_argument", "status is pending, approved or rejected", 422)
        return {"schema_version": 1, "result": {"proposals": [public_proposal(p) for p in ckpts.list_proposals(session, status)],
                                                "approval_gate": ctx.supervision.approval}}

    def pending(proposal_id):
        p = ckpts.get_proposal(proposal_id)
        if p is None:
            fail("not_found", "Unknown proposal", 404)
        if p["status"] != "pending":
            fail("not_pending", f"This proposal is already {p['status']}", 409)
        return p

    @app.post(base + "/proposals/{proposal_id}/approve")
    def approve(proposal_id: str, user=Depends(human_only)):
        """Run the agent's apply, exactly as the agent would have (same recipe, same key, the agent's session and quota)."""
        p = pending(proposal_id)
        if p["session"] != ctx.creds.principal:
            fail("session_gone", "The agent session that proposed this no longer exists", 409)
        stored = p["request"]
        if stored["recipe"]["recipe_hash"] != p["recipe_hash"]:  # guard before anything is written
            fail("proposal_stale", "The stored recipe does not match the proposal", 409)
        envelope_ = run_apply(p["session"], p["checkpoint_id"], ApplyRequest(recipe=stored["recipe"], allow_noop=stored["allow_noop"]),
                              p["idem_digest"], gate=False)  # verify_recipe makes it exactly the reviewed recipe
        child = envelope_["result"]["manifest"]["child_sha256"]
        # The quota was spent on the AGENT's session: its trace totals must say so (and the human's own request row must not claim it).
        charged = any(e.get("kind") == "quota_charged" for e in envelope_["side_effects"])
        trace.note(charged=0)
        if ckpts.decide_proposal(proposal_id, "approved", child):  # a concurrent second approve replays the ledger and records nothing
            trace.record_control(ckpts, p["session"], "approve", charged=charged, proposal_id=proposal_id, child_id=child)
        return {**envelope_, "result": {**envelope_["result"], "proposal": {"proposal_id": proposal_id, "status": "approved"}}}

    @app.post(base + "/proposals/{proposal_id}/reject")
    def reject(proposal_id: str, user=Depends(human_only)):
        p = pending(proposal_id)
        ckpts.decide_proposal(proposal_id, "rejected")
        trace.record_control(ckpts, p["session"], "reject", proposal_id=proposal_id)
        return {"schema_version": 1, "result": {"proposal_id": proposal_id, "status": "rejected"}}

    # ------------------------------------------------------------ supervision reads (human only)
    @app.get(base + "/trace/sessions")
    def trace_sessions(user=Depends(human_only)):
        """Every session the catalog knows, with the counts the supervision list needs."""
        rows = trace.sessions(ckpts)
        return {"schema_version": 1, "result": {"sessions": rows, "current_agent_session": ctx.creds.principal,
                                                "remember_full": trace.remember_full(ckpts)}}

    @app.post(base + "/selections/resolve")
    def resolve_selections(data: ResolveSelections, user=Depends(human_only)):
        """Selection ids from the trace to the canonical selectors the agent used, in one call."""
        found = trace.resolve_selections(ckpts, data.ids)
        return {"schema_version": 1, "result": {"selections": found, "unknown": [i for i in data.ids if i not in found]}}

    @app.post(base + "/interventions/resolve")
    def resolve_interventions(data: ResolveInterventions, user=Depends(human_only)):
        """Intervention ids from the trace to the activation edits an agent asked for (node, unit axis, units, op, factor): a person's view."""
        known = evalservice.Ledger(ckpts).interventions()
        found = {i: known[i] for i in data.ids if i in known}
        return {"schema_version": 1, "result": {"interventions": found, "unknown": [i for i in data.ids if i not in found]}}

    @app.get(base + "/recipes/by-hash/{recipe_hash}")
    def recipe_by_hash(recipe_hash: str, user=Depends(human_only)):
        recipe = trace.get_recipe(ckpts, recipe_hash)
        if recipe is None:
            fail("not_found", "No recipe with that hash was planned or applied in this workspace", 404)
        return {"schema_version": 1, "result": {"recipe_hash": recipe_hash, "recipe": recipe,
                                                "untrusted_text": ["recipe.hypothesis"]}}

    # ------------------------------------------------------------ blobs
    @app.post(base + "/blobs", status_code=201)
    async def upload_blob(request: Request, user=Depends(needs_write)):
        try:
            declared = int(request.headers.get("content-length", ""))
        except ValueError:
            declared = 0
        if declared <= 0:
            fail("length_required", "Content-Length is required", 411)
        if declared > ctx.blob_max:
            fail("payload_too_large", f"A blob may be at most {ctx.blob_max} bytes", 413)
        budget.charge("blob_bytes", declared)
        if declared % 4:
            fail("blob_size", "A blob is a flat little-endian float32 array: its length must be a multiple of 4", 422)
        directory = workspace / "blobs"
        directory.mkdir(exist_ok=True)
        if sum(p.stat().st_size for p in directory.iterdir() if p.is_file()) + declared > BLOB_TOTAL_MAX:
            fail("blob_quota", f"The workspace blob store is limited to {BLOB_TOTAL_MAX} bytes", 413)
        staging = workspace / "staging" / f"blob-{uuid.uuid4().hex}.part"
        hasher, received = hashlib.sha256(), 0
        try:
            def sink(chunk):
                hasher.update(chunk)
                out.write(chunk)

            with staging.open("wb") as out:
                async for chunk in request.stream():
                    received += len(chunk)
                    if received > declared:
                        fail("payload_too_large", "Body is longer than its Content-Length", 413)
                    await run_in_threadpool(sink, chunk)
            if received != declared:
                fail("length_mismatch", "Body is shorter than its Content-Length", 400)
            hex_id = hasher.hexdigest()
            target = directory / hex_id
            existing = target.exists()
            if existing:
                staging.unlink(missing_ok=True)
            else:
                os.replace(staging, target)
        except BaseException:
            staging.unlink(missing_ok=True)
            raise
        blob_id = "sha256:" + hex_id
        trace.note(blob_id=blob_id)
        return {"schema_version": 1, "resolved": {"bytes": declared},
                "result": {"blob_id": blob_id, "bytes": declared, "elems": declared // 4, "existing": existing},
                "fingerprints": {"checkpoint": None, "selection": None, "op_revision": edit.OPS_REVISION},
                "side_effects": [] if existing else [{"kind": "blob_stored", "id": blob_id, "bytes": declared}],
                "truncated": False, "cursor": None}
