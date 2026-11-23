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
import secrets
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, Request
from starlette.concurrency import run_in_threadpool

from . import budget, coupling, edit, graph_ops, group_ops, inspect_ops, onnx_provider, providers, trace
from .checkpoints import ImportFailure, register_child
from .graph import GraphMismatch, GraphStore, table_signature
from .export_provider import TorchExportProvider
from .graphview import GraphView, provider_list
from .registry import HUMAN_CAPABILITIES
from .schemas import (ApplyRequest, AttachGraph, DiffRequest, EditRequest, QueryRequest, ReadTensorRequest, ResolveSelections,
                      StatsRequest)
from .selector import InspectError, digest
from .tensors import TensorFileError, open_tensors

BLOB_TOTAL_MAX = 1 << 30
ATTACH_KEEP = 200       # finished attach jobs remembered
DIFF_HINT_MAX = 20      # selectors an apply result echoes for its diff
DERIVED_KEEP = 16       # derivations (roles and groups) kept per process
APPLY_WAIT_SECONDS = 60


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
    quota_limit: int | None = None  # the apply quota, reported as the `applies` budget


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
        return finish(inspect_ops.describe(table, name_glob=name_glob, limit=limit, cursor=unsign(cursor), checkpoint_id=cid,
                                           graph=view_for(table)))

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
    def diff(checkpoint_id: str, data: DiffRequest, user=Depends(needs("weights:read"))):
        if data.values and not ctx.has(user, "weights:values"):  # element-wise deltas are values; summaries are aggregates
            fail("forbidden", "The weights:values capability is required", 403)
        cid, table = open_checkpoint(checkpoint_id)
        other_id, other = open_checkpoint(data.other)
        return finish(inspect_ops.diff(
            table, other, selector=data.selector, name_glob=data.name_glob, unit_axis=data.unit_axis,
            values=data.values, include_same=data.include_same, limit=data.limit, max_elems=data.max_elems,
            unit_axis_implicit="unit_axis" not in data.model_fields_set,
            include_range=data.include_range, cursor=unsign(data.cursor), checkpoint_a=cid, checkpoint_b=other_id,
            graph=view_if(table, data.selector)))

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
        try:
            table = open_path(path)
            with attach_lock:
                record["status"] = "running"
            stage("building", 0)
            if "reuse" in spec:
                ir, provider = gstore.load(spec["reuse"]), gstore.provider_of(spec["reuse"])
            elif spec["provider"] == "torch_export":
                ir, provider = exporter.build(spec), "torch_export"
            elif spec["provider"] == "onnx":
                ir, provider = onnx_provider.build(spec, table, timeout=ctx.export_timeout), "onnx"
            else:
                ir, provider = providers.build_graph(spec), spec["provider"]
            stage("binding", 1)
            derived = coupling.derive(ir, table)  # a graph the rules cannot handle is refused here, never stored
            gstore.put(ir, table, provider=provider, checkpoint_id=record["checkpoint_id"])
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
    def plan(checkpoint_id: str, data: EditRequest, user=Depends(needs("weights:read"))):
        cid, table = open_checkpoint(checkpoint_id)
        recipe = build_recipe(table, cid, data)
        sources, blobs = gather(recipe)
        result = edit.plan(table, recipe, base_id=cid, sources=sources, blobs=blobs, graph=plan_graph(table, recipe))
        result["result"]["recipe"] = recipe
        trace.note(recipe_hash=recipe["recipe_hash"], ops=len(recipe["program"]),
                   selections=[o["selection_id"] for o in result["result"]["ops"]])
        remember_edit(cid, recipe, result["result"]["ops"])
        return result

    @app.post(base + "/checkpoints/{checkpoint_id}/validate")
    def validate(checkpoint_id: str, data: EditRequest, user=Depends(needs("weights:read"))):
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
        return envelope(cid, recipe["recipe_hash"], {"recipe_hash": recipe["recipe_hash"], "base_checkpoint": cid,
                                                     "ops": len(recipe["program"])}, {**report, "recipe": recipe},
                        revision=edit.revision_of(recipe))

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
    def apply(checkpoint_id: str, data: ApplyRequest, user=Depends(needs("weights:write", "weights:read")),
              idem=Depends(ctx.key)):
        cid, table = open_checkpoint(checkpoint_id)
        budget.defer_scan()  # an apply costs one `applies`; its planning reads count against `scan` only if it makes no child
        recipe = build_recipe(table, cid, data)
        report_graph = plan_graph(table, recipe)
        idem = trace.key_digest(idem) or hashlib.sha256(idem.encode("utf-8")).hexdigest()[:32]  # the ledger, like the trace, keeps only a digest of the client's key
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
            live = {**ctx.budgets.limits, "applies": ctx.quota_limit}
        data = trace.timeline(ckpts, target, after=after, limit=limit, limits=live)
        events = [{("event_id" if k == "id" else k): v for k, v in e.items()} for e in data["events"]]
        return {"schema_version": 1, "resolved": {"session": target, "after": after, "limit": limit},
                "result": {"session": target, "events": events, "summary": data["summary"], "next_after": data["next_after"]},
                "fingerprints": {"checkpoint": None, "selection": None, "op_revision": 1},
                "side_effects": [], "truncated": data["next_after"] is not None, "cursor": None}

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

    @app.get(base + "/recipes/by-hash/{recipe_hash}")
    def recipe_by_hash(recipe_hash: str, user=Depends(human_only)):
        recipe = trace.get_recipe(ckpts, recipe_hash)
        if recipe is None:
            fail("not_found", "No recipe with that hash was planned or applied in this workspace", 404)
        return {"schema_version": 1, "result": {"recipe_hash": recipe_hash, "recipe": recipe,
                                                "untrusted_text": ["recipe.hypothesis"]}}

    # ------------------------------------------------------------ blobs
    @app.post(base + "/blobs", status_code=201)
    async def upload_blob(request: Request, user=Depends(needs("weights:write", "weights:read"))):
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
