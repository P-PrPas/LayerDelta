"""v2 HTTP routes: tensor-space inspection, edit programs and blobs (v0.2-T4c).

Every route sits under /api/v1/projects/{p}/checkpoints/{checkpoint_id}/... (the id is the T4b
catalog's content-addressed id, resolved through the catalog: the URL text is never used as an
identity) except POST /blobs. Each one wraps one library call and returns its v1 envelope
(schema_version, resolved, result, fingerprints, side_effects, truncated, cursor). Capabilities:
model:read (describe), weights:read (read-tensor, stats, diff, query, plan, validate),
weights:write (apply, blobs). The human session has all of them. Pure routes always return
side_effects []. Cursors are HMAC-signed here, so a client cannot forge or edit one.
"""

import hashlib
import hmac
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import Depends, Request
from starlette.concurrency import run_in_threadpool

from . import edit, inspect_ops
from .checkpoints import ImportFailure, register_child
from .schemas import ApplyRequest, DiffRequest, EditRequest, QueryRequest, ReadTensorRequest, StatsRequest
from .selector import InspectError, digest
from .tensors import TensorFileError, open_tensors

BLOB_TOTAL_MAX = 1 << 30
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


def install_v2(app, ctx: V2Context) -> None:
    base, fail, ckpts, workspace = ctx.base, ctx.fail, ctx.ckpts, ctx.workspace
    secret = b"layerdelta-v2-cursor:" + ctx.creds.csrf.encode()

    def needs(capability):
        def dependency(user=Depends(ctx.actor)):
            if not ctx.has(user, capability):
                fail("forbidden", f"The {capability} capability is required", 403)
            return user
        return dependency

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

    # ------------------------------------------------------------ pure inspection
    @app.get(base + "/checkpoints/{checkpoint_id}/describe")
    def describe(checkpoint_id: str, user=Depends(needs("model:read")), name_glob: str | None = None,
                 limit: int | None = None, cursor: str | None = None):
        cid, table = open_checkpoint(checkpoint_id)
        return finish(inspect_ops.describe(table, name_glob=name_glob, limit=limit, cursor=unsign(cursor), checkpoint_id=cid))

    @app.post(base + "/checkpoints/{checkpoint_id}/read-tensor")
    def read_tensor(checkpoint_id: str, data: ReadTensorRequest, user=Depends(needs("weights:read"))):
        cid, table = open_checkpoint(checkpoint_id)
        return finish(inspect_ops.read_tensor(table, data.selector, max_elems=data.max_elems, cursor=unsign(data.cursor),
                                              include_range=data.include_range, checkpoint_id=cid))

    @app.post(base + "/checkpoints/{checkpoint_id}/stats")
    def stats(checkpoint_id: str, data: StatsRequest, user=Depends(needs("weights:read"))):
        cid, table = open_checkpoint(checkpoint_id)
        return finish(inspect_ops.stats(table, data.selector, reduce_over_axes=data.reduce_over_axes, bins=data.bins,
                                        limit=data.limit, cursor=unsign(data.cursor), checkpoint_id=cid))

    @app.post(base + "/checkpoints/{checkpoint_id}/diff")
    def diff(checkpoint_id: str, data: DiffRequest, user=Depends(needs("weights:read"))):
        cid, table = open_checkpoint(checkpoint_id)
        other_id, other = open_checkpoint(data.other)
        return finish(inspect_ops.diff(
            table, other, selector=data.selector, name_glob=data.name_glob, unit_axis=data.unit_axis,
            values=data.values, include_same=data.include_same, limit=data.limit, max_elems=data.max_elems,
            include_range=data.include_range, cursor=unsign(data.cursor), checkpoint_a=cid, checkpoint_b=other_id))

    @app.post(base + "/checkpoints/{checkpoint_id}/query")
    def query(checkpoint_id: str, data: QueryRequest, user=Depends(needs("weights:read"))):
        cid, table = open_checkpoint(checkpoint_id)
        fields = given(data, "name_glob", "dtype", "min_ndim", "max_ndim", "editable", "min_bytes", "max_bytes", "where",
                       "limit", "max_scan_elems")
        return finish(inspect_ops.query(table, cursor=unsign(data.cursor), checkpoint_id=cid, **fields))

    # ------------------------------------------------------------ edit programs
    def build_recipe(table, cid, data):
        if (data.recipe is None) == (data.program is None):
            raise InspectError("invalid_argument", "give exactly one of recipe or program (with hypothesis)",
                               "recipe" if data.recipe is not None else "program")
        if data.recipe is not None:
            return edit.verify_recipe(data.recipe)
        if data.hypothesis is None:
            raise InspectError("invalid_argument", "hypothesis is required with program", "hypothesis")
        return edit.freeze_recipe(table, cid, data.program, data.hypothesis, evaluation=data.evaluation)

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

    def envelope(cid, recipe_hash, resolved, result, side_effects=()):
        return {"schema_version": 1, "resolved": resolved, "result": result,
                "fingerprints": {"checkpoint": cid, "recipe": recipe_hash, "selection": None, "op_revision": edit.OPS_REVISION},
                "side_effects": list(side_effects), "truncated": False, "cursor": None}

    @app.post(base + "/checkpoints/{checkpoint_id}/plan")
    def plan(checkpoint_id: str, data: EditRequest, user=Depends(needs("weights:read"))):
        cid, table = open_checkpoint(checkpoint_id)
        recipe = build_recipe(table, cid, data)
        sources, blobs = gather(recipe)
        result = edit.plan(table, recipe, base_id=cid, sources=sources, blobs=blobs)
        result["result"]["recipe"] = recipe
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
        report = edit.validate(table, recipe, base_id=cid, sources=sources, blobs=blobs)
        return envelope(cid, recipe["recipe_hash"], {"recipe_hash": recipe["recipe_hash"], "base_checkpoint": cid,
                                                     "ops": len(recipe["program"])}, {**report, "recipe": recipe})

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
    def apply(checkpoint_id: str, data: ApplyRequest, user=Depends(needs("weights:write")), idem=Depends(ctx.key)):
        cid, table = open_checkpoint(checkpoint_id)
        recipe = build_recipe(table, cid, data)
        sources, blobs = gather(recipe)
        planned = edit.plan(table, recipe, base_id=cid, sources=sources, blobs=blobs)  # invalid programs cost nothing
        if planned["result"]["noop"] and not data.allow_noop:
            raise InspectError("no_op", "the program changes no stored value (e.g. scale by 1)", "program")
        charge = user != "human"
        capabilities = sorted(ctx.creds.capabilities) if charge else ["model:read", "weights:read", "weights:write"]
        state, stored = begin(user, idem, digest([cid, recipe["recipe_hash"], data.allow_noop]), cid,
                              recipe["recipe_hash"], capabilities, charge)
        if state == "replay":
            return {**stored, "result": {**stored["result"], "replayed": True}, "side_effects": []}
        staging = workspace / "staging" / f"apply-{uuid.uuid4().hex}.safetensors"
        try:
            envelope_ = edit.apply(table, recipe, staging, base_id=cid, sources=sources, blobs=blobs,
                                   allow_noop=data.allow_noop)
            child = envelope_["result"]["manifest"]["child_sha256"]
            final = workspace / "checkpoints" / (child + ".safetensors")
            if final.exists():
                staging.unlink(missing_ok=True)
            else:
                os.replace(staging, final)
            parent = ckpts.get(cid)
            register_child(ckpts, final, child, f"{parent['display_name']} edit {recipe['recipe_hash'][:8]}", cid,
                           recipe["recipe_hash"])
            effects = [{"kind": "checkpoint_created", "id": child, "bytes": final.stat().st_size}]
            if charge:
                effects.append({"kind": "quota_charged", "runs": 1})
            envelope_["side_effects"] = effects
            envelope_["result"].update(checkpoint=ckpts.get(child), recipe=recipe, replayed=False)
            ckpts.finish_edit(user, idem, child, envelope_)
        except BaseException:
            staging.unlink(missing_ok=True)
            ckpts.abort_edit(user, idem)  # not charged, key reusable
            raise
        return envelope_

    # ------------------------------------------------------------ blobs
    @app.post(base + "/blobs", status_code=201)
    async def upload_blob(request: Request, user=Depends(needs("weights:write"))):
        try:
            declared = int(request.headers.get("content-length", ""))
        except ValueError:
            declared = 0
        if declared <= 0:
            fail("length_required", "Content-Length is required", 411)
        if declared > ctx.blob_max:
            fail("payload_too_large", f"A blob may be at most {ctx.blob_max} bytes", 413)
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
        return {"schema_version": 1, "resolved": {"bytes": declared},
                "result": {"blob_id": blob_id, "bytes": declared, "elems": declared // 4, "existing": existing},
                "fingerprints": {"checkpoint": None, "selection": None, "op_revision": edit.OPS_REVISION},
                "side_effects": [] if existing else [{"kind": "blob_stored", "id": blob_id, "bytes": declared}],
                "truncated": False, "cursor": None}
