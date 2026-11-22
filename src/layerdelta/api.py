"""Project-scoped loopback HTTP transport shared by the desktop and Python client."""

import base64
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import tempfile
import uuid

from fastapi import Depends, FastAPI, File, Header, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .checkpoint import MAX_FILE_BYTES, load_checkpoint
from .domain import DependencyMismatch, canonical_hash, freeze_recipe, verify_recipe
from .job import checkpoint_path
from .runner import load_fixture, read_json
from .schemas import Edit, Export, Final, Preview, Run, Ticket
from .store import Store, StoreError


@dataclass
class Credentials:
    project: str
    agent: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    ticket: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    cookie: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    csrf: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    principal: str = field(default_factory=lambda: "agent-" + uuid.uuid4().hex)
    # Opt-in `weights:read` capability for the agent token. The human session always has it.
    # TODO(T4c/T5): process-wide today; make it per token/session and record the grant on each
    # job/event so a report can be traced to the capability its caller held.
    weights_read: bool = False


# Preview fields an agent without weights:read may see. `activation` and `image` are deliberately
# listed: they are NOT weight-free (see docs/api.md, weights:read), only the direct kernel
# statistics are withheld.
PREVIEW_PUBLIC = frozenset({
    "schema_version", "target_id", "channel_index", "actual_device", "hook", "parameter_name",
    "activation", "image", "sample_id", "source_sample_id", "split", "condition", "label", "index",
    "runtime", "cache_key", "checkpoint_id"})


def fail(code, message, status):
    raise StoreError(code, message, status)


def matches(value, secret):
    return secrets.compare_digest(value.encode("utf-8"), secret.encode("utf-8"))


def page(items, cursor, limit, filters, secret):
    if not 1 <= limit <= 200:
        fail("invalid_limit", "Limit must be 1..200", 422)
    start = 0
    if cursor:
        try:
            raw = base64.urlsafe_b64decode(cursor)
            body, sig = raw[:-32], raw[-32:]
            position, bound = json.loads(body)
            if (not hmac.compare_digest(hmac.digest(secret.encode(), body, "sha256"), sig)
                    or bound != filters):
                raise ValueError
            if filters[0] == "history":
                start = next(i for i, item in enumerate(items) if item["run_id"] == position) + 1
            elif type(position) is int and position >= 0:
                start = position
            else:
                raise ValueError
        except (ValueError, TypeError, StopIteration):
            fail("invalid_cursor", "Invalid cursor for this collection", 422)
    end = start + limit
    token = None
    if end < len(items):
        position = items[end - 1]["run_id"] if filters[0] == "history" else end
        body = json.dumps([position, filters], separators=(",", ":")).encode()
        token = base64.urlsafe_b64encode(body + hmac.digest(secret.encode(), body, "sha256")).decode()
    return {"items": items[start:end], "next_cursor": token}


def create_app(workspace: Path, creds: Credentials, assets: Path | None = None, quota=10):
    import torch

    store = Store(workspace, quota=quota)
    fixture, manifest, demo = load_fixture(workspace / "reference")
    default = fixture["artifacts"]["model.safetensors"]
    samples = [r for r in manifest["samples"] if r["split"] == "exploration"]
    by_sample = {r["sample_id"]: r for r in samples}
    targets = {t["target_id"]: t for t in fixture["model"]["targets"]}
    revision = manifest["dataset_revision_id"]

    @asynccontextmanager
    async def lifespan(_app):
        store.start()
        try:
            yield
        finally:
            store.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
    base = "/api/v1/projects/{project_id}"

    @app.exception_handler(StoreError)
    async def store_error(request, exc):
        return JSONResponse({"error": {"code": exc.code, "message": str(exc), "request_id": uuid.uuid4().hex}},
                            status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def invalid(request, exc):
        error = exc.errors()[0]
        return JSONResponse({"error": {"code": "unsupported_schema", "message": error["msg"],
                             "field": ".".join(map(str, error["loc"])), "request_id": uuid.uuid4().hex}}, 422)

    @app.middleware("http")
    async def local_only(request, call_next):
        host = request.headers.get("host", "")
        origin = request.headers.get("origin")
        if host.split(":")[0] not in ("127.0.0.1", "localhost") or (origin and origin != "http://" + host):
            return JSONResponse({"error": {"code": "forbidden", "message": "Local origin required",
                                           "request_id": uuid.uuid4().hex}}, 403)
        limit = MAX_FILE_BYTES + (32 << 10) if request.url.path.endswith("/checkpoints") else 128 << 10
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            length = limit + 1
        if length > limit or "transfer-encoding" in request.headers:
            return JSONResponse({"error": {"code": "payload_too_large", "message": "Request exceeds limit",
                                           "request_id": uuid.uuid4().hex}}, 413)
        response = await call_next(request)
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/v1/health")
    def health():
        if not store.ready:
            fail("service_unavailable", "Workspace supervisor is unavailable", 503)
        return {"ready": True}

    @app.post("/api/v1/session")
    async def exchange(data: Ticket, response: Response):
        if not creds.ticket or not matches(data.ticket, creds.ticket):
            fail("unauthenticated", "Invalid or expired launcher ticket", 401)
        creds.ticket = ""
        response.set_cookie("layerdelta_session", creds.cookie, httponly=True, samesite="strict")
        return {"project_id": creds.project, "csrf": creds.csrf}

    def actor(project_id: str, request: Request):
        if matches(request.headers.get("authorization", ""), "Bearer " + creds.agent):
            user = creds.principal
        elif matches(request.cookies.get("layerdelta_session", ""), creds.cookie):
            if request.method != "GET" and not matches(request.headers.get("x-csrf-token", ""), creds.csrf):
                fail("forbidden", "CSRF token required", 403)
            user = "human"
        else:
            fail("unauthenticated", "Session required", 401)
        if project_id != creds.project:
            fail("not_found", "Unknown project", 404)
        return user

    def human(user=Depends(actor)):
        if user != "human":
            fail("forbidden", "Human confirmation required", 403)
        return user

    def key(idempotency_key: str = Header(max_length=200, min_length=1)):
        return idempotency_key

    def context(digest=None):
        digest = default if digest is None else digest
        try:
            available = checkpoint_path(workspace, digest, fixture).is_file()
        except ValueError:
            available = False
        if not available:
            fail("dependency_mismatch", "Unknown checkpoint_id", 409)
        seed = fixture["training"]["seed"] if digest == default else None
        return {**fixture, "training": {**fixture["training"], "seed": seed},
                "artifacts": {**fixture["artifacts"], "model.safetensors": digest}}

    def check_revision(dataset_revision_id):
        if dataset_revision_id not in (None, revision):
            fail("dependency_mismatch", "Unknown dataset revision", 409)

    def check_evaluation(value):
        if value is None:
            return
        supported = {"split": "exploration", "condition_ids": ["normal", "blurred"],
                     "metric_ids": ["accuracy", "confusion_matrix"],
                     "slice_ids": ["overall", "normal", "blurred", "primary", "baseline_error"]}
        if value != supported:
            fail("unsupported_schema", "Evaluation differs from the frozen exploration configuration", 422)

    def ensure_device(device):
        if device == "cuda" and not torch.cuda.is_available():
            fail("device_unavailable", "CUDA unavailable on the selected runtime", 409)

    def can_read_weights(user):
        return user == "human" or creds.weights_read

    def redact_preview(result, user):
        """Allowlist: a key added to the preview later stays hidden until it is reviewed and listed here."""
        if can_read_weights(user) or not isinstance(result, dict):
            return result
        return {k: v for k, v in result.items() if k in PREVIEW_PUBLIC}

    def redact(result, user):
        """Drop weight statistics from a published run report unless the caller has weights:read."""
        if can_read_weights(user) or not isinstance(result, dict):
            return result
        result = dict(result)
        audit = result.get("intervention_audit")
        if isinstance(audit, dict):  # keep the mapping, drop before/after kernel statistics
            result["intervention_audit"] = {k: v for k, v in audit.items()
                                            if k in ("parameter_name", "channel_axis", "channel_index", "hook")}
        return result

    def public(job):
        return {k: v for k, v in job.items() if k not in ("payload", "actor", "principal")}

    @app.get(base + "/catalog")
    def catalog(user=Depends(actor)):
        digests = [default] + sorted(p.stem for p in (workspace / "checkpoints").glob("*.safetensors") if p.stem != default)
        return {"project_id": creds.project, "dataset_revision_id": revision,
                "manifest_sha256": fixture["manifest_sha256"], "model": fixture["model"],
                "devices": [{"id": "cpu", "available": True}, {"id": "cuda", "available": torch.cuda.is_available()}],
                "targets": [dict(t, operations=["zero", "scale"], scale_bounds=[0, 2]) for t in targets.values()],
                "checkpoints": [{"checkpoint_id": d, "sha256": d} for d in digests], "demo": demo}

    @app.get(base + "/samples")
    def list_samples(user=Depends(actor), source_sample_id: str | None = None, condition: str | None = None,
                     cursor: str | None = None, limit: int = 50):
        items = [r for r in samples if (source_sample_id is None or r["source_sample_id"] == source_sample_id)
                 and (condition is None or r["condition"] == condition)]
        return page(items, cursor, limit, ["samples", user, source_sample_id, condition], creds.csrf)

    @app.post(base + "/checkpoints", status_code=201)
    def upload(file: UploadFile = File(), user=Depends(human)):
        directory = workspace / "checkpoints"
        directory.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=directory) as staging:
            path = Path(staging) / "checkpoint"
            data = file.file.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                fail("payload_too_large", "Checkpoint exceeds cap", 413)
            path.write_bytes(data)
            try:
                load_checkpoint(path)
            except ValueError as exc:
                fail("invalid_checkpoint", str(exc), 422)
            digest = hashlib.sha256(data).hexdigest()
            os.replace(path, directory / (digest + ".safetensors"))
        return {"checkpoint_id": digest, "sha256": digest}

    @app.post(base + "/recipes", status_code=201)
    def propose(data: Edit, user=Depends(actor)):
        check_revision(data.dataset_revision_id)
        check_evaluation(data.evaluation)
        try:
            recipe = freeze_recipe(context(data.checkpoint_id), data.intervention, data.hypothesis)
        except ValueError as exc:
            fail("invalid_scale" if "scale" in str(exc) else "invalid_target", str(exc), 422)
        return {**recipe, "recipe_id": store.put_recipe(recipe, user)}

    @app.post(base + "/recipes/import", status_code=201)
    def imported(data: dict, user=Depends(actor)):
        data = {k: v for k, v in data.items() if k != "recipe_id"}
        try:
            recipe = verify_recipe(context(data["dependencies"]["checkpoint_sha256"]), data)
        except DependencyMismatch as exc:
            fail("dependency_mismatch", str(exc), 409)
        except (ValueError, TypeError, KeyError) as exc:
            fail("unsupported_schema", str(exc), 422)
        return {**recipe, "recipe_id": store.put_recipe(recipe, user)}

    @app.get(base + "/recipes/{recipe_id}")
    def get_recipe(recipe_id: str, user=Depends(actor)):
        return {**store.get_recipe(recipe_id, user), "recipe_id": recipe_id}

    @app.get(base + "/holdout-audit")
    def audit(user=Depends(human)):
        return {"items": store.holdout_audit(user)}

    @app.get(base + "/agent-session")
    def budget(user=Depends(actor)):
        if user == "human":
            fail("forbidden", "Agent session required", 403)
        return {**store.quota_info(user), "capabilities": ["weights:read"] if can_read_weights(user) else []}

    @app.post(base + "/previews", status_code=202)
    def submit_preview(data: Preview, user=Depends(actor), idem=Depends(key)):
        ensure_device(data.device)
        check_revision(data.dataset_revision_id)
        t = targets.get(data.target_id)
        if t is None or not 0 <= data.channel_index < t["channel_count"]:
            fail("invalid_target", "Unknown target or channel", 422)
        if data.sample_id not in by_sample:
            fail("not_found", "Unknown exploration sample", 404)
        context(data.checkpoint_id)
        p = {**data.model_dump(), "checkpoint_id": data.checkpoint_id or default}
        return public(store.submit(user, "previews", idem, p, "preview"))

    @app.post(base + "/runs", status_code=202)
    def submit_run(data: Run, user=Depends(actor), idem=Depends(key)):
        ensure_device(data.device)
        check_revision(data.dataset_revision_id)
        check_evaluation(data.evaluation)
        if bool(data.recipe_id) != (data.kind == "intervention"):
            fail("unsupported_schema", "Intervention requires a recipe ID", 422)
        recipe = store.get_recipe(data.recipe_id, user) if data.recipe_id else None
        digest = recipe["dependencies"]["checkpoint_sha256"] if recipe else data.checkpoint_id or default
        if data.checkpoint_id is not None and data.checkpoint_id != digest:
            fail("dependency_mismatch", "Recipe checkpoint mismatch", 409)
        context(digest)
        p = dict(checkpoint_id=digest, recipe=recipe, recipe_id=data.recipe_id, device=data.device)
        if data.retry_of_run_id:
            old = store.get(data.retry_of_run_id, user)
            old_payload = {k: v for k, v in old["payload"].items() if k != "retry_of_run_id"}
            if old["status"] not in ("failed", "cancelled") or old_payload != p or old["kind"] != data.kind:
                fail("idempotency_conflict", "Retry differs from the failed or cancelled run", 409)
            p["retry_of_run_id"] = data.retry_of_run_id
        return public(store.submit(user, "runs", idem, p, data.kind))

    @app.get(base + "/runs")
    def history(user=Depends(actor), cursor: str | None = None, limit: int = 50):
        items = [public(j) for j in store.list_jobs(user) if j["kind"] not in ("preview", "checkpoint_export")]
        return page(items, cursor, limit, ["history", user], creds.csrf)

    @app.get(base + "/runs/{run_id}")
    def status(run_id: str, user=Depends(actor)):
        return public(store.get(run_id, user))

    @app.get(base + "/previews/{run_id}")
    def preview_status(run_id: str, user=Depends(actor)):
        job = public(store.get(run_id, user))
        if job["kind"] != "preview":
            fail("not_found", "Unknown preview", 404)
        if job["status"] == "succeeded":
            job["result"] = redact_preview(read_json(store.artifact(run_id, user, "report.json")), user)
        return job

    @app.post(base + "/runs/{run_id}/cancel")
    def cancel(run_id: str, user=Depends(actor)):
        return public(store.cancel(run_id, user))

    @app.get(base + "/runs/{run_id}/report")
    def report(run_id: str, user=Depends(actor)):
        result = redact(read_json(store.artifact(run_id, user, "report.json")), user)
        return {**result, "run_id": run_id}

    @app.get(base + "/runs/{run_id}/comparison")
    def comparison(run_id: str, user=Depends(actor)):
        data = read_json(store.artifact(run_id, user, "comparison.json"))
        r = report(run_id, user)
        return {**data, "run_id": run_id, "baseline_run_id": r["baseline_id"],
                "recipe_hash": r["recipe_hash"], "evaluation_fingerprint": r["evaluation_fingerprint"],
                "runtime": r["runtime"]}

    @app.get(base + "/runs/{run_id}/changes")
    def changes(run_id: str, user=Depends(actor), cursor: str | None = None, limit: int = 50):
        data = read_json(store.artifact(run_id, user, "comparison.json"))
        return page(data["changes"], cursor, limit, ["changes", user, run_id], creds.csrf)

    @app.post(base + "/final-evaluations", status_code=202)
    def final(data: Final, user=Depends(human), idem=Depends(key)):
        ensure_device(data.device)
        recipe = store.get_recipe(data.recipe_id, user)
        if not data.confirmed or data.recipe_hash != recipe["recipe_hash"]:
            fail("dependency_mismatch", "Confirm the frozen recipe hash", 409)
        p = {**data.model_dump(), "recipe": recipe, "checkpoint_id": recipe["dependencies"]["checkpoint_sha256"],
             "split_sha256": manifest["split_sha256"]["holdout"]}
        return public(store.submit(user, "final-evaluations", idem, p, "final_evaluation", split="holdout"))

    @app.post(base + "/checkpoint-exports", status_code=202)
    def export(data: Export, user=Depends(human), idem=Depends(key)):
        job = store.get(data.source_run_id, user)
        if not data.confirmed or job["status"] != "succeeded" or job["kind"] not in ("intervention", "final_evaluation"):
            fail("run_not_ready", "Confirm a successful weight run", 409)
        recipe = job["payload"]["recipe"]
        if recipe["intervention"]["mode"] != "weight":
            fail("invalid_target", "Only a weight run can export a checkpoint", 422)
        r = report(data.source_run_id, user)
        if r["recipe_hash"] != recipe["recipe_hash"]:
            fail("dependency_mismatch", "Published recipe differs from source run", 409)
        p = {**data.model_dump(), "recipe": recipe, "checkpoint_id": r["parent_sha256"]}
        return public(store.submit(user, "checkpoint-exports", idem, p, "checkpoint_export", split=job["split"]))

    @app.get(base + "/checkpoint-exports/{run_id}")
    def export_status(run_id: str, user=Depends(human)):
        job = public(store.get(run_id, user))
        if job["kind"] != "checkpoint_export":
            fail("not_found", "Unknown checkpoint export", 404)
        if job["status"] == "succeeded":
            job["result"] = {**report(run_id, user),
                             "download_url": f"/api/v1/projects/{creds.project}/checkpoint-exports/{run_id}/file"}
        return job

    @app.get(base + "/checkpoint-exports/{run_id}/file")
    def child_file(run_id: str, user=Depends(human)):
        path = store.artifact(run_id, user, "child.safetensors")
        return FileResponse(path, filename="child.safetensors")

    if assets:
        app.mount("/", StaticFiles(directory=assets, html=True))
    return app
