"""Workspace catalog, serialized child execution and immutable result publication."""

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .process import popen


def _now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(value):
    return json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class StoreError(Exception):
    def __init__(self, code, message, status):
        self.code, self.message, self.status = code, message, status
        super().__init__(message)


def _lock(path):
    stream = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise
    return stream


def _guard(workspace, command):
    """Own the worker until it exits. A closed supervisor pipe means cancel."""
    lost_parent = threading.Event()

    def read_pipe():
        try:
            os.read(0, 1)
        finally:
            lost_parent.set()

    threading.Thread(target=read_pipe, daemon=True).start()
    # A new supervisor can start while an orphaned guard is still stopping its child.
    while not lost_parent.is_set():
        try:
            lease = _lock(workspace / ".worker.lock")
            break
        except OSError:
            lost_parent.wait(.05)
    else:
        return 1
    with lease, popen(command, stdin=subprocess.DEVNULL) as child:
        while child.poll() is None:
            if lost_parent.wait(.05):
                child.kill()
                break
        return child.wait()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, actor TEXT NOT NULL, route TEXT NOT NULL, key TEXT NOT NULL,
    request_digest TEXT NOT NULL, kind TEXT NOT NULL, split TEXT NOT NULL, payload TEXT NOT NULL,
    status TEXT NOT NULL, charged INTEGER NOT NULL, cancel_requested INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, error TEXT, manifest TEXT,
    UNIQUE(actor, route, key)
);
CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, created_at, id);
CREATE TABLE IF NOT EXISTS recipes (
    id TEXT NOT NULL, actor TEXT NOT NULL, content TEXT NOT NULL, PRIMARY KEY(id,actor)
);
CREATE TABLE IF NOT EXISTS holdout_audit (
    run_id TEXT PRIMARY KEY REFERENCES jobs(id), actor TEXT NOT NULL, recipe_hash TEXT NOT NULL,
    created_at TEXT NOT NULL, split_manifest_hash TEXT
);
"""


class Store:
    def __init__(self, workspace, quota=10, queue_limit=32, timeout=300, child_command=None):
        self.workspace = Path(workspace).resolve()
        self.quota, self.queue_limit, self.timeout = quota, queue_limit, timeout
        self.child_command = child_command or [sys.executable, "-m", "layerdelta.job"]
        self.db_path = self.workspace / "catalog.sqlite3"
        self._lease = self._thread = None
        self._stopping = threading.Event()
        self._wake = threading.Event()

    @contextmanager
    def _db(self, write=False):
        db = sqlite3.connect(self.db_path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA foreign_keys=ON")
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def start(self):
        if self._lease:
            return
        self.workspace.mkdir(parents=True, exist_ok=True)
        try:
            self._lease = _lock(self.workspace / ".supervisor.lock")
        except OSError as exc:
            raise StoreError("workspace_locked", "Workspace already has a supervisor", 409) from exc
        try:
            self._stopping.clear()
            for sub in ("staging", "runs"):
                (self.workspace / sub).mkdir(exist_ok=True)
            with self._db() as db:
                db.execute("PRAGMA journal_mode=WAL")
                db.executescript(_SCHEMA)
            with self._db(True) as db:
                db.execute("""UPDATE jobs SET status=CASE cancel_requested WHEN 1 THEN 'cancelled' ELSE 'failed' END,
                    finished_at=?,error=CASE cancel_requested WHEN 1 THEN NULL ELSE ? END WHERE status='running'""",
                    (_now(), _json({"code": "interrupted", "message": "Supervisor stopped during execution"})))
            self._clean_orphans()
            self._thread = threading.Thread(target=self._work, name="layerdelta-supervisor", daemon=True)
            self._thread.start()
        except BaseException:
            self._lease.close()
            self._lease = None
            raise

    def close(self):
        if not self._lease:
            return
        self._stopping.set()
        self._wake.set()
        if self._thread:
            self._thread.join()
        self._lease.close()
        self._lease = self._thread = None

    def _clean_orphans(self):
        try:
            lease = _lock(self.workspace / ".worker.lock")
        except OSError:
            return
        with lease, self._db() as db:
            for row in db.execute("SELECT id FROM jobs WHERE status IN ('failed','cancelled')"):
                run_id = row["id"]
                if re.fullmatch(r"run_[0-9a-f]{32}", run_id):
                    for name in ("staging", "runs"):
                        candidate = self.workspace / name / run_id
                        if candidate.is_dir() and candidate.resolve() == candidate:
                            shutil.rmtree(candidate, ignore_errors=True)

    @property
    def ready(self):
        return bool(self._thread and self._thread.is_alive() and not self._stopping.is_set())

    @staticmethod
    def _visible(db, run_id, actor):
        row = db.execute("SELECT * FROM jobs WHERE id=? AND (?='human' OR (actor=? AND split='exploration'))",
                         (run_id, actor, actor)).fetchone()
        if row is None:
            raise StoreError("not_found", "Job was not found", 404)
        return row

    @staticmethod
    def _public(row):
        payload = json.loads(row["payload"])
        return {
            "run_id": row["id"], "kind": row["kind"], "split": row["split"], "status": row["status"],
            "recipe_id": payload.get("recipe_id"), "payload": payload,
            "progress": {"phase": row["status"]}, "created_at": row["created_at"],
            "started_at": row["started_at"], "finished_at": row["finished_at"],
            "error": json.loads(row["error"]) if row["error"] else None,
        }

    def get(self, run_id, actor):
        with self._db() as db:
            return self._public(self._visible(db, run_id, actor))

    def list_jobs(self, actor):
        with self._db() as db:
            rows = db.execute("""SELECT * FROM jobs WHERE ?='human' OR (actor=? AND split='exploration')
                                 ORDER BY created_at DESC,id DESC""", (actor, actor))
            return [self._public(row) for row in rows]

    def quota_info(self, actor):
        with self._db() as db:
            charged = db.execute("SELECT coalesce(sum(charged),0) FROM jobs WHERE actor=?", (actor,)).fetchone()[0]
        return {"limit": self.quota, "charged": charged, "remaining": max(0, self.quota - charged),
                "queue_limit": self.queue_limit, "timeout_seconds": self.timeout}

    def submit(self, actor, route, key, payload, kind, split="exploration"):
        if not actor or not key or len(key) > 256:
            raise StoreError("invalid_request", "An actor and idempotency key are required", 422)
        if split not in ("exploration", "holdout"):
            raise StoreError("invalid_request", "Unknown evaluation split", 422)
        if actor != "human" and (split != "exploration" or kind not in ("baseline", "intervention", "preview")):
            raise StoreError("forbidden", "This operation requires a human session", 403)
        digest = hashlib.sha256(_json([kind, split, payload]).encode()).hexdigest()
        with self._db(True) as db:
            prior = db.execute("SELECT * FROM jobs WHERE actor=? AND route=? AND key=?", (actor, route, key)).fetchone()
            if prior:
                if prior["request_digest"] != digest:
                    raise StoreError("idempotency_conflict", "This key has a different request", 409)
                return self._public(prior)
            charge = int(actor != "human" and kind in ("baseline", "intervention"))
            if charge:
                total = db.execute("SELECT coalesce(sum(charged),0) FROM jobs WHERE actor=?", (actor,)).fetchone()[0]
                if total >= self.quota:
                    raise StoreError("quota_exhausted", "Session run quota is exhausted", 429)
            if db.execute("SELECT count(*) FROM jobs WHERE status='queued'").fetchone()[0] >= self.queue_limit:
                raise StoreError("queue_full", "Worker queue is full", 429)
            run_id, created = "run_" + uuid.uuid4().hex, _now()
            db.execute("""INSERT INTO jobs(id,actor,route,key,request_digest,kind,split,payload,status,charged,created_at)
                          VALUES(?,?,?,?,?,?,?,?,'queued',?,?)""",
                       (run_id, actor, route, key, digest, kind, split, _json(payload), charge, created))
            if kind == "final_evaluation":
                if split != "holdout" or payload.get("confirmed") is not True or not payload.get("recipe_hash"):
                    raise StoreError("confirmation_required", "Confirm the frozen recipe hash before holdout", 422)
                db.execute("INSERT INTO holdout_audit VALUES(?,?,?,?,?)",
                           (run_id, actor, payload["recipe_hash"], created, payload["split_sha256"]))
            result = self._public(self._visible(db, run_id, actor))
        self._wake.set()
        return result

    def cancel(self, run_id, actor):
        with self._db(True) as db:
            row = self._visible(db, run_id, actor)
            if row["status"] == "queued":
                db.execute("UPDATE jobs SET status='cancelled',cancel_requested=1,finished_at=? WHERE id=?", (_now(), run_id))
            elif row["status"] == "running":
                db.execute("UPDATE jobs SET cancel_requested=1 WHERE id=?", (run_id,))
            return self._public(self._visible(db, run_id, actor))

    def holdout_audit(self, actor):
        if actor != "human":
            raise StoreError("forbidden", "Holdout audit requires a human session", 403)
        with self._db() as db:
            return [dict(row) for row in db.execute("SELECT * FROM holdout_audit ORDER BY created_at DESC,run_id DESC")]

    def put_recipe(self, recipe, actor):
        if recipe["evaluation"]["split"] != "exploration":
            raise StoreError("forbidden", "Recipes must be frozen on exploration", 403)
        with self._db(True) as db:
            db.execute("INSERT OR IGNORE INTO recipes VALUES(?,?,?)", (recipe["recipe_hash"], actor, _json(recipe)))
        return recipe["recipe_hash"]

    def get_recipe(self, recipe_id, actor):
        with self._db() as db:
            row = db.execute("SELECT content FROM recipes WHERE id=? AND (?='human' OR actor IN (?, 'human')) LIMIT 1",
                             (recipe_id, actor, actor)).fetchone()
        if row is None:
            raise StoreError("not_found", "Recipe was not found", 404)
        return json.loads(row[0])

    def list_recipes(self, actor):
        with self._db() as db:
            rows = db.execute("SELECT DISTINCT content FROM recipes WHERE ?='human' OR actor IN (?, 'human') ORDER BY id",
                              (actor, actor))
            return [json.loads(row[0]) for row in rows]

    def artifact(self, run_id, actor, filename):
        with self._db() as db:
            row = self._visible(db, run_id, actor)
            if row["status"] != "succeeded":
                raise StoreError("run_not_ready", "Job has no published result", 409)
            manifest = json.loads(row["manifest"])
        if filename not in manifest:
            raise StoreError("not_found", "Artifact was not found", 404)
        root = self.workspace / "runs" / run_id
        path = root / filename
        try:
            valid = (root.resolve() == root and path.resolve() == path
                     and _digest(root / "manifest.json") == manifest["manifest.json"]
                     and _digest(path) == manifest[filename])
        except OSError:
            valid = False
        if not valid:
            raise StoreError("artifact_unavailable", "Published artifact is missing or corrupt", 409)
        return path

    def _work(self):
        while not self._stopping.is_set():
            with self._db(True) as db:
                row = db.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY created_at,id LIMIT 1").fetchone()
                if row:
                    db.execute("UPDATE jobs SET status='running',started_at=? WHERE id=?", (_now(), row["id"]))
            if row is None:
                self._wake.wait(.5)
                self._wake.clear()
                continue
            self._execute(row)

    def _execute(self, row):
        run_id = row["id"]
        stage = self.workspace / "staging" / run_id
        output = stage / "result"
        try:
            stage.mkdir()
            request = stage / "request.json"
            request.write_text(_json({"kind": row["kind"], "split": row["split"], "run_id": run_id,
                                      "payload": json.loads(row["payload"])}), encoding="utf-8")
            command = [*self.child_command, "--workspace", str(self.workspace), "--request", str(request),
                       "--output", str(output)]
            guard = [sys.executable, "-m", "layerdelta.store", str(self.workspace), *command]
            with popen(guard, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as child:
                start = time.monotonic()
                try:
                    while child.poll() is None:
                        if self._stopping.wait(.1):
                            raise StoreError("interrupted", "Supervisor stopped during execution", 503)
                        if time.monotonic() - start > self.timeout:
                            raise StoreError("timeout", "Worker exceeded its time limit", 503)
                        with self._db() as db:
                            cancel = db.execute("SELECT cancel_requested FROM jobs WHERE id=?", (run_id,)).fetchone()[0]
                        if cancel:
                            raise StoreError("cancelled", "Cancellation requested", 409)
                finally:
                    child.stdin.close()
                if child.returncode:
                    error_file = output / "error.json"
                    if error_file.is_file() and error_file.stat().st_size <= 16384:
                        detail = json.loads(error_file.read_bytes())
                        raise StoreError(str(detail["code"])[:128], str(detail["message"])[:2048], 500)
                    raise StoreError("worker_failed", f"Worker exited with code {child.returncode}", 500)
            self._publish(run_id, row["kind"], output)
        except Exception as exc:
            code = exc.code if isinstance(exc, StoreError) else "worker_failed"
            self._finish_error(run_id, code, str(exc))
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def _publish(self, run_id, kind, output):
        required = {"report.json"}
        if kind in ("baseline", "intervention", "final_evaluation"):
            required.update(("predictions.json", "comparison.json"))
        if kind in ("intervention", "final_evaluation"):
            required.add("recipe.json")
        if kind == "checkpoint_export":
            required.add("child.safetensors")
        if output.is_symlink() or not all((output / name).is_file() for name in required):
            raise StoreError("artifact_unavailable", "Worker result is incomplete", 409)
        manifest = {}
        for file in output.iterdir():
            if not file.is_file() or file.is_symlink() or file.name == "manifest.json":
                raise StoreError("artifact_unavailable", "Invalid worker output", 409)
            with file.open("r+b") as stream:
                manifest[file.name] = hashlib.file_digest(stream, "sha256").hexdigest()
                os.fsync(stream.fileno())
        if kind in ("baseline", "intervention", "final_evaluation", "checkpoint_export"):
            try:
                report = json.loads((output / "report.json").read_bytes())
                if kind == "checkpoint_export":
                    intact = report["child_sha256"] == manifest["child.safetensors"]
                else:
                    declared = report["artifacts"]
                    intact = (isinstance(declared, dict) and set(declared) == set(manifest) - {"report.json"}
                              and all(record["sha256"] == manifest[name]
                                      and type(record["bytes"]) is int and record["bytes"] == (output / name).stat().st_size
                                      for name, record in declared.items()))
            except (ValueError, KeyError, TypeError):
                intact = False
            if not intact:
                raise StoreError("artifact_unavailable", "Staging disagrees with the worker artifact digests", 409)
        with (output / "manifest.json").open("xb") as stream:
            content = _json(manifest).encode()
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            manifest["manifest.json"] = hashlib.sha256(content).hexdigest()
        with self._db(True) as db:
            if db.execute("SELECT cancel_requested FROM jobs WHERE id=?", (run_id,)).fetchone()[0]:
                raise StoreError("cancelled", "Cancellation requested", 409)
            output.rename(self.workspace / "runs" / run_id)
            db.execute("UPDATE jobs SET status='succeeded',manifest=?,finished_at=? WHERE id=?",
                       (_json(manifest), _now(), run_id))

    def _finish_error(self, run_id, code, message):
        with self._db(True) as db:
            db.execute("""UPDATE jobs SET status=CASE cancel_requested WHEN 1 THEN 'cancelled' ELSE 'failed' END,
                          error=CASE cancel_requested WHEN 1 THEN NULL ELSE ? END,finished_at=?
                          WHERE id=? AND status='running'""",
                       (_json({"code": code, "message": message[:2048]}), _now(), run_id))


if __name__ == "__main__":
    raise SystemExit(_guard(Path(sys.argv[1]), sys.argv[2:]))
