"""Checkpoint catalog and import (v0.2-T4b).

The catalog is a SQLite table next to the job store, independent of the reference fixture: the
fixture registers itself as one entry (origin `fixture`), imports add `imported` entries and, later,
applied edits add `child` entries. Checkpoints are content addressed: `checkpoint_id` is the sha256
of the file (single file) or of the sorted (name, sha256) list of the file set (sharded). Import
validates with the T1 reader only (no model construction, no code from the file), copies into the
workspace, re-validates the copy and publishes it atomically, so later reads never depend on the
user's original file.
"""

import argparse
import hashlib
import json
import os
import queue
import shutil
import sqlite3
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .tensors import INDEX_SUFFIX, MAX_FILE_BYTES, MAX_HEADER_BYTES, TensorFileError, open_tensors

STAGES = ("validating", "copying", "verifying", "publishing")
PICKLE_SUFFIXES = (".pt", ".pth", ".bin", ".ckpt", ".pkl", ".pickle")
PICKLE_MESSAGE = ("Pickle checkpoints can run code when loaded, so they are refused. Convert the file to "
                  "safetensors yourself, then import the .safetensors")
COPY_CHUNK = 1 << 20
# A sharded id hashes this prefix plus the sorted [file name, sha256] list (the index file's NAME is
# part of the id), so it can never equal the sha256 of a single file.
FILESET_DOMAIN = b"layerdelta-fileset-v1\n"
ORIGINS = ("imported", "fixture", "child")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS checkpoints (
    id TEXT PRIMARY KEY, display_name TEXT NOT NULL, origin TEXT NOT NULL, parent_id TEXT, recipe_hash TEXT,
    format TEXT NOT NULL, sharded INTEGER NOT NULL, shard_count INTEGER NOT NULL, file_bytes INTEGER NOT NULL,
    tensor_count INTEGER, element_count INTEGER, adapter_compatible INTEGER NOT NULL,
    location TEXT NOT NULL, imported_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS imports (
    id TEXT PRIMARY KEY, actor TEXT NOT NULL, source TEXT NOT NULL, source_name TEXT NOT NULL,
    display_name TEXT, status TEXT NOT NULL, stage TEXT, done INTEGER NOT NULL DEFAULT 0,
    total INTEGER NOT NULL DEFAULT 0, unit TEXT, error TEXT, checkpoint_id TEXT, existing INTEGER,
    tensor_count INTEGER, created_at TEXT NOT NULL, finished_at TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    principal TEXT PRIMARY KEY, kind TEXT NOT NULL, capabilities TEXT NOT NULL, created_at TEXT NOT NULL, revoked_at TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session TEXT NOT NULL, seq INTEGER NOT NULL, at TEXT NOT NULL,
    route TEXT NOT NULL, tool TEXT NOT NULL, request_id TEXT NOT NULL, status INTEGER NOT NULL, error_code TEXT,
    checkpoint_id TEXT, other_checkpoint_id TEXT, selection_id TEXT, recipe_hash TEXT, child_id TEXT, run_id TEXT,
    idempotency_key TEXT, replayed INTEGER, charged INTEGER NOT NULL, capabilities TEXT NOT NULL, detail TEXT NOT NULL,
    via TEXT NOT NULL DEFAULT 'http', duration_ms INTEGER,
    UNIQUE(session, seq)
);
CREATE INDEX IF NOT EXISTS events_session ON events(session, id);
CREATE INDEX IF NOT EXISTS events_key ON events(session, route, idempotency_key);
CREATE TABLE IF NOT EXISTS proposals (
    proposal_id TEXT PRIMARY KEY, session TEXT NOT NULL, idem_digest TEXT NOT NULL, request_digest TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL, recipe_hash TEXT NOT NULL, selections TEXT NOT NULL, request TEXT NOT NULL,
    status TEXT NOT NULL, created_at TEXT NOT NULL, decided_at TEXT, child_id TEXT, UNIQUE(session, idem_digest)
);
CREATE TABLE IF NOT EXISTS selections (
    checkpoint_id TEXT NOT NULL, selection_id TEXT NOT NULL, selector TEXT NOT NULL, first_at TEXT NOT NULL,
    PRIMARY KEY(checkpoint_id, selection_id)
);
CREATE INDEX IF NOT EXISTS selections_id ON selections(selection_id);
CREATE TABLE IF NOT EXISTS recipe_store (
    recipe_hash TEXT PRIMARY KEY, recipe TEXT NOT NULL, first_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS graphs (
    ir_sha256 TEXT NOT NULL, provider TEXT NOT NULL, spec_sha256 TEXT NOT NULL, table_signature TEXT NOT NULL,
    ir_json TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(ir_sha256, table_signature)
);
CREATE INDEX IF NOT EXISTS graphs_signature ON graphs(table_signature, created_at);
CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id TEXT PRIMARY KEY, actor TEXT NOT NULL, key TEXT NOT NULL, request_digest TEXT NOT NULL, checkpoint_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL, split TEXT NOT NULL, baseline_id TEXT, fingerprint TEXT NOT NULL, status TEXT NOT NULL, error TEXT,
    result TEXT, samples INTEGER NOT NULL, baseline_cached INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, started_at TEXT,
    finished_at TEXT, recipe_hash TEXT, UNIQUE(actor, key)
);
CREATE TABLE IF NOT EXISTS probes (
    probe_id TEXT PRIMARY KEY, actor TEXT NOT NULL, key TEXT NOT NULL, request_digest TEXT NOT NULL, checkpoint_id TEXT NOT NULL,
    dataset_id TEXT NOT NULL, status TEXT NOT NULL, error TEXT, result TEXT, samples INTEGER NOT NULL, created_at TEXT NOT NULL,
    started_at TEXT, finished_at TEXT, UNIQUE(actor, key)
);
CREATE TABLE IF NOT EXISTS executables (
    ir_sha256 TEXT PRIMARY KEY, record TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS graph_attachments (
    ir_sha256 TEXT NOT NULL, table_signature TEXT NOT NULL, family_signature TEXT NOT NULL, checkpoint_id TEXT NOT NULL,
    attached_at TEXT NOT NULL, PRIMARY KEY(ir_sha256, checkpoint_id)
);
CREATE INDEX IF NOT EXISTS graph_attachments_family ON graph_attachments(family_signature);
CREATE TABLE IF NOT EXISTS edits (
    actor TEXT NOT NULL, key TEXT NOT NULL, request_digest TEXT NOT NULL, status TEXT NOT NULL,
    charged INTEGER NOT NULL, base_id TEXT NOT NULL, recipe_hash TEXT NOT NULL, capabilities TEXT NOT NULL,
    checkpoint_id TEXT, result TEXT, created_at TEXT NOT NULL, finished_at TEXT, PRIMARY KEY(actor, key)
);
"""


def _now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json(value):
    return json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class ImportLimits:
    """Configurable caps. Defaults are the T1 reader's."""
    max_file_bytes: int = MAX_FILE_BYTES      # each file (shard)
    max_header_bytes: int = MAX_HEADER_BYTES
    max_total_bytes: int = 4 * MAX_FILE_BYTES  # a whole sharded set
    stall_seconds: float = 120.0  # no progress for this long (a hung network path) fails the import and frees the queue


class ImportFailure(Exception):
    """A refused import with a stable `code` (T1 codes plus the ones below)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code, self.message = code, message


class ImportControl:
    """Cancellation and liveness of one running import.

    The worker beats (and checks for cancellation) between chunks and stages. `cancel()` fails if the import has already
    reached its commit point (the atomic publish), so a cancelled import never leaves a published checkpoint behind.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.cancelled = self.committed = False
        self.heartbeat = time.monotonic()

    def beat(self):
        self.heartbeat = time.monotonic()

    def check(self):
        self.beat()
        if self.cancelled:
            raise ImportFailure("cancelled", "The import was cancelled")

    def cancel(self) -> bool:
        with self._lock:
            if self.committed:
                return False
            self.cancelled = True
            return True

    def commit(self):
        with self._lock:
            if self.cancelled:
                raise ImportFailure("cancelled", "The import was cancelled")
            self.committed = True


def check_display_name(value) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > 128 or any(ord(c) < 32 for c in value):
        raise ValueError("display_name must be 1 to 128 printable characters")
    return value.strip()


def adapter_compatible(table, state_schema) -> bool:
    """True when the tensor table is exactly the registered MNIST adapter's state dict (F32 shapes)."""
    if not state_schema:
        return False
    want = {name: ("F32", tuple(spec["shape"])) for name, spec in state_schema.items()
            if spec.get("dtype") == "torch.float32"}
    return len(want) == len(state_schema) and {n: (i.dtype, i.shape) for n, i in table.tensors.items()} == want


def _public(row) -> dict:
    return {"checkpoint_id": row["id"], "display_name": row["display_name"], "origin": row["origin"],
            "parent_id": row["parent_id"], "recipe_hash": row["recipe_hash"], "format": row["format"],
            "sharded": bool(row["sharded"]), "shard_count": row["shard_count"], "file_bytes": row["file_bytes"],
            "tensor_count": row["tensor_count"], "element_count": row["element_count"],
            "adapter_compatible": bool(row["adapter_compatible"]), "imported_at": row["imported_at"]}


def _public_import(row) -> dict:
    stage = row["stage"]
    return {"import_id": row["id"], "status": row["status"], "stage": stage,
            "stage_index": STAGES.index(stage) + 1 if stage in STAGES else None, "stage_count": len(STAGES),
            "done": row["done"], "total": row["total"], "unit": row["unit"],
            "error": json.loads(row["error"]) if row["error"] else None, "checkpoint_id": row["checkpoint_id"],
            "existing": None if row["existing"] is None else bool(row["existing"]),
            "tensor_count": row["tensor_count"], "display_name": row["display_name"], "source": row["source"],
            "source_name": row["source_name"], "created_at": row["created_at"], "finished_at": row["finished_at"]}


def lookup_location(workspace, digest):
    """(path, sharded) of a catalogued checkpoint, or None. Read-only: safe from job processes."""
    db_path = Path(workspace) / "catalog.sqlite3"
    try:
        db = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True, timeout=5)
    except sqlite3.Error:
        return None
    try:
        row = db.execute("SELECT location, sharded FROM checkpoints WHERE id=?", (digest,)).fetchone()
    except sqlite3.Error:
        return None
    finally:
        db.close()
    return None if row is None else (Path(workspace) / row[0], bool(row[1]))


def _summary(path: Path, state_schema) -> dict:
    """Counts for an existing file; tolerant, so an unreadable reference file still registers."""
    try:
        table = open_tensors(path)
    except (TensorFileError, OSError):
        size = path.stat().st_size if path.is_file() else 0
        return {"sharded": False, "shard_count": 1, "file_bytes": size, "tensor_count": None,
                "element_count": None, "compatible": False}
    return {"sharded": table.sharded, "shard_count": len(table.files), "file_bytes": sum(p.stat().st_size for p in table.files),
            "tensor_count": len(table.tensors), "element_count": _elements(table),
            "compatible": adapter_compatible(table, state_schema)}


def _elements(table) -> int:
    total = 0
    for info in table.tensors.values():
        n = 1
        for dim in info.shape:
            n *= dim
        total += n
    return total


class Catalog:
    def __init__(self, workspace, state_schema=None):
        self.workspace = Path(workspace).resolve()
        self.db_path = self.workspace / "catalog.sqlite3"
        self.state_schema = state_schema

    @contextmanager
    def _db(self, write=False, durable=True):
        db = sqlite3.connect(self.db_path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            if not durable:  # trace events: WAL-safe, but the last few commits may be lost on a power cut
                db.execute("PRAGMA synchronous=NORMAL")
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def start(self, fixture=None, *, clean=True):
        """Create tables, fail interrupted imports, drop stale staging, adopt legacy files, add the fixture."""
        for sub in ("staging", "checkpoints"):
            (self.workspace / sub).mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(_SCHEMA)
            have = {r["name"] for r in db.execute("PRAGMA table_info(events)")}  # a catalog from before v0.4
            for name, ddl in (("via", "TEXT NOT NULL DEFAULT 'http'"), ("duration_ms", "INTEGER")):
                if name not in have:
                    db.execute(f"ALTER TABLE events ADD COLUMN {name} {ddl}")
            if "recipe_hash" not in {r["name"] for r in db.execute("PRAGMA table_info(evaluations)")}:  # before the final evaluation of v0.5-E4c
                db.execute("ALTER TABLE evaluations ADD COLUMN recipe_hash TEXT")
            if "revoked_at" not in {r["name"] for r in db.execute("PRAGMA table_info(sessions)")}:
                db.execute("ALTER TABLE sessions ADD COLUMN revoked_at TEXT")
            from . import registry  # events from before v0.4 carry path-derived tool names: rename them to the registry's
            for route in registry.ROUTES:
                if route.tool:
                    db.execute("UPDATE events SET tool=? WHERE route=? AND tool<>?", (route.tool, f"{route.method} {route.path}", route.tool))
        if clean:
            with self._db(True) as db:
                db.execute("UPDATE imports SET status='failed', finished_at=?, error=? WHERE status IN ('queued','running')",
                           (_now(), _json({"code": "interrupted", "message": "The server stopped during this import"})))
                db.execute("DELETE FROM edits WHERE status='running'")  # an interrupted apply releases its charge
            # stale staging, including the hidden .partial-* files a hard crash can leave next to a child
            for entry in (self.workspace / "staging").iterdir():
                if entry.name.startswith(("import-", "upload-", "apply-", ".apply-", "blob-")):
                    shutil.rmtree(entry, ignore_errors=True) if entry.is_dir() else entry.unlink(missing_ok=True)
        for path in sorted((self.workspace / "checkpoints").glob("*.safetensors")):
            if len(path.stem) == 64 and self.get(path.stem) is None:  # pre-catalog upload or crashed publish
                with path.open("rb") as stream:  # the file name is not trusted as its content id
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != path.stem:
                    quarantine = self.workspace / "quarantine"
                    quarantine.mkdir(exist_ok=True)
                    os.replace(path, quarantine / f"{path.stem}.sha-{actual}.safetensors")
                    continue
                self.insert(path.stem, path.stem[:12], "imported", f"checkpoints/{path.name}", _summary(path, self.state_schema))
        if fixture is not None:
            digest = fixture["artifacts"]["model.safetensors"]
            if self.get(digest) is None:
                summary = _summary(self.workspace / "reference" / "model.safetensors", self.state_schema)
                summary["compatible"] = True  # the fixture is the adapter's own reference
                self.insert(digest, "Reference fixture", "fixture", "reference/model.safetensors", summary)

    def insert(self, digest, display_name, origin, location, summary, *, parent_id=None, recipe_hash=None) -> bool:
        assert origin in ORIGINS
        with self._db(True) as db:
            cursor = db.execute(
                "INSERT OR IGNORE INTO checkpoints VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (digest, display_name, origin, parent_id, recipe_hash, "safetensors", int(summary["sharded"]),
                 summary["shard_count"], summary["file_bytes"], summary["tensor_count"], summary["element_count"],
                 int(summary["compatible"]), location, _now()))
            return cursor.rowcount == 1

    def get(self, digest) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM checkpoints WHERE id=?", (digest,)).fetchone()
        return None if row is None else _public(row)

    def children(self, digest) -> list[str]:
        """Ids of the checkpoints whose lineage names `digest` as parent (oldest first)."""
        with self._db() as db:
            return [r["id"] for r in db.execute("SELECT id FROM checkpoints WHERE parent_id=? ORDER BY imported_at, id", (digest,))]

    def list(self, *, compatible_only=False) -> list[dict]:
        """The fixture first (agents rely on checkpoints[0] being the reference), then by import time."""
        with self._db() as db:
            rows = db.execute("SELECT * FROM checkpoints WHERE (? = 0 OR adapter_compatible = 1) "
                              "ORDER BY origin != 'fixture', imported_at, id", (int(compatible_only),))
            return [_public(row) for row in rows]

    def location(self, digest):
        return lookup_location(self.workspace, digest)

    # -- agent sessions and capabilities (T5 reads these)
    def record_session(self, principal, kind, capabilities):
        with self._db(True) as db:
            db.execute("INSERT OR REPLACE INTO sessions(principal,kind,capabilities,created_at) VALUES (?,?,?,?)",
                       (principal, kind, _json(sorted(capabilities)), _now()))

    def set_revoked(self, principal) -> None:
        with self._db(True) as db:
            db.execute("UPDATE sessions SET revoked_at=? WHERE principal=? AND revoked_at IS NULL", (_now(), principal))

    # -- approval gate: an agent apply waits here for a person
    @staticmethod
    def _proposal(row) -> dict:
        return {"proposal_id": row["proposal_id"], "session": row["session"], "idem_digest": row["idem_digest"],
                "request_digest": row["request_digest"], "checkpoint_id": row["checkpoint_id"], "recipe_hash": row["recipe_hash"],
                "selections": json.loads(row["selections"]), "request": json.loads(row["request"]), "status": row["status"],
                "created_at": row["created_at"], "decided_at": row["decided_at"], "child_id": row["child_id"]}

    def create_proposal(self, session, idem_digest, request_digest, checkpoint_id, recipe_hash, selections, request) -> dict:
        with self._db(True) as db:
            db.execute("INSERT INTO proposals(proposal_id,session,idem_digest,request_digest,checkpoint_id,recipe_hash,selections,request,"
                       "status,created_at) VALUES (?,?,?,?,?,?,?,?,'pending',?)",
                       ("p_" + uuid.uuid4().hex[:16], session, idem_digest, request_digest, checkpoint_id, recipe_hash,
                        _json(selections), _json(request), _now()))
        return self.find_proposal(session, idem_digest)

    def find_proposal(self, session, idem_digest) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM proposals WHERE session=? AND idem_digest=?", (session, idem_digest)).fetchone()
        return None if row is None else self._proposal(row)

    def get_proposal(self, proposal_id) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM proposals WHERE proposal_id=?", (proposal_id,)).fetchone()
        return None if row is None else self._proposal(row)

    def list_proposals(self, session=None, status=None) -> "list[dict]":
        with self._db() as db:
            rows = db.execute("SELECT * FROM proposals WHERE (? IS NULL OR session=?) AND (? IS NULL OR status=?) ORDER BY created_at, proposal_id",
                              (session, session, status, status)).fetchall()
        return [self._proposal(r) for r in rows]

    def decide_proposal(self, proposal_id, status, child_id=None) -> bool:
        """pending -> approved | rejected, once. False if it was not pending."""
        with self._db(True) as db:
            return db.execute("UPDATE proposals SET status=?, decided_at=?, child_id=? WHERE proposal_id=? AND status='pending'",
                              (status, _now(), child_id, proposal_id)).rowcount == 1

    def get_session(self, principal) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM sessions WHERE principal=?", (principal,)).fetchone()
        return None if row is None else {"principal": row["principal"], "kind": row["kind"],
                                         "capabilities": json.loads(row["capabilities"]), "created_at": row["created_at"]}

    # -- idempotent, quota-charged edits (apply)
    def begin_edit(self, actor, key, request_digest, base_id, recipe_hash, capabilities, *, charge, quota):
        """('new', None) | ('replay', stored result) | ('running', None). Charges once, under one write lock."""
        with self._db(True) as db:
            row = db.execute("SELECT * FROM edits WHERE actor=? AND key=?", (actor, key)).fetchone()
            if row is not None:
                if row["request_digest"] != request_digest:
                    raise ImportFailure("idempotency_conflict", "This key was used for a different request")
                return ("replay", json.loads(row["result"])) if row["status"] == "succeeded" else ("running", None)
            if charge:
                total = 0
                for table in ("jobs", "edits"):
                    try:
                        total += db.execute(f"SELECT coalesce(sum(charged),0) FROM {table} WHERE actor=?", (actor,)).fetchone()[0]
                    except sqlite3.OperationalError:  # no job store in this workspace yet
                        pass
                if total >= quota:
                    raise ImportFailure("quota_exhausted", "Session run quota is exhausted")
            db.execute("INSERT INTO edits(actor,key,request_digest,status,charged,base_id,recipe_hash,capabilities,created_at) "
                       "VALUES (?,?,?,'running',?,?,?,?,?)",
                       (actor, key, request_digest, int(charge), base_id, recipe_hash, _json(sorted(capabilities)), _now()))
        return "new", None

    def finish_edit(self, actor, key, checkpoint_id, result):
        with self._db(True) as db:
            db.execute("UPDATE edits SET status='succeeded', checkpoint_id=?, result=?, finished_at=? WHERE actor=? AND key=?",
                       (checkpoint_id, _json(result), _now(), actor, key))

    def get_edit(self, actor, key):
        """(status, stored result or None, request digest) for an apply key, or None if there is no such attempt."""
        with self._db() as db:
            row = db.execute("SELECT status, result, request_digest FROM edits WHERE actor=? AND key=?",
                             (actor, key)).fetchone()
        return None if row is None else (row["status"], json.loads(row["result"]) if row["result"] else None,
                                         row["request_digest"])

    def abort_edit(self, actor, key):
        """A failed apply is not charged and its key is reusable."""
        with self._db(True) as db:
            db.execute("DELETE FROM edits WHERE actor=? AND key=? AND status='running'", (actor, key))

    # -- import records
    def create_import(self, actor, source, source_name, display_name) -> dict:
        import_id = "imp_" + uuid.uuid4().hex
        with self._db(True) as db:
            db.execute("INSERT INTO imports(id,actor,source,source_name,display_name,status,created_at) "
                       "VALUES (?,?,?,?,?,'queued',?)", (import_id, actor, source, source_name, display_name, _now()))
        return self.get_import(import_id)

    def update_import(self, import_id, **fields):
        names = ", ".join(f"{k}=?" for k in fields)
        with self._db(True) as db:
            db.execute(f"UPDATE imports SET {names} WHERE id=?", (*fields.values(), import_id))

    def transition_import(self, import_id, expect, **fields) -> bool:
        """UPDATE only while the import is still in `expect` (a status or tuple of them): True if it applied."""
        expect = (expect,) if isinstance(expect, str) else tuple(expect)
        names = ", ".join(f"{k}=?" for k in fields)
        marks = ",".join("?" * len(expect))
        with self._db(True) as db:
            cursor = db.execute(f"UPDATE imports SET {names} WHERE id=? AND status IN ({marks})",
                                (*fields.values(), import_id, *expect))
            return cursor.rowcount == 1

    def get_import(self, import_id) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM imports WHERE id=?", (import_id,)).fetchone()
        return None if row is None else _public_import(row)


def register_child(catalog: Catalog, path: Path, digest: str, display_name: str, parent_id: str, recipe_hash: str) -> bool:
    """Catalog an applied child (origin `child`) already stored at checkpoints/<digest>.safetensors."""
    return catalog.insert(digest, display_name, "child", f"checkpoints/{path.name}", _summary(path, catalog.state_schema),
                          parent_id=parent_id, recipe_hash=recipe_hash)


def register_file(catalog: Catalog, path: Path, digest: str, display_name: str, origin="imported") -> bool:
    """Catalog a single file already stored at checkpoints/<digest>.safetensors (legacy upload route)."""
    return catalog.insert(digest, display_name, origin, f"checkpoints/{path.name}", _summary(path, catalog.state_schema))


def _sniff_pickle(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            head = stream.read(8)
    except OSError:
        return False
    return head[:4] == b"PK\x03\x04" or (head[:1] == b"\x80" and len(head) > 2 and head[1] in (2, 3, 4, 5) and any(head[2:8]))


def _failure(exc: TensorFileError) -> ImportFailure:
    return ImportFailure(exc.code, str(exc).split(": ", 1)[-1])


def _validate_source(source: Path, name: str, limits: ImportLimits, staged: bool):
    if name.lower().endswith(PICKLE_SUFFIXES):
        raise ImportFailure("unsupported_format", PICKLE_MESSAGE)
    sharded = name.endswith(INDEX_SUFFIX)
    if not sharded and not name.endswith(".safetensors"):
        raise ImportFailure("unsupported_format", "Expected a .safetensors file or a *.safetensors.index.json")
    if sharded and staged:
        raise ImportFailure("unsupported_format", "Uploads take a single .safetensors file; import shards by path")
    if not source.is_file():
        raise ImportFailure("source_not_found", f"No such file: {name}")
    try:
        table = open_tensors(source, max_file_bytes=limits.max_file_bytes, max_header_bytes=limits.max_header_bytes)
    except TensorFileError as exc:
        if exc.code in ("file_size", "header_size", "header_json") and _sniff_pickle(source):
            raise ImportFailure("unsupported_format", PICKLE_MESSAGE) from exc
        raise _failure(exc) from exc
    if sum(p.stat().st_size for p in table.files) > limits.max_total_bytes:
        raise ImportFailure("total_size", f"The file set exceeds {limits.max_total_bytes} bytes")
    files = [source, *table.files] if sharded else [source]
    if sharded:
        _confine(source, table.files)
    return table, sharded, files, {p: _identity(os.stat(p)) for p in files}


def _identity(stat) -> tuple:
    return stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev


def _confine(index: Path, shards):
    base = index.resolve().parent
    for shard in shards:  # a symlinked shard must not lead out of the index directory
        if shard.resolve().parent != base:
            raise ImportFailure("shard_outside_directory", f"{shard.name} resolves outside the index directory")


def import_checkpoint(catalog: Catalog, source, source_name: str, *, display_name=None,
                      limits: ImportLimits = ImportLimits(), progress=lambda *a, **k: None,
                      staged_sha256: str | None = None, origin="imported", control: ImportControl | None = None):
    """Validate, copy, verify and publish one checkpoint. Returns (catalog entry, already_existed).

    `staged_sha256` is given for an upload that was already streamed into workspace staging (and
    hashed on the way in): that file is consumed instead of copied.
    """
    source = Path(source)
    staged = staged_sha256 is not None
    stage_dir = catalog.workspace / "staging" / f"import-{uuid.uuid4().hex}"
    check = control.check if control else (lambda: None)
    try:
        check()
        progress("validating", 0, 1, "files")
        table, sharded, files, identities = _validate_source(source, source_name, limits, staged)
        check()
        progress("validating", 1, 1, "files", tensor_count=len(table.tensors))
        # copying: hash while copying, so the id is the hash of exactly the bytes we keep
        total = sum(identities[f][0] for f in files)
        stage_dir.mkdir(parents=True)
        done, hashes, kept = 0, {}, {}
        progress("copying", 0, total, "bytes")
        for f in files:
            if staged:
                kept[f.name], hashes[f.name], done = f, staged_sha256, done + f.stat().st_size
                continue
            target, digest = stage_dir / f.name, hashlib.sha256()
            if sharded:
                _confine(source, [f] if f != source else [])  # again at copy time: the tree may have changed
            with f.open("rb") as src, target.open("wb") as dst:
                if _identity(os.fstat(src.fileno())) != identities[f]:  # the file we validated is the file we copy
                    raise ImportFailure("file_changed", f"{f.name} changed after it was validated")
                while chunk := src.read(COPY_CHUNK):
                    check()
                    digest.update(chunk)
                    dst.write(chunk)
                    done += len(chunk)
                    progress("copying", done, total, "bytes")
                dst.flush()
                os.fsync(dst.fileno())
            kept[f.name], hashes[f.name] = target, digest.hexdigest()
        progress("copying", total, total, "bytes")
        # verifying: the copy must parse to the same table as the file we validated
        check()
        progress("verifying", 0, 1, "files")
        copy = kept[source.name]
        try:
            check = open_tensors(copy, max_file_bytes=limits.max_file_bytes, max_header_bytes=limits.max_header_bytes)
        except TensorFileError as exc:
            raise _failure(exc) from exc
        shape = lambda t: {n: (i.dtype, i.shape, i.nbytes) for n, i in t.tensors.items()}  # noqa: E731
        if shape(check) != shape(table):
            raise ImportFailure("file_changed", "The file changed while it was being copied")
        checkpoint_id = (hashes[source.name] if not sharded else
                         hashlib.sha256(FILESET_DOMAIN + _json(sorted([n, h] for n, h in hashes.items())).encode()).hexdigest())
        summary = {"sharded": sharded, "shard_count": len(check.files), "file_bytes": total,
                   "tensor_count": len(check.tensors), "element_count": _elements(check),
                   "compatible": adapter_compatible(check, catalog.state_schema)}
        progress("verifying", 1, 1, "files")
        # publishing: one atomic rename under the catalog write lock; identical bytes are idempotent.
        # From the commit point on a cancel is refused, so a cancelled import never publishes.
        if control:
            control.commit()
        progress("publishing", 0, 1, "files")
        name = (display_name or "").strip() or source_name.removesuffix(INDEX_SUFFIX if sharded else ".safetensors")
        with catalog._db(True) as db:
            if db.execute("SELECT 1 FROM checkpoints WHERE id=?", (checkpoint_id,)).fetchone():
                existing = True
            else:
                existing = False
                final = catalog.workspace / "checkpoints" / (checkpoint_id if sharded else checkpoint_id + ".safetensors")
                if sharded:
                    shutil.rmtree(final, ignore_errors=True)  # an orphan of a crashed publish
                    os.replace(stage_dir, final)
                    location = f"checkpoints/{checkpoint_id}/{source.name}"
                else:
                    os.replace(copy, final)
                    location = f"checkpoints/{checkpoint_id}.safetensors"
                db.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                           (checkpoint_id, name, origin, None, None, "safetensors", int(sharded), summary["shard_count"],
                            summary["file_bytes"], summary["tensor_count"], summary["element_count"],
                            int(summary["compatible"]), location, _now()))
        progress("publishing", 1, 1, "files")
        return catalog.get(checkpoint_id), existing
    except OSError as exc:
        raise ImportFailure("io_error", exc.strerror or str(exc)) from exc
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
        if staged:
            source.unlink(missing_ok=True)


class ImportRunner:
    """Runs imports one at a time and records progress in the catalog.

    A dispatcher thread takes queued imports in order and runs each in its own worker thread. It watches the worker: a
    cancel request or `limits.stall_seconds` without progress (a hung network path blocks inside a read that cannot be
    interrupted) ends the import at once and the dispatcher moves on; the abandoned worker aborts at its next check and
    can no longer publish (see ImportControl).
    """

    def __init__(self, catalog: Catalog, limits: ImportLimits = ImportLimits()):
        self.catalog, self.limits = catalog, limits
        self._queue: queue.Queue = queue.Queue()
        self._controls: dict[str, ImportControl] = {}
        self._staged: dict[str, Path] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._dispatch, name="layerdelta-import", daemon=True)
        self._thread.start()

    def submit(self, actor, source, source_name, display_name, *, staged_sha256=None) -> dict:
        record = self.catalog.create_import(actor, "upload" if staged_sha256 else "path", source_name, display_name)
        import_id = record["import_id"]
        self._controls[import_id] = ImportControl()
        if staged_sha256:
            self._staged[import_id] = Path(source)
        self._queue.put((import_id, Path(source), source_name, display_name, staged_sha256))
        return record

    def cancel(self, import_id) -> dict | None:
        """Cancel a queued or running import. A finished import is returned unchanged; one already past its
        commit point finishes (the cancel is refused)."""
        record = self.catalog.get_import(import_id)
        if record is None or record["status"] not in ("queued", "running"):
            return record
        if record["status"] == "queued" and self.catalog.transition_import(
                import_id, "queued", status="cancelled", finished_at=_now(),
                error=_json({"code": "cancelled", "message": "The import was cancelled before it started"})):
            self._controls.pop(import_id, None)
            staged = self._staged.pop(import_id, None)
            if staged is not None:
                staged.unlink(missing_ok=True)
            return self.catalog.get_import(import_id)
        control = self._controls.get(import_id)
        if control is not None and control.cancel():
            deadline = time.monotonic() + 2  # the dispatcher notices within a fraction of a second
            while time.monotonic() < deadline and self.catalog.get_import(import_id)["status"] == "running":
                time.sleep(.02)
        return self.catalog.get_import(import_id)

    def _dispatch(self):
        while not self._stop.is_set():
            try:
                job = self._queue.get(timeout=.2)
            except queue.Empty:
                continue
            try:
                self._execute(*job)
            except Exception as exc:  # the dispatcher must outlive any single import
                print(f"import dispatcher: {exc}", file=sys.stderr)

    def _execute(self, import_id, source, source_name, display_name, staged_sha256):
        catalog, control = self.catalog, self._controls.get(import_id)
        if control is None or not catalog.transition_import(import_id, "queued", status="running"):
            return  # cancelled while it waited
        worker = threading.Thread(target=self._work, name=f"layerdelta-import-{import_id[-6:]}", daemon=True,
                                  args=(import_id, source, source_name, display_name, staged_sha256, control))
        control.beat()
        worker.start()
        while worker.is_alive() and not self._stop.is_set():
            worker.join(.05)
            if control.cancelled and not control.committed:
                catalog.transition_import(import_id, "running", status="cancelled", finished_at=_now(),
                                          error=_json({"code": "cancelled", "message": "The import was cancelled"}))
                break
            if not control.committed and time.monotonic() - control.heartbeat > self.limits.stall_seconds:
                if control.cancel():  # the abandoned worker will stop at its next check and cannot publish
                    catalog.transition_import(
                        import_id, "running", status="failed", finished_at=_now(),
                        error=_json({"code": "import_stalled", "message": f"No progress for {self.limits.stall_seconds:g} "
                                     "seconds (is the path on a disconnected network share?)"}))
                    break
        else:
            worker.join(5)
        if self._controls.pop(import_id, None) is not None and import_id in self._staged and not worker.is_alive():
            self._staged.pop(import_id, None)
        elif import_id in self._staged and catalog.get_import(import_id)["status"] != "succeeded":
            staged = self._staged.pop(import_id)
            staged.unlink(missing_ok=True)

    def _work(self, import_id, source, source_name, display_name, staged_sha256, control):
        catalog, last = self.catalog, [0.0, None]

        def progress(stage, done, total, unit, tensor_count=None):
            control.beat()
            now = time.monotonic()
            if stage == last[1] and done < total and now - last[0] < 0.25:
                return
            last[:] = [now, stage]
            fields = {"stage": stage, "done": done, "total": total, "unit": unit}
            if tensor_count is not None:
                fields["tensor_count"] = tensor_count
            catalog.transition_import(import_id, "running", **fields)

        try:
            entry, existing = import_checkpoint(catalog, source, source_name, display_name=display_name,
                                                limits=self.limits, progress=progress, staged_sha256=staged_sha256,
                                                control=control)
            catalog.transition_import(import_id, "running", status="succeeded", checkpoint_id=entry["checkpoint_id"],
                                      existing=int(existing), finished_at=_now())
        except ImportFailure as exc:
            status = "cancelled" if exc.code == "cancelled" else "failed"
            catalog.transition_import(import_id, "running", status=status, finished_at=_now(),
                                      error=_json({"code": exc.code, "message": exc.message}))
        except Exception as exc:  # never let a worker thread die silently
            catalog.transition_import(import_id, "running", status="failed", finished_at=_now(),
                                      error=_json({"code": "import_failed", "message": str(exc)[:512]}))

    def close(self):
        self._stop.set()


def main():
    parser = argparse.ArgumentParser(description="Import a .safetensors checkpoint (or *.safetensors.index.json) "
                                                 "into a LayerDelta workspace")
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("path", type=Path)
    parser.add_argument("--name", help="display name")
    parser.add_argument("--max-file-bytes", type=int, default=ImportLimits.max_file_bytes)
    args = parser.parse_args()
    fixture_file = args.workspace / "reference" / "fixture.json"
    schema = json.loads(fixture_file.read_bytes())["model"].get("state_schema") if fixture_file.is_file() else None
    catalog = Catalog(args.workspace, schema)
    catalog.start(clean=False)
    try:
        entry, existing = import_checkpoint(
            catalog, args.path.resolve(), args.path.name, display_name=check_display_name(args.name),
            limits=ImportLimits(max_file_bytes=args.max_file_bytes),
            progress=lambda stage, done, total, unit, **k: print(f"{stage} {done}/{total} {unit}", file=sys.stderr))
    except (ImportFailure, ValueError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        raise SystemExit(2)
    print(json.dumps({**entry, "existing": existing}))


if __name__ == "__main__":
    main()
