"""Minimal session trace (v0.2-T5): one event per request an authenticated principal makes.

An event records WHO (session = principal, capability snapshot), WHAT (route template, tool, request
id, status, error code) and the ids that link a read to a plan to an apply to the child it created:
checkpoint ids, selection ids, recipe hash, child id, run id, idempotency key. It never stores tensor
values, tokens, auth headers, images, request bodies or prompts: routes can only add the whitelisted,
pattern-checked id fields below, so there is no code path that could write a value. Events live in the
catalog's SQLite file and survive restarts. Replays of an idempotent request are marked and linked to
the original, so a retry is neither a second apply nor a second charge.
"""

import json
import re
from contextvars import ContextVar

_HEX64 = re.compile("[0-9a-f]{64}")
_PATTERNS = {
    "checkpoint_id": _HEX64, "other_checkpoint_id": _HEX64, "child_id": _HEX64, "recipe_hash": _HEX64,
    "selection_id": re.compile("sel_[0-9a-f]{32}"), "run_id": re.compile("run_[0-9a-f]{32}"),
    "blob_id": re.compile("sha256:[0-9a-f]{64}"), "error_code": re.compile("[A-Za-z0-9_]{1,64}"),
}
_PLAIN = {"principal", "replayed", "existing", "charged", "ops", "selections"}
ALLOWED = frozenset(_PATTERNS) | _PLAIN
MAX_SELECTIONS = 256

_current: ContextVar = ContextVar("layerdelta_trace", default=None)


def begin():
    """Start collecting the fields of one request. Returns (fields, token for end)."""
    fields: dict = {}
    return fields, _current.set(fields)


def end(token) -> None:
    _current.reset(token)


def note(**fields) -> None:
    """Add id fields to the current request's event. Unknown names are a programming error; ids that do not
    look like ids (user-supplied URL text, say) are dropped instead of stored."""
    target = _current.get()
    if target is None:
        return
    for name, value in fields.items():
        if name not in ALLOWED:
            raise KeyError(f"{name} is not a traceable field")
        if value is None:
            continue
        if name in _PATTERNS:
            if isinstance(value, str) and _PATTERNS[name].fullmatch(value):
                target[name] = value
        elif name == "selections":
            ids = [v for v in value if isinstance(v, str) and _PATTERNS["selection_id"].fullmatch(v)]
            target[name] = ids[:MAX_SELECTIONS]
        elif name == "principal":
            target[name] = str(value)[:80]
        elif name == "ops" or name == "charged":
            target[name] = int(value)
        else:
            target[name] = bool(value)


def clean_key(value) -> str | None:
    """An idempotency key as a printable, bounded string (it is client chosen)."""
    if not isinstance(value, str) or not value or len(value) > 200 or not value.isprintable():
        return None
    return value


def tool_of(method: str, template: str) -> str:
    """Short tool name from a route template: the v2 op name, else the literal path segments."""
    segments = [s for s in template.strip("/").split("/") if s]
    if len(segments) == 3 and segments[0] == "checkpoints" and segments[1].startswith("{"):
        return segments[2].replace("-", "_")
    literal = [s.replace("-", "_") for s in segments if not s.startswith("{")]
    return "_".join(literal) or "root"


def record(catalog, session: str, *, route: str, tool: str, request_id: str, status: int, capabilities,
           idempotency_key: str | None, fields: dict) -> None:
    detail = {k: fields[k] for k in ("selections", "ops", "existing") if k in fields}
    with catalog._db(True) as db:
        seq = db.execute("SELECT coalesce(max(seq),0)+1 FROM events WHERE session=?", (session,)).fetchone()[0]
        db.execute(
            "INSERT INTO events(session,seq,at,route,tool,request_id,status,error_code,checkpoint_id,"
            "other_checkpoint_id,selection_id,recipe_hash,child_id,run_id,idempotency_key,replayed,charged,"
            "capabilities,detail) VALUES (?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now'),?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (session, seq, route, tool, request_id, status, fields.get("error_code"), fields.get("checkpoint_id"),
             fields.get("other_checkpoint_id"), fields.get("selection_id"), fields.get("recipe_hash"),
             fields.get("child_id"), fields.get("run_id"), idempotency_key,
             None if "replayed" not in fields else int(fields["replayed"]), int(fields.get("charged", 0)),
             json.dumps(sorted(capabilities)), json.dumps(detail, sort_keys=True)))


def _row(row) -> dict:
    detail = json.loads(row["detail"])
    return {"id": row["id"], "session": row["session"], "seq": row["seq"], "at": row["at"], "route": row["route"],
            "tool": row["tool"], "request_id": row["request_id"], "status": row["status"],
            "error_code": row["error_code"], "checkpoint_id": row["checkpoint_id"],
            "other_checkpoint_id": row["other_checkpoint_id"], "selection_id": row["selection_id"],
            "recipe_hash": row["recipe_hash"], "child_id": row["child_id"], "run_id": row["run_id"],
            "idempotency_key": row["idempotency_key"], "replayed": None if row["replayed"] is None else bool(row["replayed"]),
            "charged": row["charged"], "capabilities": json.loads(row["capabilities"]),
            "selections": detail.get("selections", []), "ops": detail.get("ops"), "existing": detail.get("existing")}


def timeline(catalog, session: str | None, *, after: int = 0, limit: int = 100) -> dict:
    """Events (oldest first) with replay links and, for every apply, the plan and reads that led to it."""
    with catalog._db() as db:
        where, args = ("WHERE session=?", (session,)) if session else ("", ())
        rows = [_row(r) for r in db.execute(f"SELECT * FROM events {where} ORDER BY id", args)]
    ok = [e for e in rows if e["status"] < 400]
    originals: dict = {}
    for e in ok:  # an idempotent request repeated under one key: the first non-replayed success is the original
        if e["idempotency_key"] is not None:
            slot = (e["session"], e["route"], e["idempotency_key"])
            if slot not in originals or (originals[slot]["replayed"] and not e["replayed"]):
                originals[slot] = e
    for e in rows:
        slot = (e["session"], e["route"], e["idempotency_key"])
        original = originals.get(slot) if e["status"] < 400 and e["idempotency_key"] is not None else None
        e["replay_of"] = original["seq"] if original is not None and original is not e else None
    by_session: dict = {}
    for e in rows:
        by_session.setdefault(e["session"], []).append(e)
    for e in rows:
        if e["tool"] == "apply" and e["status"] < 400:
            earlier = [x for x in by_session[e["session"]] if x["seq"] < e["seq"] and x["status"] < 400]
            e["links"] = {
                "plan_seqs": [x["seq"] for x in earlier if x["tool"] == "plan" and x["recipe_hash"] == e["recipe_hash"]],
                "selection_seqs": [x["seq"] for x in earlier if x["selection_id"] in set(e["selections"])
                                   and x["checkpoint_id"] == e["checkpoint_id"]],
                "child_id": e["child_id"]}
    applies = [e for e in ok if e["tool"] == "apply" and e["replay_of"] is None]
    summary = {"events": len(rows), "applies": len(applies), "replays": sum(1 for e in rows if e["replay_of"] is not None),
               "charged_runs": sum(e["charged"] for e in rows if e["replay_of"] is None)}
    page = [e for e in rows if e["id"] > after][:limit]
    nxt = page[-1]["id"] if len(page) == limit and page[-1]["id"] != rows[-1]["id"] else None
    return {"events": page, "summary": summary, "next_after": nxt}
