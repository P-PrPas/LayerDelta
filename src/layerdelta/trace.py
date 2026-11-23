"""Minimal session trace (v0.2-T5): one event per request an authenticated principal makes.

An event records WHO (session = principal, capability snapshot), WHAT (route template, tool, request
id, status, error code) and the ids that link a read to a plan to an apply to the child it created:
checkpoint ids, selection ids, recipe hash, child id, run id, idempotency key. It never stores tensor
values, tokens, auth headers, images, request bodies or prompts: routes can only add the whitelisted,
pattern-checked id fields below, so there is no code path that could write a value. Events live in the
catalog's SQLite file and survive restarts. Replays of an idempotent request are marked and linked to
the original, so a retry is neither a second apply nor a second charge.
"""

import hashlib
import json
import re
from contextvars import ContextVar

_HEX64 = re.compile("[0-9a-f]{64}")
_PATTERNS = {
    "checkpoint_id": _HEX64, "other_checkpoint_id": _HEX64, "child_id": _HEX64, "recipe_hash": _HEX64,
    "selection_id": re.compile("sel_[0-9a-f]{32}"), "run_id": re.compile("run_[0-9a-f]{32}"),
    "blob_id": re.compile("sha256:[0-9a-f]{64}"), "error_code": re.compile("[A-Za-z0-9_]{1,64}"),
}
_PLAIN = {"principal", "replayed", "existing", "charged", "ops", "selections"}  # budget_charged is written by the middleware
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


def key_digest(value) -> str | None:
    """A client-chosen idempotency key as sha256(key)[:32]: enough to match a retry to its original, and a key
    that holds a secret, a prompt or a value never reaches the database."""
    if not isinstance(value, str) or not value or len(value) > 200 or not value.isprintable():
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


clean_key = key_digest  # the name the middleware used first


def tool_of(method: str, template: str) -> str:
    """Short tool name from a route template: the v2 op name, else the literal path segments."""
    segments = [s for s in template.strip("/").split("/") if s]
    if len(segments) == 3 and segments[0] == "checkpoints" and segments[1].startswith("{"):
        return segments[2].replace("-", "_")
    literal = [s.replace("-", "_") for s in segments if not s.startswith("{")]
    return "_".join(literal) or "root"


def record(catalog, session: str, *, route: str, tool: str, request_id: str, status: int, capabilities,
           idempotency_key: str | None, fields: dict) -> None:
    detail = {k: fields[k] for k in ("selections", "ops", "existing", "budget_charged") if k in fields}
    with catalog._db(True, durable=False) as db:
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
            "selections": detail.get("selections", []), "ops": detail.get("ops"), "existing": detail.get("existing"),
            "budget_charged": detail.get("budget_charged", {})}


_ORIGINAL = ("(SELECT o.id FROM events o WHERE o.session=e.session AND o.route=e.route "
             "AND o.idempotency_key=e.idempotency_key AND o.status<400 ORDER BY coalesce(o.replayed,0), o.id LIMIT 1)")


def timeline(catalog, session: str | None, *, after: int = 0, limit: int = 100) -> dict:
    """One page of events (oldest first) plus a summary, all from indexed queries: the cost is the page, not the log.

    Replay links: an idempotent request repeated under one key points (`replay_of`) at the original, the first
    non-replayed success of that (session, route, key). For every successful apply, `links` names the earlier plan
    events with its recipe hash and the earlier reads of its input selections.
    """
    scope, args = ("e.session=? AND ", [session]) if session else ("", [])
    with catalog._db() as db:
        rows = db.execute(f"SELECT e.*, (SELECT x.seq FROM events x WHERE x.id = CASE WHEN e.status<400 "
                          f"AND e.idempotency_key IS NOT NULL THEN {_ORIGINAL} END AND x.id != e.id) AS replay_of "
                          f"FROM events e WHERE {scope}e.id>? ORDER BY e.id LIMIT ?", (*args, after, limit + 1)).fetchall()
        more = len(rows) > limit
        page = []
        for row in rows[:limit]:
            event = _row(row)
            event["replay_of"] = row["replay_of"]
            if event["tool"] == "apply" and event["status"] < 400:
                plans = db.execute("SELECT seq FROM events WHERE session=? AND tool='plan' AND recipe_hash=? AND seq<? "
                                   "AND status<400 ORDER BY seq", (event["session"], event["recipe_hash"], event["seq"]))
                ids = event["selections"]
                reads = db.execute(
                    f"SELECT seq FROM events WHERE session=? AND checkpoint_id=? AND seq<? AND status<400 "
                    f"AND selection_id IN ({','.join('?' * len(ids)) or 'NULL'}) ORDER BY seq",
                    (event["session"], event["checkpoint_id"], event["seq"], *ids))
                event["links"] = {"plan_seqs": [r["seq"] for r in plans], "selection_seqs": [r["seq"] for r in reads],
                                  "child_id": event["child_id"]}
            page.append(event)
        where = f"WHERE {scope[:-5]}" if session else ""
        keyed = f"{'WHERE e.session=? AND ' if session else 'WHERE '}"
        summary = {
            "events": db.execute(f"SELECT count(*) FROM events e {where}", args).fetchone()[0],
            "applies": db.execute(
                f"SELECT count(*) FROM events e {keyed}e.tool='apply' AND e.status<400 AND "
                f"(e.idempotency_key IS NULL OR e.id = {_ORIGINAL})", args).fetchone()[0],
            "replays": db.execute(
                f"SELECT count(*) FROM events e {keyed}e.status<400 AND e.idempotency_key IS NOT NULL "
                f"AND e.id != {_ORIGINAL}", args).fetchone()[0],
            "charged_runs": db.execute(f"SELECT coalesce(sum(e.charged),0) FROM events e {where}", args).fetchone()[0]}
    return {"events": page, "summary": summary, "next_after": page[-1]["id"] if more else None}
