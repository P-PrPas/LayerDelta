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
    "proposal_id": re.compile("p_[0-9a-f]{16}"),
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
    """The registry name of the route (the same for HTTP and MCP calls), else the v2 op name, else the literal path segments."""
    from . import registry
    for route in registry.ROUTES:
        if route.tool and route.method == method and route.path == template:
            return route.tool
    segments = [s for s in template.strip("/").split("/") if s]
    if len(segments) == 3 and segments[0] == "checkpoints" and segments[1].startswith("{"):
        return segments[2].replace("-", "_")
    literal = [s.replace("-", "_") for s in segments if not s.startswith("{")]
    return "_".join(literal) or "root"


VIA = ("http", "mcp")
SELECTION_MAX_BYTES = 16 << 10
RECIPE_MAX_BYTES = 256 << 10
TABLE_MAX_BYTES = 64 << 20  # per table: the selections and recipe_store an agent can fill by planning
TABLE_MAX_ROWS = 200_000
_USE: dict = {}             # (catalog file, table) -> [rows, bytes]; counted once per process, then kept up to date


def record(catalog, session: str, *, route: str, tool: str, request_id: str, status: int, capabilities,
           idempotency_key: str | None, fields: dict) -> None:
    detail = {k: fields[k] for k in ("selections", "ops", "existing", "budget_charged", "proposal_id", "control") if k in fields}
    with catalog._db(True, durable=False) as db:
        seq = db.execute("SELECT coalesce(max(seq),0)+1 FROM events WHERE session=?", (session,)).fetchone()[0]
        db.execute(
            "INSERT INTO events(session,seq,at,route,tool,request_id,status,error_code,checkpoint_id,"
            "other_checkpoint_id,selection_id,recipe_hash,child_id,run_id,idempotency_key,replayed,charged,"
            "capabilities,detail,via,duration_ms) VALUES (?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now'),?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (session, seq, route, tool, request_id, status, fields.get("error_code"), fields.get("checkpoint_id"),
             fields.get("other_checkpoint_id"), fields.get("selection_id"), fields.get("recipe_hash"),
             fields.get("child_id"), fields.get("run_id"), idempotency_key,
             None if "replayed" not in fields else int(fields["replayed"]), int(fields.get("charged", 0)),
             json.dumps(sorted(capabilities)), json.dumps(detail, sort_keys=True),
             fields.get("via") if fields.get("via") in VIA else "http", fields.get("duration_ms")))


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
            "budget_charged": detail.get("budget_charged", {}), "via": row["via"], "duration_ms": row["duration_ms"],
            "proposal_id": detail.get("proposal_id"), "control": detail.get("control")}


_ORIGINAL = ("(SELECT o.id FROM events o WHERE o.session=e.session AND o.route=e.route "
             "AND o.idempotency_key=e.idempotency_key AND o.status<400 ORDER BY coalesce(o.replayed,0), o.id LIMIT 1)")


def timeline(catalog, session: str | None, *, after: int = 0, limit: int = 100, limits: dict | None = None,
             with_budget: bool = True) -> dict:
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
            elif event["tool"] == "approve_proposal" and event["proposal_id"]:  # the child of an approved proposal: same links as an apply
                origin = db.execute("SELECT seq, recipe_hash, checkpoint_id, detail FROM events WHERE session=? AND tool='apply' AND status=202 "
                                    "AND json_extract(detail,'$.proposal_id')=? ORDER BY seq LIMIT 1", (event["session"], event["proposal_id"])).fetchone()
                if origin is not None:
                    ids = json.loads(origin["detail"]).get("selections", [])
                    plans = db.execute("SELECT seq FROM events WHERE session=? AND tool='plan' AND recipe_hash=? AND seq<? AND status<400 "
                                       "ORDER BY seq", (event["session"], origin["recipe_hash"], origin["seq"]))
                    reads = db.execute(
                        f"SELECT seq FROM events WHERE session=? AND checkpoint_id=? AND seq<? AND status<400 "
                        f"AND selection_id IN ({','.join('?' * len(ids)) or 'NULL'}) ORDER BY seq",
                        (event["session"], origin["checkpoint_id"], origin["seq"], *ids))
                    event["links"] = {"plan_seqs": [r["seq"] for r in plans], "selection_seqs": [r["seq"] for r in reads],
                                      "child_id": event["control"].get("child_id"), "proposal_seq": origin["seq"]}
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
        summary["errors"] = db.execute(f"SELECT count(*) FROM events e {keyed}e.status>=400", args).fetchone()[0]
        summary["by_status"] = {str(r[0]): r[1] for r in db.execute(
            f"SELECT e.status, count(*) FROM events e {where} GROUP BY e.status ORDER BY e.status", args)}
        summary["via"] = {r[0]: r[1] for r in db.execute(f"SELECT e.via, count(*) FROM events e {where} GROUP BY e.via", args)}
        last = db.execute(f"SELECT e.at, e.status FROM events e {where} ORDER BY e.id DESC LIMIT 1", args).fetchone()
        summary["last_at"], summary["last_status"] = (last[0], last[1]) if last else (None, None)
        if session and with_budget:  # the numbers a budget meter needs, from what each event charged; limits are the live session's, if it is one
            used = {dim: db.execute("SELECT coalesce(sum(json_extract(detail, ?)),0) FROM events WHERE session=?",
                                    (f"$.budget_charged.{dim}", session)).fetchone()[0] for dim in ("values", "scan", "blob_bytes")}
            used["applies"] = summary["charged_runs"]
            summary["budget"] = {dim: {"limit": (limits or {}).get(dim), "used": n,
                                       "remaining": None if (limits or {}).get(dim) is None else max(0, limits[dim] - n)}
                                 for dim, n in used.items()}
    return {"events": page, "summary": summary, "next_after": page[-1]["id"] if more else None}


CONTROL_ACTIONS = ("revoke", "budget_change", "approve", "reject")


def record_control(catalog, session: str, action: str, *, charged: int = 0, **control) -> None:
    """A human's supervision act, written into the AFFECTED session's timeline (`principal: human`). Ids and numbers only.
    `charged` is the apply quota the act spent on that session (an approval makes the agent's apply): the session's totals add it up."""
    import uuid
    if action not in CONTROL_ACTIONS:
        raise KeyError(action)
    clean: dict = {"principal": "human", "action": action}
    for name, value in control.items():
        if name in ("from", "to") and (value is None or (type(value) is int and value >= 0)):
            clean[name] = value
        elif name == "dimension" and value in ("values", "scan", "applies", "blob_bytes"):
            clean[name] = value
        elif name in ("proposal_id", "child_id", "recipe_hash") and isinstance(value, str) and _PATTERNS[name].fullmatch(value):
            clean[name] = value
        else:
            raise KeyError(f"{name} is not a control field")
    tool = {"revoke": "revoke_session", "budget_change": "change_budget", "approve": "approve_proposal", "reject": "reject_proposal"}[action]
    route = {"revoke": "POST /agent-sessions/{session}/revoke", "budget_change": "POST /agent-sessions/{session}/budget",
             "approve": "POST /proposals/{proposal_id}/approve", "reject": "POST /proposals/{proposal_id}/reject"}[action]
    record(catalog, session, route=route, tool=tool, request_id=uuid.uuid4().hex, status=200, capabilities=[], idempotency_key=None,
           fields={"control": clean, "charged": int(bool(charged)), **({"proposal_id": clean["proposal_id"]} if "proposal_id" in clean else {})})


def proposal_seq(catalog, session: str, proposal_id: str) -> int | None:
    """The seq of the agent apply that made the proposal."""
    with catalog._db() as db:
        row = db.execute("SELECT seq FROM events WHERE session=? AND tool='apply' AND status=202 AND json_extract(detail,'$.proposal_id')=? "
                         "ORDER BY seq LIMIT 1", (session, proposal_id)).fetchone()
    return None if row is None else row["seq"]


def egress_class(capabilities) -> int:
    return 2 if "weights:values" in capabilities else 1 if "weights:read" in capabilities else 0


def sessions(catalog) -> list[dict]:
    """One row per session the catalog knows (agent and human), for the supervision list."""
    with catalog._db() as db:
        rows = db.execute("SELECT principal, kind, capabilities, created_at, revoked_at FROM sessions ORDER BY created_at, principal").fetchall()
    out = []
    for row in rows:
        summary = timeline(catalog, row["principal"], limit=1, with_budget=False)["summary"]
        caps = json.loads(row["capabilities"])
        out.append({"session": row["principal"], "kind": row["kind"], "created_at": row["created_at"], "capabilities": caps,
                    "egress_class": egress_class(caps), "events": summary["events"], "applies": summary["applies"],
                    "charged_runs": summary["charged_runs"], "errors": summary["errors"], "last_at": summary["last_at"],
                    "last_status": summary["last_status"], "via": summary["via"], "via_mix": len(summary["via"]) > 1,
                    "revoked": row["revoked_at"] is not None, "revoked_at": row["revoked_at"]})
    return out


def _use(catalog, table: str, column: str) -> list:
    key = (str(catalog.db_path), table)
    if key not in _USE:
        with catalog._db() as db:
            rows, size = db.execute(f"SELECT count(*), coalesce(sum(length({column})),0) FROM {table}").fetchone()
        _USE[key] = [rows, size]
    return _USE[key]


def _put(catalog, table: str, column: str, sql: str, args: tuple, size: int, *, force: bool = False) -> bool:
    """Insert one row unless the table is full (`force`: an applied recipe, bounded by the apply quota, is always kept).
    Remembering is best effort: it never fails the request that produced the row. True if a row was written."""
    use = _use(catalog, table, column)
    if not force and (use[0] >= TABLE_MAX_ROWS or use[1] + size > TABLE_MAX_BYTES):
        return False
    try:
        with catalog._db(True, durable=False) as db:
            if db.execute(sql, args).rowcount == 1:
                use[0] += 1
                use[1] += size
                return True
    except Exception as exc:
        import sys
        print(f"trace: could not remember: {exc}", file=sys.stderr)
    return False


def remember_full(catalog) -> dict:
    """Which tables have stopped taking planned recipes and selections (agents cannot grow them without bound)."""
    return {table: (lambda u: u[0] >= TABLE_MAX_ROWS or u[1] >= TABLE_MAX_BYTES)(_use(catalog, table, column))
            for table, column in (("selections", "selector"), ("recipe_store", "recipe"))}


def remember_selection(catalog, checkpoint_id: str, selection_id: str, selector) -> None:
    """Keep the canonical selector of a selection id (human-only table: selectors name tensors, so they never enter the trace)."""
    text = json.dumps(selector, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if (isinstance(checkpoint_id, str) and _PATTERNS["checkpoint_id"].fullmatch(checkpoint_id) and isinstance(selection_id, str)
            and _PATTERNS["selection_id"].fullmatch(selection_id) and len(text) <= SELECTION_MAX_BYTES):
        _put(catalog, "selections", "selector", "INSERT OR IGNORE INTO selections(checkpoint_id,selection_id,selector,first_at) "
             "VALUES (?,?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now'))", (checkpoint_id, selection_id, text), len(text))


def remember_recipe(catalog, recipe: dict, *, applied: bool = False) -> None:
    text = json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if _PATTERNS["recipe_hash"].fullmatch(str(recipe.get("recipe_hash"))) and len(text) <= RECIPE_MAX_BYTES:
        _put(catalog, "recipe_store", "recipe", "INSERT OR IGNORE INTO recipe_store(recipe_hash,recipe,first_at) "
             "VALUES (?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now'))", (recipe["recipe_hash"], text), len(text), force=applied)


def resolve_selections(catalog, ids: list[str]) -> dict:
    found: dict[str, list] = {}
    with catalog._db() as db:
        for sid in ids:
            rows = db.execute("SELECT checkpoint_id, selector FROM selections WHERE selection_id=? ORDER BY first_at, checkpoint_id",
                              (sid,)).fetchall()
            if rows:
                found[sid] = [{"checkpoint_id": r["checkpoint_id"], "selector": json.loads(r["selector"])} for r in rows]
    return found


def get_recipe(catalog, recipe_hash: str) -> dict | None:
    with catalog._db() as db:
        row = db.execute("SELECT recipe FROM recipe_store WHERE recipe_hash=?", (recipe_hash,)).fetchone()
    return None if row is None else json.loads(row["recipe"])
