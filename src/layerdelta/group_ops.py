"""Group selectors for read_tensor, stats, diff and query (v0.3-G4, contract rev7).

A group selector `{"group": "g_...", "units": [17] | {"start", "stop"}, "members": {"roles": [...], "exclude": [...]}}`
expands deterministically (design 3.4) into one tensor-local selection per member: the member's axis restricted to the union
of `indices(u)` of the chosen units. Results come back per member. The expansion is echoed in `resolved.selector.expanded`,
so a client can replay it with tensor-local selectors and no graph. Nothing here writes, reads outside the table, or runs code.
"""

import math
from dataclasses import dataclass

from . import inspect_ops as ops
from .graphview import GraphView, base_role
from .selector import InspectError, Selection, _axis_ranges, _merge, digest
from .tensors import TensorTable

SELECTOR_KEYS = ("group", "units", "members", "checkpoint")
MEMBER_FILTER_KEYS = ("roles", "exclude")
EXCLUDES = {"norm_stats": lambda basis: basis.startswith("norm.stat@")}


def is_group(selector) -> bool:
    return isinstance(selector, dict) and ("group" in selector or "units" in selector or "members" in selector)


@dataclass
class Member:
    tensor: str
    axis: int
    role: str
    spec: dict          # the group's {offset, block, period, repeat}
    sel: Selection      # every selected unit's slice of this member

    @property
    def header(self) -> dict:
        return {"tensor": self.tensor, "axis": self.axis, "role": self.role}

    def unit_selection(self, unit: int) -> Selection:
        """One unit's slice of this member."""
        return Selection(self.sel.info, tuple(
            _ranges(_indices(self.spec, unit)) if a == self.axis else r for a, r in enumerate(self.sel.axes)))


@dataclass
class GroupSelection:
    group: dict
    units: list[int]
    filter: dict | None
    members: list[Member]
    ir_sha256: str

    @property
    def unit_ranges(self) -> list[dict]:
        return [{"start": s, "stop": e} for s, e in _ranges(self.units)]

    @property
    def canonical(self) -> dict:
        return {"group_id": self.group["group_id"], "ir_sha256": self.ir_sha256,
                "units": [list(r) for r in _ranges(self.units)], "members": self.filter}

    @property
    def selection_id(self) -> str:
        return "sel_" + digest(self.canonical)[:32]

    @property
    def resolved(self) -> dict:
        return {"group": self.group["group_id"], "units": self.unit_ranges, "members": self.filter,
                "expanded": [m.sel.request_form() for m in self.members]}

    @property
    def header(self) -> dict:
        g = self.group
        return {"group_id": g["group_id"], "size": g["size"], "closed": g["closed"], "kind": g["kind"],
                "unit_count": len(self.units)}


def _indices(spec: dict, unit: int):
    return [spec["offset"] + unit * spec["block"] + k * spec["period"] + j
            for k in range(spec["repeat"]) for j in range(spec["block"])]


def _ranges(values) -> tuple[tuple[int, int], ...]:
    return _merge([(v, v + 1) for v in values])


def _filter(raw, field: str) -> dict | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise InspectError("invalid_argument", "members must be an object", field)
    for key in raw:
        if key not in MEMBER_FILTER_KEYS:
            raise InspectError("unknown_field", f"unknown field {key!r}", f"{field}.{key}")
    out = {}
    if "roles" in raw:
        if not isinstance(raw["roles"], list) or not all(isinstance(r, str) for r in raw["roles"]):
            raise InspectError("invalid_argument", "roles must be a list of role names", f"{field}.roles")
        out["roles"] = sorted(set(raw["roles"]))
    if "exclude" in raw:
        if not isinstance(raw["exclude"], list) or any(e not in EXCLUDES for e in raw["exclude"]):
            raise InspectError("invalid_argument", f"exclude takes {sorted(EXCLUDES)}", f"{field}.exclude")
        out["exclude"] = sorted(set(raw["exclude"]))
    return out or None


def parse_group_selector(table: TensorTable, view: GraphView | None, selector, *, field: str = "selector",
                         checkpoint_id: str | None = None) -> GroupSelection:
    if not isinstance(selector, dict):
        raise InspectError("invalid_argument", "selector must be an object", field)
    if view is None:
        raise InspectError("no_graph", "no graph is attached to this checkpoint", f"{field}.group")
    for key in selector:
        if key not in SELECTOR_KEYS:
            raise InspectError("unknown_field", f"unknown selector field {key!r}", f"{field}.{key}")
    gid = selector.get("group")
    if not isinstance(gid, str):
        raise InspectError("invalid_argument", "selector.group must be a group id", f"{field}.group")
    group = view.by_id.get(gid)
    if group is None:
        raise InspectError("unknown_group", f"no group {gid!r} in this graph", f"{field}.group")
    if any(m["offset"] is None for m in group["members"]):
        raise InspectError("non_affine_group", "this group has a member whose indices have no {offset, block, period, repeat} "
                           "form, so units cannot be expanded; use tensor-local selectors", f"{field}.group")
    if "checkpoint" in selector and checkpoint_id is not None and selector["checkpoint"] != checkpoint_id:
        raise InspectError("invalid_argument", "selector.checkpoint differs from the checkpoint of this call",
                           f"{field}.checkpoint")
    if "units" in selector:
        ranges = _axis_ranges(selector["units"], group["size"], f"{field}.units")
    else:
        ranges = ((0, group["size"]),)
    units = [u for s, e in ranges for u in range(s, e)]
    flt = _filter(selector.get("members"), f"{field}.members")
    members = []
    for m in group["members"]:
        role = view.role_entry(m["tensor"], m["axis"])
        if flt and "roles" in flt and role["role"] not in flt["roles"] and base_role(role["role"]) not in flt["roles"]:
            continue
        if flt and any(EXCLUDES[e](role["basis"]) for e in flt.get("exclude", [])):
            continue
        indices = sorted({i for u in units for i in _indices(m, u)})
        info = table.tensors[m["tensor"]]
        axes = tuple(_ranges(indices) if a == m["axis"] else ((0, dim),) for a, dim in enumerate(info.shape))
        members.append(Member(m["tensor"], m["axis"], role["role"], m, Selection(info, axes)))
    if not members:
        raise InspectError("no_members", "the members filter leaves no member of this group", f"{field}.members")
    return GroupSelection(group, units, flt, members, view.ir["ir_sha256"])


# ---------------------------------------------------------------- read_tensor

def read_tensor(table, view, selector, *, max_elems, cursor, include_range, checkpoint_id) -> dict:
    gs = parse_group_selector(table, view, selector, checkpoint_id=checkpoint_id)
    max_elems = ops._arg("max_elems", max_elems, ops.READ_DEFAULT, 1, ops.READ_MAX)
    total = sum(m.sel.total for m in gs.members)
    rid = digest(["read_group", ops._ident(table, checkpoint_id), gs.canonical])
    position = ops._cursor_decode(cursor, "read_tensor", rid, total)
    end, truncated, next_cursor = ops._page("read_tensor", rid, position, max_elems, total)
    ops.budget.charge("values", end - position)  # once for the whole page, before any member is read
    members, at = [], 0
    for m in gs.members:
        lo, hi = max(position, at), min(end, at + m.sel.total)
        if lo < hi:
            local = lo - at
            want = ops._want_range(include_range, local)
            members.append({**m.header, "offset": local, "total_elems": m.sel.total,
                            "result": ops._read_result(table, m.sel, local, hi - at, want)})
        at += m.sel.total
    result = {"group": gs.header, "total_elems": total, "offset": position, "returned_elems": end - position,
              "members": members}
    return ops._envelope({"selector": gs.resolved, "max_elems": max_elems,
                          "include_range": ops._want_range(include_range, position)}, result, checkpoint=checkpoint_id,
                         selection=gs.selection_id, truncated=truncated, cursor=next_cursor)


# ---------------------------------------------------------------- stats

def stats(table, view, selector, *, reduce_over_axes, bins, limit, cursor, checkpoint_id) -> dict:
    gs = parse_group_selector(table, view, selector, checkpoint_id=checkpoint_id)
    if reduce_over_axes is not None:
        raise InspectError("invalid_argument", "a group selector fixes the unit axis; reduce_over_axes is not accepted",
                           "reduce_over_axes")
    if bins not in (None, 0):
        raise InspectError("invalid_argument", "histograms are not available on a group selector", "bins")
    limit = ops._arg("limit", limit, ops.STATS_DEFAULT, 1, ops.STATS_MAX)
    per, total = len(gs.units), len(gs.units) * len(gs.members)
    rid = digest(["stats_group", ops._ident(table, checkpoint_id), gs.canonical])
    position = ops._cursor_decode(cursor, "stats", rid, total)
    end, truncated, next_cursor = ops._page("stats", rid, position, limit, total)
    members = []
    for i in range(position, end):
        m, unit = gs.members[i // per], gs.units[i % per]
        if not members or members[-1]["tensor"] != m.tensor or members[-1]["axis"] != m.axis:
            members.append({**m.header, "dtype": m.sel.info.dtype, "units": []})
        sub = m.unit_selection(unit)
        ops.budget.require_aggregate(sub.total, "selector")  # the elements of one unit of one member
        members[-1]["units"].append({"unit": unit, **ops._unit_stats(table, sub)})
    result = {"group": gs.header, "unit_count": per, "member_count": len(gs.members), "offset": position,
              "accumulate_dtype": ops.ACCUMULATE_DTYPE, "members": members}
    return ops._envelope({"selector": gs.resolved, "limit": limit}, result, checkpoint=checkpoint_id,
                         selection=gs.selection_id, truncated=truncated, cursor=next_cursor)


# ---------------------------------------------------------------- diff

def diff(table_a, table_b, view, selector, *, values, limit, cursor, checkpoint_a, checkpoint_b) -> dict:
    gs = parse_group_selector(table_a, view, selector, checkpoint_id=checkpoint_a)
    if values:
        raise InspectError("invalid_argument", "values need a tensor selector: take the group's `resolved.selector.expanded` "
                           "list", "values")
    limit = ops._arg("limit", limit, ops.DIFF_DEFAULT, 1, ops.DIFF_MAX)
    for m in gs.members:
        other = table_b.tensors.get(m.tensor)
        if other is None or other.dtype != m.sel.info.dtype or other.shape != m.sel.info.shape:
            raise InspectError("incompatible_tensors", f"{m.tensor} is missing or differs in dtype/shape in the other "
                               "checkpoint", "selector.group")
    per, total = len(gs.units), len(gs.units) * len(gs.members)
    ids = (ops._ident(table_a, checkpoint_a), ops._ident(table_b, checkpoint_b))
    rid = digest(["diff_group", ids, gs.canonical])
    position = ops._cursor_decode(cursor, "diff", rid, total)
    end, truncated, next_cursor = ops._page("diff", rid, position, limit, total)
    members = []
    for i in range(position, end):
        m, unit = gs.members[i // per], gs.units[i % per]
        sa = m.unit_selection(unit)
        sb = Selection(table_b.tensors[m.tensor], sa.axes)
        ops.budget.require_aggregate(sa.total, "selector")
        pair = ops._Pair()
        for raw_a, raw_b in ops._zip_closing(ops.iter_raw(table_a, sa), ops.iter_raw(table_b, sb)):
            pair.add(sa.info.dtype, raw_a, raw_b)
        if not members or members[-1]["tensor"] != m.tensor or members[-1]["axis"] != m.axis:
            members.append({**m.header, "dtype": sa.info.dtype, "units": []})
        members[-1]["units"].append({"unit": unit, **pair.result()})
    result = {"accumulate_dtype": ops.ACCUMULATE_DTYPE, "group": gs.header, "unit_count": per,
              "member_count": len(gs.members), "offset": position, "members": members}
    return ops._envelope({"selector": gs.resolved, "unit_axis": None, "values": False, "limit": limit}, result,
                         checkpoint=None, pair=(checkpoint_a, checkpoint_b), selection=gs.selection_id,
                         truncated=truncated, cursor=next_cursor)


# ---------------------------------------------------------------- query

def query(table, view, where, *, limit, cursor, checkpoint_id, max_scan_elems=None) -> dict:
    """Rank the units of a group by a member statistic: the first `out_unit` member, else the first member (and it says which)."""
    for key in where:
        if key not in ("group", "units", "members", "stat", "op", "value"):
            raise InspectError("unknown_field", f"unknown field {key!r}", f"where.{key}")
    if where.get("stat") not in ops.QUERY_STATS:
        raise InspectError("invalid_argument", f"stat must be one of {list(ops.QUERY_STATS)}", "where.stat")
    if where.get("op") not in ops.QUERY_OPS:
        raise InspectError("invalid_argument", f"op must be one of {list(ops.QUERY_OPS)}", "where.op")
    if type(where.get("value")) not in (int, float) or not math.isfinite(where["value"]):
        raise InspectError("invalid_argument", "value must be a finite number", "where.value")
    limit = ops._arg("limit", limit, ops.QUERY_DEFAULT, 1, ops.QUERY_MAX)
    gs = parse_group_selector(table, view, {k: v for k, v in where.items() if k in SELECTOR_KEYS}, field="where",
                              checkpoint_id=checkpoint_id)
    scan_cap = ops._arg("max_scan_elems", max_scan_elems, ops.QUERY_SCAN_DEFAULT, 1, ops.QUERY_SCAN_MAX)
    chosen = next((m for m in gs.members if base_role(m.role) == "out_unit"), gs.members[0])
    if chosen.sel.total > scan_cap:  # the ranking reads the chosen member's whole slice, like a rev6 predicate scan
        raise InspectError("cap_exceeded", f"query would scan {chosen.sel.total} elements (limit {scan_cap}); narrow it "
                           "with units or the members filter", "where")
    why = "first out_unit member" if base_role(chosen.role) == "out_unit" else "no out_unit member: the first member"
    compare = ops.QUERY_OPS[where["op"]]
    hits = []
    too_small = bool(gs.units) and ops.budget.min_aggregate() and chosen.unit_selection(gs.units[0]).total < ops.budget.min_aggregate()
    for unit in ([] if too_small else gs.units):  # a statistic of a unit this small is its elements: not evaluated
        row = ops._unit_stats(table, chosen.unit_selection(unit))
        row["max_abs"] = None if row["min"] is None else max(abs(row["min"]), abs(row["max"]))
        value = row[where["stat"]]
        if value is not None and compare(value, where["value"]):
            hits.append((value, unit))
    hits.sort(key=lambda h: (h[0] if where["op"] in ("lt", "le") else -h[0], h[1]))
    resolved = {"where": {"group": gs.group["group_id"], "units": gs.unit_ranges, "members": gs.filter,
                          "stat": where["stat"], "op": where["op"], "value": where["value"]}, "limit": limit,
                "max_scan_elems": scan_cap}
    rid = digest(["query_group", ops._ident(table, checkpoint_id), gs.canonical, resolved["where"]])
    matches = []
    if hits:
        picked = GroupSelection(gs.group, sorted(u for _, u in hits), gs.filter, gs.members, gs.ir_sha256)
        matches.append({"group": gs.header, "member_used": {**chosen.header, "reason": why},
                        "selector": {"group": gs.group["group_id"], "units": picked.unit_ranges,
                                     **({"members": gs.filter} if gs.filter else {})},
                        "selection_id": picked.selection_id, "units_total": len(hits),
                        "units_truncated": len(hits) > ops.UNIT_LIST_MAX,
                        "units": [{"unit": u, "value": v} for v, u in hits[:ops.UNIT_LIST_MAX]]})
    position = ops._cursor_decode(cursor, "query", rid, len(matches))
    end, truncated, next_cursor = ops._page("query", rid, position, limit, len(matches))
    result = {"matched": len(matches), "offset": position, "matches": matches[position:end]}
    if ops.budget.min_aggregate():
        result.update(skipped_small=[{**chosen.header, "units": len(gs.units)}] if too_small else [], skipped_small_count=int(bool(too_small)))
    return ops._envelope(resolved, result, checkpoint=checkpoint_id, truncated=truncated, cursor=next_cursor)
