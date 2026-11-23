"""The graph, groups and node routes as library calls (v0.3-G4): envelopes over a GraphView, no HTTP.

`graph_view(...)` answers GET .../graph: with nothing attached it returns the attach card (`graph: null`, the provider list and
the layouts' reuse OFFERS); with a graph, the summary plus a page of nodes or of the module tree. `groups(...)` pages the coupling
groups with filters. `node(...)` explains one node: its rule or why it is opaque. All of it is pure and read-only.
"""

from . import inspect_ops as ops
from .graphview import GROUPS_DEFAULT, GROUPS_MAX, MODULES_DEFAULT, MODULES_MAX, NODES_DEFAULT, NODES_MAX, GraphView
from .selector import InspectError, digest
from .tensors import TensorTable


def _no_graph():
    return InspectError("no_graph", "no graph is attached to this checkpoint", "checkpoint")


def graph_view(table: TensorTable, view: GraphView | None, *, offered: dict, view_name: str | None = None,
               module_prefix: str | None = None, op: str | None = None, limit: int | None = None,
               cursor: str | None = None, checkpoint_id: str | None = None) -> dict:
    if view_name not in (None, "nodes", "modules"):
        raise InspectError("invalid_argument", "view must be nodes or modules", "view")
    if view is None:
        if any(v is not None for v in (view_name, module_prefix, op, cursor)):
            raise _no_graph()
        return ops._envelope({"view": None}, {"graph": None, **offered}, checkpoint=checkpoint_id)
    name = view_name or "nodes"
    if name == "modules" and op is not None:
        raise InspectError("invalid_argument", "op filters nodes, not modules", "op")
    default, cap = (MODULES_DEFAULT, MODULES_MAX) if name == "modules" else (NODES_DEFAULT, NODES_MAX)
    limit = ops._arg("limit", limit, default, 1, cap)
    rows = view.modules(module_prefix) if name == "modules" else view.list_nodes(module_prefix, op)
    rid = digest(["graph", ops._ident(table, checkpoint_id), view.ir["ir_sha256"], name, module_prefix, op])
    position = ops._cursor_decode(cursor, "graph", rid, len(rows))
    end, truncated, next_cursor = ops._page("graph", rid, position, limit, len(rows))
    page = rows[position:end]
    ir = view.ir
    result = {"graph": {**view.meta(), **offered}, "coverage": view.coverage(), "warnings": view.derived.warnings,
              "counts": {"nodes": len(ir["nodes"]), "params": len(ir["params"]), "inputs": len(ir["inputs"]),
                         "outputs": len(ir["outputs"]), "groups": len(view.derived.groups)},
              "inputs": ir["inputs"], "outputs": ir["outputs"], "view": name, "matched": len(rows), "offset": position,
              **({"modules": page} if name == "modules" else {"nodes": [view.node_row(n) for n in page]})}
    resolved = {"view": name, "module_prefix": module_prefix, "op": op, "limit": limit}
    return ops._envelope(resolved, result, checkpoint=checkpoint_id, truncated=truncated, cursor=next_cursor)


def groups(table: TensorTable, view: GraphView | None, *, tensor=None, node=None, module_prefix=None, closed=None, kind=None,
           limit: int | None = None, cursor: str | None = None, checkpoint_id: str | None = None) -> dict:
    if view is None:
        raise _no_graph()
    limit = ops._arg("limit", limit, GROUPS_DEFAULT, 1, GROUPS_MAX)
    found = view.list_groups(tensor=tensor, node=node, module_prefix=module_prefix, closed=closed, kind=kind)
    resolved = {"tensor": tensor, "node": node, "module_prefix": module_prefix, "closed": closed, "kind": kind, "limit": limit}
    rid = digest(["groups", ops._ident(table, checkpoint_id), view.ir["ir_sha256"],
                  {k: v for k, v in resolved.items() if k != "limit"}])
    position = ops._cursor_decode(cursor, "groups", rid, len(found))
    end, truncated, next_cursor = ops._page("groups", rid, position, limit, len(found))
    result = {"graph": view.meta(), "matched": len(found), "offset": position,
              "groups": [view.group_record(g) for g in found[position:end]]}
    return ops._envelope(resolved, result, checkpoint=checkpoint_id, truncated=truncated, cursor=next_cursor)


def node(table: TensorTable, view: GraphView | None, node_id: str, *, checkpoint_id: str | None = None) -> dict:
    if view is None:
        raise _no_graph()
    return ops._envelope({"node": node_id}, {"graph": view.meta(), "node": view.node_detail(node_id)}, checkpoint=checkpoint_id)
