"""A checkpoint's attached graph as the inspect-ops layer sees it (v0.3-G4, contract rev7).

`GraphView` pairs a verified GraphIR with what `coupling.derive` makes of it for ONE tensor table: roles, groups, coverage.
Everything here is pure and generic: no tensor name, no model family, no HTTP. The functions return plain dicts that the
envelope helpers of inspect_ops wrap. With no view, nothing here is called and every rev6 response is unchanged.
"""

import fnmatch
import math
from dataclasses import dataclass, field

from . import coupling
from .graph import bind, table_signature
from .selector import InspectError, digest
from .tensors import TensorTable

SPANS_MAX = 200           # activation spans echoed per group; `spans_total` says how many there are
NODES_DEFAULT, NODES_MAX = 500, 2000
GROUPS_DEFAULT, GROUPS_MAX = 100, 500
MODULES_DEFAULT, MODULES_MAX = 500, 2000

KIND_WORDS = {"channel": "channels", "head": "heads", "qk_dim": "query/key dimensions", "v_dim": "value dimensions",
              "hidden": "hidden units", "position": "positions", "vocab": "vocabulary entries"}
BOUNDARY_NOTES = {
    "graph_output": "These units reach a model output: permuting them reorders the output, so they cannot be permuted.",
    "graph_input": "These units are an input's: permuting them changes the function.",
    "opaque_op": "An operation without a coupling rule touches these units, so their coupling is not known.",
    "data_dimension": "These units are tied to a data dimension (positions, frequency bins), not to learned units.",
    "indexed_unit": "One unit of this dimension is picked by index, so the order of units matters.",
    "recurrent_state_used": "The final recurrent state is used, so the order of hidden units is observable.",
    "recurrent_initial_state": "The initial recurrent state is not zeros, so the order of hidden units is observable.",
    "non_affine_member": "A member's indices do not form a regular pattern, so the group cannot be described exactly.",
    "overlapping_group": "Another group cuts the same tensor axis into the same units.",
}


def base_role(role: str) -> str:
    """A packed axis (stacked projections, recurrent gates) is an out_unit axis with extra structure."""
    return "out_unit" if role == "packed" else role


@dataclass
class GraphView:
    ir: dict
    derived: coupling.Derived
    table: TensorTable
    record: dict                      # the graph store row: provider, created_at, ...
    bound: object = None              # graph.BoundGraph
    by_id: dict = field(default_factory=dict)
    by_tensor: dict = field(default_factory=dict)   # tensor -> [(axis, group)]
    nodes: dict = field(default_factory=dict)
    offered: dict = field(default_factory=dict)     # {"providers": [...], "reusable": [...]} for the attach card

    @classmethod
    def build(cls, ir: dict, table: TensorTable, record: dict, derived: coupling.Derived | None = None) -> "GraphView":
        view = cls(ir, derived or coupling.derive(ir, table), table, record)
        view.bound = bind(ir, table)
        view.by_id = {g["group_id"]: g for g in view.derived.groups}
        for g in view.derived.groups:
            for m in g["members"]:
                view.by_tensor.setdefault(m["tensor"], []).append((m["axis"], g))
        view.nodes = {n["id"]: n for n in ir["nodes"]}
        return view

    # -------------------------------------------------------------- summaries
    def meta(self) -> dict:
        return {"ir_sha256": self.ir["ir_sha256"], "provider": self.record["provider"],
                "table_signature": table_signature(self.table), "attached_at": self.record["created_at"], **self.offered}

    def coverage(self) -> dict:
        return self.derived.coverage

    def role_entry(self, tensor: str, axis: int) -> dict:
        return self.derived.roles[tensor][axis]

    def group_record(self, group: dict) -> dict:
        """The public form of a group: members with their roles, boundary, label, kind, notes, capped spans."""
        members = [{**m, "role": self.role_entry(m["tensor"], m["axis"])["role"]} for m in group["members"]]
        spans = group["spans"]
        return {"group_id": group["group_id"], "size": group["size"], "closed": group["closed"], "kind": group["kind"],
                "label": self.label(group, members), "boundary": group["boundary"], "constraints": group["constraints"],
                "members": members, "notes": self.notes(group), "spans": spans[:SPANS_MAX], "spans_total": len(spans),
                "spans_truncated": len(spans) > SPANS_MAX}

    def label(self, group: dict, members: list[dict]) -> str:
        """`<module path> <kind>`, from the first member that produces the units (else the first member)."""
        chosen = next((m for m in members if base_role(m["role"]) == "out_unit"), members[0])
        module = chosen["tensor"].rsplit(".", 1)[0] if "." in chosen["tensor"] else chosen["tensor"]
        return f"{module} {KIND_WORDS.get(group['kind'], group['kind'])}"

    def notes(self, group: dict) -> list[str]:
        out = []
        for reason in sorted({b["reason"] for b in group["boundary"]}):
            if reason in BOUNDARY_NOTES:
                out.append(BOUNDARY_NOTES[reason])
        if any(self.role_entry(m["tensor"], m["axis"])["basis"].startswith("norm.stat@") for m in group["members"]):
            out.append("An edit through this group also changes the running statistics of its normalisation layers; "
                       "exclude them with the members filter {\"exclude\": [\"norm_stats\"]}.")
        if group["constraints"].get("kind") == "within_blocks":
            out.append(f"Units must stay inside blocks of {group['constraints']['n']} (group normalisation).")
        return out

    # -------------------------------------------------------------- describe
    def describe_row(self, name: str) -> dict:
        """What a describe row gains when a graph is attached."""
        groups = sorted(({"group_id": g["group_id"], "axis": axis} for axis, g in self.by_tensor.get(name, [])),
                        key=lambda e: (e["axis"], e["group_id"]))
        return {"roles": self.derived.roles[name], "binding": self.bound.tensors[name].status, "groups": groups}

    # -------------------------------------------------------------- groups
    def list_groups(self, *, tensor=None, node=None, module_prefix=None, closed=None, kind=None) -> list[dict]:
        for field_, value in (("tensor", tensor), ("node", node), ("module_prefix", module_prefix), ("kind", kind)):
            if value is not None and not isinstance(value, str):
                raise InspectError("invalid_argument", f"{field_} must be a string", field_)
        if closed is not None and type(closed) is not bool:
            raise InspectError("invalid_argument", "closed must be true or false", "closed")
        if node is not None and node not in self.nodes:
            raise InspectError("unknown_node", f"no node {node!r} in this graph", "node")
        touching = None
        if node is not None:
            n = self.nodes[node]
            values = {o["value"] for o in n["outputs"]} | _refs(n["args"]) | _refs(list(n["kwargs"].values()))
            touching = values
        out = []
        for g in self.derived.groups:
            if closed is not None and g["closed"] != closed:
                continue
            if kind is not None and g["kind"] != kind:
                continue
            if tensor is not None and not any(m["tensor"] == tensor for m in g["members"]):
                continue
            if module_prefix is not None and not any(_under(m["tensor"], module_prefix) for m in g["members"]):
                continue
            if touching is not None and not (any(b["node"] == node for b in g["boundary"])
                                             or any(s["value"] in touching for s in g["spans"])
                                             or any(self._param_used_by(m["tensor"], node) for m in g["members"])):
                continue
            out.append(g)
        return out

    def _param_used_by(self, tensor: str, node: str) -> bool:
        info = self.bound.tensors.get(tensor)
        return bool(info and any(u["node"] == node for u in info.uses))

    # -------------------------------------------------------------- nodes and modules
    def node_row(self, n: dict) -> dict:
        rule = self.derived.node_rules.get(n["id"], {"rule": None, "opaque": None})
        return {"id": n["id"], "op": n["op"], "module_path": n["module_path"], "module_type": n["module_type"],
                "outputs": [{"value": o["value"], "kind": o["kind"], **({"shape": o["shape"]} if "shape" in o else {})}
                            for o in n["outputs"]], "rule": rule["rule"], "opaque_reason": rule["opaque"]}

    def node_detail(self, node_id: str) -> dict:
        n = self.nodes.get(node_id)
        if n is None:
            raise InspectError("unknown_node", f"no node {node_id!r} in this graph", "node_id")
        params = sorted(t for t, info in self.bound.tensors.items() if any(u["node"] == node_id for u in info.uses))
        groups = sorted(g["group_id"] for g in self.derived.groups
                        if any(b["node"] == node_id for b in g["boundary"]) or any(m["tensor"] in params for m in g["members"]))
        return {**self.node_row(n), "args": n["args"], "kwargs": n["kwargs"], "parameters": params, "groups": groups}

    def list_nodes(self, module_prefix: str | None = None, op: str | None = None) -> list[dict]:
        for field_, value in (("module_prefix", module_prefix), ("op", op)):
            if value is not None and not isinstance(value, str):
                raise InspectError("invalid_argument", f"{field_} must be a string", field_)
        return [n for n in self.ir["nodes"]
                if (module_prefix is None or _under(n["module_path"] or "", module_prefix))
                and (op is None or fnmatch.fnmatchcase(n["op"], op))]

    def modules(self, module_prefix: str | None = None) -> list[dict]:
        """The module tree: one row per module path seen on a node, with counts over its whole subtree."""
        paths: dict[str, dict] = {}

        def row(path):
            return paths.setdefault(path, {"path": path, "parent": None, "type": None, "node_count": 0, "tensor_count": 0,
                                           "axes": {"derived": 0, "declared": 0, "unknown": 0}, "opaque_count": 0,
                                           "open_groups": 0})
        for n in self.ir["nodes"]:
            path = n["module_path"]
            if path is None:
                continue
            opaque = self.derived.node_rules.get(n["id"], {}).get("opaque") is not None
            for prefix in _prefixes(path):
                r = row(prefix)
                r["node_count"] += 1
                r["opaque_count"] += opaque
            row(path)["type"] = row(path)["type"] or n["module_type"]
        owner = {}
        for name, info in self.bound.tensors.items():
            use = next((u for u in info.uses if u["module_path"] is not None), None)
            owner[name] = use["module_path"] if use else (name.rsplit(".", 1)[0] if "." in name else "")
        for name, path in owner.items():
            for prefix in _prefixes(path):
                if prefix not in paths:
                    continue
                r = paths[prefix]
                r["tensor_count"] += 1
                for role in self.derived.roles[name]:
                    r["axes"]["derived" if role["provenance"] == "derived" else "unknown"] += 1
        for g in self.derived.groups:
            if g["closed"]:
                continue
            seen = set()
            for m in g["members"]:
                for prefix in _prefixes(owner.get(m["tensor"], "")):
                    if prefix in paths and prefix not in seen:
                        seen.add(prefix)
                        paths[prefix]["open_groups"] += 1
        for path, r in paths.items():
            r["parent"] = path.rsplit(".", 1)[0] if "." in path else ("" if path else None)
        rows = [paths[p] for p in sorted(paths) if module_prefix is None or _under(p, module_prefix)]
        return rows


def _prefixes(path: str):
    if path is None:
        return
    parts = path.split(".") if path else []
    yield ""
    for i in range(1, len(parts) + 1):
        yield ".".join(parts[:i])


def _under(path: str, prefix: str) -> bool:
    return prefix == "" or path == prefix or path.startswith(prefix + ".")


def _refs(value) -> set:
    out = set()
    if isinstance(value, str):
        out.add(value)
    elif isinstance(value, list):
        for v in value:
            out |= _refs(v)
    return out


def view_digest(view: GraphView) -> str:
    return digest([view.ir["ir_sha256"], table_signature(view.table)])


def provider_list(registered: list[str], user_code: bool = False) -> list[dict]:
    """The attach card's provider list. `builtin` always; `torch_export` only when the project opted in to running user code."""
    return [{"name": "builtin", "enabled": True, "reason": None, "models": registered},
            {"name": "torch_export", "enabled": user_code,
             "reason": None if user_code else "Running your model code is off for this project; start the server with --allow-user-code to turn it on."},
            {"name": "onnx", "enabled": False, "reason": "Not implemented yet."}]
