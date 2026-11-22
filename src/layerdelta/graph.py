"""GraphIR (ir_version 1), parameter binding and the graph store (v0.3-G2).

Everything here is generic and torch-free: a provider (see providers.py) turns a model into a JSON
graph, and from then on only this JSON is used. `bind` joins a graph to a tensor table by name and
shape and says, for every tensor, whether the graph uses it (`bound`), carries it without using it
(`unused`, e.g. a BatchNorm counter) or does not know it (`unbound`, e.g. a parameter forward never
touches). The store keeps graphs keyed by a signature of the table (names, dtypes, shapes), so every
checkpoint with the same layout, such as each child written by `apply`, finds the graph its parent had.
No roles or coupling yet (G3).
"""

import json
import re
from dataclasses import dataclass, field

from .selector import InspectError, canonical_json, digest
from .tensors import DTYPES, TensorTable

IR_VERSION = 1
_VALUE = re.compile(r"[vp][0-9]+")
_NODE = re.compile(r"n[0-9]+")
_FLOATS = frozenset({"F16", "BF16", "F32", "F64", "F8_E4M3", "F8_E5M2"})
_INTS = frozenset({"I8", "U8", "I16", "U16", "I32", "U32", "I64", "U64"})


class GraphMismatch(InspectError):
    """A graph that does not fit the checkpoint it is attached to; `names` lists the offending tensors."""

    def __init__(self, problems: list[tuple[str, str]]):
        self.problems, self.names = problems, sorted(n for n, _ in problems)
        listed = ", ".join(f"{n} ({why})" for n, why in sorted(problems)[:20])
        super().__init__("graph_mismatch", f"the graph does not fit this checkpoint: {listed}", "graph")


# ---------------------------------------------------------------- the IR value

def finalize_ir(ir: dict) -> dict:
    """Add `ir_sha256` (sha256 of the canonical JSON of everything else) and validate."""
    body = {k: v for k, v in ir.items() if k != "ir_sha256"}
    try:
        sealed = {**body, "ir_sha256": digest(body)}
    except (ValueError, TypeError) as exc:  # NaN/Inf or a value JSON cannot hold
        raise InspectError("invalid_graph", f"the graph is not canonical JSON: {exc}", "graph") from exc
    verify_ir(sealed)
    return sealed


def ir_json(ir: dict) -> str:
    return canonical_json(ir)


def _bad(why: str, field_: str = "graph"):
    raise InspectError("invalid_graph", why, field_)


def _arg(value, known: set, where: str):
    """Check one encoded argument: bare strings are value references, everything else is a literal or a wrapper."""
    if isinstance(value, str):
        if value not in known:
            _bad(f"{where} refers to {value!r}, which is not defined before it")
    elif isinstance(value, list):
        for item in value:
            _arg(item, known, where)
    elif isinstance(value, dict):
        keys = set(value)
        ok = (keys == {"str"} and isinstance(value["str"], str)) or (keys == {"enum"} and isinstance(value["enum"], str)) \
            or (keys == {"float"} and value["float"] in ("inf", "-inf", "nan"))
        if not ok:
            _bad(f"{where} has an unknown literal wrapper {sorted(keys)}")
    elif value is not None and type(value) not in (bool, int, float):
        _bad(f"{where} has an unsupported literal of type {type(value).__name__}")
    elif isinstance(value, float) and value != value:
        _bad(f"{where} has a NaN literal (use the float wrapper)")


def _shape(value, where):
    if not isinstance(value, list) or any(type(n) is not int or n < 0 for n in value):
        _bad(f"{where}: shape must be a list of non-negative integers")


def _dtype(value, where):
    if value not in DTYPES:
        _bad(f"{where}: unknown dtype {value!r}")


def verify_ir(ir) -> dict:
    """Strict schema, reference and hash check. Raises InspectError('invalid_graph' | 'unsupported_ir_version' | 'ir_hash_mismatch')."""
    keys = {"ir_version", "provider", "inputs", "params", "ties", "nodes", "outputs", "ir_sha256"}
    if not isinstance(ir, dict) or ir.get("ir_version") != IR_VERSION:
        raise InspectError("unsupported_ir_version", f"expected a GraphIR with ir_version {IR_VERSION}", "graph.ir_version")
    if set(ir) != keys:
        _bad(f"fields must be exactly {sorted(keys)}")
    provider = ir["provider"]
    if not isinstance(provider, dict) or set(provider) != {"name", "version", "spec_sha256"} \
            or not all(isinstance(v, str) and v for v in provider.values()):
        _bad("provider must be {name, version, spec_sha256}")
    known: set = set()
    for i, item in enumerate(ir["inputs"]):
        where = f"inputs[{i}]"
        if not isinstance(item, dict) or set(item) != {"value", "shape", "dtype"} or not _VALUE.fullmatch(str(item["value"])):
            _bad(f"{where} must be {{value, shape, dtype}}")
        _shape(item["shape"], where)
        _dtype(item["dtype"], where)
        if item["value"] in known:
            _bad(f"{where}: duplicate value {item['value']}")
        known.add(item["value"])
    names = set()
    for i, item in enumerate(ir["params"]):
        where = f"params[{i}]"
        if not isinstance(item, dict) or set(item) != {"name", "value", "kind", "shape", "dtype"} \
                or not _VALUE.fullmatch(str(item["value"])) or item["kind"] not in ("parameter", "buffer") \
                or not isinstance(item["name"], str) or not item["name"]:
            _bad(f"{where} must be {{name, value, kind, shape, dtype}}")
        _shape(item["shape"], where)
        _dtype(item["dtype"], where)
        if item["value"] in known or item["name"] in names:
            _bad(f"{where}: duplicate parameter {item['name']}")
        known.add(item["value"])
        names.add(item["name"])
    ties = ir["ties"]
    if not isinstance(ties, list) or any(not isinstance(g, list) or len(g) < 2 or g != sorted(set(g))
                                         or any(not isinstance(n, str) for n in g) for g in ties) or ties != sorted(ties):
        _bad("ties must be a sorted list of sorted groups of at least two distinct names")
    seen_nodes: set = set()
    for i, node in enumerate(ir["nodes"]):
        where = f"nodes[{i}]"
        if not isinstance(node, dict) or set(node) != {"id", "op", "args", "kwargs", "outputs", "module_path", "module_type"}:
            _bad(f"{where} has the wrong fields")
        if not _NODE.fullmatch(str(node["id"])) or node["id"] in seen_nodes or not isinstance(node["op"], str) or not node["op"]:
            _bad(f"{where}: bad id or op")
        seen_nodes.add(node["id"])
        if not isinstance(node["args"], list) or not isinstance(node["kwargs"], dict):
            _bad(f"{where}: args must be a list and kwargs an object")
        for j, a in enumerate(node["args"]):
            _arg(a, known, f"{where}.args[{j}]")
        for k, a in node["kwargs"].items():
            _arg(a, known, f"{where}.kwargs.{k}")
        for k in ("module_path", "module_type"):
            if node[k] is not None and not isinstance(node[k], str):
                _bad(f"{where}.{k} must be a string or null")
        if not node["outputs"]:
            _bad(f"{where} has no outputs")
        for out in node["outputs"]:
            if not isinstance(out, dict) or not _VALUE.fullmatch(str(out.get("value"))) or out["value"] in known \
                    or out.get("kind") not in ("tensor", "tuple", "other"):
                _bad(f"{where}: bad output {out!r}")
            if out["kind"] == "tensor":
                if set(out) != {"value", "kind", "shape", "dtype"}:
                    _bad(f"{where}: a tensor output is {{value, kind, shape, dtype}}")
                _shape(out["shape"], where)
                _dtype(out["dtype"], where)
            elif out["kind"] == "tuple":
                if set(out) != {"value", "kind", "items"} or not isinstance(out["items"], list):
                    _bad(f"{where}: a tuple output is {{value, kind, items}}")
                for item in out["items"]:
                    if item is not None:
                        _shape(item.get("shape"), where)
                        _dtype(item.get("dtype"), where)
            elif set(out) != {"value", "kind"}:
                _bad(f"{where}: an 'other' output is {{value, kind}}")
            known.add(out["value"])
    if not isinstance(ir["outputs"], list) or not ir["outputs"] or any(v not in known for v in ir["outputs"]):
        _bad("outputs must name defined values")
    body = {k: v for k, v in ir.items() if k != "ir_sha256"}
    if digest(body) != ir["ir_sha256"]:
        raise InspectError("ir_hash_mismatch", "ir_sha256 does not match the graph content", "graph.ir_sha256")
    return ir


# ---------------------------------------------------------------- table signature and binding

def table_signature(table: TensorTable) -> str:
    """sha256 of the sorted (name, dtype, shape) list: what two checkpoints must share to share a graph."""
    return digest([[n, i.dtype, list(i.shape)] for n, i in sorted(table.tensors.items())])


def _family(dtype: str) -> str:
    return "float" if dtype in _FLOATS else "int" if dtype in _INTS else dtype


def _refs(value, path=()):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, list):
        for i, item in enumerate(value):
            yield from _refs(item, (*path, i))


@dataclass
class TensorBinding:
    status: str                      # bound | unused | unbound
    uses: list = field(default_factory=list)   # {node, op, arg (position or kwarg name), path (index inside a list argument), module_path, module_type}
    tied_with: list = field(default_factory=list)


@dataclass
class BoundGraph:
    ir_sha256: str
    table_signature: str
    tensors: dict[str, TensorBinding]
    ties: list

    def names(self, status: str) -> list[str]:
        return sorted(n for n, b in self.tensors.items() if b.status == status)

    @property
    def bound(self):
        return self.names("bound")

    @property
    def unused(self):
        return self.names("unused")

    @property
    def unbound(self):
        return self.names("unbound")


def bind(ir: dict, table: TensorTable) -> BoundGraph:
    """Join a graph to a table. Raises GraphMismatch if the graph names tensors the checkpoint lacks or reshapes.

    dtype is not part of the meaning of a graph (an F32 graph binds to a BF16 checkpoint), only its family is checked.
    """
    verify_ir(ir)
    params = {p["name"]: p for p in ir["params"]}
    problems = []
    tie_of = {n: g for g in ir["ties"] for n in g}
    for name, p in params.items():
        info = table.tensors.get(name)
        if info is None:
            if name not in tie_of or not any(m in table.tensors for m in tie_of[name]):  # a tied name may be stored once
                problems.append((name, "missing from the checkpoint"))
        elif list(info.shape) != p["shape"]:
            problems.append((name, f"shape {list(info.shape)} in the checkpoint, {p['shape']} in the graph"))
        elif _family(info.dtype) != _family(p["dtype"]):
            problems.append((name, f"dtype {info.dtype} in the checkpoint, {p['dtype']} in the graph"))
    if problems:
        raise GraphMismatch(problems)
    by_value = {p["value"]: p["name"] for p in ir["params"]}
    uses: dict[str, list] = {name: [] for name in params}
    for node in ir["nodes"]:
        for position, arg in [*enumerate(node["args"]), *sorted(node["kwargs"].items())]:
            for path, ref in _refs(arg):
                if ref in by_value:
                    uses[by_value[ref]].append({"node": node["id"], "op": node["op"], "arg": position, "path": list(path),
                                                "module_path": node["module_path"], "module_type": node["module_type"]})
    tensors: dict[str, TensorBinding] = {}
    for name in sorted(table.tensors):
        group = tie_of.get(name, [])
        rep = name if name in params else next((m for m in group if m in params), None)
        if rep is None:
            tensors[name] = TensorBinding("unbound", [], [m for m in group if m != name])
        else:  # a tied alias is bound through the name the graph carries
            tensors[name] = TensorBinding("bound" if uses[rep] else "unused", list(uses[rep]), [m for m in group if m != name])
    return BoundGraph(ir["ir_sha256"], table_signature(table), tensors, ir["ties"])


# ---------------------------------------------------------------- the graph store (catalog tables)

class GraphStore:
    """Graphs attached to table signatures. One signature may carry several graphs (one per provider)."""

    def __init__(self, catalog):
        self.catalog = catalog

    def put(self, ir: dict, table: TensorTable) -> dict:
        """Attach `ir` to this table's signature (after proving it binds). Idempotent for the same graph and signature."""
        bind(ir, table)
        signature = table_signature(table)
        with self.catalog._db(True) as db:
            db.execute("INSERT OR IGNORE INTO graphs(ir_sha256,provider,spec_sha256,table_signature,ir_json,created_at) "
                       "VALUES (?,?,?,?,?,strftime('%Y-%m-%dT%H:%M:%fZ','now'))",
                       (ir["ir_sha256"], ir["provider"]["name"], ir["provider"]["spec_sha256"], signature, ir_json(ir)))
        return next(r for r in self.for_signature(signature) if r["ir_sha256"] == ir["ir_sha256"])

    def for_signature(self, signature: str) -> list[dict]:
        with self.catalog._db() as db:
            rows = db.execute("SELECT ir_sha256,provider,spec_sha256,table_signature,created_at FROM graphs "
                              "WHERE table_signature=? ORDER BY created_at, ir_sha256", (signature,)).fetchall()
        return [dict(r) for r in rows]

    def for_table(self, table: TensorTable, provider: str | None = None) -> list[dict]:
        """The graphs a checkpoint inherits through its table signature (all of its parent's, for a child)."""
        return [r for r in self.for_signature(table_signature(table)) if provider in (None, r["provider"])]

    def load(self, ir_sha256: str, signature: str | None = None) -> dict:
        with self.catalog._db() as db:
            row = db.execute("SELECT ir_json FROM graphs WHERE ir_sha256=?" + (" AND table_signature=?" if signature else ""),
                             (ir_sha256, *([signature] if signature else []))).fetchone()
        if row is None:
            raise InspectError("not_found", "no such graph", "ir_sha256")
        return verify_ir(json.loads(row["ir_json"]))

    def graph_for(self, table: TensorTable, provider: str | None = None) -> dict | None:
        """The newest graph attached to this table's signature, verified, or None."""
        records = self.for_table(table, provider)
        return self.load(records[-1]["ir_sha256"], records[-1]["table_signature"]) if records else None
