"""Edit programs (recipe v2): plan, validate and apply weight edits to an immutable base checkpoint.

A program is an ordered list of ops (zero, scale, set, add, clamp, lerp, copy_from), each with a
selector resolved against the BASE tensor table. Ops run in order on a per-tensor float32 working
copy and the result of EVERY op is rounded, to nearest-even, into the stored dtype (F32 as is, F16
by IEEE, BF16 by bit rounding) before the next op sees it ("cpu-f32-round-per-op"), so a program
[a, b] equals apply(apply(a), b) bit for bit. Every byte outside the selections is copied from the
parent, so the child has the parent's exact header and layout. Everything is deterministic CPU numpy float32 elementwise math.
Nothing is written until the whole program has validated, and the child becomes visible only through
one final atomic rename. Library level: no HTTP, no catalog (child registration is T4c).

Ops revision 2 (v0.3-G5) adds coupled-group edits. A group selector in a program is expanded, when the program is frozen, into
one tensor-local op per member (`select` stays a canonical tensor-local selector); the group is kept as `via` provenance, so
verifying, planning and applying a recipe never need the graph. `permute_units` is a pure gather along one axis (bit-exact, no
rounding) and is the only op that needs a closed group. A recipe made only of revision-1 features keeps the revision-1 engine and
its hash, so every v0.2 recipe still verifies and hashes as before.
"""

import hashlib
import math
import os
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import group_ops
from .selector import CHUNK_ELEMS, InspectError, Selection, canonical_json, decode, digest, iter_raw, parse_selector, raw_bits
from .tensors import DTYPES, EDITABLE_DTYPES, TensorFileError, TensorTable, open_tensors

SCHEMA_VERSION = 2
OPS_REVISION = 1
ENGINE = {"layerdelta_ops_revision": OPS_REVISION, "plugins": {}, "numeric": "cpu-f32-round-per-op"}
ENGINE_V2 = {**ENGINE, "layerdelta_ops_revision": 2}
SUPPORTED_ENGINES = (ENGINE, ENGINE_V2)  # revision 1 recipes stay valid; revision 2 may use permute_units and `via`
REV2_OPS = ("permute_units",)
PERM_MAX = 1 << 24
MOVED_LISTED = 1000
VIA_KEYS = {"group", "ir_sha256", "units", "members", "member"}
F32_MAX = float(np.finfo(np.float32).max)
HEX64 = re.compile("[0-9a-f]{64}")
BLOB_ID = re.compile("sha256:[0-9a-f]{64}")
OVERLAPS_LISTED = 1000


@dataclass(frozen=True)
class Limits:
    max_program_ops: int = 256
    max_touched_elems: int = 1 << 30  # selected elements summed over the program
    max_tensor_elems: int = 1 << 27   # one touched tensor is held in memory as float32
    max_blob_bytes: int = 1 << 28


# ---------------------------------------------------------------- recipe schema and hashing

def _number(value, field, *, low=None, high=None) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > F32_MAX:
        raise InspectError("invalid_argument", "expected a finite number within float32 range", field)
    value = 0.0 if value == 0 else float(value)
    if (low is not None and value < low) or (high is not None and value > high):
        raise InspectError("invalid_argument", f"expected a number in [{low}, {high}]", field)
    return value


def _blob(value, field) -> str:
    if not isinstance(value, str) or not BLOB_ID.fullmatch(value):
        raise InspectError("invalid_argument", "blob must be 'sha256:<64 hex>'", field)
    return value


def _source(value, field) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise InspectError("invalid_argument", "source must be a checkpoint id (64 hex)", field)
    return value


def _permutation(perm, field) -> list[int]:
    if (not isinstance(perm, list) or not perm or len(perm) > PERM_MAX or any(type(i) is not int for i in perm)
            or sorted(perm) != list(range(len(perm)))):
        raise InspectError("invalid_permutation", "perm must be a list holding each of 0..n-1 exactly once", field)
    return list(perm)


def _args(op, args, field) -> dict:
    """Canonical args for one op (floats, strict key sets)."""
    if not isinstance(args, dict):
        raise InspectError("invalid_argument", "args must be an object", field)

    def only(*allowed):
        for key in args:
            if key not in allowed:
                raise InspectError("unknown_field", f"unknown argument {key!r} for {op}", f"{field}.{key}")

    def need(key):
        if key not in args:
            raise InspectError("invalid_argument", f"{op} needs {key}", f"{field}.{key}")

    if op == "zero":
        only()
        return {}
    if op == "scale":
        only("k"), need("k")
        return {"k": _number(args["k"], f"{field}.k")}
    if op in ("set", "add"):
        only("value", "blob")
        if ("value" in args) == ("blob" in args):
            raise InspectError("invalid_argument", f"{op} takes exactly one of value or blob", field)
        return ({"value": _number(args["value"], f"{field}.value")} if "value" in args
                else {"blob": _blob(args["blob"], f"{field}.blob")})
    if op == "clamp":
        only("min", "max")
        out = {k: _number(args[k], f"{field}.{k}") for k in ("min", "max") if k in args}
        if not out:
            raise InspectError("invalid_argument", "clamp needs min and/or max", field)
        if len(out) == 2 and out["min"] > out["max"]:
            raise InspectError("invalid_argument", "clamp min must not exceed max", f"{field}.min")
        return out
    if op == "lerp":
        only("source", "t"), need("source"), need("t")
        return {"source": _source(args["source"], f"{field}.source"), "t": _number(args["t"], f"{field}.t", low=0, high=1)}
    if op == "copy_from":
        only("source"), need("source")
        return {"source": _source(args["source"], f"{field}.source")}
    if op == "permute_units":  # tensor-local form: gather `perm` along `axis` of the selection
        only("axis", "perm"), need("axis"), need("perm")
        axis = args["axis"]
        if type(axis) is not int or axis < 0:
            raise InspectError("invalid_axis", "axis must be a non-negative integer", f"{field}.axis")
        return {"axis": axis, "perm": _permutation(args["perm"], f"{field}.perm")}
    raise InspectError("invalid_argument", f"unknown op {op!r}", field.rsplit(".", 1)[0] + ".op")


OPS = ("zero", "scale", "set", "add", "clamp", "lerp", "copy_from", "permute_units")


def revision_of(recipe: dict) -> int:
    return recipe["engine"]["layerdelta_ops_revision"]


def _canonical_select(select, field) -> None:
    """A recipe's selector must be exactly the resolved form freeze_recipe writes (one hash per selection)."""
    bad = lambda why: InspectError("noncanonical_recipe", f"select is not canonical: {why}", field)  # noqa: E731
    if not isinstance(select, dict) or set(select) != {"tensor", "shape", "index"}:
        raise bad("keys must be exactly tensor, shape, index")
    shape, index = select["shape"], select["index"]
    if (not isinstance(select["tensor"], str) or not isinstance(shape, list)
            or any(type(n) is not int or n < 0 for n in shape)):
        raise bad("tensor must be a string and shape a list of non-negative integers")
    if not isinstance(index, dict) or set(index) != {str(i) for i in range(len(shape))}:
        raise bad("index must hold every axis 0..ndim-1")
    for axis, ranges in index.items():
        n = shape[int(axis)]
        if not isinstance(ranges, list) or not ranges:
            raise bad(f"axis {axis} needs a non-empty list of ranges")
        last = -1
        for r in ranges:
            if (not isinstance(r, dict) or set(r) != {"start", "stop"} or type(r["start"]) is not int
                    or type(r["stop"]) is not int or not 0 <= r["start"] < r["stop"] <= n):
                raise bad(f"axis {axis} has an invalid {{start, stop}} range")
            if r["start"] <= last:  # sorted, non-overlapping and non-adjacent
                raise bad(f"axis {axis} ranges must be sorted, disjoint and not adjacent")
            last = r["stop"]


def _canonical_via(via, field) -> None:
    """Provenance only: where a group-expanded op came from. Nothing can check it without the graph, so it is checked for shape."""
    bad = lambda why: InspectError("noncanonical_recipe", f"via is not canonical: {why}", field)  # noqa: E731
    if not isinstance(via, dict) or not (VIA_KEYS <= set(via) <= VIA_KEYS | {"perm"}):
        raise bad("keys must be group, ir_sha256, units, members, member (and perm for permute_units)")
    if not isinstance(via["group"], str) or not re.fullmatch("g_[0-9a-f]{24}", via["group"]):
        raise bad("group must be a group id")
    if not isinstance(via["ir_sha256"], str) or not HEX64.fullmatch(via["ir_sha256"]):
        raise bad("ir_sha256 must be 64 hex characters")
    last = -1
    if not isinstance(via["units"], list) or not via["units"]:
        raise bad("units must be a non-empty list of {start, stop}")
    for r in via["units"]:
        if (not isinstance(r, dict) or set(r) != {"start", "stop"} or type(r["start"]) is not int or type(r["stop"]) is not int
                or not 0 <= r["start"] < r["stop"] or r["start"] <= last):
            raise bad("units must be sorted, disjoint, non-adjacent {start, stop} ranges")
        last = r["stop"]
    members = via["members"]
    if members is not None and (not isinstance(members, dict) or not set(members) <= {"roles", "exclude"}
                                or any(not isinstance(v, list) or not all(isinstance(x, str) for x in v) for v in members.values())):
        raise bad("members must be null or {roles?, exclude?} lists of names")
    member = via["member"]
    if not isinstance(member, dict) or set(member) != {"tensor", "axis"} or not isinstance(member["tensor"], str) \
            or type(member["axis"]) is not int or member["axis"] < 0:
        raise bad("member must be {tensor, axis}")
    if "perm" in via:
        _permutation(via["perm"], f"{field}.perm")


def _finish(base_id, program, hypothesis, evaluation, engine=None) -> dict:
    if not isinstance(hypothesis, str) or not hypothesis.strip() or len(hypothesis) > 4096:
        raise InspectError("invalid_argument", "hypothesis must contain 1 to 4096 characters", "hypothesis")
    if evaluation is not None and not isinstance(evaluation, dict):
        raise InspectError("invalid_argument", "evaluation must be null or an object", "evaluation")
    try:
        canonical_json(evaluation)
    except (ValueError, TypeError) as exc:  # NaN/Inf or a non-JSON value
        raise InspectError("invalid_argument", f"evaluation is not canonical JSON: {exc}", "evaluation") from exc
    if not isinstance(base_id, str) or not HEX64.fullmatch(base_id):
        raise InspectError("invalid_argument", "base checkpoint id must be 64 hex characters", "base.checkpoint_sha256")
    recipe = {"schema_version": SCHEMA_VERSION,
              "base": {"checkpoint_sha256": base_id, "format": "safetensors", "profile_sha256": None},
              "program": program, "engine": dict(engine or ENGINE), "hypothesis": hypothesis, "evaluation": evaluation}
    return {**recipe, "recipe_hash": digest(recipe)}


def freeze_recipe(table: TensorTable, base_id: str, program, hypothesis: str, *, evaluation=None, graph=None) -> dict:
    """Canonical recipe v2: selectors resolved against the base table (so equivalent spellings hash alike).

    A group selector needs `graph` (a GraphView of the base table); it is expanded here into tensor-local ops with `via`.
    """
    if not isinstance(program, list) or not program:
        raise InspectError("invalid_argument", "program must be a non-empty list", "program")
    frozen = []
    for i, entry in enumerate(program):
        field = f"program[{i}]"
        if not isinstance(entry, dict) or set(entry) != {"op", "args", "select"}:
            raise InspectError("invalid_argument", "each op needs exactly op, args and select", field)
        if entry["op"] not in OPS:
            raise InspectError("invalid_argument", f"op must be one of {list(OPS)}", f"{field}.op")
        if group_ops.is_group(entry["select"]):
            frozen += _expand_group(table, graph, entry, field)
            continue
        if entry["op"] in REV2_OPS:
            raise InspectError("invalid_argument", f"{entry['op']} takes a group selector", f"{field}.select")
        sel = parse_selector(table, entry["select"], field=f"{field}.select")
        if sel.total == 0:  # also keeps freeze_recipe from writing a selector verify_recipe would call noncanonical
            raise InspectError("empty_selection", "the selection contains no elements", f"{field}.select")
        frozen.append({"op": entry["op"], "args": _args(entry["op"], entry["args"], f"{field}.args"),
                       "select": sel.resolved})
    rev2 = any(e["op"] in REV2_OPS or "via" in e for e in frozen)
    return _finish(base_id, frozen, hypothesis, evaluation, ENGINE_V2 if rev2 else ENGINE)


def _expand_group(table, graph, entry, field) -> list[dict]:
    """One tensor-local op per member of the group selector, each carrying the group as `via`."""
    gs = group_ops.parse_group_selector(table, graph, entry["select"], field=f"{field}.select", checkpoint_id=None)
    op, group = entry["op"], gs.group
    via = {"group": group["group_id"], "ir_sha256": gs.ir_sha256, "units": gs.unit_ranges, "members": gs.filter}
    if op != "permute_units":
        args = _args(op, entry["args"], f"{field}.args")
        if "blob" in args:
            raise InspectError("invalid_argument", "a blob holds values for one selection: use a tensor selector", f"{field}.args.blob")
        return [{"op": op, "args": args, "select": m.sel.resolved, "via": {**via, "member": {"tensor": m.tensor, "axis": m.axis}}}
                for m in gs.members]
    # permute_units: the whole group, every member, and only if reordering its units cannot change the function
    if not isinstance(entry["args"], dict) or set(entry["args"]) != {"perm"}:
        raise InspectError("invalid_argument", "permute_units on a group takes exactly {perm}", f"{field}.args")
    perm = _permutation(entry["args"]["perm"], f"{field}.args.perm")
    if len(perm) != group["size"]:
        raise InspectError("invalid_permutation", f"perm must reorder all {group['size']} units of the group", f"{field}.args.perm")
    if gs.units != list(range(group["size"])) or gs.filter is not None:
        raise InspectError("invalid_argument", "permute_units needs the whole group with every member: no units or members filter",
                           f"{field}.select")
    if not group["closed"]:
        reasons = sorted({b["reason"] for b in group["boundary"]})
        raise InspectError("group_not_closed", f"this group is open ({', '.join(reasons)}): reordering its units can change the "
                           "function", f"{field}.select.group")
    blocks = group["constraints"]
    if blocks.get("kind") == "within_blocks":
        n = blocks["n"]
        if any(j // n != perm[j] // n for j in range(len(perm))):
            raise InspectError("constraint_violation", f"units may only move inside blocks of {n} (group normalisation)",
                               f"{field}.args.perm")
    out = []
    for m in gs.members:
        indices = sorted({i for u in gs.units for i in group_ops._indices(m.spec, u)})
        where = {v: k for k, v in enumerate(indices)}
        pi = [0] * len(indices)
        for j in range(group["size"]):
            for dst, src in zip(sorted(group_ops._indices(m.spec, j)), sorted(group_ops._indices(m.spec, perm[j]))):
                pi[where[dst]] = where[src]
        out.append({"op": op, "args": {"axis": m.axis, "perm": pi}, "select": m.sel.resolved,
                    "via": {**via, "member": {"tensor": m.tensor, "axis": m.axis}, "perm": perm}})
    return out


def verify_recipe(recipe) -> dict:
    """Strict schema and hash check of a recipe v2 (no table needed)."""
    keys = {"schema_version", "base", "program", "engine", "hypothesis", "evaluation", "recipe_hash"}
    if not isinstance(recipe, dict) or recipe.get("schema_version") != SCHEMA_VERSION or set(recipe) != keys:
        raise InspectError("unsupported_schema", "not a recipe v2 (schema_version 2 with the documented fields)", "schema_version")
    if recipe["engine"] not in SUPPORTED_ENGINES:
        raise InspectError("unsupported_engine", "recipe was made for a different ops engine", "engine")
    revision = revision_of(recipe)
    base = recipe["base"]
    if (not isinstance(base, dict) or set(base) != {"checkpoint_sha256", "format", "profile_sha256"}
            or base["format"] != "safetensors" or base["profile_sha256"] is not None):
        raise InspectError("unsupported_schema", "base must be a safetensors checkpoint without a profile", "base")
    program = recipe["program"]
    if not isinstance(program, list) or not program:
        raise InspectError("invalid_argument", "program must be a non-empty list", "program")
    for i, entry in enumerate(program):
        field = f"program[{i}]"
        if (not isinstance(entry, dict) or not {"op", "args", "select"} <= set(entry) <= {"op", "args", "select", "via"}
                or entry["op"] not in OPS):
            raise InspectError("invalid_argument", "each op needs a known op, args and select (and optionally via)", field)
        if revision < 2 and (entry["op"] in REV2_OPS or "via" in entry):
            raise InspectError("unsupported_engine", "this recipe uses ops revision 2 features but declares revision 1", "engine")
        if _args(entry["op"], entry["args"], f"{field}.args") != entry["args"]:
            raise InspectError("noncanonical_recipe", "args are not in canonical form", f"{field}.args")
        _canonical_select(entry["select"], f"{field}.select")
        if "via" in entry:
            _canonical_via(entry["via"], f"{field}.via")
    rebuilt = _finish(base["checkpoint_sha256"], program, recipe["hypothesis"], recipe["evaluation"], recipe["engine"])
    if rebuilt["recipe_hash"] != recipe["recipe_hash"]:
        raise InspectError("recipe_hash_mismatch", "recipe_hash does not match the recipe content", "recipe_hash")
    return recipe


def translate_v1(recipe: dict, table: TensorTable) -> dict:
    """v1 weight-mode recipe -> v2 program of one op. Activation-mode recipes stay on the v1 path.

    The table is required: the v2 selector is stored in its resolved form, so one v1 recipe has exactly one v2 hash.
    """
    if not isinstance(recipe, dict) or recipe.get("schema_version") != 1:
        raise InspectError("unsupported_schema", "not a recipe v1", "schema_version")
    payload = {k: v for k, v in recipe.items() if k != "recipe_hash"}
    if digest(payload) != recipe.get("recipe_hash"):
        raise InspectError("recipe_hash_mismatch", "v1 recipe_hash does not match its content", "recipe_hash")
    iv, mapping = recipe["intervention"], recipe["mapping"]
    if iv.get("mode") != "weight":
        raise InspectError("unsupported_mode", "only weight-mode v1 recipes translate (activation edits stay on v1)",
                           "intervention.mode")
    select = {"tensor": mapping["parameter_name"], "index": {str(mapping["channel_axis"]): [iv["channel_index"]]}}
    step = ({"op": "zero", "args": {}} if iv["operation"] == "zero"
            else {"op": "scale", "args": {"k": iv["scale"]}})
    program = [{**step, "select": select}]
    base_id = recipe["dependencies"]["checkpoint_sha256"]
    return freeze_recipe(table, base_id, program, recipe["hypothesis"], evaluation=recipe.get("evaluation"))


# ---------------------------------------------------------------- resolution against the base table

@dataclass
class _Op:
    index: int
    op: str
    args: dict
    sel: Selection
    blob: "np.ndarray | None" = None
    source: "TensorTable | None" = None


def _resolve(table, recipe, base_id, sources, blobs, limits) -> list[_Op]:
    recipe = verify_recipe(recipe)
    if recipe["base"]["checkpoint_sha256"] != base_id:
        raise InspectError("base_mismatch", "recipe was made for a different base checkpoint", "base.checkpoint_sha256")
    if table.sharded:
        raise InspectError("unsupported_base", "editing a sharded checkpoint is not supported yet", "base")
    program = recipe["program"]
    if len(program) > limits.max_program_ops:
        raise InspectError("cap_exceeded", f"a program may have at most {limits.max_program_ops} ops", "program")
    ops, selected = [], 0
    for i, entry in enumerate(program):
        field = f"program[{i}]"
        sel = parse_selector(table, entry["select"], field=f"{field}.select")
        if sel.resolved != entry["select"]:
            raise InspectError("noncanonical_recipe", "select does not resolve to itself against the base table",
                               f"{field}.select")
        if sel.info.dtype not in EDITABLE_DTYPES:
            raise InspectError("read_only_dtype", f"{sel.info.name} is {sel.info.dtype}; only F32/F16/BF16 can be edited",
                               f"{field}.select.tensor")
        if sel.total == 0:
            raise InspectError("empty_selection", "the selection contains no elements", f"{field}.select")
        selected += sel.total
        if selected > limits.max_touched_elems:
            raise InspectError("cap_exceeded", f"the program selects more than {limits.max_touched_elems} elements", "program")
        op = _Op(i, entry["op"], entry["args"], sel)
        if op.op == "permute_units":
            axis, perm = op.args["axis"], op.args["perm"]
            if axis >= len(sel.shape):
                raise InspectError("invalid_axis", f"axis {axis} does not exist (tensor has {len(sel.shape)} axes)", f"{field}.args.axis")
            if len(perm) != sel.counts[axis]:
                raise InspectError("invalid_permutation", f"perm has {len(perm)} entries, the selection has {sel.counts[axis]} "
                                   f"on axis {axis}", f"{field}.args.perm")
        if "blob" in op.args:
            op.blob = _load_blob(blobs, op, f"{field}.args.blob", limits)
        if "source" in op.args:
            op.source = _check_source(sources, op, f"{field}.args.source")
        ops.append(op)
    return ops


def _load_blob(blobs, op, field, limits) -> np.ndarray:
    blob_id = op.args["blob"]
    data = (blobs or {}).get(blob_id)
    if data is None:
        raise InspectError("blob_missing", f"blob {blob_id} was not supplied", field)
    if len(data) > limits.max_blob_bytes:
        raise InspectError("cap_exceeded", f"blob exceeds {limits.max_blob_bytes} bytes", field)
    if "sha256:" + hashlib.sha256(data).hexdigest() != blob_id:
        raise InspectError("blob_mismatch", "blob content does not hash to its id", field)
    if len(data) != 4 * op.sel.total:
        raise InspectError("blob_size", f"blob holds {len(data) // 4} float32 values, the selection has {op.sel.total}", field)
    values = np.frombuffer(data, dtype="<f4").astype(np.float32)
    if not np.isfinite(values).all():
        raise InspectError("non_finite", "blob contains NaN or Inf", field)
    return values.reshape(op.sel.counts)


def _check_source(sources, op, field) -> TensorTable:
    source = (sources or {}).get(op.args["source"])
    if source is None:
        raise InspectError("source_missing", f"checkpoint {op.args['source']} was not supplied", field)
    other = source.tensors.get(op.sel.info.name)
    if other is None or other.dtype != op.sel.info.dtype or other.shape != op.sel.info.shape:
        raise InspectError("incompatible_source", f"{op.sel.info.name} is missing or differs in dtype/shape in the source",
                           field)
    return source


# ---------------------------------------------------------------- computing one tensor

def _index(sel: Selection):
    """Index tuple selecting the box; one multi-range axis may mix with slices, more use a grid."""
    def span(ranges):
        return np.concatenate([np.arange(s, e) for s, e in ranges])

    if sum(len(ranges) > 1 for ranges in sel.axes) <= 1:
        return tuple(span(r) if len(r) > 1 else slice(*r[0]) for r in sel.axes)
    return np.ix_(*[span(r) for r in sel.axes])


def _source_values(op: _Op) -> np.ndarray:
    info = op.source.tensors[op.sel.info.name]
    raw = bytearray().join(iter_raw(op.source, Selection(info, op.sel.axes)))
    return decode(info.dtype, raw).astype(np.float32).reshape(op.sel.counts)


def _f32(value) -> np.float32:
    return np.float32(value)


def _apply(op: _Op, region: np.ndarray) -> np.ndarray:
    a = op.args
    if op.op == "zero":
        return np.zeros_like(region)
    if op.op == "scale":
        return region * _f32(a["k"])
    if op.op == "set":
        return op.blob.copy() if op.blob is not None else np.full_like(region, _f32(a["value"]))
    if op.op == "add":
        return region + (op.blob if op.blob is not None else _f32(a["value"]))
    if op.op == "clamp":
        return np.clip(region, _f32(a["min"]) if "min" in a else None, _f32(a["max"]) if "max" in a else None)
    if op.op == "permute_units":  # a gather: values are moved, never recomputed
        return np.take(region, a["perm"], axis=a["axis"])
    other = _source_values(op)
    if op.op == "copy_from" or a["t"] == 1:
        return other
    if a["t"] == 0:
        return region.copy()
    return region + (other - region) * _f32(a["t"])  # lerp: a + (b - a) * t, float32


def _decode_bits(bits: np.ndarray, dtype: str) -> np.ndarray:
    """Stored-dtype bits -> float32 (exact)."""
    if dtype == "F32":
        return bits.view(np.float32)
    if dtype == "F16":
        return bits.view(np.float16).astype(np.float32)
    return (bits.astype(np.uint32) << 16).view(np.float32)


def _encode(values: np.ndarray, dtype: str, name: str, op_index) -> np.ndarray:
    """float32 -> stored-dtype bits, round to nearest even; refuses values the dtype cannot hold."""
    x = np.array(values, dtype=np.float32, order="C", copy=True)
    if dtype == "F32":
        return x.view(np.uint32)
    if dtype == "F16":
        with np.errstate(over="ignore"):  # overflow to inf is detected below and refused
            half = x.astype(np.float16)
        bits = half.view(np.uint16)
        finite = np.isfinite(half)
    else:  # BF16: keep the top 16 bits of the float32, rounding the dropped half to nearest even
        b = x.view(np.uint32)
        bits = ((b.astype(np.uint64) + 0x7FFF + ((b >> 16) & 1).astype(np.uint64)) >> 16).astype(np.uint16)
        finite = (bits & 0x7F80) != 0x7F80
    if not finite.all():
        raise InspectError("dtype_overflow", f"op {op_index} overflows {dtype} in {name}", f"program[{op_index}]")
    return bits


def _run_tensor(table: TensorTable, name: str, ops: list[_Op], limits: Limits) -> dict:
    info = table.tensors[name]
    dtype, shape = info.dtype, info.shape
    if math.prod(shape) > limits.max_tensor_elems:
        raise InspectError("cap_exceeded", f"{name} has more than {limits.max_tensor_elems} elements", f"program[{ops[0].index}]")
    parent_raw = table.read_bytes(name)
    parent_bits = raw_bits(dtype, parent_raw)
    work = decode(dtype, parent_raw).astype(np.float32).reshape(shape)
    mask = np.zeros(shape, dtype=bool)
    with np.errstate(all="ignore"):
        for op in ops:
            idx = _index(op.sel)
            new = _apply(op, np.asarray(work[idx]))
            if not np.isfinite(new).all():
                raise InspectError("non_finite", f"op {op.op} produced NaN or Inf in {name}", f"program[{op.index}]")
            # round into the stored dtype now, so the next op sees what a stored child would hold
            work[idx] = _decode_bits(_encode(new, dtype, name, op.index), dtype).reshape(new.shape)
            mask[idx] = True
    flat_mask = mask.reshape(-1)
    bits = parent_bits.copy()
    bits[flat_mask] = _encode(work.reshape(-1)[flat_mask], dtype, name, ops[-1].index)  # already representable
    changed = bits != parent_bits
    assert not changed[~flat_mask].any()  # touched-mask is contained in the selection union by construction
    return {"bits": bits, "touched_elems": int(flat_mask.sum()), "changed_elems": int(changed.sum()),
            "before_sha": hashlib.sha256(parent_raw).hexdigest(), "after_sha": hashlib.sha256(bits).hexdigest()}


def _box_overlap(a: Selection, b: Selection) -> int:
    total = 1
    for ra, rb in zip(a.axes, b.axes):
        n = sum(max(0, min(e1, e2) - max(s1, s2)) for s1, e1 in ra for s2, e2 in rb)
        total *= n
    return total


# ---------------------------------------------------------------- plan / validate / apply

def _envelope(base_id, recipe, result, *, side_effects=()) -> dict:
    return {"schema_version": 1, "resolved": {"recipe_hash": recipe["recipe_hash"], "base_checkpoint": base_id,
                                              "ops": len(recipe["program"])},
            "result": result,
            "fingerprints": {"checkpoint": base_id, "recipe": recipe["recipe_hash"], "selection": None,
                             "op_revision": revision_of(recipe)},
            "side_effects": list(side_effects), "truncated": False, "cursor": None}


def _group_report(recipe, ops, graph) -> tuple[list, list, list]:
    """(groups, moved_units, warnings) for the group-expanded ops of a recipe. Works from `via` alone; a GraphView adds
    whether each group is closed and what its boundary and notes say."""
    seen: dict = {}
    for op, entry in zip(ops, recipe["program"]):
        via = entry.get("via")
        if via is None:
            continue
        key = canonical_json({k: via[k] for k in ("group", "ir_sha256", "units", "members")} | {"op": op.op})
        item = seen.setdefault(key, {"group": via["group"], "ir_sha256": via["ir_sha256"], "units": via["units"],
                                     "members_filter": via["members"], "op": op.op, "ops": [], "members": [],
                                     "perm": via.get("perm")})
        item["ops"].append(op.index)
        item["members"].append({"tensor": via["member"]["tensor"], "axis": via["member"]["axis"]})
    groups, moved, warnings = [], [], []
    for item in seen.values():
        record = None
        if graph is not None and graph.ir["ir_sha256"] == item["ir_sha256"] and item["group"] in graph.by_id:
            record = graph.group_record(graph.by_id[item["group"]])
        item["closed"] = record["closed"] if record else None
        item["boundary"] = record["boundary"] if record else None
        item["notes"] = record["notes"] if record else []
        item["label"] = record["label"] if record else None
        if item["op"] == "permute_units":
            perm = item.pop("perm")
            shifted = [{"from": perm[j], "to": j} for j in range(len(perm)) if perm[j] != j]
            moved.append({"group": item["group"], "count": len(shifted), "total": len(perm), "units": shifted[:MOVED_LISTED],
                          "truncated": len(shifted) > MOVED_LISTED})
        else:
            item.pop("perm")
            if record and not record["closed"]:  # only permute_units needs a closed group; an edit through an open one is allowed
                warnings.append({"code": "group_open", "group": item["group"], "boundary": record["boundary"],
                                 "message": "This group is open: the edit applies to its known members only."})
        groups.append(item)
    return groups, moved, warnings


def _plan(table, recipe, base_id, sources, blobs, limits, graph=None):
    ops = _resolve(table, recipe, base_id, sources, blobs, limits)
    by_tensor: dict[str, list[_Op]] = {}
    for op in ops:
        by_tensor.setdefault(op.sel.info.name, []).append(op)
    tensors = []
    for name in sorted(by_tensor):
        stats = _run_tensor(table, name, by_tensor[name], limits)
        stats.pop("bits")
        tensors.append({"tensor": name, "dtype": table.tensors[name].dtype, "ops": [o.index for o in by_tensor[name]],
                        **stats})
    overlaps = []
    for group in by_tensor.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                n = _box_overlap(a.sel, b.sel)
                if n:
                    overlaps.append({"tensor": a.sel.info.name, "ops": [a.index, b.index], "elems": n})
    result = {
        "ops": [{"index": o.index, "op": o.op, "tensor": o.sel.info.name, "selection_id": o.sel.selection_id,
                 "selected_elems": o.sel.total} for o in ops],
        "tensors": tensors,
        "touched_elems_total": sum(t["touched_elems"] for t in tensors),
        "changed_elems_total": sum(t["changed_elems"] for t in tensors),
        "overlaps": overlaps[:OVERLAPS_LISTED], "overlaps_total": len(overlaps),
        "noop": all(t["changed_elems"] == 0 for t in tensors),
        "child_bytes": table.file_bytes, "numeric": ENGINE["numeric"], "rounding": "nearest-even, after every op",
    }
    if any("via" in e for e in recipe["program"]):
        for entry, row in zip(recipe["program"], result["ops"]):
            if "via" in entry:
                row["via"] = entry["via"]
        result["groups"], result["moved_units"], result["warnings"] = _group_report(recipe, ops, graph)
    return ops, by_tensor, result


def plan(table: TensorTable, recipe: dict, *, base_id: str, sources=None, blobs=None, limits: Limits = Limits(),
         graph=None) -> dict:
    """Dry run: validates everything apply() would and reports the effect. Writes nothing. A GraphView only adds what a
    group-expanded recipe says about its groups (closed, boundary, notes); the plan never depends on it."""
    _, _, result = _plan(table, recipe, base_id, sources, blobs, limits, graph)
    return _envelope(base_id, recipe, result)


CHECKS = ("recipe_schema", "selectors_in_range", "dtype_editable", "blobs_and_sources", "finite",
          "dtype_overflow", "shape_preserved", "touched_mask_within_selection", "caps")


def validate(table: TensorTable, recipe: dict, *, base_id: str, sources=None, blobs=None, limits: Limits = Limits(),
             graph=None) -> dict:
    """Same checks as plan(), reported as data: {"valid": bool, "checks": [...], "error": {...} | None}."""
    try:
        plan(table, recipe, base_id=base_id, sources=sources, blobs=blobs, limits=limits, graph=graph)
    except InspectError as exc:
        return {"valid": False, "checks": list(CHECKS), "error": exc.to_dict()}
    return {"valid": True, "checks": list(CHECKS), "error": None}


def apply(table: TensorTable, recipe: dict, out_path, *, base_id: str, sources=None, blobs=None,
          limits: Limits = Limits(), allow_noop: bool = False, graph=None) -> dict:
    """Write the child checkpoint. Validation (plan) runs first, so a bad program writes nothing.

    The child is built in a hidden partial file, verified with the T1 reader, then renamed into
    place: a crash at any point leaves no visible child. A program that changes no byte is refused
    as `no_op` unless allow_noop, in which case the child is byte-identical to the parent.
    """
    ops, by_tensor, result = _plan(table, recipe, base_id, sources, blobs, limits, graph)
    if result["noop"] and not allow_noop:
        raise InspectError("no_op", "the program changes no stored value (e.g. scale by 1)", "program")
    out = Path(out_path)
    if out.exists():
        raise InspectError("output_exists", f"{out.name} already exists", "out_path")
    expected = {t["tensor"]: t for t in result["tensors"]}
    partial = out.with_name(f".{out.name}.partial-{uuid.uuid4().hex}")
    try:
        child_sha = _write_child(table, by_tensor, expected, limits, partial)
        _verify_child(table, partial)
        _publish(partial, out)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    manifest = {"schema_version": 1, "parent": base_id, "child_sha256": child_sha, "recipe_hash": recipe["recipe_hash"],
                "child_bytes": table.file_bytes, "engine": dict(recipe["engine"]),
                "touched": [{"tensor": t["tensor"], "elems": t["touched_elems"], "changed_elems": t["changed_elems"],
                             "before_sha": t["before_sha"], "after_sha": t["after_sha"]} for t in result["tensors"]]}
    envelope = _envelope(base_id, recipe, {**result, "manifest": manifest},
                         side_effects=[{"kind": "checkpoint_created", "id": child_sha, "bytes": table.file_bytes}])
    return envelope


def _publish(partial: Path, out: Path) -> None:
    """Make the child visible without ever replacing an existing file (hard link, then drop the partial)."""
    try:
        os.link(partial, out)
    except FileExistsError as exc:
        raise InspectError("output_exists", f"{out.name} already exists", "out_path") from exc
    except OSError:  # a filesystem without hard links: check, then rename (narrow race remains)
        if out.exists():
            raise InspectError("output_exists", f"{out.name} already exists", "out_path") from None
        os.replace(partial, out)
        return
    partial.unlink(missing_ok=True)


def _write_child(table, by_tensor, expected, limits, partial: Path) -> str:
    digest_ = hashlib.sha256()
    with partial.open("wb") as out:
        def put(chunk):
            out.write(chunk)
            digest_.update(chunk)

        put(table.header_bytes())  # the parent's exact header: same names, dtypes, shapes, offsets, metadata
        for info in sorted(table.tensors.values(), key=lambda i: i.start):
            if info.name in by_tensor:
                done = _run_tensor(table, info.name, by_tensor[info.name], limits)
                if done["after_sha"] != expected[info.name]["after_sha"]:
                    raise InspectError("nondeterministic_edit", f"{info.name} computed differently on the second pass", "program")
                put(memoryview(done["bits"]).cast("B"))
            else:
                step = CHUNK_ELEMS * 4
                for chunk in table.read_ranges(info.name, ((o, min(step, info.nbytes - o)) for o in range(0, info.nbytes, step))):
                    put(chunk)
        out.flush()
        os.fsync(out.fileno())
    return digest_.hexdigest()


def _verify_child(table: TensorTable, partial: Path) -> None:
    try:
        child = open_tensors(partial)
    except TensorFileError as exc:
        raise InspectError("child_invalid", str(exc), "out_path") from exc
    shape = lambda t: {n: (i.dtype, i.shape, i.start, i.end) for n, i in t.tensors.items()}  # noqa: E731
    if shape(child) != shape(table) or child.metadata != table.metadata:
        raise InspectError("child_invalid", "the child's table differs from the parent's", "out_path")
