"""Pure inspect/compare/select operations over TensorTable: describe, read_tensor, stats, diff, query.

Library level (T2): every function returns the v1 response envelope as a plain dict and raises
InspectError for bad requests. Nothing here writes, builds a model or runs code from the file; all
numbers are accumulated in float64 from the stored values in fixed 1 Mi-element chunks, so repeated
calls are byte-identical (`to_json`). Schema: see the T2 draft / docs.
"""

import base64
import fnmatch
import itertools
import json
import math

import numpy as np

from . import budget, group_ops
from .selector import (CHUNK_ELEMS, InspectError, Selection, canonical_json, decode, digest, is_float,
                       iter_raw, parse_selector, raw_bits)
from .tensors import DTYPES, EDITABLE_DTYPES, TensorTable

SCHEMA_VERSION = 1
OP_REVISION = 1  # unchanged by rev7: the graph layer only adds keys when a graph is attached
ACCUMULATE_DTYPE = "float64"

DESCRIBE_DEFAULT, DESCRIBE_MAX = 5000, 20_000
READ_DEFAULT, READ_MAX = 1000, 10_000
RANGE_SCAN_MAX = 1 << 24
STATS_DEFAULT, STATS_MAX, BINS_MAX = 2048, 2048, 256
HISTOGRAM_SCAN_MAX = 1 << 26
DIFF_DEFAULT, DIFF_MAX = 500, 1000
UNIT_LIST_MAX = 1000
UNIT_BLOCK_MAX = 1 << 26
SELECTOR_RANGES_MAX = 1000  # ranges per axis a query match may echo as a selector
QUERY_DEFAULT, QUERY_MAX = 100, 500
QUERY_SCAN_DEFAULT, QUERY_SCAN_MAX = 1 << 27, 1 << 30
QUERY_STATS = ("l1", "l2", "max_abs", "mean", "std", "min", "max", "zero_frac")
QUERY_OPS = {"lt": lambda v, x: v < x, "le": lambda v, x: v <= x, "gt": lambda v, x: v > x, "ge": lambda v, x: v >= x}


# ---------------------------------------------------------------- envelope, arguments, cursors

def to_json(envelope: dict) -> str:
    """Canonical JSON: sorted keys, no spaces, no NaN. Byte-identical for repeated calls."""
    return canonical_json(envelope)


def _envelope(resolved, result, *, checkpoint, selection=None, truncated=False, cursor=None, pair=None):
    ids = {"checkpoint_a": pair[0], "checkpoint_b": pair[1]} if pair else {"checkpoint": checkpoint}
    fingerprints = {**ids, "selection": selection, "op_revision": OP_REVISION}
    return {"schema_version": SCHEMA_VERSION, "resolved": resolved, "result": result, "fingerprints": fingerprints,
            "side_effects": [], "truncated": truncated, "cursor": cursor}


def _arg(field, value, default, low, high):
    if value is None:
        return default
    if type(value) is not int or value < low:
        raise InspectError("invalid_argument", f"{field} must be an integer >= {low}", field)
    if value > high:
        raise InspectError("cap_exceeded", f"{field} may be at most {high}", field)
    return value


def _flag(field, value):
    if type(value) is not bool:
        raise InspectError("invalid_argument", f"{field} must be true or false", field)
    return value


def _cursor_encode(op, rid, position):
    body = canonical_json({"op": op, "rid": rid, "pos": position}).encode()
    return "c1." + base64.urlsafe_b64encode(body).decode()


def _cursor_decode(cursor, op, rid, total):
    if cursor is None:
        return 0
    try:
        if not isinstance(cursor, str) or not cursor.startswith("c1."):
            raise ValueError
        body = json.loads(base64.urlsafe_b64decode(cursor[3:]))
        if body["op"] != op or body["rid"] != rid or type(body["pos"]) is not int or not 0 <= body["pos"] <= total:
            raise ValueError
    except (ValueError, KeyError, TypeError):
        raise InspectError("invalid_cursor", "cursor does not belong to this request", "cursor") from None
    return body["pos"]


def _ident(table: TensorTable, checkpoint_id):
    """What a cursor is bound to besides the request: the checkpoint id, else the parsed file identity."""
    return checkpoint_id if checkpoint_id is not None else table.identity


def _zip_closing(*generators):
    """zip() that closes every generator, so no file handle outlives the comparison."""
    try:
        yield from zip(*generators)
    finally:
        for generator in generators:
            getattr(generator, "close", lambda: None)()


def _page(op, rid, position, size, total):
    """(end, truncated, next_cursor) for a page [position, position+size) of `total` items."""
    end = min(position + size, total)
    more = end < total
    return end, more, (_cursor_encode(op, rid, end) if more else None)


# ---------------------------------------------------------------- number formatting

_SPECIAL = {math.inf: "Inf", -math.inf: "-Inf"}


def _bf16_round(value: float) -> float:
    bits = int(np.float32(value).view(np.uint32))
    bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF) >> 16 << 16
    return float(np.uint32(bits).view(np.float32))


def _roundtrip(candidate: float, dtype: str) -> float:
    if dtype == "F32":
        return float(np.float32(candidate))
    if dtype == "F16":
        return float(np.float16(candidate))
    if dtype == "BF16":
        return _bf16_round(candidate)
    import torch  # F8

    return float(torch.tensor(candidate, dtype=torch.float32).to(getattr(torch, DTYPES[dtype][1])).float())


def shortest(value: float, dtype: str) -> float:
    """Shortest decimal that parses back to `value` in the STORED dtype (F64 keeps Python repr)."""
    if dtype == "F64" or value == 0:
        return value
    for digits in range(1, 18):
        candidate = float(f"{value:.{digits - 1}e}")
        if _roundtrip(candidate, dtype) == value:
            return candidate
    return value


def format_values(array: np.ndarray, dtype: str) -> list:
    """JSON values for stored-dtype elements. Non-finite floats become "NaN", "Inf", "-Inf"."""
    if dtype in ("I64", "U64"):
        return [str(v) for v in array.tolist()]  # beyond 2**53 a JSON number is not exact in JS
    if not is_float(dtype):
        return array.tolist()
    out = []
    for value in array.tolist():
        if value != value:
            out.append("NaN")
        elif value in _SPECIAL:
            out.append(_SPECIAL[value])
        else:
            out.append(shortest(value, dtype))
    return out


def format_f64(array: np.ndarray) -> list:
    return [("NaN" if v != v else _SPECIAL[v] if v in _SPECIAL else v) for v in array.tolist()]


def _num(value):
    """Explicit null for undefined or non-finite metrics."""
    return None if value is None or not math.isfinite(value) else float(value)


def _f64(array: np.ndarray) -> np.ndarray:
    return array.astype(np.float64)


# ---------------------------------------------------------------- accumulators

class _Acc:
    """Per-unit statistics over finite elements, merged chunk by chunk (Chan's algorithm)."""

    def __init__(self):
        self.count = self.nonfinite = self.n = self.zeros = 0
        self.mean = self.m2 = self.l1 = self.sumsq = 0.0
        self.mn = self.mx = None

    def add(self, x: np.ndarray):
        self.count += x.size
        finite = np.isfinite(x)
        if not finite.all():
            self.nonfinite += int(x.size - finite.sum())
            x = x[finite]
        n = x.size
        if n == 0:
            return
        self.zeros += int((x == 0).sum())
        mean = float(x.mean())
        m2 = float(((x - mean) ** 2).sum())
        total = self.n + n
        delta = mean - self.mean
        self.mean += delta * n / total
        self.m2 += m2 + delta * delta * self.n * n / total
        self.n = total
        self.l1 += float(np.abs(x).sum())
        self.sumsq += float((x * x).sum())
        lo, hi = float(x.min()), float(x.max())
        self.mn = lo if self.mn is None else min(self.mn, lo)
        self.mx = hi if self.mx is None else max(self.mx, hi)

    def result(self) -> dict:
        has = self.n > 0
        return {
            "count": self.count, "nonfinite_count": self.nonfinite,
            "min": _num(self.mn), "max": _num(self.mx),
            "mean": _num(self.mean) if has else None,
            "std": _num(math.sqrt(self.m2 / self.n)) if has else None,
            "l1": _num(self.l1) if has else None, "l2": _num(math.sqrt(self.sumsq)) if has else None,
            "zero_frac": self.zeros / self.count if self.count else None,
            "constant": bool(has and self.nonfinite == 0 and self.mn == self.mx),
        }


def _unit_stats(table: TensorTable, sel: Selection) -> dict:
    acc = _Acc()
    for raw in iter_raw(table, sel):
        acc.add(_f64(decode(sel.info.dtype, raw)))
    return acc.result()


def _range_pass(chunks, total: int, dtype_wide: bool) -> dict:
    """min/max/abs_max/abs_p99 (exact nearest-rank) and non-finite count over f64 chunks."""
    buffer = np.empty(total, dtype=np.float64 if dtype_wide else np.float32)
    n = nonfinite = 0
    lo = hi = None
    for x in chunks:
        finite = np.isfinite(x)
        nonfinite += int(x.size - finite.sum())
        x = x[finite]
        if x.size:
            buffer[n:n + x.size] = np.abs(x)
            n += x.size
            lo = float(x.min()) if lo is None else min(lo, float(x.min()))
            hi = float(x.max()) if hi is None else max(hi, float(x.max()))
    if n == 0:
        return {"range": None, "nonfinite_count": nonfinite}
    rank = (99 * n + 99) // 100 - 1  # ceil(0.99 n) - 1
    p99 = float(np.partition(buffer[:n], rank)[rank])
    return {"range": {"min": lo, "max": hi, "abs_max": max(abs(lo), abs(hi)), "abs_p99": p99},
            "nonfinite_count": nonfinite}


def _wide(dtype: str) -> bool:
    return dtype in ("F64", "I32", "U32", "I64", "U64")


class _Pair:
    """Paired comparison of two equally shaped element streams; `changed` is bitwise inequality."""

    def __init__(self):
        self.changed = self.total = self.nboth = 0
        self.maxabs = None
        self.d2 = self.na = self.nb = self.dot = 0.0

    def add(self, dtype, raw_a, raw_b) -> np.ndarray:
        neq = raw_bits(dtype, raw_a) != raw_bits(dtype, raw_b)
        self.total += neq.size
        self.changed += int(neq.sum())
        a, b = _f64(decode(dtype, raw_a)), _f64(decode(dtype, raw_b))
        both = np.isfinite(a) & np.isfinite(b)
        if both.any():
            a, b = a[both], b[both]
            delta = b - a
            top = float(np.abs(delta).max())
            self.maxabs = top if self.maxabs is None else max(self.maxabs, top)
            self.d2 += float((delta * delta).sum())
            self.na += float((a * a).sum())
            self.nb += float((b * b).sum())
            self.dot += float((a * b).sum())
            self.nboth += int(both.sum())
        return neq

    def result(self) -> dict:
        identical = self.d2 == 0
        rel = 0.0 if identical and self.nboth else (None if self.na == 0 or not self.nboth else math.sqrt(self.d2 / self.na))
        if not self.nboth or self.na == 0 or self.nb == 0:
            cosine = None
        else:
            cosine = 1.0 if identical else max(-1.0, min(1.0, self.dot / math.sqrt(self.na * self.nb)))
        return {"changed_elems": self.changed, "max_abs": _num(self.maxabs), "rel_l2": _num(rel),
                "cosine": _num(cosine)}


# ---------------------------------------------------------------- describe

def describe(table: TensorTable, *, name_glob: str | None = None, limit: int | None = None,
             cursor: str | None = None, checkpoint_id: str | None = None, graph=None) -> dict:
    """The tensor table. Never reads tensor data and never hashes. With a `graph` (a GraphView) rows gain roles, binding
    and groups, and the result gains `graph` and the graph's coverage; without one the response is the rev6 one, byte for byte."""
    limit = _arg("limit", limit, DESCRIBE_DEFAULT, 1, DESCRIBE_MAX)
    if name_glob is not None and not isinstance(name_glob, str):
        raise InspectError("invalid_argument", "name_glob must be a string", "name_glob")
    names = sorted(n for n in table.tensors if name_glob is None or fnmatch.fnmatchcase(n, name_glob))
    rid = digest(["describe", _ident(table, checkpoint_id), name_glob] + ([graph.ir["ir_sha256"]] if graph else []))
    position = _cursor_decode(cursor, "describe", rid, len(names))
    end, truncated, next_cursor = _page("describe", rid, position, limit, len(names))
    dtypes: dict[str, int] = {}
    for info in table.tensors.values():
        dtypes[info.dtype] = dtypes.get(info.dtype, 0) + 1
    tensors = []
    for name in names[position:end]:
        info = table.tensors[name]
        tensors.append({
            "name": name, "dtype": info.dtype, "shape": list(info.shape), "numel": math.prod(info.shape),
            "nbytes": info.nbytes, "editable": info.editable,
            "read_only_reason": None if info.dtype in EDITABLE_DTYPES else "dtype_not_editable",
            "shard": info.path.name if table.sharded else None, "tensor_sha256": None,
            "identical_bytes_group": None, "roles": []})
        if graph is not None:
            tensors[-1].update(graph.describe_row(name))
    result = {
        "checkpoint": {"format": "safetensors", "sharded": table.sharded, "shard_count": table.file_count,
                       "file_bytes": table.file_bytes, "tensor_count": len(table.tensors),
                       "element_count": sum(math.prod(i.shape) for i in table.tensors.values()),
                       "metadata": dict(sorted(table.metadata.items()))},
        # Axes, not tensors. No role is ever guessed in v0.2: every axis is unknown.
        "coverage": {"unit": "axes", "derived": 0, "declared": 0, "inferred": 0,
                     "unknown": sum(len(i.shape) for i in table.tensors.values()), "opaque_ops": []},
        "identical_bytes": None, "dtypes": dict(sorted(dtypes.items())), "matched": len(names),
        "offset": position, "tensors": tensors}
    if graph is not None:
        result["graph"] = graph.meta()
        result["coverage"] = graph.coverage()
    return _envelope({"name_glob": name_glob, "limit": limit}, result, checkpoint=checkpoint_id,
                     truncated=truncated, cursor=next_cursor)


# ---------------------------------------------------------------- read_tensor

def _want_range(include_range, position):
    """Range/non-finite pass: first page by default, later pages only when asked (it rescans the selection)."""
    return position == 0 if include_range is None else _flag("include_range", include_range)


def read_tensor(table: TensorTable, selector, *, max_elems: int | None = None, cursor: str | None = None,
                include_range: bool | None = None, checkpoint_id: str | None = None, graph=None) -> dict:
    if _is_group(selector):
        return group_ops.read_tensor(table, graph, selector, max_elems=max_elems, cursor=cursor,
                                     include_range=include_range, checkpoint_id=checkpoint_id)
    sel = parse_selector(table, selector, checkpoint_id=checkpoint_id)
    max_elems = _arg("max_elems", max_elems, READ_DEFAULT, 1, READ_MAX)
    rid = digest(["read_tensor", _ident(table, checkpoint_id), sel.canonical])
    position = _cursor_decode(cursor, "read_tensor", rid, sel.total)
    want_range = _want_range(include_range, position)
    end, truncated, next_cursor = _page("read_tensor", rid, position, max_elems, sel.total)
    budget.charge("values", end - position)  # before anything is read
    result = _read_result(table, sel, position, end, want_range)
    return _envelope({"selector": sel.resolved, "max_elems": max_elems, "include_range": want_range}, result,
                     checkpoint=checkpoint_id,
                     selection=sel.selection_id, truncated=truncated, cursor=next_cursor)


def _read_result(table, sel, position, end, want_range):
    """The result of reading elements [position, end) of a selection (shared by tensor and group selectors)."""
    dtype = sel.info.dtype
    raw = bytearray().join(iter_raw(table, sel, skip=position, take=end - position))
    result = {"tensor": sel.info.name, "dtype": dtype, "shape": list(sel.shape),
              "selection_shape": list(sel.counts), "total_elems": sel.total, "offset": position,
              "returned_elems": end - position, "values": format_values(decode(dtype, raw), dtype)}
    if want_range and sel.total <= RANGE_SCAN_MAX:
        chunks = (_f64(decode(dtype, r)) for r in iter_raw(table, sel))
        result.update(_range_pass(chunks, sel.total, _wide(dtype)))
    elif want_range:
        result.update(range=None, nonfinite_count=None, range_skipped="selection_too_large")
    return result


def _is_group(selector) -> bool:
    return group_ops.is_group(selector)


# ---------------------------------------------------------------- stats

def _axes_arg(field, value, ndim):
    if value is None:
        return list(range(ndim))
    if not isinstance(value, list):
        raise InspectError("invalid_argument", f"{field} must be a list of axes", field)
    for i, axis in enumerate(value):
        if type(axis) is not int or not 0 <= axis < ndim:
            raise InspectError("invalid_axis", f"axis {axis!r} does not exist (tensor has {ndim} axes)",
                               f"{field}[{i}]")
    if len(set(value)) != len(value):
        raise InspectError("invalid_argument", f"{field} has duplicate axes", field)
    return sorted(value)


def stats(table: TensorTable, selector, *, reduce_over_axes=None, bins: int | None = None, limit: int | None = None,
          cursor: str | None = None, checkpoint_id: str | None = None, graph=None) -> dict:
    if _is_group(selector):
        return group_ops.stats(table, graph, selector, reduce_over_axes=reduce_over_axes, bins=bins, limit=limit,
                               cursor=cursor, checkpoint_id=checkpoint_id)
    sel = parse_selector(table, selector, checkpoint_id=checkpoint_id)
    reduce = _axes_arg("reduce_over_axes", reduce_over_axes, len(sel.shape))
    bins = _arg("bins", bins, 0, 0, BINS_MAX)
    limit = _arg("limit", limit, STATS_DEFAULT, 1, STATS_MAX)
    kept = [a for a in range(len(sel.shape)) if a not in reduce]
    unit_count = math.prod(sel.counts[a] for a in kept)
    budget.require_aggregate(sel.total // unit_count if unit_count else sel.total, "reduce_over_axes")  # the elements of each unit
    rid = digest(["stats", _ident(table, checkpoint_id), sel.canonical, reduce, bins])
    position = _cursor_decode(cursor, "stats", rid, unit_count)
    end, truncated, next_cursor = _page("stats", rid, position, limit, unit_count)
    dtype = sel.info.dtype
    edges = None
    if bins:
        if sel.total > HISTOGRAM_SCAN_MAX:
            raise InspectError("cap_exceeded", f"a shared histogram scans the whole selection (at most "
                               f"{HISTOGRAM_SCAN_MAX} elements)", "bins")
        whole = _Acc()
        for raw in iter_raw(table, sel):
            whole.add(_f64(decode(dtype, raw)))
        lo, hi = whole.mn, whole.mx
        edges = [] if lo is None else [lo + (hi - lo) * i / bins for i in range(bins)] + [hi]
    picks = itertools.islice(itertools.product(*[sel.indices(a) for a in kept]), position, end)
    units, counts = [], []
    for index in picks:
        sub = sel.narrow(dict(zip(kept, index)))
        units.append({"index": {str(a): i for a, i in zip(kept, index)}, **_unit_stats(table, sub)})
        if bins:
            histogram = np.zeros(bins, dtype=np.int64)
            for raw in iter_raw(table, sub):
                x = _f64(decode(dtype, raw))
                x = x[np.isfinite(x)]
                if x.size and hi > lo:
                    x = np.clip(np.floor((x - lo) / (hi - lo) * bins).astype(np.int64), 0, bins - 1)
                elif x.size:
                    x = np.zeros(x.size, dtype=np.int64)
                histogram += np.bincount(x.astype(np.int64), minlength=bins)
            counts.append(histogram.tolist())
    result = {"tensor": sel.info.name, "dtype": dtype, "reduce_over_axes": reduce, "unit_axes": kept,
              "unit_count": unit_count, "offset": position, "accumulate_dtype": ACCUMULATE_DTYPE, "units": units}
    if bins:
        result["histogram"] = {"edges": edges, "counts": counts}
    return _envelope({"selector": sel.resolved, "reduce_over_axes": reduce, "bins": bins, "limit": limit},
                     result, checkpoint=checkpoint_id, selection=sel.selection_id, truncated=truncated,
                     cursor=next_cursor)


# ---------------------------------------------------------------- diff

def _unit_axis_arg(value, ndim, field="unit_axis"):
    if value is None or ndim == 0:
        return None
    if type(value) is not int or not 0 <= value < ndim:
        raise InspectError("invalid_axis", f"axis {value!r} does not exist (tensor has {ndim} axes)", field)
    return value


def _compare_whole(ta: TensorTable, tb: TensorTable, name: str, unit_axis) -> dict:
    """Bitwise-compare one tensor present in both files; per-unit changed indices along `unit_axis`."""
    info = ta.tensors[name]
    dtype, shape, item = info.dtype, info.shape, DTYPES[info.dtype][0]
    numel = math.prod(shape)
    block = 1
    if unit_axis is not None and numel:
        inner = math.prod(shape[unit_axis + 1:])
        block = inner if unit_axis == 0 else shape[unit_axis] * inner
        if block > UNIT_BLOCK_MAX:
            raise InspectError("cap_exceeded", f"unit block of {block} elements exceeds {UNIT_BLOCK_MAX}", "unit_axis")
    per = max(1, CHUNK_ELEMS // block) * block
    ranges = [(lo * item, min(per, numel - lo) * item) for lo in range(0, numel, per)]
    pair, flags = _Pair(), (np.zeros(shape[unit_axis], dtype=bool) if unit_axis is not None else None)
    for lo, raw_a, raw_b in _zip_closing(iter(range(0, numel, per)), ta.read_ranges(name, iter(ranges)),
                                         tb.read_ranges(name, iter(ranges))):
        neq = pair.add(dtype, raw_a, raw_b)
        if flags is not None:
            if unit_axis == 0:
                rows = neq.reshape(-1, inner).any(axis=1)
                flags[lo // inner: lo // inner + rows.size] |= rows
            else:
                flags |= neq.reshape(-1, shape[unit_axis], inner).any(axis=(0, 2))
    entry = {"name": name, "dtype": dtype, "shape": list(shape), "numel": numel, "basis": "values",
             **pair.result()}
    entry["status"] = "changed" if pair.changed else "same"
    if flags is not None:
        changed = np.nonzero(flags)[0].tolist()
        entry["unit_axis"] = unit_axis
        entry["changed_units"] = {"total": len(changed), "listed": changed[:UNIT_LIST_MAX],
                                  "truncated": len(changed) > UNIT_LIST_MAX}
    return entry


def diff(table_a: TensorTable, table_b: TensorTable, *, selector=None, name_glob: str | None = None,
         unit_axis=0, values: bool = False, include_same: bool = False, limit: int | None = None,
         max_elems: int | None = None, include_range: bool | None = None, cursor: str | None = None,
         checkpoint_a: str | None = None, checkpoint_b: str | None = None, graph=None) -> dict:
    if _is_group(selector):
        return group_ops.diff(table_a, table_b, graph, selector, values=_flag("values", values), limit=limit,
                              cursor=cursor, checkpoint_a=checkpoint_a, checkpoint_b=checkpoint_b)
    limit = _arg("limit", limit, DIFF_DEFAULT, 1, DIFF_MAX)
    values, include_same = _flag("values", values), _flag("include_same", include_same)
    if name_glob is not None and not isinstance(name_glob, str):
        raise InspectError("invalid_argument", "name_glob must be a string", "name_glob")
    if selector is not None:
        return _diff_selection(table_a, table_b, selector, unit_axis, values, limit, max_elems, include_range,
                               cursor, checkpoint_a, checkpoint_b)
    if values:
        raise InspectError("invalid_argument", "values needs a selector", "values")
    names = sorted(n for n in table_a.tensors.keys() | table_b.tensors.keys()
                   if name_glob is None or fnmatch.fnmatchcase(n, name_glob))
    if unit_axis is not None and (type(unit_axis) is not int or unit_axis < 0):
        raise InspectError("invalid_axis", "unit_axis must be a non-negative integer or null", "unit_axis")
    rows, only_a, only_b, shape_bad, dtype_bad, skipped_small = [], [], [], [], [], []
    same = changed_total = 0
    max_abs = None
    for name in names:
        ia, ib = table_a.tensors.get(name), table_b.tensors.get(name)
        if ib is None:
            only_a.append(name)
            rows.append({"name": name, "status": "only_a", "dtype": ia.dtype, "shape": list(ia.shape)})
        elif ia is None:
            only_b.append(name)
            rows.append({"name": name, "status": "only_b", "dtype": ib.dtype, "shape": list(ib.shape)})
        elif ia.dtype != ib.dtype:
            dtype_bad.append(name)
            rows.append({"name": name, "status": "dtype_mismatch", "a": ia.dtype, "b": ib.dtype})
        elif ia.shape != ib.shape:
            shape_bad.append(name)
            rows.append({"name": name, "status": "shape_mismatch", "a": list(ia.shape), "b": list(ib.shape)})
        else:
            axis = unit_axis if unit_axis is not None and unit_axis < len(ia.shape) else None  # fewer axes: no units
            elems = math.prod(ia.shape)
            if budget.min_aggregate() and elems < budget.min_aggregate():  # a summary of fewer than K elements is those elements
                skipped_small.append(name)
                rows.append({"name": name, "status": "skipped_small", "dtype": ia.dtype, "shape": list(ia.shape)})
                continue
            if axis is not None and budget.min_aggregate() and elems // max(1, ia.shape[axis]) < budget.min_aggregate():
                axis = None  # units too small to report: the tensor's own summary only
            entry = _compare_whole(table_a, table_b, name, axis)
            changed_total += entry["changed_elems"]
            if entry["max_abs"] is not None:
                max_abs = entry["max_abs"] if max_abs is None else max(max_abs, entry["max_abs"])
            if entry["status"] == "same":
                same += 1
                if not include_same:
                    continue
            rows.append(entry)
    listing = rows
    rid = digest(["diff", _ident(table_a, checkpoint_a), _ident(table_b, checkpoint_b), name_glob, unit_axis, include_same])
    position = _cursor_decode(cursor, "diff", rid, len(listing))
    end, truncated, next_cursor = _page("diff", rid, position, limit, len(listing))
    cap = UNIT_LIST_MAX
    summary = {"tensors_total": len(names), "changed": len(names) - same - len(only_a) - len(only_b)
               - len(shape_bad) - len(dtype_bad), "same": same,
               "only_a": only_a[:cap], "only_b": only_b[:cap], "shape_mismatch": shape_bad[:cap],
               "dtype_mismatch": dtype_bad[:cap], "only_a_count": len(only_a), "only_b_count": len(only_b),
               **({"skipped_small": skipped_small[:cap], "skipped_small_count": len(skipped_small)} if budget.min_aggregate() else {}),
               "shape_mismatch_count": len(shape_bad), "dtype_mismatch_count": len(dtype_bad),
               "changed_elems_total": changed_total, "max_abs": max_abs}
    result = {"accumulate_dtype": ACCUMULATE_DTYPE, "summary": summary, "offset": position,
              "tensors": listing[position:end]}
    resolved = {"selector": None, "name_glob": name_glob, "unit_axis": unit_axis, "include_same": include_same,
                "limit": limit}
    return _envelope(resolved, result, checkpoint=None, pair=(checkpoint_a, checkpoint_b), truncated=truncated,
                     cursor=next_cursor)


def _diff_selection(table_a, table_b, selector, unit_axis, values, limit, max_elems, include_range, cursor,
                    checkpoint_a, checkpoint_b):
    sel = parse_selector(table_a, selector, checkpoint_id=checkpoint_a)
    name, info = sel.info.name, sel.info
    other = table_b.tensors.get(name)
    if other is None or other.dtype != info.dtype or other.shape != info.shape:
        raise InspectError("incompatible_tensors", f"{name} is missing or differs in dtype/shape in the other "
                           "checkpoint", "selector.tensor")
    sel_b = Selection(other, sel.axes)
    dtype = info.dtype
    axis = None if values else _unit_axis_arg(unit_axis, len(info.shape))  # values mode returns no unit rows
    if not values:
        budget.require_aggregate(sel.total, "selector")
        if axis is not None and sel.counts[axis]:
            budget.require_aggregate(sel.total // sel.counts[axis], "unit_axis")  # pass unit_axis: null for the whole-selection summary
    ids = (_ident(table_a, checkpoint_a), _ident(table_b, checkpoint_b))
    pair = _Pair()
    for raw_a, raw_b in _zip_closing(iter_raw(table_a, sel), iter_raw(table_b, sel_b)):
        pair.add(dtype, raw_a, raw_b)
    entry = {"name": name, "dtype": dtype, "shape": list(info.shape), "total_elems": sel.total, "basis": "values",
             **pair.result()}
    entry["status"] = "changed" if pair.changed else "same"
    result = {"accumulate_dtype": ACCUMULATE_DTYPE, "tensor": entry}
    resolved = {"selector": sel.resolved, "unit_axis": None if values else axis, "values": values, "limit": limit,
                "max_elems": max_elems}
    if values:
        page = _arg("max_elems", max_elems, READ_DEFAULT, 1, READ_MAX)
        rid = digest(["diff_values", ids, sel.canonical])
        position = _cursor_decode(cursor, "diff", rid, sel.total)
        end, truncated, next_cursor = _page("diff", rid, position, page, sel.total)
        budget.charge("values", end - position)
        raw_a = bytearray().join(iter_raw(table_a, sel, skip=position, take=end - position))
        raw_b = bytearray().join(iter_raw(table_b, sel_b, skip=position, take=end - position))
        a, b = _f64(decode(dtype, raw_a)), _f64(decode(dtype, raw_b))
        result.update(delta_dtype="float64", offset=position, returned_elems=end - position,
                      values=format_f64(b - a),
                      changed=(raw_bits(dtype, raw_a) != raw_bits(dtype, raw_b)).astype(np.int8).tolist())
        want_range = _want_range(include_range, position)
        if want_range and sel.total <= RANGE_SCAN_MAX:
            deltas = (_f64(decode(dtype, rb)) - _f64(decode(dtype, ra))
                      for ra, rb in _zip_closing(iter_raw(table_a, sel), iter_raw(table_b, sel_b)))
            result.update(_range_pass(deltas, sel.total, True))
        elif want_range:
            result.update(range=None, nonfinite_count=None, range_skipped="selection_too_large")
        resolved.update(max_elems=page, include_range=want_range)
        return _envelope(resolved, result, checkpoint=None, pair=(checkpoint_a, checkpoint_b),
                         selection=sel.selection_id, truncated=truncated, cursor=next_cursor)
    if axis is None:
        return _envelope(resolved, result, checkpoint=None, pair=(checkpoint_a, checkpoint_b),
                         selection=sel.selection_id)
    unit_total = sel.counts[axis]
    rid = digest(["diff_units", ids, sel.canonical, axis])
    position = _cursor_decode(cursor, "diff", rid, unit_total)
    end, truncated, next_cursor = _page("diff", rid, position, limit, unit_total)
    rows = []
    for index in sel.indices(axis)[position:end]:
        unit = _Pair()
        for raw_a, raw_b in _zip_closing(iter_raw(table_a, sel.narrow({axis: index})),
                                         iter_raw(table_b, sel_b.narrow({axis: index}))):
            unit.add(dtype, raw_a, raw_b)
        rows.append({"index": {str(axis): index}, **unit.result()})
    result.update(unit_axis=axis, unit_count=unit_total, offset=position, units=rows)
    return _envelope(resolved, result, checkpoint=None, pair=(checkpoint_a, checkpoint_b),
                     selection=sel.selection_id, truncated=truncated, cursor=next_cursor)


# ---------------------------------------------------------------- query

def query(table: TensorTable, *, name_glob: str | None = None, dtype=None, min_ndim: int | None = None,
          max_ndim: int | None = None, editable: bool | None = None, min_bytes: int | None = None,
          max_bytes: int | None = None, where: dict | None = None, limit: int | None = None,
          cursor: str | None = None, max_scan_elems: int | None = None, checkpoint_id: str | None = None,
          graph=None) -> dict:
    if isinstance(where, dict) and "group" in where:
        for field, value in (("name_glob", name_glob), ("dtype", dtype), ("min_ndim", min_ndim), ("max_ndim", max_ndim),
                             ("editable", editable), ("min_bytes", min_bytes), ("max_bytes", max_bytes)):
            if value is not None:  # tensor filters do not apply to a group: refuse them instead of ignoring them
                raise InspectError("invalid_argument", f"{field} cannot be combined with where.group", field)
        return group_ops.query(table, graph, where, limit=limit, cursor=cursor, checkpoint_id=checkpoint_id,
                               max_scan_elems=max_scan_elems)
    limit = _arg("limit", limit, QUERY_DEFAULT, 1, QUERY_MAX)
    scan_cap = _arg("max_scan_elems", max_scan_elems, QUERY_SCAN_DEFAULT, 1, QUERY_SCAN_MAX)
    if name_glob is not None and not isinstance(name_glob, str):
        raise InspectError("invalid_argument", "name_glob must be a string", "name_glob")
    if dtype is not None and (not isinstance(dtype, list) or any(d not in DTYPES for d in dtype)):
        raise InspectError("invalid_argument", "dtype must be a list of safetensors dtypes", "dtype")
    for field, value in (("min_ndim", min_ndim), ("max_ndim", max_ndim), ("min_bytes", min_bytes),
                         ("max_bytes", max_bytes)):
        if value is not None and (type(value) is not int or value < 0):
            raise InspectError("invalid_argument", f"{field} must be a non-negative integer", field)
    if editable is not None and type(editable) is not bool:
        raise InspectError("invalid_argument", "editable must be true or false", "editable")
    predicate = None
    if where is not None:
        if not isinstance(where, dict):
            raise InspectError("invalid_argument", "where must be an object", "where")
        for key in where:
            if key not in ("stat", "axis", "op", "value"):
                raise InspectError("unknown_field", f"unknown field {key!r}", f"where.{key}")
        if where.get("stat") not in QUERY_STATS:
            raise InspectError("invalid_argument", f"stat must be one of {list(QUERY_STATS)}", "where.stat")
        if where.get("op") not in QUERY_OPS:
            raise InspectError("invalid_argument", f"op must be one of {list(QUERY_OPS)}", "where.op")
        if type(where.get("axis")) is not int or where["axis"] < 0:
            raise InspectError("invalid_axis", "axis must be a non-negative integer", "where.axis")
        if type(where.get("value")) not in (int, float) or not math.isfinite(where["value"]):
            raise InspectError("invalid_argument", "value must be a finite number", "where.value")
        predicate = where
    names = []
    for name in sorted(table.tensors):
        info = table.tensors[name]
        if ((name_glob is not None and not fnmatch.fnmatchcase(name, name_glob))
                or (dtype is not None and info.dtype not in dtype)
                or (min_ndim is not None and len(info.shape) < min_ndim)
                or (max_ndim is not None and len(info.shape) > max_ndim)
                or (editable is not None and info.editable != editable)
                or (min_bytes is not None and info.nbytes < min_bytes)
                or (max_bytes is not None and info.nbytes > max_bytes)):
            continue
        if predicate is not None and predicate["axis"] >= len(info.shape):
            continue  # the predicate axis does not exist on this tensor
        names.append(name)
    if predicate is not None:
        scanned = sum(math.prod(table.tensors[n].shape) for n in names)
        if scanned > scan_cap:
            raise InspectError("cap_exceeded", f"query would scan {scanned} elements (limit {scan_cap}); narrow it "
                               "with name_glob/dtype/ndim", "where")  # the predicate is what forces the scan
    matches, skipped_small = [], []
    for name in names:
        whole = parse_selector(table, {"tensor": name})
        if predicate is None:
            matches.append({"name": name, "selector": whole.request_form(), "selector_truncated": False,
                            "selection_id": whole.selection_id,
                            "total_elems": whole.total})
            continue
        axis, compare = predicate["axis"], QUERY_OPS[predicate["op"]]
        if budget.min_aggregate() and whole.total // max(1, whole.shape[axis]) < budget.min_aggregate():
            skipped_small.append(name)  # a statistic of a unit this small is its elements: not evaluated
            continue
        hits = []
        for unit in range(whole.shape[axis]):
            row = _unit_stats(table, whole.narrow({axis: unit}))
            row["max_abs"] = None if row["min"] is None else max(abs(row["min"]), abs(row["max"]))
            value = row[predicate["stat"]]
            if value is not None and compare(value, predicate["value"]):
                hits.append((value, unit))
        if hits:
            hits.sort(key=lambda h: (h[0] if predicate["op"] in ("lt", "le") else -h[0], h[1]))
            chosen = parse_selector(table, {"tensor": name, "index": {str(axis): sorted(u for _, u in hits)}})
            echoed = max(len(r) for r in chosen.axes) <= SELECTOR_RANGES_MAX
            matches.append({"name": name, "selector": chosen.request_form() if echoed else None,
                            "selector_truncated": not echoed, "selection_id": chosen.selection_id,
                            "total_elems": chosen.total, "units_total": len(hits),
                            "units_truncated": len(hits) > UNIT_LIST_MAX,
                            "units": [{"index": {str(axis): u}, "value": v} for v, u in hits[:UNIT_LIST_MAX]]})
    resolved = {"name_glob": name_glob, "dtype": dtype, "min_ndim": min_ndim, "max_ndim": max_ndim,
                "editable": editable, "min_bytes": min_bytes, "max_bytes": max_bytes, "where": predicate,
                "limit": limit, "max_scan_elems": scan_cap}
    rid = digest(["query", _ident(table, checkpoint_id), {k: v for k, v in resolved.items() if k not in ("limit", "max_scan_elems")}])
    position = _cursor_decode(cursor, "query", rid, len(matches))
    end, truncated, next_cursor = _page("query", rid, position, limit, len(matches))
    result = {"matched": len(matches), "offset": position, "matches": matches[position:end]}
    if budget.min_aggregate() and predicate is not None:
        result.update(skipped_small=skipped_small[:UNIT_LIST_MAX], skipped_small_count=len(skipped_small))
    return _envelope(resolved, result, checkpoint=checkpoint_id, truncated=truncated, cursor=next_cursor)
