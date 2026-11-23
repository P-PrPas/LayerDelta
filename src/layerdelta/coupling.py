"""Roles, coupling groups, opaque boundaries and coverage derived from a GraphIR (v0.3-G3).

Generic and torch-free: the input is the JSON graph (graph.py) plus a tensor table. No tensor name and no model
family appears here; every rule is keyed by an aten op name (torch 2.6 export, see RULES_FOR).

Model. Every activation dimension and every parameter axis is a `Dim`: an ordered list of *pieces* (what `cat` makes),
each piece a list of *factors* (what `flatten`/`reshape` make, major to minor), each factor a `Space`: a unit space with a
size and, for each parameter axis that depends on it, the explicit axis indices that belong to each of its units. Ties
(residual adds, convolution in/out channels, ...) merge spaces. `reshape` splits a space into mixed-radix factors,
`cat`/`slice` split it into pieces, and `select`/`unbind` of a packed factor restricts its sibling factors to the chosen
slice, so each of q, k and v gets its own head-dimension space. A *group* is a live space that some parameter axis depends
on; its members are the explicit per-unit index sets compressed into the `{offset, block, period, repeat}` formula
(design section 3.4), and it is *closed* unless a graph input, a graph output, a data dimension or an op without a rule
touches it. Roles come from the rule that consumes each parameter, with a `basis` of `rule@node`.
"""

import math
from dataclasses import dataclass, field

from .graph import bind, table_signature
from .selector import canonical_json, digest
from .tensors import TensorTable

RULES_VERSION = 1
SUPPORTED_TORCH = ("2.6",)


# ---------------------------------------------------------------- spaces and dims

class Space:
    __slots__ = ("id", "size", "members", "dims", "boundary", "free", "kind", "packed", "superseded", "origin", "repl", "restr",
                 "constraints", "dead")

    def __init__(self, sid: int, size: int, free: bool = False, kind: str | None = None):
        self.id, self.size, self.free, self.kind = sid, size, free, kind
        self.members: dict[tuple, list[list[int]]] = {}
        self.dims: set = set()
        self.boundary: dict[tuple, dict] = {}
        self.packed = self.superseded = self.dead = False
        self.origin = None  # (Root, {digit space: fixed value}) when this space is a mixed-radix digit of a root space
        self.restr: list = []  # restricted copies made from this (superseded) digit: a mark or tie here reaches them
        self.repl: list = []  # the live spaces this one became (merge target or decomposition/split parts)
        self.constraints: dict[tuple, dict] = {}


class Root:
    """The members of a space before it was decomposed, and its digits (major to minor); shared by every digit."""
    __slots__ = ("snapshot", "digits")

    def __init__(self, snapshot, digits):
        self.snapshot, self.digits = snapshot, digits


class Dim:
    __slots__ = ("pieces",)

    def __init__(self, pieces):
        self.pieces = pieces
        for piece in pieces:
            for space in piece:
                space.dims.add(self)

    @property
    def size(self) -> int:
        return sum(math.prod(s.size for s in piece) for piece in self.pieces)

    def spaces(self):
        return [s for piece in self.pieces for s in piece]


def _merge_lists(a, b):
    return [sorted(set(x) | set(y)) if (x or y) else [] for x, y in zip(a, b)]


class _Opaque(Exception):
    pass


class Ctx:
    """One derivation: walks the nodes in order, applying the rule of each op."""

    def __init__(self, ir: dict, tensors: dict):
        self.ir, self.tensors = ir, tensors  # tensors: name -> shape (from the table)
        self.next_space = 0
        self.v: dict[str, list[Dim]] = {}
        self.t: dict[str, list] = {}
        self.producer: dict[str, str] = {}
        self.params = {p["value"]: p for p in ir["params"]}
        self.by_name = {p["name"]: p for p in ir["params"]}
        self.axis_dims: dict[tuple, Dim] = {}
        self.roles: dict[tuple, dict] = {}
        self.opaque: dict[str, int] = {}
        self.consumed: set = set()  # parameter names a rule has described
        self.pending_free: dict[str, str] = {}  # committed to free_params only if the rule applied (not opaque)
        self.free_params: dict[str, str] = {}  # parameters that flow through shape/elementwise ops: name -> first node
        self.node = None
        self.used = _used_values(ir)
        self.node_rules: dict = {}
        self.restricted_cache: dict = {}  # (digit, selected digit, k) -> the one restricted space

    # ---- construction
    def space(self, size, free=False, kind=None) -> Space:
        self.next_space += 1
        return Space(self.next_space, size, free, kind)

    def fresh(self, size, kind=None) -> Dim:
        return Dim([[]]) if size == 1 else Dim([[self.space(size, True, kind)]])

    def axis(self, name, axis) -> Dim:
        key = (name, axis)
        if key not in self.axis_dims:
            size = self.tensors[name][axis]
            if size == 1:
                self.axis_dims[key] = Dim([[]])
            else:
                s = self.space(size)
                s.members[key] = [[i] for i in range(size)]
                self.axis_dims[key] = Dim([[s]])
        return self.axis_dims[key]

    def pname(self, ref):
        p = self.params.get(ref) if isinstance(ref, str) else None
        return p["name"] if p else None

    def role(self, name, axis, role, rule):
        key = (name, axis)
        entry = {"role": role, "basis": f"{rule}@{self.node['id']}"}
        old = self.roles.get(key)
        if old is None:
            self.roles[key] = entry
        elif old["role"] != role:
            self.roles[key] = {"role": "unknown", "basis": f"conflict:{old['basis']}|{entry['basis']}"}
        self.consumed.add(name)

    def mark(self, spaces, reason, node=None, **extra):
        node = node or self.node
        for s in self.live(spaces):
            key = (node["id"] if node else "graph", reason, *sorted(extra.items()))
            s.boundary[key] = {"node": node["id"] if node else "graph", "op": node["op"] if node else reason,
                               "reason": reason, **extra}

    # ---- merging, splitting
    def merge(self, a: Space, b: Space) -> Space:
        if a is b:
            return a
        touched = [x for x in (a, b) if x.restr]
        if touched:  # the whole digit is used as one after it was restricted: its copies are no longer independent
            self.mark(touched, "opaque_op")
        for key, lists in b.members.items():
            a.members[key] = _merge_lists(a.members[key], lists) if key in a.members else [list(x) for x in lists]
        a.boundary.update(b.boundary)
        if a.free != b.free:
            self.mark([a], "data_dimension")  # a unit space tied to a data dimension cannot be permuted on its own
        a.free = a.free and b.free
        a.kind = a.kind or b.kind
        a.packed = a.packed or b.packed
        a.superseded = a.superseded or b.superseded
        a.constraints.update(b.constraints)
        for dim in list(b.dims):
            for piece in dim.pieces:
                for i, s in enumerate(piece):
                    if s is b:
                        piece[i] = a
            a.dims.add(dim)
        b.dims.clear()
        b.dead = True
        b.repl = [a]
        if b.origin is not None:
            b.origin[0].digits[:] = [a if d is b else d for d in b.origin[0].digits]
        return a

    def decompose(self, space: Space, sizes: list[int]) -> list[Space]:
        """Mixed-radix split (major to minor); every dim holding `space` now holds its factors."""
        assert math.prod(sizes) == space.size
        if space.origin is not None and space in space.origin[0].digits:
            root, fixed = space.origin
        else:
            root, fixed = Root({k: [list(x) for x in v] for k, v in space.members.items()}, [space]), {}
        factors = []
        for k, n in enumerate(sizes):
            f = self.space(n, space.free, space.kind)
            f.boundary = dict(space.boundary)
            f.constraints, spans_blocks = _factor_blocks(space.constraints, math.prod(sizes[k + 1:]), n)
            if spans_blocks:  # a block of the parent cuts across this factor's units in a way no block size describes
                self.mark([f], "opaque_op")
            f.restr = list(space.restr)  # a mark on any part still reaches the restricted copies of the whole
            f.origin = (root, dict(fixed))
            factors.append(f)
        at = root.digits.index(space)
        root.digits[at:at + 1] = factors
        for f in factors:
            f.members = self._digit_members(root, f, fixed)
        for dim in list(space.dims):
            for piece in dim.pieces:
                if space in piece:
                    i = piece.index(space)
                    piece[i:i + 1] = factors
            for f in factors:
                f.dims.add(dim)
        space.dims.clear()
        space.superseded = space.dead = True
        space.repl = factors
        return factors

    @staticmethod
    def _digit_members(root: Root, f: Space, fixed: dict):
        sizes = [d.size for d in root.digits]
        total = math.prod(sizes)
        strides = [math.prod(sizes[i + 1:]) for i in range(len(sizes))]
        at = root.digits.index(f)
        pins = [(root.digits.index(d), k) for d, k in fixed.items()]
        out = {}
        for key, lists in root.snapshot.items():
            per = [set() for _ in range(f.size)]
            for u in range(total):
                if all((u // strides[i]) % sizes[i] == k for i, k in pins):
                    per[(u // strides[at]) % sizes[at]].update(lists[u])
            if any(per):
                out[key] = [sorted(x) for x in per]
        return out

    def restricted(self, f: Space, sel: Space, k: int) -> Space:
        """`f` with the sibling digit `sel` fixed to k: the head dimension of q, of k or of v."""
        if (f, sel, k) in self.restricted_cache:
            return self.restricted_cache[(f, sel, k)]
        root, fixed = f.origin
        fixed = {**fixed, sel: k}
        g = self.space(f.size, f.free, f.kind)
        g.boundary, g.constraints = dict(f.boundary), dict(f.constraints)
        g.origin = (root, fixed)
        g.members = self._digit_members(root, f, fixed)
        f.restr.append(g)
        self.restricted_cache[(f, sel, k)] = g
        return g

    @staticmethod
    def live(spaces) -> list:
        """The live spaces `spaces` have become (a merged or split space is followed to its successors)."""
        out, todo = [], list(spaces)
        while todo:
            s = todo.pop()
            if s.dead:
                todo.extend(s.repl)
            elif s not in out:
                out.append(s)
                todo.extend(s.restr)  # a restricted digit's copies (q, k, v) see what happens to the whole
        return out

    def split(self, space: Space, sizes: list[int]):
        """Concatenation split into pieces; None (nothing changed) if a dim holds the space in a way that cannot split."""
        assert sum(sizes) == space.size
        refs = []
        for dim in space.dims:
            for pi, piece in enumerate(dim.pieces):
                if space in piece:
                    if piece.index(space) != 0:
                        return None
                    refs.append((dim, pi))
        offsets = [sum(sizes[:i]) for i in range(len(sizes))]
        parts = []
        for n, off in zip(sizes, offsets):
            s = self.space(n, space.free, space.kind)
            s.boundary, s.packed, s.superseded = dict(space.boundary), space.packed, space.superseded
            s.constraints, spans_blocks = _part_blocks(space.constraints, off, n)
            if spans_blocks:
                self.mark([s], "opaque_op")
            s.restr = list(space.restr)
            s.members = {k: [list(x) for x in v[off:off + n]] for k, v in space.members.items()}
            s.members = {k: v for k, v in s.members.items() if any(v)}
            parts.append(s)
        for dim, pi in refs:
            piece = dim.pieces[pi]
            replacement = [[p, *piece[1:]] for p in parts]
            dim.pieces[pi:pi + 1] = replacement
            for p in parts:
                p.dims.add(dim)
            for s in piece[1:]:
                s.dims.add(dim)
        space.dims.clear()
        space.superseded = space.dead = True
        space.repl = parts
        return parts

    # ---- ties
    def _align_pieces(self, da: Dim, db: Dim) -> bool:
        def cuts(d):
            out, at = set(), 0
            for piece in d.pieces:
                at += math.prod(s.size for s in piece)
                out.add(at)
            return out
        for d, other in ((da, db), (db, da)):
            for cut in sorted(cuts(other) - cuts(d)):
                at = 0
                for piece in d.pieces:
                    n = math.prod(s.size for s in piece)
                    if at < cut < at + n:
                        major = piece[0]
                        minor = n // major.size
                        if (cut - at) % minor:
                            return False
                        if self.split(major, [(cut - at) // minor, major.size - (cut - at) // minor]) is None:
                            return False
                        break
                    at += n
        return True

    def _align_factors(self, da: Dim, db: Dim, pi: int) -> bool:
        for d in (da, db):  # a unit-size factor carries no information: drop it so [1, n] lines up with [n]
            d.pieces[pi][:] = [x for x in d.pieces[pi] if x.size != 1]
        i = 0
        while i < min(len(da.pieces[pi]), len(db.pieces[pi])):
            x, y = da.pieces[pi][i], db.pieces[pi][i]
            if x.size != y.size:
                big, small = (x, y) if x.size > y.size else (y, x)
                if big.size % small.size:
                    return False
                self.decompose(big, [small.size, big.size // small.size])
                continue
            i += 1
        return len(da.pieces[pi]) == len(db.pieces[pi])

    def tie(self, da: Dim, db: Dim) -> bool:
        if da is db:
            return True
        if da.size != db.size:
            return False
        if da.size == 1:
            return True
        if not self._align_pieces(da, db) or len(da.pieces) != len(db.pieces):
            return self._tie_failed(da, db)
        for pi in range(len(da.pieces)):
            if not self._align_factors(da, db, pi):
                return self._tie_failed(da, db)
        for pa, pb in zip(da.pieces, db.pieces):
            for sa, sb in zip(list(pa), list(pb)):
                self.merge(sa, sb)
        return True

    def _tie_failed(self, da, db):
        self.mark(da.spaces() + db.spaces(), "opaque_op")
        return False

    # ---- reshape and selection
    def reshape(self, old: list[Dim], shape: list[int]) -> list[Dim]:
        tokens = []  # a space, or a whole multi-piece dim
        for d in old:
            if d.size == 1:
                continue
            tokens.extend(d.pieces[0] if len(d.pieces) == 1 else [d])
        out = []
        i = 0
        for s in shape:
            if s == 1:
                out.append(Dim([[]]))
                continue
            taken, product = [], 1
            while product < s:
                if i >= len(tokens):
                    raise _Opaque("reshape")
                tok = tokens[i]
                size = tok.size
                if product * size <= s:
                    if s % (product * size) and product * size != s:
                        raise _Opaque("reshape")
                    taken.append(tok)
                    product *= size
                    i += 1
                else:
                    need = s // product
                    if isinstance(tok, Dim) or s % product or tok.size % need:
                        raise _Opaque("reshape")
                    first, rest = self.decompose(tok, [need, tok.size // need])
                    taken.append(first)
                    tokens[i] = rest
                    product *= need
            if product != s:
                raise _Opaque("reshape")
            if len(taken) == 1 and isinstance(taken[0], Dim):
                out.append(taken[0])
            elif any(isinstance(t, Dim) for t in taken):
                raise _Opaque("reshape")
            else:
                out.append(Dim([list(taken)]))
        if any(t.size != 1 for t in tokens[i:]):
            raise _Opaque("reshape")
        return out

    def select(self, dims: list[Dim], d: int, k: int) -> list[Dim]:
        target = dims[d]
        rest = dims[:d] + dims[d + 1:]
        spaces = target.spaces()
        if target.size == 1 or not any(s.members for s in spaces):
            return rest
        if len(target.pieces) == 1 and len(target.pieces[0]) == 1 and target.pieces[0][0].origin is not None:
            sel = target.pieces[0][0]
            sel.packed = True
            cache: dict = {}
            result = []
            for dim in rest:
                pieces = []
                for piece in dim.pieces:
                    new = []
                    for s in piece:
                        if s.origin is not None and s.origin[0] is sel.origin[0] and s is not sel and s in s.origin[0].digits:
                            if s not in cache:
                                cache[s] = self.restricted(s, sel, k)
                            s.superseded = True
                            new.append(cache[s])
                        else:
                            new.append(s)
                    pieces.append(new)
                result.append(dim if all(a is b for pa, pb in zip(pieces, dim.pieces) for a, b in zip(pa, pb)) else Dim(pieces))
            return result
        # an indexed unit: the result depends on which unit was picked, so permuting that space is not function-preserving
        self.mark(spaces, "indexed_unit")
        return rest

    def piece_range(self, dim: Dim, start: int, n: int) -> Dim | None:
        """The units [start, start+n) of a dim as a new dim; None if they cannot be addressed."""
        if n == dim.size and start == 0:
            return dim
        if not any(s.members for s in dim.spaces()):
            return self.fresh(n)
        pieces, at = dim.pieces, 0
        out = []
        for piece in pieces:
            size = math.prod(s.size for s in piece)
            lo, hi = max(start, at), min(start + n, at + size)
            if lo < hi:
                if (lo, hi) == (at, at + size):
                    out.append(piece)
                elif len(piece) == 1:
                    parts = self.split(piece[0], [x for x in (lo - at, hi - lo, at + size - hi) if x])
                    if parts is None:
                        return None
                    out.append([parts[1 if lo > at else 0]])
                else:
                    return None
            at += size
        if len(out) != 1 and not out:
            return None
        return Dim([list(p) for p in out])

    # ---- values
    def get(self, ref):
        if not isinstance(ref, str):
            return None
        if ref in self.v:
            return self.v[ref]
        p = self.params.get(ref)
        if p:
            self.pending_free.setdefault(p["name"], self.node["id"])
            return [self.axis(p["name"], a) for a in range(len(p["shape"]))]
        return None


def _factor_blocks(constraints: dict, stride: int, size: int):
    """A `within_blocks n` constraint counts units of the space a factor was cut from. For the factor whose units are `stride`
    parent units apart and `size` of them: n a multiple of stride*size keeps every factor unit inside one block (no constraint),
    n a multiple of stride that divides stride*size becomes n/stride, anything else cannot be described (second value True)."""
    out, bad = {}, False
    for key, c in constraints.items():
        if c["kind"] != "within_blocks":
            out[key] = c
            continue
        n = c["n"]
        if n % (stride * size) == 0:
            continue
        if n % stride == 0 and (stride * size) % n == 0:
            out[("within_blocks", n // stride)] = {"kind": "within_blocks", "n": n // stride}
        else:
            bad = True
    return out, bad


def _part_blocks(constraints: dict, offset: int, size: int):
    """The same for a piece of a concatenation: units [offset, offset+size) of the parent."""
    out, bad = {}, False
    for key, c in constraints.items():
        if c["kind"] != "within_blocks":
            out[key] = c
            continue
        n = c["n"]
        if offset % n == 0 and size % n == 0:
            out[key] = c
        elif offset // n != (offset + size - 1) // n:
            bad = True  # the piece starts or ends inside a block and spans more than one
    return out, bad


def _used_values(ir: dict) -> set:
    used = set(ir["outputs"])

    def walk(x):
        if isinstance(x, str):
            used.add(x)
        elif isinstance(x, list):
            for y in x:
                walk(y)
        elif isinstance(x, dict) and set(x) not in ({"str"}, {"enum"}, {"float"}):
            for y in x.values():
                walk(y)
    for node in ir["nodes"]:
        walk(node["args"])
        walk(list(node["kwargs"].values()))
    return used


def _shape(node, i=0):
    out = node["outputs"][i]
    if out["kind"] != "tensor":  # no static shape (a provider could not infer one): the op cannot be tracked
        raise _Opaque("no_shape")
    return out["shape"]


def _arg(node, i, default=None):
    a = node["args"]
    return a[i] if i < len(a) else default


def _kw(node, name, i, default=None):
    if name in node["kwargs"]:
        return node["kwargs"][name]
    return _arg(node, i, default)


def _norm(d, n):
    return d + n if d < 0 else d


# ---------------------------------------------------------------- rules

def r_conv(c: Ctx, node, transposed=False):
    x, w, b = c.get(_arg(node, 0)), c.pname(_arg(node, 1)), c.pname(_arg(node, 2))
    groups = _kw(node, "groups", 6, 1)
    if x is None or w is None:
        raise _Opaque("conv")
    nd = len(c.tensors[w]) - 2
    rule = "convT.kernel" if transposed else "conv.kernel"
    c.role(w, 0, "in_unit" if transposed else "out_unit", rule)
    c.role(w, 1, "out_unit" if transposed else "in_unit", rule)
    for a in range(2, nd + 2):
        c.role(w, a, "spatial", rule)
    if b:
        c.role(b, 0, "channel", "conv.shift")
    shape = _shape(node)
    out_dim = c.axis(w, 1 if transposed else 0)
    in_dim = c.axis(w, 0 if transposed else 1)
    if transposed:
        if groups == 1:
            c.tie(x[1], c.axis(w, 0))
        else:
            c.mark(x[1].spaces() + c.axis(w, 0).spaces() + out_dim.spaces(), "opaque_op")
    else:
        i_total, o_total = c.tensors[w][1] * groups, c.tensors[w][0]
        if groups == 1:
            c.tie(x[1], in_dim)
        elif groups == o_total == x[1].size and c.tensors[w][1] == 1:  # depthwise: one channel space
            c.tie(x[1], out_dim)
        else:
            c.mark(x[1].spaces() + in_dim.spaces() + out_dim.spaces(), "opaque_op")
    if b:
        c.tie(c.axis(b, 0), out_dim)
    c.v[node["outputs"][0]["value"]] = [x[0], out_dim] + [c.fresh(n) for n in shape[2:]]


def r_linear(c: Ctx, node):
    x, w, b = c.get(_arg(node, 0)), c.pname(_arg(node, 1)), c.pname(_arg(node, 2))
    if x is None or w is None:
        raise _Opaque("linear")
    c.role(w, 0, "out_unit", "linear.kernel")
    c.role(w, 1, "in_unit", "linear.kernel")
    c.tie(x[-1], c.axis(w, 1))
    out = c.axis(w, 0)
    if b:
        c.role(b, 0, "channel", "linear.shift")
        c.tie(c.axis(b, 0), out)
    c.v[node["outputs"][0]["value"]] = x[:-1] + [out]


def r_batch_norm(c: Ctx, node):
    x = c.get(_arg(node, 0))
    if x is None:
        raise _Opaque("batch_norm")
    for i in (1, 2, 3, 4):
        p = c.pname(_arg(node, i))
        if p:
            c.role(p, 0, "channel", "norm.param" if i < 3 else "norm.stat")  # weight, bias | running mean, running variance
            c.tie(c.axis(p, 0), x[1])
    c.v[node["outputs"][0]["value"]] = x


def r_layer_norm(c: Ctx, node):
    x, shape = c.get(_arg(node, 0)), _arg(node, 1)
    if x is None or not isinstance(shape, list):
        raise _Opaque("layer_norm")
    for i in (2, 3):
        p = c.pname(_arg(node, i))
        if p:
            for j in range(len(shape)):
                c.role(p, j, "channel", "norm.param")
                c.tie(c.axis(p, j), x[len(x) - len(shape) + j])
    c.v[node["outputs"][0]["value"]] = x


def r_group_norm(c: Ctx, node):
    x, groups = c.get(_arg(node, 0)), _arg(node, 1)
    if x is None:
        raise _Opaque("group_norm")
    for i in (2, 3):
        p = c.pname(_arg(node, i))
        if p:
            c.role(p, 0, "channel", "norm.param")
            c.tie(c.axis(p, 0), x[1])
    n, at = x[1].size // groups, 0
    for piece in x[1].pieces:  # a block of n channels must lie inside one unit space: only then is it a constraint on it
        size = math.prod(sp.size for sp in piece)
        if len(piece) == 1 and size % n == 0 and at % n == 0:
            piece[0].constraints[("within_blocks", n)] = {"kind": "within_blocks", "n": n}
        elif size > 1:
            c.mark(piece, "opaque_op")
        at += size
    c.v[node["outputs"][0]["value"]] = x


def r_embedding(c: Ctx, node):
    w, idx = c.pname(_arg(node, 0)), c.get(_arg(node, 1))
    if w is None or idx is None:
        raise _Opaque("embedding")
    c.role(w, 0, "vocab", "embedding.table")
    c.role(w, 1, "channel", "embedding.table")
    c.v[node["outputs"][0]["value"]] = idx + [c.axis(w, 1)]


def r_recurrent(c: Ctx, node, gates: int, cell: bool):
    x, hx, params = c.get(_arg(node, 0)), _arg(node, 1), _arg(node, 2)
    has_bias, layers, bidir, batch_first = _arg(node, 3), _arg(node, 4), _arg(node, 7), _arg(node, 8)
    if x is None or not isinstance(params, list) or not all(isinstance(p, str) for p in params):
        raise _Opaque("recurrent")
    dirs = 2 if bidir else 1
    per = (4 if has_bias else 2)
    proj = len(params) == layers * dirs * (per + 1)  # the node itself says whether projection weights are present
    if len(params) != layers * dirs * (per + (1 if proj else 0)):
        raise _Opaque("recurrent")
    stride = per + (1 if proj else 0)
    time_dim = 1 if batch_first else 0
    prev = x[-1]
    state_spaces = []
    for layer in range(layers):
        outs = []
        for d in range(dirs):
            names = [c.pname(p) for p in params[(layer * dirs + d) * stride:(layer * dirs + d + 1) * stride]]
            w_ih, w_hh = names[0], names[1]
            b_ih, b_hh = (names[2], names[3]) if has_bias else (None, None)
            w_hr = names[-1] if proj else None
            if None in [w_ih, w_hh] or (has_bias and None in (b_ih, b_hh)) or (proj and w_hr is None):
                raise _Opaque("recurrent")
            c.role(w_ih, 0, "out_unit", "rnn.input_kernel")
            c.role(w_ih, 1, "in_unit", "rnn.input_kernel")
            c.role(w_hh, 0, "out_unit", "rnn.recurrent_kernel")
            c.role(w_hh, 1, "in_unit", "rnn.recurrent_kernel")
            rows = c.axis(w_ih, 0)
            size = c.tensors[w_ih][0] // gates
            gate, hidden = c.decompose(rows.pieces[0][0], [gates, size])
            gate.packed, gate.kind, hidden.kind = True, "gate", "hidden"
            c.tie(c.axis(w_hh, 0), rows)
            for b_ in (b_ih, b_hh):
                if b_:
                    c.role(b_, 0, "channel", "rnn.shift")
                    c.tie(c.axis(b_, 0), rows)
            hid = Dim([c.live([hidden])])  # the row ties above may have merged `hidden` into another space
            if proj:
                c.role(w_hr, 0, "out_unit", "rnn.projection_kernel")
                c.role(w_hr, 1, "in_unit", "rnn.projection_kernel")
                c.tie(c.axis(w_hr, 1), hid)
                state = c.axis(w_hr, 0)
            else:
                state = hid
            c.tie(c.axis(w_hh, 1), state)
            c.tie(c.axis(w_ih, 1), prev) if dirs == 1 or d == 0 else None
            outs.append(state)
            state_spaces += state.spaces() + hid.spaces()
            if d == 0 and dirs == 2:
                in_dim = c.axis(w_ih, 1)
            elif d == 1:
                c.tie(c.axis(w_ih, 1), in_dim)
        merged = Dim([p for o in outs for p in o.pieces])
        prev = merged
    hx_refs = hx if isinstance(hx, list) else [hx]
    for r in hx_refs:
        if isinstance(r, str) and c.producer.get(r) not in ("aten.zeros.default",):
            c.mark(state_spaces, "recurrent_initial_state")
    items = node["outputs"][0].get("items") or []
    dims = x[:-1] + [prev]
    value = node["outputs"][0]["value"]
    c.t[value] = [dims] + [[c.fresh(n) for n in it["shape"]] if it else None for it in items[1:]]
    c.t[value + ":state"] = state_spaces
    c.v[value] = dims


def r_cat(c: Ctx, node):
    refs, dim = _arg(node, 0), _arg(node, 1, 0)
    xs = [c.get(r) for r in refs] if isinstance(refs, list) else None
    if not xs or any(v is None for v in xs):
        raise _Opaque("cat")
    rank = len(xs[0])
    dim = _norm(dim, rank)
    out = []
    for i in range(rank):
        if i == dim:
            pieces = [p for v in xs for p in v[i].pieces]
            out.append(Dim([list(p) for p in pieces]) if len(xs) > 1 else xs[0][i])
        else:
            for v in xs[1:]:
                c.tie(xs[0][i], v[i])
            out.append(xs[0][i])
    c.v[node["outputs"][0]["value"]] = out


def r_binary(c: Ctx, node):
    a, b = c.get(_arg(node, 0)), c.get(_arg(node, 1))
    if a is None and b is None:
        raise _Opaque("binary")
    if a is None or b is None:
        c.v[node["outputs"][0]["value"]] = a if a is not None else b
        return
    n = max(len(a), len(b))
    pa, pb = [None] * (n - len(a)) + a, [None] * (n - len(b)) + b
    out = []
    for x, y in zip(pa, pb):
        if x is None or (y is not None and x.size == 1 and y.size > 1):
            out.append(y)
        elif y is None or y.size == 1:
            out.append(x)
        else:
            c.tie(x, y)
            out.append(x)
    c.v[node["outputs"][0]["value"]] = out


def r_passthrough(c: Ctx, node):
    x = c.get(_arg(node, 0))
    if x is None:
        raise _Opaque("unary")
    c.v[node["outputs"][0]["value"]] = x


def r_reshape(c: Ctx, node):
    x = c.get(_arg(node, 0))
    if x is None:
        raise _Opaque("reshape")
    c.v[node["outputs"][0]["value"]] = c.reshape(x, _shape(node))


def r_permute(c: Ctx, node):
    x, dims = c.get(_arg(node, 0)), _arg(node, 1)
    if x is None or not isinstance(dims, list):
        raise _Opaque("permute")
    c.v[node["outputs"][0]["value"]] = [x[_norm(d, len(x))] for d in dims]


def r_transpose(c: Ctx, node):
    x = c.get(_arg(node, 0))
    if x is None:
        raise _Opaque("transpose")
    a, b = _norm(_arg(node, 1), len(x)), _norm(_arg(node, 2), len(x))
    out = list(x)
    out[a], out[b] = out[b], out[a]
    c.v[node["outputs"][0]["value"]] = out


def r_t(c: Ctx, node):
    x = c.get(_arg(node, 0))
    if x is None:
        raise _Opaque("t")
    c.v[node["outputs"][0]["value"]] = list(reversed(x)) if len(x) == 2 else x


def r_expand(c: Ctx, node):
    x = c.get(_arg(node, 0))
    if x is None:
        raise _Opaque("expand")
    shape = _shape(node)
    lead = len(shape) - len(x)
    out = [c.fresh(n) for n in shape[:lead]]
    for d, n in zip(x, shape[lead:]):
        out.append(d if d.size == n else c.fresh(n))
    c.v[node["outputs"][0]["value"]] = out


def r_select(c: Ctx, node):
    x = c.get(_arg(node, 0))
    if x is None:
        raise _Opaque("select")
    d = _norm(_arg(node, 1), len(x))
    k = _arg(node, 2)
    n = x[d].size
    c.v[node["outputs"][0]["value"]] = c.select(x, d, k % n if n else k)


def r_unbind(c: Ctx, node):
    x = c.get(_arg(node, 0))
    if x is None:
        raise _Opaque("unbind")
    d = _norm(_arg(node, 1, 0), len(x))
    c.t[node["outputs"][0]["value"]] = [c.select(x, d, k) for k in range(x[d].size)]


def r_slice(c: Ctx, node):
    x = c.get(_arg(node, 0))
    if x is None:
        raise _Opaque("slice")
    d = _norm(_arg(node, 1, 0), len(x))
    n = _shape(node)[d]
    if _kw(node, "step", 4, 1) != 1:  # every other unit: the result is not a contiguous range of the input
        c.mark(x[d].spaces(), "opaque_op")
        c.v[node["outputs"][0]["value"]] = x[:d] + [c.fresh(n)] + x[d + 1:]
        return
    start = _arg(node, 2, 0) or 0
    start = start + x[d].size if start < 0 else start
    new = c.piece_range(x[d], min(start, x[d].size - n), n)
    if new is None:
        c.mark(x[d].spaces(), "opaque_op")
        new = c.fresh(n)
    c.v[node["outputs"][0]["value"]] = x[:d] + [new] + x[d + 1:]


def r_split(c: Ctx, node):
    x = c.get(_arg(node, 0))
    if x is None:
        raise _Opaque("split")
    d = _norm(_arg(node, 2, 0), len(x))
    items = node["outputs"][0].get("items") or []
    start, out = 0, []
    for it in items:
        n = it["shape"][d]
        new = c.piece_range(x[d], start, n)
        if new is None:
            c.mark(x[d].spaces(), "opaque_op")
            new = c.fresh(n)
        out.append(x[:d] + [new] + x[d + 1:])
        start += n
    c.t[node["outputs"][0]["value"]] = out


def r_getitem(c: Ctx, node):
    ref, i = _arg(node, 0), _arg(node, 1)
    if ref in c.t and isinstance(i, int) and 0 <= i < len(c.t[ref]) and c.t[ref][i] is not None:
        c.v[node["outputs"][0]["value"]] = c.t[ref][i]
        if i >= 1 and ref + ":state" in c.t and node["outputs"][0]["value"] in c.used:  # a consumed hidden/cell state is observable
            c.mark(c.t[ref + ":state"], "recurrent_state_used")
        return
    raise _Opaque("getitem")


def r_reduce(c: Ctx, node, keep_at: int = 2):
    x = c.get(_arg(node, 0))
    dims, keep = _kw(node, "dim", 1), _kw(node, "keepdim", keep_at, False)
    if x is None:
        raise _Opaque("reduce")
    dims = [dims] if isinstance(dims, int) else (dims if isinstance(dims, list) else list(range(len(x))))
    dims = {_norm(d, len(x)) for d in dims}
    out = []
    for i, d in enumerate(x):
        if i in dims:
            if keep:
                out.append(Dim([[]]))
        else:
            out.append(d)
    c.v[node["outputs"][0]["value"]] = out


def r_pool(c: Ctx, node):
    x = c.get(_arg(node, 0))
    if x is None:
        raise _Opaque("pool")
    shape = _shape(node)
    c.v[node["outputs"][0]["value"]] = x[:2] + [c.fresh(n) for n in shape[2:]]


def r_matmul(c: Ctx, node):
    a, b = c.get(_arg(node, 0)), c.get(_arg(node, 1))
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        raise _Opaque("matmul")
    c.tie(a[-1], b[-2])
    for ref, nd, left in ((_arg(node, 0), len(a), True), (_arg(node, 1), len(b), False)):
        name = c.pname(ref)  # a parameter used as a matmul operand: its contraction axis is the input side
        if name:
            c.role(name, nd - 1 if left else nd - 2, "in_unit", "matmul.operand")
            c.role(name, nd - 2 if left else nd - 1, "out_unit", "matmul.operand")
    n = max(len(a), len(b)) - 2
    ba, bb = [None] * (n - len(a) + 2) + a[:-2], [None] * (n - len(b) + 2) + b[:-2]
    batch = []
    for x, y in zip(ba, bb):
        if x is None or (y is not None and x.size == 1 and y.size > 1):
            batch.append(y)
        elif y is None or y.size == 1:
            batch.append(x)
        else:
            c.tie(x, y)
            batch.append(x)
    if len(a) >= 4:  # labels only: the leading batch dim is the batch, the others are heads
        for i, d in enumerate(batch):
            if i > 0:
                for s in d.spaces():
                    s.kind = s.kind or ("head" if s.members else None)
        for s in a[-1].spaces():
            s.kind = s.kind or ("qk_dim" if s.members else None)  # only a unit space that parameters index is a dimension
        for s in b[-1].spaces():
            s.kind = s.kind or ("v_dim" if s.members else None)
    c.v[node["outputs"][0]["value"]] = batch + [a[-2], b[-1]]


def r_sdpa(c: Ctx, node):
    q, k, v = c.get(_arg(node, 0)), c.get(_arg(node, 1)), c.get(_arg(node, 2))
    if None in (q, k, v):
        raise _Opaque("sdpa")
    c.tie(q[-1], k[-1])
    c.tie(k[-2], v[-2])
    for d in q[1:-2]:
        for sp in d.spaces():
            sp.kind = sp.kind or ("head" if sp.members else None)
    for sp in q[-1].spaces():
        sp.kind = sp.kind or ("qk_dim" if sp.members else None)
    for sp in v[-1].spaces():
        sp.kind = sp.kind or ("v_dim" if sp.members else None)
    for x, y in zip(q[:-2], k[:-2]):
        c.tie(x, y)
    for x, y in zip(q[:-2], v[:-2]):
        c.tie(x, y)
    mask = _kw(node, "attn_mask", 3)
    if isinstance(mask, str):  # tie the mask's non-broadcast dims to the units they are indexed by (a data mask goes open)
        mk = c.get(mask)
        if mk is None:
            c.mark([sp for d in q[:-2] for sp in d.spaces()], "opaque_op")
        else:
            pairs = [(mk[-1], k[-2]), (mk[-2], q[-2])] + list(zip(reversed(mk[:-2]), reversed(q[:-2])))
            for md, qd in pairs:
                if md.size > 1:
                    c.tie(qd, md)
    c.v[node["outputs"][0]["value"]] = q[:-1] + [v[-1]]


def r_create(c: Ctx, node):
    out = node["outputs"][0]
    if out["kind"] == "tensor":
        c.v[out["value"]] = [c.fresh(n) for n in out["shape"]]
    else:
        raise _Opaque("create")


def _table() -> dict:
    t = {}
    for name in ("conv1d", "conv2d", "conv3d"):
        t[f"aten.{name}.default"] = r_conv
    for name in ("conv_transpose1d", "conv_transpose2d", "conv_transpose3d"):
        t[f"aten.{name}.input"] = lambda c, n: r_conv(c, n, transposed=True)
    t["aten.linear.default"] = r_linear
    t["aten.batch_norm.default"] = r_batch_norm
    t["aten.layer_norm.default"] = r_layer_norm
    t["aten.group_norm.default"] = r_group_norm
    t["aten.embedding.default"] = r_embedding
    t["aten.lstm.input"] = lambda c, n: r_recurrent(c, n, 4, True)
    t["aten.gru.input"] = lambda c, n: r_recurrent(c, n, 3, False)
    t["aten.rnn_tanh.input"] = t["aten.rnn_relu.input"] = lambda c, n: r_recurrent(c, n, 1, False)
    t["aten.cat.default"] = t["aten.concat.default"] = r_cat
    for name in ("add", "sub", "mul", "div", "maximum", "minimum"):
        t[f"aten.{name}.Tensor"] = r_binary
    for name in ("add", "sub", "mul", "div", "pow"):
        t[f"aten.{name}.Scalar"] = r_passthrough
    t["aten.pow.Tensor_Scalar"] = r_passthrough
    for name in ("relu", "gelu", "silu", "sigmoid", "tanh", "elu", "leaky_relu", "hardtanh", "hardswish", "hardsigmoid",
                 "softplus", "mish", "dropout", "clone", "contiguous", "alias", "detach", "neg", "abs", "exp", "log", "sqrt",
                 "rsqrt", "erf", "sin", "cos", "clamp", "relu6", "zeros_like", "ones_like", "_to_copy"):
        t[f"aten.{name}.default"] = r_passthrough
    t["aten.to.dtype"] = t["aten.dropout.default"] = r_passthrough
    for name in ("softmax.int", "log_softmax.int", "_softmax.default", "_log_softmax.default"):
        t[f"aten.{name}"] = r_passthrough
    for name in ("reshape.default", "view.default", "flatten.using_ints", "unflatten.int", "squeeze.dim", "squeeze.dims",
                 "squeeze.default", "unsqueeze.default", "view_as.default"):
        t[f"aten.{name}"] = r_reshape
    t["aten.permute.default"] = r_permute
    t["aten.transpose.int"] = t["aten.swapaxes.default"] = r_transpose
    t["aten.t.default"] = r_t
    t["aten.expand.default"] = r_expand
    t["aten.select.int"] = r_select
    t["aten.unbind.int"] = r_unbind
    t["aten.slice.Tensor"] = r_slice
    t["aten.narrow.default"] = r_slice
    for name in ("split.Tensor", "chunk.default", "split_with_sizes.default"):
        t[f"aten.{name}"] = r_split
    t["_operator.getitem"] = r_getitem
    for name in ("mean.dim", "sum.dim_IntList", "amax.default", "amin.default", "var.correction", "prod.dim_int"):
        t[f"aten.{name}"] = r_reduce
    t["aten.var.dim"] = lambda c, n: r_reduce(c, n, keep_at=3)  # (self, dim, unbiased, keepdim)
    for name in ("max_pool1d", "max_pool2d", "max_pool3d", "avg_pool1d", "avg_pool2d", "avg_pool3d", "adaptive_avg_pool1d",
                 "adaptive_avg_pool2d", "adaptive_avg_pool3d"):
        t[f"aten.{name}.default"] = r_pool
    for name in ("upsample_nearest1d", "upsample_nearest2d", "upsample_nearest3d", "upsample_bilinear2d", "upsample_bicubic2d"):
        t[f"aten.{name}.vec"] = r_pool
    t["aten.matmul.default"] = t["aten.bmm.default"] = t["aten.mm.default"] = r_matmul
    t["aten.scaled_dot_product_attention.default"] = r_sdpa
    for name in ("zeros", "ones", "full", "empty", "arange", "rand", "randn"):
        t[f"aten.{name}.default"] = r_create
    return t


# The rule table is versioned by the torch release whose export op names it matches; any other release is opaque.
RULES_FOR = {"2.6": _table()}


# ---------------------------------------------------------------- derivation

def _fit(lists: list[list[int]]):
    """Compress per-unit index lists into (offset, block, period, repeat), or None if they are not that shape."""
    if not lists or any(not x for x in lists):
        return None
    first = sorted(set(lists[0]))
    runs, run = [], [first[0]]
    for v in first[1:]:
        if v == run[-1] + 1:
            run.append(v)
        else:
            runs.append(run)
            run = [v]
    runs.append(run)
    block, repeat = len(runs[0]), len(runs)
    if any(len(r) != block for r in runs):
        return None
    period = runs[1][0] - runs[0][0] if repeat > 1 else block
    if repeat > 1 and any(runs[i + 1][0] - runs[i][0] != period for i in range(repeat - 1)):
        return None
    offset = first[0]
    for u, got in enumerate(lists):
        want = sorted(offset + u * block + k * period + j for k in range(repeat) for j in range(block))
        if want != sorted(set(got)):
            return None
    if repeat > 1 and period == block:
        block, repeat = block * repeat, 1
    return offset, block, period if repeat > 1 else block, repeat


@dataclass
class Derived:
    ir_sha256: str
    table_signature: str
    roles: dict[str, list[dict]]
    groups: list[dict]
    opaque_ops: list[dict]
    coverage: dict
    rules: dict
    warnings: list = field(default_factory=list)
    node_rules: dict = field(default_factory=dict)  # node id -> {"rule": op name, "opaque": None} | {"rule": None, "opaque": reason}

    def to_dict(self) -> dict:
        return {"ir_sha256": self.ir_sha256, "table_signature": self.table_signature, "roles": self.roles,
                "groups": self.groups, "opaque_ops": self.opaque_ops, "coverage": self.coverage, "rules": self.rules,
                "warnings": self.warnings, "node_rules": self.node_rules}

    def sha256(self) -> str:
        return digest(self.to_dict())


def derive(ir: dict, table: TensorTable) -> Derived:
    bound = bind(ir, table)
    shapes = {n: list(i.shape) for n, i in table.tensors.items()}
    for p in ir["params"]:
        shapes.setdefault(p["name"], p["shape"])
    c = Ctx(ir, {**shapes, **{p["name"]: p["shape"] for p in ir["params"]}})
    version = ir["provider"]["version"].removeprefix("torch-")
    warnings = []
    if ir["provider"]["name"] == "onnx":
        from .coupling_onnx import rules_for
        rules = rules_for(version)
        if not rules:
            warnings.append(f"no rule table for {version}: every op is treated as opaque")
    else:
        release = ".".join(version.split(".")[:2])
        rules = RULES_FOR.get(release)
        if rules is None:
            warnings.append(f"no rule table for torch {version}: every op is treated as opaque")
            rules = {}
    for item in ir["inputs"]:
        dims = [c.fresh(n) for n in item["shape"]]
        for d in dims:
            c.mark(d.spaces(), "graph_input", None)
        c.v[item["value"]] = dims
    for node in ir["nodes"]:
        c.node = node
        for out in node["outputs"]:
            c.producer[out["value"]] = node["op"]
        rule = rules.get(node["op"])
        c.node_rules[node["id"]] = {"rule": node["op"] if rule else None, "opaque": None if rule else "no_rule"}
        try:
            if rule is None:
                raise _Opaque(node["op"])
            c.pending_free = {}
            rule(c, node)
            for name, at in c.pending_free.items():
                c.free_params.setdefault(name, at)
        except _Opaque:
            c.pending_free = {}
            c.node_rules[node["id"]] = {"rule": None, "opaque": "untracked_dimensions" if rule else "no_rule"}
            _opaque(c, node)
    for index, out in enumerate(ir["outputs"]):
        for dims in ([c.v[out]] if out in c.v else c.t.get(out, [])):
            for d in dims or []:
                c.mark(d.spaces(), "graph_output", None, output=index)
    return _finish(c, ir, bound, table, warnings)


def _opaque(c: Ctx, node):
    c.opaque[node["op"]] = c.opaque.get(node["op"], 0) + 1
    for ref in [*node["args"], *node["kwargs"].values()]:
        for r in (ref if isinstance(ref, list) else [ref]):
            dims = c.get(r) if isinstance(r, str) else None
            if dims:
                for d in dims:
                    c.mark(d.spaces(), "opaque_op")
    for out in node["outputs"]:
        if out["kind"] == "tensor":
            dims = [c.fresh(n) for n in out["shape"]]
            c.v[out["value"]] = dims
        elif out["kind"] == "tuple":
            c.t[out["value"]] = [[c.fresh(n) for n in it["shape"]] if it else None for it in out["items"]]
            dims = [d for t in c.t[out["value"]] if t for d in t]
        else:
            continue
        for d in dims:
            c.mark(d.spaces(), "opaque_op")


def _group_id(ir_sha, size, members) -> str:
    return "g_" + digest({"ir": ir_sha, "size": size, "members": members})[:24]


def _finish(c: Ctx, ir: dict, bound, table: TensorTable, warnings: list) -> Derived:
    # ---- roles: one entry per axis, derived by a rule or unknown
    roles: dict[str, list[dict]] = {}
    derived = unknown = 0
    for name, info in sorted(table.tensors.items()):
        entries = []
        param = c.by_name.get(name)
        if not info.shape and param and param["kind"] == "buffer" and not bound.tensors[name].uses \
                and info.dtype in ("I8", "U8", "I16", "U16", "I32", "U32", "I64", "U64"):
            entries.append({"axis": None, "role": "counter", "provenance": "derived", "basis": "counter.buffer@graph"})
            derived += 1
        elif not info.shape:  # a 0-d tensor has no axis to describe: one entry with axis null
            if name in c.free_params:
                entries.append({"axis": None, "role": "scalar", "provenance": "derived",
                                "basis": f"param_operand@{c.free_params[name]}"})
                derived += 1
            else:
                entries.append({"axis": None, "role": "unknown", "provenance": "unknown",
                                "basis": "unbound" if bound.tensors[name].status == "unbound" else "no rule"})
                unknown += 1
        for axis in range(len(info.shape)):
            entry = c.roles.get((name, axis))
            if entry is None:
                entry = _free_role(c, name, axis, len(info.shape))
            if entry is None or entry["role"] == "unknown":
                unknown += 1
                entries.append({"axis": axis, "role": "unknown", "provenance": "unknown",
                                "basis": (entry or {}).get("basis", "unbound" if bound.tensors[name].status == "unbound"
                                                           else "no rule")})
                continue
            out = {"axis": axis, "role": entry["role"], "provenance": "derived", "basis": entry["basis"]}
            structure = _factors(c, name, axis)
            if structure:
                out["factors"] = structure["factors"]
                if structure["packed"]:
                    out["role"], out["k"] = "packed", structure["packed"]
            derived += 1
            entries.append(out)
        roles[name] = entries
    # ---- groups
    groups, seen = [], set()
    spaces = {}
    for dim in c.axis_dims.values():
        for s in dim.spaces():
            spaces[s.id] = s
    for dim_list in list(c.v.values()):
        for dim in dim_list:
            for s in dim.spaces():
                spaces.setdefault(s.id, s)
    for s in sorted(spaces.values(), key=lambda s: s.id):
        if s.dead or s.superseded or s.packed or s.size < 2 or not s.members:
            continue
        members, closed_ok = [], True
        for (tensor, axis), lists in sorted(s.members.items()):
            if not any(lists):
                continue
            fit = _fit(lists)
            if fit is None:
                closed_ok = False
                members.append({"tensor": tensor, "axis": axis, "offset": None, "block": None, "period": None, "repeat": None})
            else:
                members.append({"tensor": tensor, "axis": axis, "offset": fit[0], "block": fit[1], "period": fit[2], "repeat": fit[3]})
        if not members:
            continue
        gid = _group_id(ir["ir_sha256"], s.size, members)
        boundary = list(s.boundary.values())
        if not closed_ok:
            boundary.append({"node": "graph", "op": "members", "reason": "non_affine_member"})
        data_dim = any(b["reason"] == "data_dimension" for b in boundary)  # a position axis is not a head or a value dim
        blocks = sorted(v["n"] for v in s.constraints.values() if v["kind"] == "within_blocks")
        constraint = {"kind": "within_blocks", "n": math.gcd(*blocks), "of": blocks} if blocks else {"kind": "none"}
        if gid in seen:  # the same member set reached twice: keep every boundary either copy saw
            twin = next(g for g in groups if g["group_id"] == gid)
            twin["boundary"] += boundary
            continue
        seen.add(gid)
        groups.append({"group_id": gid, "size": s.size, "closed": True, "boundary": boundary,
                       "constraints": constraint, "members": members, "kinds": [s.kind] if s.kind and not data_dim else [],
                       "kind": _plain_kind(members, roles) if data_dim or not s.kind else s.kind, "spans": _spans(c, s)})
    _open_conflicting(groups)
    for g in groups:
        unique = {(b["reason"], b["node"], b["op"], b.get("output", -1)): b for b in g["boundary"]}
        g["boundary"] = [unique[k] for k in sorted(unique)]
        g["closed"] = not g["boundary"]
    groups.sort(key=lambda g: (g["members"][0]["tensor"], g["members"][0]["axis"], g["members"][0]["offset"] or 0, g["group_id"]))
    opaque = [{"op": op, "count": n} for op, n in sorted(c.opaque.items())]
    coverage = {"unit": "axes", "derived": derived, "declared": 0, "inferred": 0, "unknown": unknown, "opaque_ops": opaque,
                "tensors": {"bound": len(bound.bound), "unused": len(bound.unused), "unbound": len(bound.unbound)},
                "groups": {"closed": sum(1 for g in groups if g["closed"]), "open": sum(1 for g in groups if not g["closed"])}}
    return Derived(ir["ir_sha256"], table_signature(table), roles, groups, opaque, coverage,
                   {"table": RULES_VERSION, "torch": sorted(RULES_FOR)}, warnings, c.node_rules)


def _plain_kind(members: list[dict], roles: dict) -> str:
    seen = {roles[m["tensor"]][m["axis"]]["role"] for m in members if m["axis"] < len(roles[m["tensor"]])}
    return "position" if seen == {"position"} else "vocab" if seen == {"vocab"} else "channel"


def _spans(c: Ctx, space: Space) -> list[dict]:
    """Where the group's units live in activations: [{value, dim, offset, block, period, repeat}], same formula as members."""
    out = []
    for value, dims in c.v.items():
        for index, dim in enumerate(dims):
            at = 0
            for piece in dim.pieces:
                size = math.prod(sp.size for sp in piece)
                minor, major = size, 1
                for sp in piece:
                    minor //= sp.size
                    if sp is space:
                        out.append({"value": value, "dim": index, "offset": at, "block": minor,
                                    "period": minor * space.size if major > 1 else minor, "repeat": major})
                    major *= sp.size
                at += size
    return out


def _open_conflicting(groups: list[dict]):
    """Backstop: two groups that cut the same tensor axis into the same units cannot both be permuted on their own."""
    seen: dict = {}
    for g in groups:
        for m in g["members"]:
            if m["offset"] is None:
                continue
            key = (m["tensor"], m["axis"], frozenset(frozenset(_indices(m, u)) for u in range(g["size"])))
            seen.setdefault(key, []).append(g)
    for found in seen.values():
        if len({id(g) for g in found}) > 1:
            for g in found:
                g["boundary"].append({"node": "graph", "op": "members", "reason": "overlapping_group"})


def _indices(m: dict, u: int):
    return (m["offset"] + u * m["block"] + k * m["period"] + j for k in range(m["repeat"]) for j in range(m["block"]))


def _free_role(c: Ctx, name: str, axis: int, rank: int):
    """A parameter no rule described but which flows through shape/elementwise ops (a token, a positional table)."""
    if (name, axis) in c.axis_dims and name in c.free_params:
        return {"role": "channel" if axis == rank - 1 else "position", "basis": f"param_operand@{c.free_params[name]}"}
    return None


def _factors(c: Ctx, name: str, axis: int):
    d = c.axis_dims.get((name, axis))
    if d is None or len(d.pieces) != 1 or len(d.pieces[0]) < 2:
        return None
    packed = next((s.size for s in d.pieces[0] if s.packed), 0)
    return {"packed": packed, "factors": [{"size": s.size, "name": "packed" if s.packed and not s.kind else s.kind}
                                          for s in d.pieces[0]]}
