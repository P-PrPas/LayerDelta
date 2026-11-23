"""Rule table for graphs from the `onnx` provider (v0.3-G7): ONNX ops mapped onto the torch rules of coupling.py.

Each ONNX op is turned into the call shape its torch rule reads (inputs are positional, attributes become what the rule
expects) and the existing rule does the work, so roles and groups mean the same thing for both providers. Anything not listed,
or with an attribute that changes the meaning (`Gemm` with `transA` or `transB=0`, `Gather` on an axis other than 0), is an
opaque boundary, reported like every other opaque op. Valid for default-domain opsets 11 to 21.
"""

from . import coupling as C

OPSETS = range(11, 22)
_PASSTHROUGH = ("Relu", "Sigmoid", "Tanh", "LeakyRelu", "Elu", "Selu", "Gelu", "HardSwish", "HardSigmoid", "Softplus", "Softsign",
                "Clip", "Dropout", "Identity", "Neg", "Abs", "Exp", "Log", "Sqrt", "Erf", "Cast", "Softmax", "LogSoftmax",
                "Sin", "Cos", "Reciprocal", "Floor", "Ceil", "Round", "Sign")


def _attr(node, name, default):
    return node["kwargs"].get(name, default)


def _bound(c, node, *slots):
    """Every weight, bias, scale or statistic input present must be a bound parameter: a constant the checkpoint does not hold
    (folded, edited or renamed) is a value the rule cannot permute, so the op is a boundary instead of a closed group."""
    for i in slots:
        ref = C._arg(node, i)
        if ref is not None and c.pname(ref) is None:
            raise C._Opaque("unbound_weight")


def _const(c, node):
    """A constant the checkpoint does not hold. Integer ones (shapes, indices) are free; a float one with more than one element is
    data the graph adds or multiplies in, so its axes are opaque (a group that reaches them stays open)."""
    out = node["outputs"][0]
    if out["kind"] != "tensor":
        raise C._Opaque("const")
    dims = [c.fresh(n) for n in out["shape"]]
    c.v[out["value"]] = dims
    if out["dtype"] in ("F16", "BF16", "F32", "F64"):
        for d in dims:
            c.mark(d.spaces(), "opaque_op")


def _binary(c, node):
    if len(node["args"]) > 2:  # Max and Min are variadic; r_binary ties the first two only
        raise C._Opaque("variadic")
    C.r_binary(c, node)


def _conv(transposed):
    def rule(c, node):
        _bound(c, node, 1, 2)
        a = node["args"] + [None] * (3 - len(node["args"]))
        C.r_conv(c, {**node, "args": a[:3], "kwargs": {"groups": _attr(node, "group", 1)}}, transposed=transposed)
    return rule


def _gemm(c, node):
    if _attr(node, "transA", 0) or not _attr(node, "transB", 0):
        raise C._Opaque("gemm")
    _bound(c, node, 1, 2)
    a = node["args"] + [None] * (3 - len(node["args"]))
    C.r_linear(c, {**node, "args": a[:3], "kwargs": {}})


def _layer_norm(c, node):
    x = c.get(C._arg(node, 0))
    if x is None:
        raise C._Opaque("layer_norm")
    _bound(c, node, 1, 2)
    axis = C._norm(_attr(node, "axis", -1), len(x))
    a = node["args"] + [None] * (3 - len(node["args"]))
    C.r_layer_norm(c, {**node, "args": [a[0], [d.size for d in x[axis:]], a[1], a[2]], "kwargs": {}})


def _batch_norm(c, node):
    _bound(c, node, 1, 2, 3, 4)
    C.r_batch_norm(c, node)


def _concat(c, node):
    C.r_cat(c, {**node, "args": [node["args"], _attr(node, "axis", 0)], "kwargs": {}})


def _transpose(c, node):
    x = c.get(C._arg(node, 0))
    if x is None:
        raise C._Opaque("transpose")
    perm = _attr(node, "perm", list(reversed(range(len(x)))))
    C.r_permute(c, {**node, "args": [node["args"][0], perm], "kwargs": {}})


def _gather(c, node):
    if _attr(node, "axis", 0) != 0 or c.pname(C._arg(node, 0)) is None:
        raise C._Opaque("gather")
    C.r_embedding(c, node)


def onnx_table() -> dict:
    t = {"onnx.Conv": _conv(False), "onnx.ConvTranspose": _conv(True), "onnx.Gemm": _gemm, "onnx.MatMul": C.r_matmul,
         "onnx.BatchNormalization": _batch_norm, "onnx.LayerNormalization": _layer_norm, "onnx.Concat": _concat,
         "onnx.Transpose": _transpose, "onnx.Gather": _gather, "_operator.getitem": C.r_getitem,
         "onnx.Initializer": _const, "onnx.Constant": _const}
    for name in ("Add", "Sub", "Mul", "Div", "Max", "Min"):
        t[f"onnx.{name}"] = _binary
    for name in _PASSTHROUGH:
        t[f"onnx.{name}"] = C.r_passthrough
    for name in ("Flatten", "Reshape", "Squeeze", "Unsqueeze"):
        t[f"onnx.{name}"] = C.r_reshape
    for name in ("MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool"):
        t[f"onnx.{name}"] = C.r_pool
    return t


def rules_for(version: str) -> dict:
    """The table for an `onnx-ir<N>-opset<M>` provider version; empty (everything opaque) outside the supported opsets."""
    try:
        opset = int(version.rsplit("opset", 1)[1])
    except (IndexError, ValueError):
        return {}
    return _TABLE if opset in OPSETS else {}


_TABLE = onnx_table()
