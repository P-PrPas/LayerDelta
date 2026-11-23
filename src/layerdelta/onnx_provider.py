"""The `onnx` graph provider (v0.3-G7): an `.onnx` file becomes a GraphIR that is bound to the checkpoint it is attached to.

No user code runs: the file is parsed as protobuf (the optional `onnx` package, `pip install layerdelta[onnx]`) and read as
data. Unlike `builtin`/`torch_export`, the provider needs the checkpoint's table, because an ONNX file carries its own weights
and they may no longer be the checkpoint's weights (BatchNorm folded into Conv, renamed or transposed constants):

- an initializer becomes a graph parameter only if its NAME is a checkpoint tensor, the SHAPE is equal and the BYTES (sha256) are
  equal. Anything else is an `onnx.Initializer` node that carries its name and the reason (`not_in_checkpoint`,
  `shape_differs`, `dtype_differs`, `value_differs`, `external_data`, `unreadable`), and the checkpoint tensor it might have been
  stays `unbound`. Coverage is reported as it is; nothing is guessed.
- ops are named `onnx.<OpType>` (`<domain>.<OpType>` for other domains), inputs are the positional `args` (an omitted optional
  input is `null`), attributes are `kwargs` (strings are `{"str": ...}`, small tensors lists, subgraphs and large tensors a `{"str": ...}` stub).
- inputs must have static shapes; a dynamic input dimension is set to `dynamic_dim` (default 1). External tensor data is never read.

Spec: `{"provider": "onnx", "path": "<absolute .onnx file>", "dynamic_dim": 1}` (`dynamic_dim` optional).
"""

import hashlib
import math
from pathlib import Path

from .graph import finalize_ir, table_signature
from .providers import GraphBuildError
from .selector import digest

MODEL_MAX = 512 << 20          # bytes; the in-memory parse and shape inference are the ceiling (ponytail: external-data/large models later)
NODES_MAX = 200_000
SMALL_TENSOR = 64              # elements: a tensor attribute up to this size is kept as a list
_ONNX_DTYPE = {1: "F32", 2: "U8", 3: "I8", 4: "U16", 5: "I16", 6: "I32", 7: "I64", 9: "BOOL", 10: "F16", 11: "F64",
               12: "U32", 13: "U64", 16: "BF16"}


def _onnx():
    try:
        import onnx
        from onnx import numpy_helper, shape_inference
    except ImportError:
        return None
    return onnx, numpy_helper, shape_inference


def available() -> bool:
    return _onnx() is not None


def reason_unavailable() -> str | None:
    return None if available() else "The onnx package is not installed; install it with: pip install \"layerdelta[onnx]\""


def check_spec(spec) -> dict:
    bad = lambda message, field: GraphBuildError("invalid_spec", f"{message} ({field})")  # noqa: E731
    if not isinstance(spec, dict) or spec.get("provider") != "onnx" or not {"provider", "path"} <= set(spec) <= {"provider", "path", "dynamic_dim"}:
        raise GraphBuildError("invalid_spec", "an onnx spec is {provider: 'onnx', path, dynamic_dim?}")
    path, dim = spec["path"], spec.get("dynamic_dim", 1)
    if not isinstance(path, str) or "\0" in path or not Path(path).is_absolute() or Path(path).suffix.lower() != ".onnx" or not Path(path).is_file():
        raise bad("path must be an absolute path to an existing .onnx file", "path")
    if type(dim) is not int or not 1 <= dim <= 4096:
        raise bad("dynamic_dim must be an integer from 1 to 4096", "dynamic_dim")
    if Path(path).stat().st_size > MODEL_MAX:
        raise bad(f"the model is larger than {MODEL_MAX} bytes", "path")
    return {"provider": "onnx", "path": str(Path(path)), "dynamic_dim": dim}


def _dims(shape, dynamic_dim) -> list | None:
    """A static shape, or None. Unknown dimensions are None here; the caller decides."""
    out = []
    for d in shape.dim:
        kind = d.WhichOneof("value")
        if kind == "dim_value" and d.dim_value >= 0:
            out.append(int(d.dim_value))
        else:
            out.append(None)
    return out


def _tensor_type(value_info):
    """(dtype, shape-with-None) of a ValueInfo, or None when it is not a plain tensor."""
    if not value_info.type.HasField("tensor_type"):
        return None
    t = value_info.type.tensor_type
    dtype = _ONNX_DTYPE.get(t.elem_type)
    if dtype is None or not t.HasField("shape"):
        return None
    return dtype, _dims(t.shape, None)


def _initializer_bytes(onnx, helper, t):
    """The tensor's little-endian bytes, or (None, why)."""
    if t.data_location == onnx.TensorProto.EXTERNAL:
        return None, "external_data"
    if t.raw_data:
        return bytes(t.raw_data), None
    if t.data_type == 16:  # bfloat16 without raw_data: numpy has no such type
        return None, "unreadable"
    try:
        return helper.to_array(t).tobytes(), None
    except Exception:  # noqa: BLE001 a malformed tensor is data, not a crash
        return None, "unreadable"


def _literal_tensor(helper, t):
    elems = math.prod(t.dims)
    if t.data_location == 1 or elems > SMALL_TENSOR or t.data_type not in (1, 6, 7, 9, 11):
        return {"str": f"tensor{list(t.dims)}:{_ONNX_DTYPE.get(t.data_type, t.data_type)}"}
    try:
        values = helper.to_array(t).reshape(-1).tolist()
    except Exception:  # noqa: BLE001
        return {"str": f"tensor{list(t.dims)}"}
    return [_number(v) for v in values]


def _number(v):
    if isinstance(v, float) and not math.isfinite(v):
        return {"float": "nan" if v != v else "inf" if v > 0 else "-inf"}
    return v


def _attribute(onnx, helper, a):
    A = onnx.AttributeProto
    if a.type == A.INT:
        return int(a.i)
    if a.type == A.FLOAT:
        return _number(float(a.f))
    if a.type == A.STRING:
        return {"str": a.s.decode("utf-8", errors="replace")}
    if a.type == A.INTS:
        return [int(v) for v in a.ints]
    if a.type == A.FLOATS:
        return [_number(float(v)) for v in a.floats]
    if a.type == A.STRINGS:
        return [{"str": s.decode("utf-8", errors="replace")} for s in a.strings]
    if a.type == A.TENSOR:
        return _literal_tensor(helper, a.t)
    return {"str": "<" + A.AttributeType.Name(a.type).lower() + ">"}  # subgraphs, sparse tensors, types: opaque to the IR


def build(spec: dict, table) -> dict:
    """Parse the file, bind its initializers to `table` by name, shape and bytes, and return the finalized GraphIR."""
    loaded = _onnx()
    if loaded is None:
        raise GraphBuildError("provider_disabled", reason_unavailable())
    onnx, helper, inference = loaded
    spec = check_spec(spec)
    data = Path(spec["path"]).read_bytes()
    model = onnx.ModelProto()
    try:
        model.ParseFromString(data)
    except Exception as exc:  # noqa: BLE001
        raise GraphBuildError("invalid_model", f"not an ONNX model: {type(exc).__name__}") from exc
    graph = model.graph
    if not graph.node or len(graph.node) > NODES_MAX:
        raise GraphBuildError("invalid_model", f"the model has {len(graph.node)} nodes; 1 to {NODES_MAX} are supported")
    opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), None)
    if opset is None:
        raise GraphBuildError("invalid_model", "the model declares no default-domain opset")
    initializers = {t.name: t for t in graph.initializer}
    dynamic = spec["dynamic_dim"]
    for item in graph.input:  # fix dynamic dimensions before shape inference
        if item.name not in initializers and item.type.HasField("tensor_type"):
            for d in item.type.tensor_type.shape.dim:
                if d.WhichOneof("value") != "dim_value" or d.dim_value <= 0:
                    d.dim_value = dynamic
    try:
        inferred = inference.infer_shapes(model)
    except Exception:  # noqa: BLE001 shape inference is a convenience: values without a shape become kind 'other'
        inferred = model
    types = {v.name: _tensor_type(v) for v in [*inferred.graph.value_info, *inferred.graph.output, *inferred.graph.input]}

    ids = {"v": 0, "p": 0, "n": 0}

    def fresh(kind):
        ids[kind] += 1
        return f"{kind}{ids[kind]}"

    ref: dict[str, str] = {}
    inputs, params, nodes = [], [], []
    name_of: dict[str, str] = {}  # parameter value id -> checkpoint name

    def define(name, node_kind="v"):
        ref[name] = fresh(node_kind)
        return ref[name]

    def output_of(name, value_id):
        known = types.get(name)
        if known is None or None in known[1]:
            return {"value": value_id, "kind": "other"}
        return {"value": value_id, "kind": "tensor", "shape": known[1], "dtype": known[0]}

    for item in graph.input:
        if item.name in initializers:
            continue
        known = _tensor_type(item)
        if known is None or None in known[1]:
            raise GraphBuildError("invalid_model", f"input {item.name!r} must be a tensor of a supported dtype with a known rank")
        inputs.append({"value": define(item.name), "shape": known[1], "dtype": known[0]})

    used_by_node = {name for n in graph.node for name in n.input if name}
    consumed = used_by_node | {o.name for o in graph.output}
    checkpoint = table.tensors

    def use_initializer(name):
        """Define the initializer on first use: a parameter if it is the checkpoint's tensor, else a described constant."""
        t = initializers[name]
        shape, dtype = [int(n) for n in t.dims], _ONNX_DTYPE.get(t.data_type)
        reason = None
        if dtype is None:
            reason = "unreadable"
        elif name not in checkpoint:
            reason = "not_in_checkpoint"
        elif list(checkpoint[name].shape) != shape:
            reason = "shape_differs"
        elif checkpoint[name].dtype != dtype:
            reason = "dtype_differs"
        else:
            payload, reason = _initializer_bytes(onnx, helper, t)
            if reason is None and hashlib.sha256(payload).hexdigest() != table.sha256(name):
                reason = "value_differs"
        if reason is None:
            params.append({"name": name, "value": define(name, "p"), "kind": "parameter", "shape": shape, "dtype": dtype})
            name_of[ref[name]] = name
            return
        value = define(name)
        out = ({"value": value, "kind": "tensor", "shape": shape, "dtype": dtype} if dtype else {"value": value, "kind": "other"})
        nodes.append({"id": fresh("n"), "op": "onnx.Initializer", "args": [], "kwargs": {"name": {"str": name}, "reason": {"str": reason}},
                      "outputs": [out], "module_path": None, "module_type": None})

    for node in graph.node:
        args = []
        for name in node.input:
            if not name:
                args.append(None)
                continue
            if name not in ref:
                if name not in initializers:
                    raise GraphBuildError("invalid_model", f"node {node.name or node.op_type!r} reads {name!r} before it is defined")
                use_initializer(name)
            args.append(ref[name])
        names = [n for n in node.output]
        used = [i for i, n in enumerate(names) if n and n in consumed]
        out_ids = []
        if len(names) <= 1 or used in ([], [0]):
            value = define(names[0]) if names and names[0] else fresh("v")
            outputs = [output_of(names[0] if names else "", value)]
        else:
            value = fresh("v")
            items = []
            for n in names:
                known = types.get(n) if n else None
                items.append(None if known is None or None in known[1] else {"shape": known[1], "dtype": known[0]})
            outputs = [{"value": value, "kind": "tuple", "items": items}]
            out_ids = [(i, names[i]) for i in used]
        domain = node.domain if node.domain not in ("", "ai.onnx") else ""
        weights = sorted({name_of[a] for a in args if a in name_of})
        prefix = _module_of(weights)
        nodes.append({"id": fresh("n"), "op": f"{domain + '.' if domain else 'onnx.'}{node.op_type}", "args": args,
                      "kwargs": {a.name: _attribute(onnx, helper, a) for a in sorted(node.attribute, key=lambda a: a.name)},
                      "outputs": outputs, "module_path": prefix, "module_type": None})
        for i, n in out_ids:
            known = types.get(n)
            got = define(n)
            tensor = {"value": got, "kind": "other"} if known is None or None in known[1] else \
                {"value": got, "kind": "tensor", "shape": known[1], "dtype": known[0]}
            nodes.append({"id": fresh("n"), "op": "_operator.getitem", "args": [value, i], "kwargs": {}, "outputs": [tensor],
                          "module_path": prefix, "module_type": None})
    for name in (o.name for o in graph.output):
        if name not in ref:
            if name not in initializers:
                raise GraphBuildError("invalid_model", f"graph output {name!r} is never defined")
            use_initializer(name)
    provider = {"name": "onnx", "version": f"onnx-ir{model.ir_version}-opset{opset}",
                "spec_sha256": digest({"spec": spec, "model_sha256": hashlib.sha256(data).hexdigest(),
                                       "table_signature": table_signature(table)})}
    return finalize_ir({"ir_version": 1, "provider": provider, "inputs": inputs, "params": params, "ties": [], "nodes": nodes,
                        "outputs": [ref[o.name] for o in graph.output]})


def _module_of(weights: list[str]) -> str | None:
    """The module a node's weights live in: their common dotted prefix without the last component ('' is the root). ONNX has no
    module stack; node names are not reliable, tensor names are what binds. ponytail: module_type stays null."""
    if not weights:
        return None
    parents = {w.rpartition(".")[0] for w in weights}
    return parents.pop() if len(parents) == 1 else None
