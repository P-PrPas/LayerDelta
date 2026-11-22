"""Graph providers (v0.3-G2): the one place a model becomes a GraphIR.

`GraphProvider.build(spec)` returns a finalized GraphIR (graph.py); nothing downstream imports model
code. Only the `builtin` provider exists now. It builds graphs for modules that OUR code registered in
this process (the MNIST reference CNN and, from tests, the corpus families): the registry is a plain dict
filled by code, and a spec can only name a registry key, never a module path, so no user code can be
reached through it. `export_to_ir` is the generic torch.export-to-IR conversion that the later isolated
`torch_export` provider will reuse inside its worker.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Callable, Protocol

import torch

from .graph import IR_VERSION, finalize_ir
from .selector import InspectError, digest

_NAME = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
_ST_DTYPE = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16", torch.float64: "F64", torch.int64: "I64",
             torch.int32: "I32", torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8", torch.bool: "BOOL"}
_ENUMS = (torch.dtype, torch.layout, torch.device, torch.memory_format)


class GraphBuildError(InspectError):
    def __init__(self, code: str, message: str):
        super().__init__(code, message, "spec")


class GraphProvider(Protocol):
    name: str

    def build(self, spec: dict) -> dict: ...


@dataclass(frozen=True)
class BuiltinModel:
    factory: Callable[[], torch.nn.Module]
    example_inputs: tuple  # ((shape, torch dtype name), ...)


_REGISTRY: dict[str, BuiltinModel] = {}


def register_builtin(name: str, factory, example_inputs, *, replace: bool = False) -> None:
    """Make a module of OUR code buildable by the builtin provider. Process-local; there is no way to call this from a request."""
    if not _NAME.fullmatch(name):
        raise ValueError(f"invalid builtin model name {name!r}")
    if name in _REGISTRY and not replace:
        raise ValueError(f"builtin model {name!r} is already registered")
    _REGISTRY[name] = BuiltinModel(factory, tuple((tuple(shape), str(dtype)) for shape, dtype in example_inputs))


def registered_models() -> list[str]:
    return sorted(_REGISTRY)


class BuiltinProvider:
    name = "builtin"

    def build(self, spec: dict) -> dict:
        if not isinstance(spec, dict) or set(spec) != {"provider", "model"} or spec["provider"] != "builtin":
            raise GraphBuildError("invalid_spec", "a builtin spec is exactly {provider: 'builtin', model: <registry name>}")
        model = spec["model"]
        if not isinstance(model, str) or not _NAME.fullmatch(model):
            raise GraphBuildError("invalid_spec", "model must be a registry name (no module paths)")
        entry = _REGISTRY.get(model)
        if entry is None:
            raise GraphBuildError("unknown_model", f"no builtin model named {model!r}; registered: {registered_models()}")
        inputs = tuple(torch.zeros(shape, dtype=getattr(torch, dtype)) for shape, dtype in entry.example_inputs)
        provider = {"name": "builtin", "version": "torch-" + torch.__version__.split("+")[0],
                    "spec_sha256": digest({"spec": spec, "inputs": [[list(s), d] for s, d in entry.example_inputs]})}
        return export_to_ir(entry.factory().eval(), inputs, provider)


PROVIDERS: dict[str, GraphProvider] = {"builtin": BuiltinProvider()}


def build_graph(spec: dict) -> dict:
    """Dispatch a spec to its provider. Only providers in PROVIDERS exist; there is no dynamic loading."""
    provider = PROVIDERS.get(spec.get("provider") if isinstance(spec, dict) else None)
    if provider is None:
        raise GraphBuildError("unknown_provider", f"providers available: {sorted(PROVIDERS)}")
    return provider.build(spec)


# ---------------------------------------------------------------- torch.export -> GraphIR

def _dtype(value) -> str:
    try:
        return _ST_DTYPE[value]
    except KeyError:
        raise GraphBuildError("unsupported_dtype", f"no safetensors dtype for {value}") from None


def _op_name(target) -> str:
    if isinstance(target, torch._ops.OpOverload) or isinstance(target, torch._ops.OpOverloadPacket):
        return str(target)
    return f"{getattr(target, '__module__', '')}.{getattr(target, '__name__', repr(target))}"


def _output(value_id: str, meta) -> dict:
    if isinstance(meta, torch.Tensor):
        return {"value": value_id, "kind": "tensor", "shape": [int(n) for n in meta.shape], "dtype": _dtype(meta.dtype)}
    if isinstance(meta, (list, tuple)):
        items = [None if not isinstance(m, torch.Tensor) else {"shape": [int(n) for n in m.shape], "dtype": _dtype(m.dtype)}
                 for m in meta]
        return {"value": value_id, "kind": "tuple", "items": items}
    return {"value": value_id, "kind": "other"}


def export_to_ir(module: torch.nn.Module, example_inputs: tuple, provider: dict) -> dict:
    """Export `module` on CPU with torch.export and describe its graph as a finalized GraphIR.

    Parameters and buffers come from the export signature; a parameter the forward never reads is simply absent
    (binding reports it as unbound). Two placeholders that alias one tensor (tied weights) become one parameter, and the
    tie is declared from the module itself.
    """
    module = module.eval()
    state = module.state_dict()
    program = torch.export.export(module, tuple(example_inputs))
    signature = program.graph_signature
    param_of, buffer_of = dict(signature.inputs_to_parameters), dict(signature.inputs_to_buffers)
    ref: dict = {}
    inputs, params, counters = [], [], {"v": 0, "p": 0}
    param_value: dict[str, str] = {}

    def fresh(prefix: str) -> str:
        counters[prefix] += 1
        return f"{prefix}{counters[prefix] - 1}"

    for node in program.graph.nodes:
        if node.op != "placeholder":
            continue
        fqn = param_of.get(node.target) or buffer_of.get(node.target)
        if fqn is not None:
            if fqn not in state:
                raise GraphBuildError("non_persistent_buffer", f"{fqn} is not in the state dict, so no checkpoint can carry it")
            if fqn not in param_value:
                param_value[fqn] = fresh("p")
                val = node.meta["val"]
                params.append({"name": fqn, "value": param_value[fqn], "kind": "parameter" if node.target in param_of else "buffer",
                               "shape": [int(n) for n in val.shape], "dtype": _dtype(val.dtype)})
            ref[node] = param_value[fqn]  # aliased placeholders share one value
        elif node.target in signature.user_inputs:
            ref[node] = fresh("v")
            val = node.meta["val"]
            inputs.append({"value": ref[node], "shape": [int(n) for n in val.shape], "dtype": _dtype(val.dtype)})
        else:
            raise GraphBuildError("unsupported_placeholder", f"placeholder {node.target} is neither a user input nor a state tensor")

    def encode(x):
        if isinstance(x, torch.fx.Node):
            return ref[x]
        if x is None or isinstance(x, (bool, int)):
            return x
        if isinstance(x, float):
            return x if x == x and abs(x) != float("inf") else {"float": "nan" if x != x else ("inf" if x > 0 else "-inf")}
        if isinstance(x, str):
            return {"str": x}
        if isinstance(x, (list, tuple)):
            return [encode(i) for i in x]
        if isinstance(x, _ENUMS):
            return {"enum": str(x)}
        raise GraphBuildError("unsupported_literal", f"cannot encode a {type(x).__name__} argument in the graph")

    nodes, outputs = [], []
    for node in program.graph.nodes:
        if node.op == "call_function":
            node_id = f"n{len(nodes)}"
            ref[node] = fresh("v")
            stack = node.meta.get("nn_module_stack") or {}
            path, kind = list(stack.values())[-1] if stack else (None, None)
            kind = None if kind is None else (kind if isinstance(kind, str) else kind.__name__).rsplit(".", 1)[-1]
            nodes.append({"id": node_id, "op": _op_name(node.target), "args": [encode(a) for a in node.args],
                          "kwargs": {k: encode(v) for k, v in sorted(node.kwargs.items())},
                          "outputs": [_output(ref[node], node.meta.get("val"))], "module_path": path, "module_type": kind})
        elif node.op == "output":
            outputs = [ref[n] for n in torch.fx.node.map_arg(node.args[0], lambda n: n) if isinstance(n, torch.fx.Node)] \
                if not isinstance(node.args[0], torch.fx.Node) else [ref[node.args[0]]]
        elif node.op != "placeholder":
            raise GraphBuildError("unsupported_node", f"graph node kind {node.op!r} is not supported")
    shared: dict[int, list[str]] = {}
    for name, tensor in module.named_parameters(remove_duplicate=False):
        shared.setdefault(id(tensor), []).append(name)
    ties = sorted(sorted(set(names)) for names in shared.values() if len(set(names)) > 1)
    return finalize_ir({"ir_version": IR_VERSION, "provider": provider, "inputs": inputs, "params": params, "ties": ties,
                        "nodes": nodes, "outputs": outputs})


from . import builtin_models  # noqa: E402,F401  registers the MNIST reference CNN
