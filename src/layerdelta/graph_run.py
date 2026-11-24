"""Run an exported program node by node (v0.5-E5): read the output of a graph node, or change some of its units, while it runs.

The graph IR names nodes `n0, n1, ...`: the `call_function` nodes of the `ExportedProgram`'s graph, in order (providers.export_to_ir). The
same program runs here through `torch.fx.Interpreter` with a checkpoint's weights substituted by name, so a node id of the IR is the same
node here. Used by the probe and the intervention workers only (they run in the isolated worker process, like forward_worker).

Covers the `builtin` and `torch_export` executables, which are both `ExportedProgram`s. A node whose output is not one tensor (a tuple) is not
reachable (`NodeUnsupported`), and an ONNX executable is not run here at all.
"""

import torch
from torch.export.graph_signature import InputKind


class NodeUnsupported(Exception):
    pass


def node_ids(program) -> dict:
    """{fx node name: IR node id} for the call_function nodes, in the order the IR numbers them."""
    out, count = {}, 0
    for node in program.graph.nodes:
        if node.op == "call_function":
            out[node.name] = f"n{count}"
            count += 1
    return out


class _Tap(torch.fx.Interpreter):
    def __init__(self, module, ids, watch, change):
        super().__init__(module)
        self.ids, self.watch, self.change, self.seen = ids, watch, change, {}

    def run_node(self, node):
        result = super().run_node(node)
        node_id = self.ids.get(node.name)
        if node_id is None:
            return result
        if node_id in self.change:
            if not isinstance(result, torch.Tensor):
                raise NodeUnsupported(f"{node_id} does not return one tensor")
            result = self.change[node_id](result)
        if node_id in self.watch:
            if not isinstance(result, torch.Tensor):
                raise NodeUnsupported(f"{node_id} does not return one tensor")
            self.seen[node_id] = result.detach()
        return result


def run(program, weights: dict, prefix: str, inputs: list, *, watch=(), change=None) -> tuple[list, dict]:
    """(outputs, {node id: tensor}) of the program on `inputs` (batch-first tensors), with `weights` (name -> tensor, full checkpoint names)
    substituted for its parameters and buffers. `change` maps a node id to a function applied to that node's output before it flows on."""
    signature = program.graph_signature
    args, users = [], iter(inputs)
    for spec in signature.input_specs:
        if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER):
            fqn = spec.target
            if prefix + fqn not in weights:
                raise NodeUnsupported(f"the checkpoint has no tensor {prefix + fqn!r}")
            own = program.state_dict[fqn]
            args.append(weights[prefix + fqn].to(own.dtype))
        elif spec.kind == InputKind.CONSTANT_TENSOR:
            args.append(program.constants[spec.target])
        elif spec.kind == InputKind.USER_INPUT:
            args.append(next(users))
        else:
            raise NodeUnsupported(f"an input of kind {spec.kind} is not supported")
    ids = node_ids(program)
    unknown = (set(watch) | set(change or ())) - set(ids.values())
    if unknown:
        raise NodeUnsupported(f"no such node: {sorted(unknown)}")
    tap = _Tap(program.graph_module, ids, set(watch), change or {})
    with torch.no_grad():
        out = tap.run(*args)
    flat = list(out) if isinstance(out, (list, tuple)) else [out]
    return flat, tap.seen
