"""A coupling group as a graph node (v0.5-E5c): where do the units of a group show up as activations?

A group's units are produced by the weight of its first `out_unit` member (a convolution's output channels, a linear layer's features). The node that
applies that weight is where the group's activations are: its output, along the channel axis (axis 1 of a convolution's NCHW output, the last axis of a
linear layer's output), has exactly `group.size` units. This is the node a probe reads and an activation edit changes when a request names a group.
The node is the layer's own output (before its activation function); the activation after it is the next node. A group that does not resolve to one
such node (a recurrent or packed group, a matmul with an unusual layout) is `probe_unsupported`: never a guess.
"""

from .graphview import base_role
from .selector import InspectError

PRODUCERS = ("conv", "linear", "gemm", "addmm", "matmul")


def resolve(view, ir: dict, group_id: str) -> tuple[str, int, int]:
    """(node id, unit axis, units) of a group, or an error."""
    group = view.by_id.get(group_id) if view is not None and isinstance(group_id, str) else None
    if group is None:
        raise InspectError("not_found", f"Unknown group {group_id!r}: list them with the groups tool", "group")
    record = view.group_record(group)
    producers = [m for m in record["members"] if base_role(m["role"]) == "out_unit"]
    if not producers:
        raise InspectError("probe_unsupported", "this group has no member that produces its units", "group")
    uses = view.bound.tensors[producers[0]["tensor"]].uses
    use = next((u for u in uses if any(key in u["op"].lower() for key in PRODUCERS)), None)
    node = next((n for n in ir["nodes"] if use is not None and n["id"] == use["node"]), None)
    out = node["outputs"][0] if node is not None and len(node["outputs"]) == 1 else None
    if out is None or out["kind"] != "tensor" or len(out["shape"]) < 2:
        raise InspectError("probe_unsupported", "this group does not resolve to one node that outputs a tensor", "group")
    axis = 1 if "conv" in node["op"].lower() else len(out["shape"]) - 1
    if out["shape"][axis] != group["size"]:
        raise InspectError("probe_unsupported", f"the group has {group['size']} units but its node {node['id']} has {out['shape'][axis]} along axis {axis}", "group")
    return node["id"], axis, group["size"]
