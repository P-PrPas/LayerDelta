"""Built-in MNIST CNN and its versioned inspection targets."""

from collections import OrderedDict
from dataclasses import asdict, dataclass

import torch

from torch import Tensor, nn


ARCHITECTURE_ID = "mnist-two-conv"
ARCHITECTURE_REVISION = 1
ADAPTER_VERSION = 1


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(OrderedDict(
        conv=nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        relu=nn.ReLU(),
        pool=nn.MaxPool2d(kernel_size=2),
    ))


class ReferenceCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(OrderedDict(
            block1=_block(1, 8),
            block2=_block(8, 24),
        ))
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(24 * 7 * 7, 10)

    def forward(self, images: Tensor) -> Tensor:
        return self.classifier(self.flatten(self.features(images)))


@dataclass(frozen=True)
class Target:
    target_id: str
    mode: str
    module_path: str
    channel_count: int
    channel_axis: int
    parameter_name: str | None = None

    def check_channel(self, channel: int) -> None:
        if type(channel) is not int or not 0 <= channel < self.channel_count:
            raise ValueError(f"{self.target_id} requires channel 0..{self.channel_count - 1}")


def target_catalog(model: ReferenceCNN) -> dict[str, Target]:
    targets = {}
    for name, block in model.features.named_children():
        prefix = f"features.{name}"
        channels = block.conv.out_channels
        relu = f"{prefix}.relu"
        conv = f"{prefix}.conv"
        activation = Target(f"{relu}.output", "activation", relu, channels, 1)
        weight = Target(f"{conv}.kernel", "weight", conv, channels, 0, f"{conv}.weight")
        targets[activation.target_id] = activation
        targets[weight.target_id] = weight
    return targets


def describe_model(model: ReferenceCNN) -> dict:
    """Trace an example input so inspect-only nodes and target shapes agree with forward."""
    layers = []
    targets = target_catalog(model)
    handles = []

    def capture(name, module, inputs, output):
        edit_targets = [t.target_id for t in targets.values() if t.module_path == name]
        layers.append({
            "layer_id": name,
            "type": type(module).__name__,
            "input_shape": list(inputs[0].shape[1:]),
            "output_shape": list(output.shape[1:]),
            "target_ids": edit_targets,
            "read_only": not edit_targets,
        })

    try:
        for name, module in model.named_modules():
            if not list(module.children()):
                handles.append(module.register_forward_hook(
                    lambda mod, args, out, name=name: capture(name, mod, args, out)
                ))
        reference = model.classifier.weight
        with torch.inference_mode():
            model(reference.new_zeros((1, 1, 28, 28)))
    finally:
        for handle in handles:
            handle.remove()
    return {
        "architecture_id": ARCHITECTURE_ID,
        "architecture_revision": ARCHITECTURE_REVISION,
        "adapter_version": ADAPTER_VERSION,
        "input_shape": [1, 28, 28],
        "layers": layers,
        "targets": [asdict(target) for target in targets.values()],
        "state_schema": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in model.state_dict().items()
        },
    }
