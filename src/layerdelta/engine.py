"""Inference and channel editing for the two-block reference adapter."""

from contextlib import contextmanager

import torch
from torch import Tensor

from .domain import normalize_intervention
from .model import ReferenceCNN, Target, target_catalog


def _channel_view(tensor: Tensor, target: Target, channel: int) -> Tensor:
    if (tensor.ndim != 4 or tensor.dtype != torch.float32
            or tensor.shape[target.channel_axis] != target.channel_count):
        raise ValueError(f"Tensor disagrees with adapter target {target.target_id}")
    target.check_channel(channel)
    return tensor.select(target.channel_axis, channel)


def _statistics(tensor: Tensor) -> dict:
    if not torch.isfinite(tensor).all():
        raise ValueError("Channel contains non-finite values")
    return {
        "shape": list(tensor.shape),
        "min": tensor.amin().item(),
        "max": tensor.amax().item(),
        "mean": tensor.double().mean().item(),
        "zero_fraction": tensor.eq(0).double().mean().item(),
    }


def _edit(channel: Tensor, spec: dict) -> Tensor:
    if spec["operation"] == "zero":
        return torch.zeros_like(channel)
    return channel * spec["scale"]


def _patch_kernel(kernel: Tensor, target: Target, spec: dict) -> dict:
    before = _statistics(kernel)
    updated = _edit(kernel, spec)
    after = _statistics(updated)
    kernel.copy_(updated)
    return {"parameter_name": target.parameter_name, "channel_axis": target.channel_axis,
            "channel_index": spec["channel_index"], "before": before, "after": after}


@torch.no_grad()
def patch_weight(model: ReferenceCNN, spec: dict) -> dict:
    """Permanently edit the requested kernel on a fresh model for child export."""
    spec = normalize_intervention(spec)
    target = target_catalog(model)[spec["target_id"]]
    if target.mode != "weight":
        raise ValueError("Checkpoint export requires a weight target")
    kernel = _channel_view(model.get_parameter(target.parameter_name), target, spec["channel_index"])
    return _patch_kernel(kernel, target, spec)


@contextmanager
def _once_per_forward(model, target, callback):
    calls = 0

    def reset(*_):
        nonlocal calls
        calls = 0

    def visit(module, args, output):
        nonlocal calls
        calls += 1
        if calls != 1:
            raise ValueError(f"Repeated forward invocation for {target.target_id}")
        return callback(output)

    def check(*_):
        if calls != 1:
            raise ValueError(f"Missing forward invocation for {target.target_id}")

    module = model.get_submodule(target.module_path)
    handles = [model.register_forward_pre_hook(reset), module.register_forward_hook(visit),
               model.register_forward_hook(check)]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def intervention(model: ReferenceCNN, spec: dict | None):
    """Scope a post-ReLU hook or kernel patch to this inference operation."""
    if spec is None:
        yield {}
        return
    spec = normalize_intervention(spec)
    target = target_catalog(model)[spec["target_id"]]
    channel = spec["channel_index"]
    if target.mode == "weight":
        with torch.no_grad():
            kernel = _channel_view(model.get_parameter(target.parameter_name), target, channel)
            original = kernel.clone()
            changes = _patch_kernel(kernel, target, spec)
        try:
            yield changes
        finally:
            with torch.no_grad():
                kernel.copy_(original)
        return

    def edit_output(output):
        edited = output.clone()
        selected = _channel_view(edited, target, channel)
        value = _edit(selected, spec)
        if not torch.isfinite(value).all():
            raise ValueError("Activation edit produced non-finite values")
        selected.copy_(value)
        return edited

    with _once_per_forward(model, target, edit_output):
        yield {"hook": target.module_path, "channel_index": channel}


@torch.inference_mode()
def infer(model: ReferenceCNN, images: Tensor, rows: list[dict], device, batch_size: int,
          spec: dict | None = None, *, audit: dict | None = None) -> list[dict]:
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("Batch size must be a positive integer")
    if images.dtype != torch.float32 or tuple(images.shape) != (len(rows), 1, 28, 28):
        raise ValueError("Inference requires one float32 MNIST image per manifest row")
    # Keep model parameters ordinary tensors when moving them between devices.
    # Export can apply a persistent patch after this inference scope has ended.
    with torch.inference_mode(False):
        model.to(device).eval()
    predictions = []
    with intervention(model, spec) as metadata:
        for start in range(0, len(rows), batch_size):
            batch = images[start:start + batch_size].to(device)
            logits = model(batch).cpu()
            if logits.shape != (len(batch), 10) or not torch.isfinite(logits).all():
                raise ValueError("Expected ten finite logits per image")
            for row, label, scores in zip(rows[start:start + batch_size], logits.argmax(1).tolist(),
                                          logits.tolist(), strict=True):
                predictions.append({**row, "prediction": label, "logits": scores})
        if audit is not None:
            audit.update(metadata)
    return predictions


@torch.inference_mode()
def preview(model: ReferenceCNN, image: Tensor, target_id: str, channel: int, device) -> dict:
    """Return one channel's kernel statistics, post-ReLU map and source pixels."""
    catalog = target_catalog(model)
    if target_id not in catalog:
        raise ValueError(f"Unknown target: {target_id}")
    target = catalog[target_id]
    target.check_channel(channel)
    block = target.module_path.rsplit(".", 1)[0]
    activation = catalog[f"{block}.relu.output"]
    weight = catalog[f"{block}.conv.kernel"]
    if image.shape == (1, 28, 28):
        image = image.unsqueeze(0)
    if image.shape != (1, 1, 28, 28) or image.dtype != torch.float32 or not torch.isfinite(image).all():
        raise ValueError("Preview requires one finite float32 MNIST image")
    with torch.inference_mode(False):
        model.to(device).eval()
    image = image.to(device)
    result = {}

    def capture(output):
        values = _channel_view(output, activation, channel)[0]
        result.update(**_statistics(values), values=values.cpu().tolist())

    with _once_per_forward(model, activation, capture):
        model(image)
    kernel = _channel_view(model.get_parameter(weight.parameter_name), weight, channel)
    return {
        "schema_version": 1, "target_id": target_id, "channel_index": channel,
        "actual_device": str(image.device), "hook": activation.module_path,
        "parameter_name": weight.parameter_name, "activation": result, "kernel": _statistics(kernel),
        "image": image[0, 0].cpu().tolist(),
    }
