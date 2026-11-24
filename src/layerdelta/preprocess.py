"""Declarative preprocessing (v0.5-E2): decode a stored PNG or BMP and apply a `datasets.check_preprocessing` spec, labels included.

Runs in the evaluation worker only (it decodes untrusted bytes with Pillow, the one place pixels are decoded). One implementation:
`torch.nn.functional.interpolate`; PIL resampling is never used, because it differs across libraries and versions. The implementation
version (`implementation_version()`) is part of an evaluation's fingerprint, so an upgrade cannot change results silently.

Order: resize (or resize_height), pad_to, grayscale, channel_order, scale, mean/std. Pixels are float32 0..255 until `scale`.
Labels follow the geometric steps: a mask is resized with nearest-neighbour and padded with the ignore index; boxes are scaled and
offset, clipped to the output, and dropped when they become empty; text is unchanged. `geometry` records the transform so boxes can be
mapped back to source-image coordinates (`to_source`).
"""

import io

import numpy as np
import torch
import torch.nn.functional as F

from . import datasets

PIXELS_MAX = datasets.Limits().max_pixels
VERSION = 1


def implementation_version() -> str:
    import PIL
    return f"preprocess-v{VERSION}/torch-{torch.__version__.split('+')[0]}/pillow-{PIL.__version__}"


def decode(data: bytes, *, mask: bool = False) -> np.ndarray:
    """uint8 array [H, W, C] (C = 1 or 3; a mask is [H, W]). PNG and BMP only, with the pixel cap; alpha is dropped."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = PIXELS_MAX
    datasets.image_header(data, mask=mask)  # the closed subset, again: the worker trusts nothing it was handed
    with Image.open(io.BytesIO(data), formats=("PNG", "BMP")) as image:
        image.load()
        if mask:
            return np.asarray(image.convert("L") if image.mode not in ("L", "P") else image, dtype=np.uint8).copy()
        if image.mode == "L":
            return np.asarray(image, dtype=np.uint8)[:, :, None].copy()
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _resize(x: torch.Tensor, size, mode: str, antialias: bool) -> torch.Tensor:
    kwargs = {} if mode == "nearest" else {"align_corners": False, "antialias": antialias}
    return F.interpolate(x[None], size=tuple(size), mode=mode, **kwargs)[0]


def apply(spec: dict, image: np.ndarray, *, mask: np.ndarray | None = None, boxes: list | None = None, ignore_index: int = 255) -> dict:
    """{"image": float32 [C, H, W], "mask": int64 [H, W] | None, "boxes": float32 [K, 4] xyxy + "box_index" (rows kept), "geometry"}."""
    x = torch.from_numpy(image).permute(2, 0, 1).to(torch.float32)
    source_h, source_w = x.shape[1:]
    m = None if mask is None else torch.from_numpy(mask).to(torch.float32)[None]
    b = None if boxes is None else torch.tensor([[bx["bbox"][0], bx["bbox"][1], bx["bbox"][0] + bx["bbox"][2], bx["bbox"][1] + bx["bbox"][3]]
                                                 for bx in boxes], dtype=torch.float32).reshape(-1, 4)
    sx = sy = 1.0
    ox = oy = 0.0
    size = None
    if "resize" in spec:
        size = spec["resize"]["size"]
    elif "resize_height" in spec:
        height = spec["resize_height"]["height"]
        size = [height, max(1, round(source_w * height / source_h))]
    if size is not None:
        x = _resize(x, size, spec["resize"]["mode"] if "resize" in spec else "bilinear", spec.get("resize", {}).get("antialias", False))
        if m is not None:
            m = _resize(m, size, "nearest", False)
        sy, sx = size[0] / source_h, size[1] / source_w
    if b is not None:
        b = b * torch.tensor([sx, sy, sx, sy])
    if "pad_to" in spec:
        target_h, target_w = spec["pad_to"]["size"]
        h, w = x.shape[1:]
        if h > target_h or w > target_w:
            raise datasets.DatasetError("invalid_preprocessing", f"pad_to {[target_h, target_w]} is smaller than the image {[h, w]}", "pad_to")
        top, left = ((target_h - h) // 2, (target_w - w) // 2) if spec["pad_to"]["anchor"] == "center" else (0, 0)
        window = (left, target_w - w - left, top, target_h - h - top)
        x = F.pad(x, window, value=spec["pad_to"]["value"])
        if m is not None:
            m = F.pad(m, window, value=float(ignore_index))
        if b is not None:
            b = b + torch.tensor([left, top, left, top], dtype=torch.float32)
        ox, oy = float(left), float(top)
    if "grayscale" in spec and x.shape[0] == 3:  # Rec. 601 luma, the same weights every time
        x = (x * torch.tensor([0.299, 0.587, 0.114])[:, None, None]).sum(0, keepdim=True)
    if spec.get("channel_order") == "bgr" and x.shape[0] == 3:
        x = x.flip(0)
    if "scale" in spec:
        x = x * spec["scale"]
    if "mean" in spec:
        channels = x.shape[0]
        mean, std = torch.tensor(spec["mean"]), torch.tensor(spec["std"])
        if mean.numel() not in (1, channels):
            raise datasets.DatasetError("invalid_preprocessing", f"mean/std have {mean.numel()} values, the image has {channels} channels", "mean")
        x = (x - mean.reshape(-1, 1, 1)) / std.reshape(-1, 1, 1)
    out_h, out_w = x.shape[1:]
    keep = None
    if b is not None:
        b = torch.stack([b[:, 0].clamp(0, out_w), b[:, 1].clamp(0, out_h), b[:, 2].clamp(0, out_w), b[:, 3].clamp(0, out_h)], dim=1)
        keep = ((b[:, 2] - b[:, 0]) > 1e-3) & ((b[:, 3] - b[:, 1]) > 1e-3)  # a box that became empty is dropped
        b = b[keep]
    return {"image": x.contiguous(), "mask": None if m is None else m[0].to(torch.int64), "boxes": b,
            "box_index": None if keep is None else torch.nonzero(keep)[:, 0],
            "geometry": {"source": [source_h, source_w], "scale": [sx, sy], "offset": [ox, oy], "output": [out_h, out_w]}}


def to_source(boxes: torch.Tensor, geometry: dict) -> torch.Tensor:
    """Boxes (xyxy) in output coordinates mapped back through the inverse of the geometric steps, clipped to the source image."""
    (sx, sy), (ox, oy), (h, w) = geometry["scale"], geometry["offset"], geometry["source"]
    b = (boxes - torch.tensor([ox, oy, ox, oy])) / torch.tensor([sx, sy, sx, sy])
    return torch.stack([b[:, 0].clamp(0, w), b[:, 1].clamp(0, h), b[:, 2].clamp(0, w), b[:, 3].clamp(0, h)], dim=1)
