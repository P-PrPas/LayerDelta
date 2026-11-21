"""Content-verified MNIST sources and the LayerDelta v1 experiment split."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import struct
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING
from urllib.request import urlopen

if TYPE_CHECKING:
    import torch


# Archives and SHA-256 values are published by TensorFlow Datasets:
# https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/mnist.txt
BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
SOURCE_SHA256 = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}
BLUR_KERNEL = 5
SPLITS = ("train", "exploration", "holdout")


def _read_archive(path: Path) -> bytes:
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256[path.name]:
        raise ValueError(f"MNIST archive digest mismatch: {path.name}")
    return data


def download_mnist(cache_dir: Path) -> dict[str, str]:
    """Fetch missing archives and verify cached bytes on every call."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, expected in SOURCE_SHA256.items():
        destination = cache_dir / name
        if destination.exists():
            _read_archive(destination)
            continue
        with NamedTemporaryFile(dir=cache_dir, delete=False) as temporary:
            staged = Path(temporary.name)
        try:
            with urlopen(BASE_URL + name, timeout=60) as response, staged.open("wb") as output:
                shutil.copyfileobj(response, output)
            if hashlib.sha256(staged.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Downloaded MNIST archive digest mismatch: {name}")
            staged.replace(destination)
        finally:
            staged.unlink(missing_ok=True)
    return SOURCE_SHA256.copy()


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _labels(cache_dir: Path, partition: str) -> bytes:
    raw = gzip.decompress(_read_archive(cache_dir / f"{partition}-labels-idx1-ubyte.gz"))
    if len(raw) < 8:
        raise ValueError("Truncated MNIST label header")
    magic, count = struct.unpack_from(">II", raw)
    labels = raw[8:]
    if magic != 2049 or len(labels) != count or any(label > 9 for label in labels):
        raise ValueError("Invalid MNIST labels")
    return labels


def _test_membership(labels: bytes) -> dict[int, str]:
    """Stratify by truth with hash-ranked indices, independent of RNG versions."""
    membership = {}
    for digit in range(10):
        indices = [i for i, label in enumerate(labels) if label == digit]
        indices.sort(key=lambda i: hashlib.sha256(f"layerdelta-v1/t10k/{i}".encode()).digest())
        midpoint = len(indices) // 2
        membership.update((i, "exploration") for i in indices[:midpoint])
        membership.update((i, "holdout") for i in indices[midpoint:])
    return membership


def build_manifest(cache_dir: Path) -> dict:
    """Describe all split members and transforms using verified local archives."""
    samples = []
    counts = {split: Counter() for split in SPLITS}
    for partition in ("train", "t10k"):
        _read_archive(cache_dir / f"{partition}-images-idx3-ubyte.gz")
        labels = _labels(cache_dir, partition)
        membership = _test_membership(labels) if partition == "t10k" else {}
        for index, label in enumerate(labels):
            split = "train" if partition == "train" else membership[index]
            source_id = f"mnist/{partition}/{index:05d}"
            conditions = ("normal",) if split == "train" else ("normal", "blurred")
            for condition in conditions:
                samples.append({
                    "sample_id": f"{source_id}/{condition}",
                    "source_sample_id": source_id,
                    "split": split,
                    "condition": condition,
                    "label": label,
                    "index": index,
                })
                counts[split][condition] += 1
    manifest = {
        "dataset_revision_id": "mnist-v1",
        "source_sha256": SOURCE_SHA256.copy(),
        "source_url_prefix": BASE_URL,
        "split_revision": "stratified-hash-half-v1",
        "preprocessing": {
            "revision": "box-blur-v1",
            "normalization": "float32 pixel / 255",
            "kernel_width": BLUR_KERNEL,
            "padding": "zeros",
            "stride": 1,
        },
        "counts": {split: dict(count) for split, count in counts.items()},
        "samples": samples,
        "split_sha256": {
            split: _digest([row for row in samples if row["split"] == split])
            for split in SPLITS
        },
    }
    manifest["manifest_sha256"] = _digest(manifest)
    return manifest


def load_split(cache_dir: Path, manifest: dict, split: str) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    """Return CPU float32 images [N,1,28,28], integer labels, and aligned rows."""
    import torch
    import torch.nn.functional as F

    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    expected_digest = build_manifest(cache_dir)["manifest_sha256"]
    payload = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != expected_digest or _digest(payload) != expected_digest:
        raise ValueError("MNIST manifest mismatch with verified source data")
    rows = [row for row in manifest["samples"] if row["split"] == split]
    partition = "train" if split == "train" else "t10k"
    raw = gzip.decompress(_read_archive(cache_dir / f"{partition}-images-idx3-ubyte.gz"))
    if len(raw) < 16:
        raise ValueError("Truncated MNIST image header")
    magic, count, height, width = struct.unpack_from(">IIII", raw)
    if (magic, height, width) != (2051, 28, 28) or len(raw) != 16 + count * 784:
        raise ValueError("Invalid MNIST images")
    if count != len(_labels(cache_dir, partition)):
        raise ValueError("MNIST image and label counts differ")
    pixels = torch.frombuffer(bytearray(raw), dtype=torch.uint8, offset=16).view(count, 1, 28, 28)
    labels = torch.tensor([row["label"] for row in rows], dtype=torch.long)
    indices = torch.tensor([row["index"] for row in rows], dtype=torch.long)
    images = pixels.index_select(0, indices).float()
    blurred = torch.tensor([row["condition"] == "blurred" for row in rows], dtype=torch.bool)
    if blurred.any():
        images[blurred] = F.avg_pool2d(images[blurred], BLUR_KERNEL, stride=1, padding=BLUR_KERNEL // 2)
    return images.div_(255), labels, rows
