"""Seeded synthetic datasets for the evaluator oracles and the v0.5 benchmark (task E2s). No licence needed: every pixel is generated here.

    python tools/bench/synthetic.py shapes-seg --out DIR [--n 400] [--seed 0]
    python tools/bench/synthetic.py boxes-det  --out DIR
    python tools/bench/synthetic.py glyph-ocr  --out DIR

Each writes a directory in one of the import formats of docs/reference/datasets.md (a person then imports it like any dataset), plus
`synthetic.json` with the generator, version, seed and counts. The same (generator, version, seed, n) gives byte-identical files.
Spec: docs/reference/synthetic-datasets.md. Images are 8-bit PNG written by a stdlib encoder (zlib), so generating needs only numpy.

Every dataset has two conditions, so the splits are stratified by condition and a "degraded condition" task (stratum S-nat, design §5.3)
has the same number of samples on both sides: `clean`, and `noisy` (shapes) or `blur` (glyphs).
"""

import argparse
import json
import struct
import zlib
from pathlib import Path

import numpy as np

VERSION = 1
SHAPES = ("disc", "square", "triangle")

# 5x7 digit glyphs, drawn for this repository
GLYPHS = {
    "0": (" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "),
    "1": ("  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "),
    "2": (" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"),
    "3": ("#####", "   # ", "  #  ", "   # ", "    #", "#   #", " ### "),
    "4": ("   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "),
    "5": ("#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "),
    "6": ("  ## ", " #   ", "#    ", "#### ", "#   #", "#   #", " ### "),
    "7": ("#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "),
    "8": (" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "),
    "9": (" ### ", "#   #", "#   #", " ####", "    #", "   # ", " ##  "),
}
GLYPH = {k: np.array([[c == "#" for c in row] for row in v], dtype=bool) for k, v in GLYPHS.items()}


def png(array: np.ndarray) -> bytes:
    """An 8-bit, non-interlaced PNG of a [H, W] (grayscale) or [H, W, 3] (RGB) uint8 array. Deterministic."""
    a = np.ascontiguousarray(array, dtype=np.uint8)
    h, w = a.shape[:2]
    color = 0 if a.ndim == 2 else 2

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def shape_mask(kind: str, cx: float, cy: float, r: float, h: int, w: int) -> np.ndarray:
    """Pixels whose centre lies in the shape: a disc of radius r, a square of half-side r, an upward triangle of half-base and height 2r."""
    y, x = np.mgrid[0:h, 0:w] + 0.5
    if kind == "disc":
        return (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    if kind == "square":
        return (np.abs(x - cx) <= r) & (np.abs(y - cy) <= r)
    top = cy - r
    depth = (y - top) / (2 * r)
    return (depth >= 0) & (depth <= 1) & (np.abs(x - cx) <= depth * r)


def _canvas(rng, h: int, w: int, condition: str) -> np.ndarray:
    base = rng.uniform(40, 215, size=3)
    noise = rng.normal(0, 8 if condition == "clean" else 40, size=(h, w, 3))
    return base + noise


def _paint(image: np.ndarray, mask: np.ndarray, rng, condition: str) -> None:
    color = rng.uniform(0, 255, size=3)
    image[mask] = color + rng.normal(0, 8 if condition == "clean" else 40, size=(int(mask.sum()), 3))


def _condition(rng) -> str:
    return "clean" if rng.random() < 0.5 else "noisy"


def shapes_seg(out: Path, n: int, seed: int, size: int = 32) -> dict:
    """1 to 3 shapes per image, later ones painted over earlier ones (the mask follows the paint). Classes: background, disc, square,
    triangle; ignore index 255 (never in a mask here: preprocessing pads with it)."""
    rng = np.random.default_rng([seed, 1])
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)
    rows = ["image,mask,condition"]
    for i in range(n):
        condition = _condition(rng)
        image, labels = _canvas(rng, size, size, condition), np.zeros((size, size), dtype=np.uint8)
        for _ in range(int(rng.integers(1, 4))):
            k = int(rng.integers(len(SHAPES)))
            r = rng.uniform(3, size / 4)
            mask = shape_mask(SHAPES[k], *rng.uniform(r, size - r, size=2), r, size, size)
            _paint(image, mask, rng, condition)
            labels[mask] = k + 1
        name = f"s{i:05d}.png"
        (out / "images" / name).write_bytes(png(np.clip(image, 0, 255).round()))
        (out / "masks" / name).write_bytes(png(labels))
        rows.append(f"images/{name},masks/{name},{condition}")
    (out / "index.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (out / "classes.json").write_text(json.dumps({"classes": ["background", *SHAPES], "ignore_index": 255}), encoding="utf-8")
    return {"format": "image-mask", "task": "segmentation", "size": [size, size]}


def boxes_det(out: Path, n: int, seed: int, size: int = 48) -> dict:
    """0 to 4 non-overlapping shapes per image; each box is the tight extent of the shape's pixels. Categories 1 disc, 2 square,
    3 triangle (COCO-style ids, not indices)."""
    rng = np.random.default_rng([seed, 2])
    (out / "images").mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for i in range(n):
        condition = _condition(rng)
        image, taken = _canvas(rng, size, size, condition), np.zeros((size, size), dtype=bool)
        for _ in range(int(rng.integers(0, 5))):
            for _attempt in range(20):
                k = int(rng.integers(len(SHAPES)))
                r = rng.uniform(3, 8)
                mask = shape_mask(SHAPES[k], *rng.uniform(r, size - r, size=2), r, size, size)
                grown = mask | np.roll(mask, 1, 0) | np.roll(mask, -1, 0) | np.roll(mask, 1, 1) | np.roll(mask, -1, 1)
                if mask.sum() >= 4 and not (grown & taken).any():  # a one-pixel gap: boxes may touch, shapes never do
                    break
            else:
                continue
            taken |= mask
            _paint(image, mask, rng, condition)
            ys, xs = np.nonzero(mask)
            annotations.append({"id": len(annotations) + 1, "image_id": i, "category_id": k + 1,
                                "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]})
        name = f"d{i:05d}.png"
        (out / "images" / name).write_bytes(png(np.clip(image, 0, 255).round()))
        images.append({"id": i, "file_name": f"images/{name}", "width": size, "height": size, "condition": condition})
    document = {"images": images, "annotations": annotations,
                "categories": [{"id": k + 1, "name": name} for k, name in enumerate(SHAPES)]}
    (out / "annotations.json").write_text(json.dumps(document), encoding="utf-8")
    return {"format": "image-boxes", "task": "detection", "size": [size, size]}


def render(text: str, rng, condition: str, scale: int = 2) -> np.ndarray:
    """Dark digits on a light line: glyphs scaled by `scale`, `scale` pixels apart, a random vertical jitter of up to one pixel each."""
    margin, gap, gh, gw = 2, scale, 7 * scale, 5 * scale
    h = gh + 2 * margin + 2
    w = 2 * margin + len(text) * gw + (len(text) - 1) * gap
    ink = np.zeros((h, w), dtype=bool)
    for k, ch in enumerate(text):
        top = margin + 1 + int(rng.integers(-1, 2))
        left = margin + k * (gw + gap)
        ink[top:top + gh, left:left + gw] |= np.kron(GLYPH[ch], np.ones((scale, scale), dtype=bool))
    image = np.where(ink, rng.uniform(0, 60), rng.uniform(190, 255)) + rng.normal(0, 6, size=(h, w))
    if condition == "blur":  # two passes of a 3x3 box blur, edges repeated
        for _ in range(2):
            padded = np.pad(image, 1, mode="edge")
            image = sum(padded[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)) / 9
    return np.clip(image, 0, 255).round().astype(np.uint8)


def glyph_ocr(out: Path, n: int, seed: int) -> dict:
    """Digit strings of 3 to 6 characters, rendered from the 5x7 glyphs at scale 2 (height 20, width by length)."""
    rng = np.random.default_rng([seed, 3])
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows = ["image,text,condition"]
    for i in range(n):
        condition = "clean" if rng.random() < 0.5 else "blur"
        text = "".join(rng.choice(list("0123456789"), size=int(rng.integers(3, 7))))
        name = f"g{i:05d}.png"
        (out / "images" / name).write_bytes(png(render(text, rng, condition)))
        rows.append(f"images/{name},{text},{condition}")
    (out / "index.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return {"format": "image-text", "task": "ocr", "height": 20}


GENERATORS = {"shapes-seg": shapes_seg, "boxes-det": boxes_det, "glyph-ocr": glyph_ocr}
DEFAULT_N = {"shapes-seg": 400, "boxes-det": 400, "glyph-ocr": 600}


def generate(name: str, out: Path, n: int | None = None, seed: int = 0) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    n = DEFAULT_N[name] if n is None else n
    info = {"generator": name, "version": VERSION, "seed": seed, "n": n, **GENERATORS[name](out, n, seed)}
    (out / "synthetic.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Write a seeded synthetic dataset in a LayerDelta import format")
    parser.add_argument("generator", choices=sorted(GENERATORS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    print(json.dumps(generate(args.generator, args.out, args.n, args.seed)))


if __name__ == "__main__":
    main()
