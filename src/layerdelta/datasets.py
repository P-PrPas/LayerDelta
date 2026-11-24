"""Datasets (v0.5-E2): a human-only import of labelled images into a content-addressed, closed-format manifest.

A dataset is imported by a person, like a checkpoint; agents choose among registered datasets and never name a path. Import reads FILES AS
DATA: it hashes them, parses their PNG/BMP headers (dimensions, bit depth) and copies them into `datasets/files/<sha256>`; it never
decodes pixels (decoding runs only in the evaluation worker, preprocess.py), so a hostile image cannot reach the server process.

Formats (a closed set; a new one is a code change, with no model-specific code):

- `image-folder-labels` (classification): `labels.csv` with columns `image,label[,condition]`;
- `image-mask` (segmentation): `index.csv` with `image,mask[,condition]` and `classes.json` `{"classes": [...], "ignore_index": 255}`;
- `image-boxes` (detection): `annotations.json`, a COCO-style subset `{images: [{id, file_name, width, height}], annotations: [{image_id,
  bbox: [x, y, w, h], category_id}], categories: [{id, name}]}` (+ optional `condition` on an image);
- `image-text` (OCR): `index.csv` with `image,text[,condition]`.

Splits: `exploration` and `holdout`, by the v0.1 rule `stratified-hash-half-v1` (per stratum, samples ranked by SHA-256 of their id, the first
half exploration). Agents may evaluate `exploration` only; the holdout count, hash and rows are never in an agent's view.

Preprocessing is declarative (`check_preprocessing`): a closed list of steps in a fixed order, applied by preprocess.py. Stdlib only.
"""

import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

from .selector import InspectError, canonical_json, digest

DIR = "datasets"
FORMATS = {"image-folder-labels": "classification", "image-mask": "segmentation", "image-boxes": "detection", "image-text": "ocr"}
SPLIT_REVISION = "stratified-hash-half-v1"
_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{1,63}")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_CHARS = re.compile(r"[ -~]{0,256}")  # printable ASCII: an OCR line, or a class label
INDEX_MAX = 32 << 20


@dataclass(frozen=True)
class Limits:
    max_samples: int = 200_000
    max_file_bytes: int = 64 << 20
    max_total_bytes: int = 8 << 30
    max_pixels: int = 1 << 26
    max_side: int = 1 << 14


class DatasetError(InspectError):
    def __init__(self, code: str, message: str, field: str = "path"):
        super().__init__(code, message, field)


# ---------------------------------------------------------------- headers (no decoding)

def image_header(data: bytes, *, mask: bool = False) -> tuple[int, int]:
    """(width, height) of a PNG or BMP from its header alone; refuses anything outside the supported subset."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR" and len(data) >= 29:
        width, height, depth, color, _compression, _filter, interlace = struct.unpack(">IIBBBBB", data[16:29])
        if depth != 8 or interlace != 0:
            raise DatasetError("unsupported_image", "only non-interlaced 8-bit PNG files are supported", "image")
        if color not in ((0, 3) if mask else (0, 2, 3, 6)):
            raise DatasetError("unsupported_image", "a mask must be an 8-bit grayscale or palette PNG" if mask else "unsupported PNG colour type", "image")
        return width, height
    if mask:
        raise DatasetError("unsupported_image", "a mask must be a PNG file", "mask")
    if data[:2] == b"BM" and len(data) >= 34:
        width, height = struct.unpack("<ii", data[18:26])
        planes, bits, compression = struct.unpack("<HHI", data[26:34])
        if compression != 0 or bits not in (8, 24, 32) or planes != 1 or width <= 0 or height == 0:
            raise DatasetError("unsupported_image", "only uncompressed 8, 24 or 32-bit BMP files are supported", "image")
        return width, abs(height)
    raise DatasetError("unsupported_image", "an image must be a PNG or BMP file (no JPEG in v0.5)", "image")


# ---------------------------------------------------------------- preprocessing spec

_STEP_KEYS = ("resize", "resize_height", "pad_to", "grayscale", "channel_order", "scale", "mean", "std")


def check_preprocessing(spec) -> dict:
    """The canonical preprocessing spec. Steps run in this order: resize (or resize_height), pad_to, grayscale, channel_order, scale,
    mean and std. `{}` is the identity (pixels as float32 0..255)."""
    bad = lambda message, field: DatasetError("invalid_preprocessing", f"{message} ({field})", field)  # noqa: E731
    if spec is None:
        spec = {}
    if not isinstance(spec, dict) or not set(spec) <= set(_STEP_KEYS):
        raise bad(f"preprocessing keys are {list(_STEP_KEYS)}", "preprocessing")
    out: dict = {}
    size = lambda v, f: ([int(v[0]), int(v[1])] if isinstance(v, list) and len(v) == 2 and all(type(n) is int and 1 <= n <= 4096 for n in v)  # noqa: E731
                         else (_ for _ in ()).throw(bad("size is [height, width], integers 1 to 4096", f)))
    if "resize" in spec and "resize_height" in spec:
        raise bad("resize and resize_height are exclusive", "resize")
    if "resize" in spec:
        r = spec["resize"]
        if (not isinstance(r, dict) or set(r) - {"size", "mode", "antialias"} or "size" not in r or r.get("mode", "bilinear") not in ("nearest", "bilinear")
                or type(r.get("antialias", False)) is not bool):
            raise bad("resize is {size, mode: nearest|bilinear, antialias?}", "resize")
        out["resize"] = {"size": size(r["size"], "resize.size"), "mode": r.get("mode", "bilinear"), "antialias": r.get("antialias", False)}
    if "resize_height" in spec:
        r = spec["resize_height"]
        if not isinstance(r, dict) or set(r) != {"height", "keep_aspect"} or type(r["height"]) is not int or not 1 <= r["height"] <= 4096 or r["keep_aspect"] is not True:
            raise bad("resize_height is {height, keep_aspect: true}", "resize_height")
        out["resize_height"] = {"height": r["height"], "keep_aspect": True}
    if "pad_to" in spec:
        p = spec["pad_to"]
        if (not isinstance(p, dict) or set(p) - {"size", "value", "anchor"} or "size" not in p or p.get("anchor", "top_left") not in ("top_left", "center")
                or isinstance(p.get("value", 0), bool) or not isinstance(p.get("value", 0), (int, float)) or not math.isfinite(p.get("value", 0))):
            raise bad("pad_to is {size, value?, anchor: top_left|center}", "pad_to")
        out["pad_to"] = {"size": size(p["size"], "pad_to.size"), "value": float(p.get("value", 0)), "anchor": p.get("anchor", "top_left")}
    if "grayscale" in spec:
        if spec["grayscale"] is not True:
            raise bad("grayscale is true or absent", "grayscale")
        out["grayscale"] = True
    if "channel_order" in spec:
        if spec["channel_order"] not in ("rgb", "bgr"):
            raise bad("channel_order is rgb or bgr", "channel_order")
        out["channel_order"] = spec["channel_order"]
    if "scale" in spec:
        s = spec["scale"]
        if isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) or s <= 0:
            raise bad("scale is a positive number", "scale")
        out["scale"] = float(s)
    for key in ("mean", "std"):
        if key in spec:
            v = spec[key]
            if (not isinstance(v, list) or len(v) not in (1, 3) or not all(isinstance(n, (int, float)) and not isinstance(n, bool) and math.isfinite(n) for n in v)
                    or (key == "std" and any(n <= 0 for n in v))):
                raise bad(f"{key} is 1 or 3 finite numbers" + (", all positive" if key == "std" else ""), key)
            out[key] = [float(n) for n in v]
    if ("mean" in out) != ("std" in out):
        raise bad("give mean and std together", "mean")
    if "mean" in out and len(out["mean"]) != len(out["std"]):
        raise bad("mean and std have the same length", "mean")
    return out


# ---------------------------------------------------------------- reading a source directory

def _safe_file(root: Path, name, limits: Limits, field: str) -> Path:
    if not isinstance(name, str) or not name or "\0" in name or name != name.strip() or len(name) > 255:
        raise DatasetError("invalid_index", f"{field} is not a file name", field)
    parts = Path(name.replace("\\", "/")).parts
    if Path(name).is_absolute() or ".." in parts or ":" in name:
        raise DatasetError("invalid_index", f"{name!r} must be a relative path inside the dataset directory", field)
    path = root.joinpath(*parts)
    if path.is_symlink() or not path.is_file() or root.resolve() not in path.resolve().parents:
        raise DatasetError("invalid_index", f"{name!r} is not a regular file inside the dataset directory", field)
    if path.stat().st_size > limits.max_file_bytes:
        raise DatasetError("dataset_too_large", f"{name!r} is larger than {limits.max_file_bytes} bytes", field)
    return path


def _read_index(root: Path, name: str) -> bytes:
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise DatasetError("invalid_index", f"the dataset directory has no {name}", name)
    if path.stat().st_size > INDEX_MAX:
        raise DatasetError("dataset_too_large", f"{name} is larger than {INDEX_MAX} bytes", name)
    return path.read_bytes()


def _csv_rows(data: bytes, columns: tuple, optional: tuple, name: str) -> list[dict]:
    try:
        reader = csv.DictReader(io.StringIO(data.decode("utf-8")))
        names = reader.fieldnames or []
        if not set(columns) <= set(names) or not set(names) <= set(columns) | set(optional):
            raise DatasetError("invalid_index", f"{name} has columns {list(columns)} (and optionally {list(optional)})", name)
        rows = [dict(r) for r in reader]
    except (UnicodeDecodeError, csv.Error) as exc:
        raise DatasetError("invalid_index", f"{name} is not UTF-8 CSV: {type(exc).__name__}", name) from exc
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise DatasetError("invalid_index", f"{name} has a row with the wrong number of fields", name)
    return rows


def _sample_id(image: str, seen: set) -> str:
    sid = Path(image.replace("\\", "/")).with_suffix("").as_posix().replace("/", "_")
    if not _ID.fullmatch(sid) or sid in seen:
        raise DatasetError("invalid_index", f"the image name {image!r} does not give a unique sample id of [A-Za-z0-9._-], up to 64 characters", "image")
    seen.add(sid)
    return sid


def _condition(value) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9_.-]{1,32}", value):
        raise DatasetError("invalid_index", "a condition is 1 to 32 characters of [a-z0-9_.-]", "condition")
    return value


def _text(value, field) -> str:
    if not isinstance(value, str) or not _CHARS.fullmatch(value):
        raise DatasetError("invalid_index", f"{field} must be printable ASCII, up to 256 characters", field)
    return value


def _parse(root: Path, fmt: str, limits: Limits) -> tuple[list[dict], dict]:
    """(sample rows with file names and labels, format metadata). Files are checked later, one by one."""
    seen: set = set()
    if fmt == "image-folder-labels":
        rows = _csv_rows(_read_index(root, "labels.csv"), ("image", "label"), ("condition",), "labels.csv")
        classes = sorted({_text(r["label"], "label") for r in rows})
        if not 2 <= len(classes) <= 4096:
            raise DatasetError("invalid_index", "a classification dataset has 2 to 4096 classes", "label")
        return ([{"sample_id": _sample_id(r["image"], seen), "image_file": r["image"], "label": r["label"], "condition": _condition(r.get("condition"))}
                 for r in rows], {"classes": classes})
    if fmt == "image-text":
        rows = _csv_rows(_read_index(root, "index.csv"), ("image", "text"), ("condition",), "index.csv")
        alphabet = "".join(sorted({c for r in rows for c in _text(r["text"], "text")}))
        return ([{"sample_id": _sample_id(r["image"], seen), "image_file": r["image"], "text": r["text"], "condition": _condition(r.get("condition"))}
                 for r in rows], {"alphabet": alphabet})
    if fmt == "image-mask":
        try:
            meta = json.loads(_read_index(root, "classes.json"))
            classes, ignore = meta["classes"], meta.get("ignore_index", 255)
            ok = (isinstance(classes, list) and 2 <= len(classes) <= 255 and all(isinstance(c, str) for c in classes)
                  and type(ignore) is int and 0 <= ignore <= 255 and ignore >= len(classes))
        except (ValueError, KeyError, TypeError):
            ok = False
        if not ok:
            raise DatasetError("invalid_index", 'classes.json is {"classes": [2 to 255 names], "ignore_index": an integer from len(classes) to 255}', "classes.json")
        rows = _csv_rows(_read_index(root, "index.csv"), ("image", "mask"), ("condition",), "index.csv")
        return ([{"sample_id": _sample_id(r["image"], seen), "image_file": r["image"], "mask_file": r["mask"], "condition": _condition(r.get("condition"))}
                 for r in rows], {"classes": [_text(c, "classes") for c in classes], "ignore_index": ignore})
    try:  # image-boxes
        data = json.loads(_read_index(root, "annotations.json"))
        categories = {c["id"]: _text(c["name"], "categories") for c in data["categories"]}
        images, annotations = data["images"], data["annotations"]
        if not isinstance(images, list) or not isinstance(annotations, list) or not 1 <= len(categories) <= 4096 or not all(type(i) is int for i in categories):
            raise ValueError
    except (ValueError, KeyError, TypeError) as exc:
        raise DatasetError("invalid_index", "annotations.json is {images, annotations, categories} (COCO subset)", "annotations.json") from exc
    boxes: dict = {}
    for a in annotations:
        try:
            x, y, w, h = a["bbox"]
            ok = all(isinstance(n, (int, float)) and not isinstance(n, bool) and math.isfinite(n) for n in (x, y, w, h)) and w > 0 and h > 0
            if not ok or a["category_id"] not in categories:
                raise ValueError
        except (ValueError, KeyError, TypeError) as exc:
            raise DatasetError("invalid_index", "each annotation is {image_id, bbox: [x, y, w, h] with w, h > 0, category_id of a category}", "annotations") from exc
        boxes.setdefault(a["image_id"], []).append({"bbox": [float(x), float(y), float(w), float(h)], "category_id": a["category_id"]})
    out = []
    for item in images:
        try:
            out.append({"sample_id": _sample_id(item["file_name"], seen), "image_file": item["file_name"], "width": item["width"], "height": item["height"],
                        "boxes": boxes.get(item["id"], []), "condition": _condition(item.get("condition"))})
        except (KeyError, TypeError) as exc:
            raise DatasetError("invalid_index", "each image is {id, file_name, width, height}", "images") from exc
    return out, {"categories": [{"id": i, "name": categories[i]} for i in sorted(categories)]}


# ---------------------------------------------------------------- splits and the manifest

def assign_splits(rows: list[dict], stratum) -> None:
    """stratified-hash-half-v1: per stratum, samples ranked by SHA-256 of their id; the first half is exploration, the rest holdout."""
    strata: dict = {}
    for row in rows:
        strata.setdefault(stratum(row), []).append(row)
    for members in strata.values():
        members.sort(key=lambda r: hashlib.sha256(f"layerdelta-dataset-v1/{r['sample_id']}".encode()).digest())
        half = len(members) // 2
        for k, row in enumerate(members):
            row["split"] = "exploration" if k < half else "holdout"
    rows.sort(key=lambda r: r["sample_id"])


def build_manifest(root: Path, fmt: str, limits: Limits = Limits(), store: Path | None = None) -> dict:
    """Parse, check and (when `store` is given) copy every file of a source directory; the manifest of the dataset."""
    if fmt not in FORMATS:
        raise DatasetError("unknown_format", f"formats: {sorted(FORMATS)}", "format")
    rows, meta = _parse(root, fmt, limits)
    if not 4 <= len(rows) <= limits.max_samples:
        raise DatasetError("dataset_too_small" if len(rows) < 4 else "dataset_too_large", f"a dataset has 4 to {limits.max_samples} samples, this has {len(rows)}", "samples")
    total = 0
    known_ids = set()
    for row in rows:
        for field, mask in (("image_file", False), ("mask_file", True)):
            if field not in row:
                continue
            path = _safe_file(root, row[field], limits, field.split("_")[0])
            data = path.read_bytes()
            total += len(data)
            if total > limits.max_total_bytes:
                raise DatasetError("dataset_too_large", f"the files are larger than {limits.max_total_bytes} bytes in total", "path")
            width, height = image_header(data, mask=mask)
            if not (1 <= width <= limits.max_side and 1 <= height <= limits.max_side and width * height <= limits.max_pixels):
                raise DatasetError("unsupported_image", f"{row[field]!r} is {width}x{height}; the limit is {limits.max_pixels} pixels and {limits.max_side} on a side", field.split("_")[0])
            sha = hashlib.sha256(data).hexdigest()
            if store is not None:
                _store(store, sha, data)
            kind = "image" if not mask else "mask"
            row[kind] = sha
            row.pop(field)
            if mask and (row["width"], row["height"]) != (width, height) if "width" in row else False:
                raise DatasetError("invalid_index", "a mask and its image differ in size", "mask")
            row.setdefault("width", width)
            row.setdefault("height", height)
            if (row["width"], row["height"]) != (width, height):
                raise DatasetError("invalid_index", f"{row['sample_id']}: the file is {width}x{height}, the index says {row['width']}x{row['height']}", "image")
        known_ids.add(row["sample_id"])
    if fmt == "image-boxes":
        for row in rows:
            if any(b["bbox"][0] < 0 or b["bbox"][1] < 0 or b["bbox"][0] + b["bbox"][2] > row["width"] + 1e-6 or b["bbox"][1] + b["bbox"][3] > row["height"] + 1e-6
                   for b in row["boxes"]):
                raise DatasetError("invalid_index", f"{row['sample_id']}: a box lies outside the image", "annotations")
    assign_splits(rows, (lambda r: r["label"]) if fmt == "image-folder-labels" else (lambda r: r["condition"] or ""))
    if not any(r["split"] == "exploration" for r in rows) or not any(r["split"] == "holdout" for r in rows):
        raise DatasetError("dataset_too_small", "each split needs at least one sample", "samples")
    manifest = {"schema_version": 1, "format": fmt, "task": FORMATS[fmt], "split_revision": SPLIT_REVISION, **meta, "samples": rows}
    manifest["split_sha256"] = {s: digest([r for r in rows if r["split"] == s]) for s in ("exploration", "holdout")}
    manifest["manifest_sha256"] = digest(manifest)
    return manifest


def _store(store: Path, sha: str, data: bytes) -> None:
    store.mkdir(parents=True, exist_ok=True)
    target = store / sha
    if not target.exists():
        partial = store / f".{sha}.partial-{os.getpid()}"
        partial.write_bytes(data)
        os.replace(partial, target)


# ---------------------------------------------------------------- the catalog of datasets

def dataset_id(manifest: dict, preprocessing: dict) -> str:
    return "ds_" + digest({"manifest": manifest["manifest_sha256"], "preprocessing": preprocessing})[:32]


def public_record(manifest: dict, preprocessing: dict, name: str, ident: str) -> dict:
    """What a person sees (the whole dataset); `agent_view` is what a token sees."""
    extra = {k: manifest[k] for k in ("classes", "categories", "alphabet", "ignore_index") if k in manifest}
    counts = {s: sum(1 for r in manifest["samples"] if r["split"] == s) for s in ("exploration", "holdout")}
    return {"dataset_id": ident, "name": name, "format": manifest["format"], "task": manifest["task"], "preprocessing": preprocessing,
            "split_revision": manifest["split_revision"], "counts": counts, "split_sha256": manifest["split_sha256"],
            "manifest_sha256": manifest["manifest_sha256"], **extra}


MNIST_ID = "mnist-v1"


def mnist_record(manifest: dict) -> dict:
    """The v0.1 MNIST manifest as the first dataset, `mnist-v1`, unchanged (its samples come from the idx archives, not from files)."""
    counts, hashes = manifest.get("counts", {}), manifest.get("split_sha256", {})  # a fixture without them still lists
    return {"dataset_id": MNIST_ID, "name": MNIST_ID, "format": "mnist-idx", "task": "classification", "preprocessing": manifest.get("preprocessing", {}),
            "split_revision": manifest.get("split_revision"), "classes": [str(d) for d in range(10)],
            "counts": {s: sum(counts.get(s, {}).values()) for s in ("exploration", "holdout")},
            "split_sha256": {s: hashes.get(s) for s in ("exploration", "holdout")}, "manifest_sha256": manifest.get("manifest_sha256")}


def agent_view(record: dict) -> dict:
    """A token's view: no holdout count and no holdout hash (agents evaluate `exploration` only)."""
    view = {k: v for k, v in record.items() if k not in ("counts", "split_sha256")}
    view["counts"] = {"exploration": record["counts"]["exploration"]}
    view["split_sha256"] = {"exploration": record["split_sha256"]["exploration"]}
    return view


def import_dataset(workspace: Path, source: Path, *, fmt: str, name: str, preprocessing=None, limits: Limits = Limits()) -> dict:
    """Import `source` (a directory in one of the closed formats) as dataset `name`. Idempotent for identical content."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise DatasetError("invalid_name", "name is 2 to 64 characters of [a-z0-9._-], starting with a letter or digit", "name")
    preprocessing = check_preprocessing(preprocessing)
    source = Path(source)
    if not source.is_absolute() or not source.is_dir():
        raise DatasetError("invalid_path", "path must be an absolute directory that exists", "path")
    root = Path(workspace) / DIR
    manifest = build_manifest(source, fmt, limits, store=root / "files")
    ident = dataset_id(manifest, preprocessing)
    folder = root / ident
    folder.mkdir(parents=True, exist_ok=True)
    record = public_record(manifest, preprocessing, name, ident)
    (folder / "manifest.json").write_text(canonical_json(manifest), encoding="utf-8")
    (folder / "record.json").write_text(canonical_json(record), encoding="utf-8")
    return record


@dataclass(frozen=True)
class Dataset:
    record: dict
    manifest: dict
    files: Path

    def path(self, sha: str) -> Path:
        """A stored file by hash; it is re-hashed when read by the worker (executable-style trust: only files this module wrote)."""
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise DatasetError("invalid_argument", "not a file hash", "sha256")
        return self.files / sha

    def samples(self, split: str) -> list[dict]:
        return [r for r in self.manifest["samples"] if r["split"] == split]


def list_records(workspace: Path) -> list[dict]:
    root = Path(workspace) / DIR
    out = []
    for folder in sorted(root.glob("ds_*")) if root.is_dir() else []:
        try:
            out.append(json.loads((folder / "record.json").read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return sorted(out, key=lambda r: (r["name"], r["dataset_id"]))


def open_dataset(workspace: Path, ident: str) -> Dataset:
    if not isinstance(ident, str) or not re.fullmatch(r"ds_[0-9a-f]{32}", ident):
        raise DatasetError("not_found", "Unknown dataset", "dataset_id")
    folder = Path(workspace) / DIR / ident
    try:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        record = json.loads((folder / "record.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DatasetError("not_found", "Unknown dataset", "dataset_id") from exc
    return Dataset(record, manifest, Path(workspace) / DIR / "files")
