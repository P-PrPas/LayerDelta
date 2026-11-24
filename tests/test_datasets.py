"""Datasets (v0.5-E2): human-only import into closed formats, header-only checks, hash splits, declarative preprocessing.

The import tests write their PNG and BMP files by hand (struct + zlib), so they prove that importing needs no image decoder; only the
preprocessing tests need Pillow (the optional `eval` extra).
"""

import json
import struct
import zlib

import pytest

from layerdelta import datasets
from layerdelta.datasets import DatasetError, Limits


def png(width, height, *, color=0, depth=8, interlace=0, fill=0):
    channels = {0: 1, 2: 3, 3: 1, 6: 4}[color]

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    row = b"\x00" + bytes([fill % 256]) * (width * channels)
    ihdr = struct.pack(">IIBBBBB", width, height, depth, color, 0, 0, interlace)
    extra = chunk(b"PLTE", bytes(range(3)) * 3) if color == 3 else b""
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + extra + chunk(b"IDAT", zlib.compress(row * height)) + chunk(b"IEND", b"")


def bmp(width, height, bits=24):
    return b"BM" + struct.pack("<IHHI", 54 + 4 * width * abs(height), 0, 0, 54) + struct.pack("<IiiHHIIiiII", 40, width, height, 1, bits, 0, 0, 0, 0, 0, 0) + b"\0" * (4 * width * height)


def write_folder(root, n_per_class=4, classes=("cat", "dog"), size=(6, 5)):
    root.mkdir()
    rows = ["image,label,condition"]
    for c, name in enumerate(classes):
        for k in range(n_per_class):
            file = f"{name}_{k}.png"
            (root / file).write_bytes(png(*size, fill=10 * c + k))
            rows.append(f"{file},{name},{'blurred' if k % 2 else ''}")
    (root / "labels.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


def test_headers_accept_the_closed_subset_and_refuse_the_rest():
    assert datasets.image_header(png(7, 3)) == (7, 3)
    assert datasets.image_header(png(7, 3, color=2)) == (7, 3) and datasets.image_header(png(7, 3, color=6)) == (7, 3)
    assert datasets.image_header(bmp(5, 4)) == (5, 4) and datasets.image_header(bmp(5, -4, bits=8)) == (5, 4)
    assert datasets.image_header(png(4, 4, color=3), mask=True) == (4, 4) and datasets.image_header(png(4, 4), mask=True) == (4, 4)
    for bad in (b"\xff\xd8\xff\xe0" + b"\0" * 40,                                   # JPEG: decoders differ by libjpeg version
                png(4, 4, depth=16), png(4, 4, interlace=1), png(4, 4)[:20], b"", b"BM" + b"\0" * 10):
        with pytest.raises(DatasetError) as raised:
            datasets.image_header(bad)
        assert raised.value.code == "unsupported_image"
    with pytest.raises(DatasetError):
        datasets.image_header(png(4, 4, color=2), mask=True)  # a mask is grayscale or palette
    with pytest.raises(DatasetError):
        datasets.image_header(bmp(4, 4), mask=True)          # and a PNG


def test_a_classification_folder_imports_with_stratified_hash_halves(tmp_path):
    source = write_folder(tmp_path / "src")
    record = datasets.import_dataset(tmp_path / "ws", source, fmt="image-folder-labels", name="pets-v1", preprocessing={"scale": 1 / 255})
    assert record["task"] == "classification" and record["classes"] == ["cat", "dog"] and record["counts"] == {"exploration": 4, "holdout": 4}
    data = datasets.open_dataset(tmp_path / "ws", record["dataset_id"])
    for cls in ("cat", "dog"):  # stratified: every class has half of its samples in each split
        rows = [r for r in data.manifest["samples"] if r["label"] == cls]
        assert sorted(r["split"] for r in rows) == ["exploration"] * 2 + ["holdout"] * 2
    assert {r["sample_id"] for r in data.manifest["samples"]} == {f"{c}_{k}" for c in ("cat", "dog") for k in range(4)}
    row = data.manifest["samples"][0]
    assert data.path(row["image"]).is_file() and (row["width"], row["height"]) == (6, 5)  # the file is stored by its hash
    assert data.path(row["image"]).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    again = datasets.import_dataset(tmp_path / "ws", source, fmt="image-folder-labels", name="pets-v1", preprocessing={"scale": 1 / 255})
    assert again == record  # the same content, the same dataset
    other = datasets.import_dataset(tmp_path / "ws", source, fmt="image-folder-labels", name="pets-v1", preprocessing={})
    assert other["dataset_id"] != record["dataset_id"] and other["manifest_sha256"] == record["manifest_sha256"]  # preprocessing is part of the id
    assert [r["name"] for r in datasets.list_records(tmp_path / "ws")] == ["pets-v1", "pets-v1"]


def test_an_agent_never_sees_the_holdout():
    record = {"dataset_id": "ds_x", "counts": {"exploration": 5, "holdout": 6}, "split_sha256": {"exploration": "a", "holdout": "b"}, "name": "n"}
    view = datasets.agent_view(record)
    assert view["counts"] == {"exploration": 5} and view["split_sha256"] == {"exploration": "a"} and "holdout" not in json.dumps(view)


def test_masks_boxes_and_text_formats(tmp_path):
    root = tmp_path / "m"
    root.mkdir()
    rows = ["image,mask,condition"]
    for k in range(6):
        (root / f"i{k}.png").write_bytes(png(8, 8, color=2))
        (root / f"m{k}.png").write_bytes(png(8, 8, color=0, fill=k % 2))
        rows.append(f"i{k}.png,m{k}.png,{'noisy' if k % 2 else ''}")
    (root / "index.csv").write_text("\n".join(rows), encoding="utf-8")
    (root / "classes.json").write_text(json.dumps({"classes": ["bg", "disc"], "ignore_index": 255}), encoding="utf-8")
    seg = datasets.import_dataset(tmp_path / "ws", root, fmt="image-mask", name="seg")
    assert seg["task"] == "segmentation" and seg["ignore_index"] == 255 and seg["counts"]["exploration"] + seg["counts"]["holdout"] == 6

    boxes = tmp_path / "b"
    boxes.mkdir()
    images, annotations = [], []
    for k in range(4):
        (boxes / f"b{k}.bmp").write_bytes(bmp(10, 6))
        images.append({"id": k, "file_name": f"b{k}.bmp", "width": 10, "height": 6})
        annotations.append({"image_id": k, "bbox": [1, 1, 4, 3], "category_id": 7})
    (boxes / "annotations.json").write_text(json.dumps({"images": images, "annotations": annotations, "categories": [{"id": 7, "name": "disc"}]}), encoding="utf-8")
    det = datasets.import_dataset(tmp_path / "ws", boxes, fmt="image-boxes", name="det")
    assert det["task"] == "detection" and det["categories"] == [{"id": 7, "name": "disc"}]

    text = tmp_path / "t"
    text.mkdir()
    lines = ["image,text"]
    for k in range(4):
        (text / f"t{k}.png").write_bytes(png(20, 8))
        lines.append(f"t{k}.png,{k}{k + 1}")
    (text / "index.csv").write_text("\n".join(lines), encoding="utf-8")
    ocr = datasets.import_dataset(tmp_path / "ws", text, fmt="image-text", name="ocr")
    assert ocr["task"] == "ocr" and ocr["alphabet"] == "01234"


@pytest.mark.parametrize("break_it, code", [
    (lambda r: (r / "labels.csv").write_text("image,label\n../escape.png,cat\na.png,dog\nb.png,dog\nc.png,cat\n"), "invalid_index"),
    (lambda r: (r / "labels.csv").write_text("image,label\n/etc/passwd,cat\na.png,dog\nb.png,dog\nc.png,cat\n"), "invalid_index"),
    (lambda r: (r / "labels.csv").write_text("image,label\nnope.png,cat\na.png,dog\nb.png,dog\nc.png,cat\n"), "invalid_index"),
    (lambda r: (r / "labels.csv").write_text("image,label\na.png,cat\na.png,dog\nb.png,dog\nc.png,cat\n"), "invalid_index"),   # a repeated id
    (lambda r: (r / "labels.csv").write_text("file,label\na.png,cat\n"), "invalid_index"),
    (lambda r: (r / "labels.csv").write_text("image,label\na.png,cat\nb.png,dog\n"), "dataset_too_small"),
    (lambda r: (r / "a.png").write_bytes(b"\xff\xd8\xff" + b"\0" * 40), "unsupported_image"),
    (lambda r: (r / "labels.csv").unlink(), "invalid_index"),
])
def test_hostile_or_broken_sources_are_refused_before_anything_is_stored(tmp_path, break_it, code):
    root = tmp_path / "src"
    root.mkdir()
    for name in "abc":
        (root / f"{name}.png").write_bytes(png(4, 4))
    (root / "labels.csv").write_text("image,label\na.png,cat\nb.png,dog\nc.png,dog\nd.png,cat\n")
    (root / "d.png").write_bytes(png(4, 4))
    break_it(root)
    with pytest.raises(DatasetError) as raised:
        datasets.import_dataset(tmp_path / "ws", root, fmt="image-folder-labels", name="bad")
    assert raised.value.code == code, raised.value.message
    assert not list((tmp_path / "ws" / "datasets").glob("ds_*"))  # nothing registered


def test_limits_and_inputs_are_checked(tmp_path):
    root = write_folder(tmp_path / "src")
    for limits, code in ((Limits(max_pixels=10), "unsupported_image"), (Limits(max_file_bytes=20), "dataset_too_large"), (Limits(max_samples=3), "dataset_too_large"),
                         (Limits(max_total_bytes=100), "dataset_too_large")):
        with pytest.raises(DatasetError) as raised:
            datasets.build_manifest(root, "image-folder-labels", limits)
        assert raised.value.code == code, (limits, raised.value.message)
    for kwargs, code in (({"fmt": "tiff-folder", "name": "x1"}, "unknown_format"), ({"fmt": "image-folder-labels", "name": "Bad Name"}, "invalid_name"),
                         ({"fmt": "image-folder-labels", "name": "x1", "preprocessing": {"resize": {"size": [0, 4]}}}, "invalid_preprocessing")):
        with pytest.raises(DatasetError) as raised:
            datasets.import_dataset(tmp_path / "ws", root, **kwargs)
        assert raised.value.code == code
    with pytest.raises(DatasetError) as raised:
        datasets.import_dataset(tmp_path / "ws", tmp_path / "missing", fmt="image-folder-labels", name="x1")
    assert raised.value.code == "invalid_path"
    for ident in ("ds_x", "../x", "ds_" + "0" * 32):
        with pytest.raises(DatasetError):
            datasets.open_dataset(tmp_path / "ws", ident)


def test_the_boxes_must_fit_their_image_and_the_sizes_must_agree(tmp_path):
    root = tmp_path / "b"
    root.mkdir()
    images = []
    for k in range(4):
        (root / f"b{k}.png").write_bytes(png(10, 6))
        images.append({"id": k, "file_name": f"b{k}.png", "width": 10, "height": 6})
    annotation = {"images": images, "annotations": [{"image_id": 0, "bbox": [8, 1, 4, 3], "category_id": 1}], "categories": [{"id": 1, "name": "a"}]}
    (root / "annotations.json").write_text(json.dumps(annotation), encoding="utf-8")
    with pytest.raises(DatasetError, match="outside the image"):
        datasets.build_manifest(root, "image-boxes")
    annotation["annotations"][0]["bbox"] = [1, 1, 4, 3]
    annotation["images"][1]["width"] = 11
    (root / "annotations.json").write_text(json.dumps(annotation), encoding="utf-8")
    with pytest.raises(DatasetError, match="the index says"):
        datasets.build_manifest(root, "image-boxes")


def test_the_preprocessing_spec_is_closed_and_canonical():
    assert datasets.check_preprocessing(None) == {} == datasets.check_preprocessing({})
    spec = {"resize": {"size": [8, 8], "mode": "nearest"}, "pad_to": {"size": [10, 10], "anchor": "center"}, "grayscale": True, "channel_order": "bgr",
            "scale": 1 / 255, "mean": [0.5], "std": [0.25]}
    assert datasets.check_preprocessing(spec)["resize"] == {"size": [8, 8], "mode": "nearest", "antialias": False}
    assert datasets.check_preprocessing(spec)["pad_to"]["value"] == 0.0
    for bad in ({"crop": 1}, {"resize": {"size": [8]}}, {"resize": {"size": [8, 8]}, "resize_height": {"height": 8, "keep_aspect": True}},
                {"resize_height": {"height": 8, "keep_aspect": False}}, {"mean": [0.5]}, {"mean": [0.5], "std": [0]}, {"mean": [0.5, 0.5], "std": [1, 1]},
                {"scale": 0}, {"scale": True}, {"channel_order": "gbr"}, {"grayscale": False}, {"pad_to": {"size": [4, 4], "value": float("nan")}}):
        with pytest.raises(DatasetError) as raised:
            datasets.check_preprocessing(bad)
        assert raised.value.code == "invalid_preprocessing", bad


# ---------------------------------------------------------------- preprocessing (needs Pillow)

class TestPreprocess:
    @pytest.fixture(autouse=True)
    def need_pillow(self):
        pytest.importorskip("PIL")

    def gradient(self, width=8, height=4):
        from PIL import Image
        import io
        import numpy as np
        array = np.zeros((height, width, 3), np.uint8)
        array[:, :, 0] = np.arange(width) * 20
        array[:, :, 1] = np.arange(height)[:, None] * 30
        array[:, :, 2] = 7
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="PNG")
        return buffer.getvalue(), array

    def test_decode_is_exact_and_refuses_what_the_header_check_refuses(self):
        from layerdelta import preprocess
        import numpy as np
        data, array = self.gradient()
        assert np.array_equal(preprocess.decode(data), array)
        with pytest.raises(DatasetError):
            preprocess.decode(b"\xff\xd8\xff" + b"\0" * 40)
        assert preprocess.implementation_version().startswith("preprocess-v1/torch-")

    def test_geometry_and_labels_follow_each_other(self):
        from layerdelta import preprocess
        import numpy as np
        import torch
        data, array = self.gradient(8, 4)
        mask = np.tile(np.array([0, 1], np.uint8), (4, 4))
        spec = datasets.check_preprocessing({"resize": {"size": [8, 16], "mode": "nearest"}, "pad_to": {"size": [12, 20], "value": 5, "anchor": "center"}})
        out = preprocess.apply(spec, preprocess.decode(data), mask=mask, boxes=[{"bbox": [2, 1, 4, 2]}, {"bbox": [0, 0, 1, 1]}], ignore_index=255)
        assert out["image"].shape == (3, 12, 20) and out["mask"].shape == (12, 20)
        assert out["geometry"] == {"source": [4, 8], "scale": [2.0, 2.0], "offset": [2.0, 2.0], "output": [12, 20]}
        assert out["image"][0, 0, 0] == 5 and out["mask"][0, 0] == 255  # pad value for pixels, ignore index for the mask
        assert out["boxes"].tolist() == [[6.0, 4.0, 14.0, 8.0], [2.0, 2.0, 4.0, 4.0]]
        back = preprocess.to_source(out["boxes"], out["geometry"])
        assert back.tolist() == [[2.0, 1.0, 6.0, 3.0], [0.0, 0.0, 1.0, 1.0]]  # the inverse of the geometric steps
        assert torch.equal(preprocess.apply(spec, preprocess.decode(data))["image"], out["image"])  # deterministic

    def test_a_box_that_becomes_empty_is_dropped_and_resize_height_keeps_the_aspect(self):
        from layerdelta import preprocess
        data, _ = self.gradient(8, 4)
        spec = datasets.check_preprocessing({"resize_height": {"height": 8, "keep_aspect": True}, "pad_to": {"size": [8, 20]}})
        out = preprocess.apply(spec, preprocess.decode(data), boxes=[{"bbox": [7.9, 0, 0.1, 4]}])
        assert out["geometry"]["output"] == [8, 20] and out["geometry"]["scale"] == [2.0, 2.0]
        assert out["boxes"].shape == (1, 4)  # 0.2 px wide after scaling is still a box
        tiny = preprocess.apply(datasets.check_preprocessing({"resize": {"size": [1, 1], "mode": "nearest"}}), preprocess.decode(data),
                                boxes=[{"bbox": [0, 0, 0.001, 0.001]}])
        assert tiny["boxes"].shape == (0, 4) and tiny["box_index"].numel() == 0

    def test_colour_steps_are_applied_in_the_fixed_order(self):
        from layerdelta import preprocess
        import torch
        data, array = self.gradient(8, 4)
        spec = datasets.check_preprocessing({"channel_order": "bgr", "scale": 1 / 255, "mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]})
        out = preprocess.apply(spec, preprocess.decode(data))["image"]
        expected = (torch.from_numpy(array).permute(2, 0, 1).float().flip(0) / 255 - 0.5) / 0.25
        assert torch.allclose(out, expected, atol=1e-6)
        gray = preprocess.apply(datasets.check_preprocessing({"grayscale": True}), preprocess.decode(data))["image"]
        assert gray.shape == (1, 4, 8)
        with pytest.raises(DatasetError):
            preprocess.apply(datasets.check_preprocessing({"mean": [0, 0], "std": [1, 1]}), preprocess.decode(data))
        with pytest.raises(DatasetError):
            preprocess.apply(datasets.check_preprocessing({"pad_to": {"size": [2, 2]}}), preprocess.decode(data))


# ---------------------------------------------------------------- HTTP: a person imports, a token lists and reads

def test_import_is_human_only_and_a_token_sees_the_exploration_split(tmp_path, monkeypatch):
    import time
    from fastapi.testclient import TestClient
    from layerdelta import api
    from test_v2_api import BASE, make_workspace
    workspace = make_workspace(tmp_path, monkeypatch)
    source = write_folder(tmp_path / "src")
    creds = api.Credentials("demo")
    agent = {"Authorization": "Bearer " + creds.agent}
    with TestClient(api.create_app(workspace, creds), base_url="http://127.0.0.1") as client:
        human = {"X-CSRF-Token": client.post("/api/v1/session", json={"ticket": creds.ticket}).json()["csrf"]}
        body = {"path": str(source), "format": "image-folder-labels", "name": "pets-v1", "preprocessing": {"scale": 1 / 255}}
        assert client.post(BASE + "/dataset-imports", headers=agent, json=body).status_code == 403  # an agent never imports (nor names a path)
        assert client.get(BASE + "/dataset-imports/dsi_x", headers=agent).status_code == 403
        started = client.post(BASE + "/dataset-imports", headers=human, json=body)
        assert started.status_code == 202
        for _ in range(100):
            done = client.get(BASE + f"/dataset-imports/{started.json()['import_id']}", headers=human).json()
            if done["status"] in ("succeeded", "failed"):
                break
            time.sleep(.05)
        assert done["status"] == "succeeded", done
        ident = done["dataset"]["dataset_id"]
        failed = client.post(BASE + "/dataset-imports", headers=human, json={**body, "path": str(tmp_path / "nowhere")}).json()
        for _ in range(100):
            bad = client.get(BASE + f"/dataset-imports/{failed['import_id']}", headers=human).json()
            if bad["status"] in ("succeeded", "failed"):
                break
            time.sleep(.05)
        assert bad["status"] == "failed" and bad["error"]["code"] == "invalid_path"
        listed = client.get(BASE + "/datasets", headers=agent).json()["datasets"]
        assert [d["name"] for d in listed] == ["mnist-v1", "pets-v1"]
        mine = next(d for d in listed if d["name"] == "pets-v1")
        assert mine["counts"] == {"exploration": 4} and "holdout" not in json.dumps(listed)
        assert client.get(BASE + f"/datasets/{ident}", headers=agent).json() == mine
        person = client.get(BASE + f"/datasets/{ident}", headers=human).json()
        assert person["counts"] == {"exploration": 4, "holdout": 4} and "holdout" in person["split_sha256"]
        assert client.get(BASE + "/datasets/mnist-v1", headers=agent).json()["classes"] == [str(d) for d in range(10)]
        assert client.get(BASE + "/datasets/ds_" + "0" * 32, headers=agent).status_code == 404
        assert client.get(BASE + "/datasets/..%2Fx", headers=agent).status_code == 404
