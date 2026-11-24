"""Synthetic datasets (v0.5-E2s): seeded, byte-reproducible, importable through the real (human-only) import path, and what they claim.

The generators live in tools/bench (benchmark tooling, not the product). The checks decode through preprocess.decode, the worker's own
decoder, so a file that imports and decodes here is a file an evaluation can read.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from layerdelta import datasets

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bench"))
import synthetic  # noqa: E402

FORMAT = {"shapes-seg": "image-mask", "boxes-det": "image-boxes", "glyph-ocr": "image-text"}


def files(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.parametrize("name", sorted(synthetic.GENERATORS))
def test_same_seed_same_bytes_and_another_seed_differs(tmp_path, name):
    a, b, c = (synthetic.generate(name, tmp_path / d, n=12, seed=s) for d, s in (("a", 0), ("b", 0), ("c", 1)))
    assert a == b and a["version"] == synthetic.VERSION and a["format"] == FORMAT[name]
    assert files(tmp_path / "a") == files(tmp_path / "b")
    assert files(tmp_path / "a") != files(tmp_path / "c")


def test_the_png_encoder_round_trips_through_the_worker_decoder():
    pytest.importorskip("PIL")
    from layerdelta import preprocess
    rng = np.random.default_rng(0)
    gray, rgb = rng.integers(0, 256, size=(7, 5), dtype=np.uint8), rng.integers(0, 256, size=(4, 9, 3), dtype=np.uint8)
    assert np.array_equal(preprocess.decode(synthetic.png(gray))[:, :, 0], gray)
    assert np.array_equal(preprocess.decode(synthetic.png(rgb)), rgb)
    assert np.array_equal(preprocess.decode(synthetic.png(gray), mask=True), gray)
    assert datasets.image_header(synthetic.png(rgb)) == (9, 4)


@pytest.mark.parametrize("name", sorted(synthetic.GENERATORS))
def test_every_dataset_imports_with_both_conditions_on_both_splits(tmp_path, name):
    source = tmp_path / "source"
    info = synthetic.generate(name, source, n=40, seed=3)
    record = datasets.import_dataset(tmp_path / "ws", source.resolve(), fmt=info["format"], name=name)
    assert record["task"] == info["task"]
    data = datasets.open_dataset(tmp_path / "ws", record["dataset_id"])
    for split in ("exploration", "holdout"):
        assert {r["condition"] for r in data.samples(split)} == ({"clean", "blur"} if name == "glyph-ocr" else {"clean", "noisy"})
    assert len(data.samples("exploration")) + len(data.samples("holdout")) == 40


def _imported(tmp_path, name, n=30, seed=5):
    source = tmp_path / "source"
    info = synthetic.generate(name, source, n=n, seed=seed)
    record = datasets.import_dataset(tmp_path / "ws", source.resolve(), fmt=info["format"], name=name)
    return datasets.open_dataset(tmp_path / "ws", record["dataset_id"])


def test_shapes_seg_masks_hold_the_four_classes_and_match_the_image_size(tmp_path):
    pytest.importorskip("PIL")
    from layerdelta import preprocess
    data = _imported(tmp_path, "shapes-seg")
    assert data.record["classes"] == ["background", "disc", "square", "triangle"] and data.record["ignore_index"] == 255
    seen = set()
    for row in data.manifest["samples"]:
        mask = preprocess.decode(data.path(row["mask"]).read_bytes(), mask=True)
        image = preprocess.decode(data.path(row["image"]).read_bytes())
        assert mask.shape == image.shape[:2] == (32, 32)
        seen |= set(np.unique(mask).tolist())
        assert (mask > 0).any()  # every image has at least one shape
    assert seen == {0, 1, 2, 3}


def test_shape_masks_are_the_shapes_they_name():
    # a disc of radius 3 centred on a pixel corner (5, 5): centre offsets (±.5, ±1.5, ±2.5)^2 within 9 -> 8 per quadrant -> 32 pixels
    disc = synthetic.shape_mask("disc", 5, 5, 3, 10, 10)
    assert disc.sum() == 32 and disc[5, 5] and not disc[0, 0]
    square = synthetic.shape_mask("square", 5, 5, 2, 10, 10)  # centres 3.5..6.5 -> 4 x 4
    assert square.sum() == 16
    triangle = synthetic.shape_mask("triangle", 5, 5, 3, 10, 10)
    rows = triangle.sum(axis=1)
    assert rows[rows > 0].tolist() == sorted(rows[rows > 0].tolist())  # widens from the apex down


def test_boxes_det_boxes_are_inside_and_tight(tmp_path):
    data = _imported(tmp_path, "boxes-det", n=60)
    assert [c["name"] for c in data.record["categories"]] == list(synthetic.SHAPES)
    counts = [len(r["boxes"]) for r in data.manifest["samples"]]
    assert 0 in counts and max(counts) >= 3  # empty images exist: the F1 = 1.0 case is exercised
    for row in data.manifest["samples"]:
        boxes = [b["bbox"] for b in row["boxes"]]
        for x, y, w, h in boxes:
            assert 0 <= x and 0 <= y and x + w <= 48 and y + h <= 48 and w >= 2 and h >= 2
    # tightness: the box rule on a known square (centres 7.5..12.5 by 9.5..14.5 within half-side 3 of (10, 12)) -> x 7..12, y 9..14
    mask = synthetic.shape_mask("square", 10.0, 12.0, 3.0, 48, 48)
    ys, xs = np.nonzero(mask)
    assert [xs.min(), ys.min(), xs.max() - xs.min() + 1, ys.max() - ys.min() + 1] == [7, 9, 6, 6]


def test_glyph_ocr_lines_are_legible_one_ink_run_per_character(tmp_path):
    pytest.importorskip("PIL")
    from layerdelta import preprocess
    data = _imported(tmp_path, "glyph-ocr", n=40)
    assert set(data.record["alphabet"]) <= set("0123456789")
    checked = 0
    for row in data.manifest["samples"]:
        image = preprocess.decode(data.path(row["image"]).read_bytes())[:, :, 0]
        assert image.shape[0] == 20 and 3 <= len(row["text"]) <= 6
        if row["condition"] != "clean":
            continue
        ink = (image < 128).any(axis=0)
        runs = int(ink[0]) + int(np.count_nonzero(ink[1:] & ~ink[:-1]))
        assert runs == len(row["text"]), row  # every glyph's ink columns are contiguous and glyphs are `scale` pixels apart
        checked += 1
    assert checked >= 10


def test_every_glyph_is_distinct_and_has_contiguous_ink_columns():
    shapes = {k: g.tobytes() for k, g in synthetic.GLYPH.items()}
    assert len(set(shapes.values())) == 10
    for k, g in synthetic.GLYPH.items():
        cols = np.nonzero(g.any(axis=0))[0]
        assert g.shape == (7, 5) and cols.tolist() == list(range(cols.min(), cols.max() + 1)), k


def test_the_cli_writes_the_info_file(tmp_path, capsys):
    synthetic.main(["glyph-ocr", "--out", str(tmp_path / "g"), "--n", "6", "--seed", "2"])
    printed = json.loads(capsys.readouterr().out)
    assert printed == json.loads((tmp_path / "g" / "synthetic.json").read_text(encoding="utf-8"))
    assert printed["n"] == 6 and printed["seed"] == 2
