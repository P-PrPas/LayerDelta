import gzip
import hashlib
import io
import json
import struct

import pytest
import torch

from layerdelta import data


@pytest.fixture
def archive_dir(tmp_path, monkeypatch):
    sources = {}
    for partition, labels in (("train", [2, 5]), ("t10k", [2, 2, 5, 5])):
        images = bytearray(len(labels) * 784)
        for i in range(len(labels)):
            images[i * 784 + 14 * 28 + 14] = 255
        entries = {
            f"{partition}-images-idx3-ubyte.gz": struct.pack(">IIII", 2051, len(labels), 28, 28) + images,
            f"{partition}-labels-idx1-ubyte.gz": struct.pack(">II", 2049, len(labels)) + bytes(labels),
        }
        for filename, payload in entries.items():
            archive = gzip.compress(payload, mtime=0)
            (tmp_path / filename).write_bytes(archive)
            sources[filename] = hashlib.sha256(archive).hexdigest()
    monkeypatch.setattr(data, "SOURCE_SHA256", sources)
    return tmp_path


def test_replayable_splits_and_paired_blur(archive_dir):
    manifest = data.build_manifest(archive_dir)
    assert json.loads(json.dumps(manifest)) == data.build_manifest(archive_dir)
    train_x, _, train_rows = data.load_split(archive_dir, manifest, "train")
    exploration_x, y, rows = data.load_split(archive_dir, manifest, "exploration")
    _, _, holdout = data.load_split(archive_dir, manifest, "holdout")
    assert train_x.shape == (2, 1, 28, 28) and exploration_x.shape == (4, 1, 28, 28)
    assert y.tolist() == [row["label"] for row in rows]
    split_sources = [{r["source_sample_id"] for r in subset} for subset in (train_rows, rows, holdout)]
    assert sum(map(len, split_sources)) == len(set.union(*split_sources)) == 6
    assert len({row["sample_id"] for row in manifest["samples"]}) == 10
    for i in (0, 2):
        assert rows[i]["source_sample_id"] == rows[i + 1]["source_sample_id"]
        assert (rows[i]["condition"], rows[i + 1]["condition"]) == ("normal", "blurred")
        assert exploration_x[i, 0, 14, 14].item() == 1
        expected = torch.zeros(28, 28)
        expected[12:17, 12:17] = 1 / 25
        torch.testing.assert_close(exploration_x[i + 1, 0], expected)


def test_reject_changed_row_and_source_bytes(archive_dir):
    manifest = data.build_manifest(archive_dir)
    manifest["samples"][0]["split"] = "exploration"
    with pytest.raises(ValueError, match="manifest mismatch"):
        data.load_split(archive_dir, manifest, "exploration")
    (archive_dir / "train-images-idx3-ubyte.gz").write_bytes(b"broken")
    with pytest.raises(ValueError, match="digest mismatch"):
        data.build_manifest(archive_dir)


def test_failed_download_cannot_poison_cache(archive_dir, monkeypatch):
    archive = archive_dir / "t10k-labels-idx1-ubyte.gz"
    valid = archive.read_bytes()
    archive.unlink()
    monkeypatch.setattr(data, "urlopen", lambda *a, **kw: io.BytesIO(b"corrupt"))
    with pytest.raises(ValueError, match="digest mismatch"):
        data.download_mnist(archive_dir)
    assert not archive.exists()
    assert {p.name for p in archive_dir.iterdir()} == set(data.SOURCE_SHA256) - {archive.name}
    monkeypatch.setattr(data, "urlopen", lambda *a, **kw: io.BytesIO(valid))
    assert data.download_mnist(archive_dir) == data.SOURCE_SHA256
    assert archive.read_bytes() == valid
