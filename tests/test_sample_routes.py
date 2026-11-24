"""Human-only sample routes (v0.5-E4b): the image and the label of one sample, for the person supervising (E9). Never for a token."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from layerdelta import api, datasets
from test_datasets import bmp, png, write_folder
from test_v2_api import BASE, make_workspace


@pytest.fixture
def world(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path, monkeypatch)
    creds = api.Credentials("demo", weights_read=True)
    with TestClient(api.create_app(workspace, creds), base_url="http://127.0.0.1") as client:
        human = {"X-CSRF-Token": client.post("/api/v1/session", json={"ticket": creds.ticket}).json()["csrf"]}
        agent = {"Authorization": "Bearer " + creds.agent}

        def imported(source, fmt, name):
            record = datasets.import_dataset(workspace, source, fmt=fmt, name=name)
            return record["dataset_id"], datasets.open_dataset(workspace, record["dataset_id"])
        yield client, human, agent, imported, workspace


def test_a_person_gets_the_stored_image_and_the_label_a_token_gets_403(world, tmp_path):
    client, human, agent, imported, workspace = world
    ident, data = imported(write_folder(tmp_path / "src"), "image-folder-labels", "pets")
    row = next(r for r in data.manifest["samples"] if r["split"] == "holdout")  # a person may see holdout; the gate is on agents
    image = client.get(BASE + f"/datasets/{ident}/samples/{row['sample_id']}/image", headers=human)
    assert image.status_code == 200 and image.headers["content-type"] == "image/png" and image.headers["x-content-type-options"] == "nosniff"
    assert image.content == data.path(row["image"]).read_bytes() and 'filename="sample.png"' in image.headers["content-disposition"]
    label = client.get(BASE + f"/datasets/{ident}/samples/{row['sample_id']}/label", headers=human).json()
    assert label == {"label": row["label"], "label_index": data.record["classes"].index(row["label"])}
    for suffix in ("image", "label"):
        refused = client.get(BASE + f"/datasets/{ident}/samples/{row['sample_id']}/{suffix}", headers=agent)
        assert refused.status_code == 403 and refused.json()["error"]["code"] == "forbidden"
    for path in (f"/datasets/{ident}/samples/nope/image", f"/datasets/ds_{'0' * 32}/samples/x/image", "/datasets/mnist-v1/samples/x/label",
                 f"/datasets/{ident}/samples/..%2F..%2Fmanifest/image"):
        assert client.get(BASE + path, headers=human).status_code == 404, path


def test_a_file_that_changed_on_disk_is_never_served(world, tmp_path):
    client, human, agent, imported, workspace = world
    ident, data = imported(write_folder(tmp_path / "src"), "image-folder-labels", "pets")
    row = data.manifest["samples"][0]
    data.path(row["image"]).write_bytes(b"not the image")
    answer = client.get(BASE + f"/datasets/{ident}/samples/{row['sample_id']}/image", headers=human)
    assert answer.status_code == 409 and answer.json()["error"]["code"] == "dataset_changed"


def test_masks_boxes_and_text_labels_come_in_their_own_types(world, tmp_path):
    client, human, agent, imported, workspace = world
    root = tmp_path / "m"
    root.mkdir()
    rows = ["image,mask"]
    for k in range(4):
        (root / f"i{k}.png").write_bytes(png(8, 8, color=2))
        (root / f"m{k}.png").write_bytes(png(8, 8, fill=k % 2))
        rows.append(f"i{k}.png,m{k}.png")
    (root / "index.csv").write_text("\n".join(rows))
    (root / "classes.json").write_text(json.dumps({"classes": ["bg", "disc"], "ignore_index": 255}))
    ident, data = imported(root, "image-mask", "seg")
    sample = data.manifest["samples"][0]
    mask = client.get(BASE + f"/datasets/{ident}/samples/{sample['sample_id']}/label", headers=human)
    assert mask.headers["content-type"] == "image/png" and mask.content == data.path(sample["mask"]).read_bytes()

    boxes = tmp_path / "b"
    boxes.mkdir()
    images = []
    for k in range(4):
        (boxes / f"b{k}.bmp").write_bytes(bmp(10, 6))
        images.append({"id": k, "file_name": f"b{k}.bmp", "width": 10, "height": 6})
    (boxes / "annotations.json").write_text(json.dumps({"images": images, "annotations": [{"image_id": 0, "bbox": [1, 1, 4, 3], "category_id": 7}],
                                                        "categories": [{"id": 7, "name": "disc"}]}))
    ident, data = imported(boxes, "image-boxes", "det")
    first = next(r for r in data.manifest["samples"] if r["boxes"])
    shown = client.get(BASE + f"/datasets/{ident}/samples/{first['sample_id']}/image", headers=human)
    assert shown.headers["content-type"] == "image/bmp"  # from the manifest, not sniffed
    assert client.get(BASE + f"/datasets/{ident}/samples/{first['sample_id']}/label", headers=human).json()["boxes"] == first["boxes"]

    text = tmp_path / "t"
    text.mkdir()
    lines = ["image,text"]
    for k in range(4):
        (text / f"t{k}.png").write_bytes(png(20, 8))
        lines.append(f"t{k}.png,<b>{k}</b>")
    (text / "index.csv").write_text("\n".join(lines))
    ident, data = imported(text, "image-text", "ocr")
    line = client.get(BASE + f"/datasets/{ident}/samples/{data.manifest['samples'][0]['sample_id']}/label", headers=human)
    assert line.headers["content-type"].startswith("text/plain") and line.text.startswith("<b>")  # served as text: never as markup
