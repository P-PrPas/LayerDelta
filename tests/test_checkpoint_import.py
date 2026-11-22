"""Checkpoint catalog and import: path, streamed upload, sharded sets, refusals, restart, legacy route."""

import hashlib
import json
import os
import pickle
import socket
import struct
import threading
import time

import httpx
import psutil
import pytest
import torch
import uvicorn
from fastapi.testclient import TestClient
from safetensors.torch import save, save_file

from layerdelta import api
from layerdelta.checkpoints import Catalog, ImportFailure, ImportLimits, import_checkpoint
from layerdelta.job import checkpoint_path
from layerdelta.model import ReferenceCNN, describe_model

BASE = "/api/v1/projects/demo"


def unet_like(seed=0):
    """A small non-MNIST random-init encoder/decoder with skip-style names."""
    g = torch.Generator().manual_seed(seed)
    r = lambda *shape: torch.randn(*shape, generator=g)  # noqa: E731
    return {"enc1.conv.weight": r(8, 3, 3, 3), "enc1.conv.bias": r(8), "enc2.conv.weight": r(16, 8, 3, 3),
            "enc2.conv.bias": r(16), "up1.deconv.weight": r(16, 8, 2, 2), "dec1.conv.weight": r(8, 16, 3, 3),
            "head.weight": r(2, 8, 1, 1), "head.bias": r(2), "steps": torch.tensor([3])}


def write(tmp_path, tensors, name="model.safetensors"):
    path = tmp_path / name
    save_file(tensors, path)
    return path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def env(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    (workspace / "reference").mkdir(parents=True)
    (workspace / "reference" / "model.safetensors").touch()
    fixture = {
        "artifacts": {"model.safetensors": "a" * 64, "dataset.json": "b" * 64, "exploration.json": "c" * 64},
        "manifest_sha256": "d" * 64, "training": {"seed": 1}, "model": describe_model(ReferenceCNN()),
    }
    manifest = {"dataset_revision_id": "mnist-v1", "split_sha256": {"holdout": "e" * 64}, "samples": [
        {"sample_id": "exploration/normal", "source_sample_id": "exploration", "split": "exploration",
         "condition": "normal", "label": 2}]}
    monkeypatch.setattr(api, "load_fixture", lambda _: (fixture, manifest, {}))
    creds = api.Credentials("demo")

    def build(limits=None):
        app = api.create_app(workspace, creds, quota=2, import_limits=limits)
        client = TestClient(app, base_url="http://127.0.0.1")
        client.__enter__()
        csrf = client.post("/api/v1/session", json={"ticket": creds.ticket}).json()["csrf"]
        return app, client, {"X-CSRF-Token": csrf}

    clients = []

    def make(limits=None):
        built = build(limits)
        clients.append(built[1])
        return (*built, {"Authorization": "Bearer " + creds.agent}, workspace, creds)

    yield make
    for client in clients:
        client.__exit__(None, None, None)


def wait_import(client, human, import_id, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = client.get(f"{BASE}/checkpoint-imports/{import_id}", headers=human).json()
        if record["status"] in ("succeeded", "failed"):
            return record
        time.sleep(.02)
    pytest.fail("import did not finish")


def start_import(client, human, path, **extra):
    response = client.post(BASE + "/checkpoint-imports", headers=human, json={"path": str(path), **extra})
    assert response.status_code == 202, response.text
    return wait_import(client, human, response.json()["import_id"])


# ------------------------------------------------------------------ library: single file

def test_import_non_mnist_file_is_content_addressed_and_idempotent(tmp_path):
    workspace = tmp_path / "ws"
    catalog = Catalog(workspace, describe_model(ReferenceCNN())["state_schema"])
    catalog.start()
    source = write(tmp_path, unet_like())
    stages = []
    entry, existing = import_checkpoint(catalog, source, source.name, progress=lambda *a, **k: stages.append(a[0]))
    assert existing is False and entry["checkpoint_id"] == sha(source)
    assert (entry["origin"], entry["display_name"], entry["parent_id"], entry["sharded"]) == ("imported", "model", None, False)
    assert entry["tensor_count"] == 9 and entry["element_count"] == sum(t.numel() for t in unet_like().values())
    assert entry["file_bytes"] == source.stat().st_size and entry["adapter_compatible"] is False
    assert [s for i, s in enumerate(stages) if s not in stages[:i]] == ["validating", "copying", "verifying", "publishing"]
    stored = workspace / "checkpoints" / (sha(source) + ".safetensors")
    assert stored.read_bytes() == source.read_bytes()
    source.unlink()  # the workspace keeps its own copy
    assert stored.is_file()
    again_src = write(tmp_path, unet_like(), "renamed.safetensors")
    again, existing = import_checkpoint(catalog, again_src, again_src.name, display_name="ignored")
    assert existing is True and again == entry and len(catalog.list()) == 1
    assert list((workspace / "staging").iterdir()) == []
    other, _ = import_checkpoint(catalog, write(tmp_path, unet_like(1), "o.safetensors"), "o.safetensors", display_name="  Second ")
    assert other["checkpoint_id"] != entry["checkpoint_id"] and other["display_name"] == "Second"


def test_mnist_schema_checkpoint_is_adapter_compatible(tmp_path):
    catalog = Catalog(tmp_path / "ws", describe_model(ReferenceCNN())["state_schema"])
    catalog.start()
    source = write(tmp_path, ReferenceCNN().state_dict())
    entry, _ = import_checkpoint(catalog, source, source.name)
    assert entry["adapter_compatible"] is True and entry["tensor_count"] == 6


@pytest.mark.parametrize("case,code", [
    ("truncated", "data_size"), ("header", "header_json"), ("pt_name", "unsupported_format"),
    ("bin_name", "unsupported_format"), ("pickle", "unsupported_format"), ("zip", "unsupported_format"),
    ("text", "unsupported_format"), ("missing", "source_not_found"), ("dtype", "dtype_unknown"),
])
def test_refused_with_stable_codes(tmp_path, case, code):
    catalog = Catalog(tmp_path / "ws")
    catalog.start()
    good = write(tmp_path, {"w": torch.ones(4)}).read_bytes()
    name, data = "m.safetensors", good
    if case == "truncated":
        data = good[:-4]
    elif case == "header":
        data = struct.pack("<Q", 5) + b"{nope" + bytes(8)
    elif case == "pt_name":
        name = "model.pt"
    elif case == "bin_name":
        name = "pytorch_model.bin"
    elif case == "pickle":
        data = pickle.dumps({"w": [1.0] * 40}, protocol=4)
    elif case == "zip":
        data = b"PK\x03\x04" + bytes(100)
    elif case == "text":
        name = "weights.h5"
    elif case == "dtype":
        header = json.dumps({"w": {"dtype": "F4", "shape": [1], "data_offsets": [0, 1]}}).encode()
        data = struct.pack("<Q", len(header)) + header + b"\0"
    path = tmp_path / name
    if case != "missing":
        path.write_bytes(data)
    with pytest.raises(ImportFailure) as info:
        import_checkpoint(catalog, path, name)
    assert info.value.code == code
    assert catalog.list() == [] and list((tmp_path / "ws" / "staging").iterdir()) == []
    assert ("convert" in info.value.message.lower()) == (code == "unsupported_format" and case in ("pt_name", "bin_name", "pickle", "zip"))


def test_caps_are_configurable(tmp_path):
    catalog = Catalog(tmp_path / "ws")
    catalog.start()
    source = write(tmp_path, {"w": torch.ones(1000)})
    with pytest.raises(ImportFailure) as info:
        import_checkpoint(catalog, source, source.name, limits=ImportLimits(max_file_bytes=1000))
    assert info.value.code == "file_size"
    with pytest.raises(ImportFailure) as info:
        import_checkpoint(catalog, source, source.name, limits=ImportLimits(max_header_bytes=4))
    assert info.value.code == "header_size"
    with pytest.raises(ImportFailure) as info:
        import_checkpoint(catalog, source, source.name, limits=ImportLimits(max_total_bytes=100))
    assert info.value.code == "total_size"
    assert import_checkpoint(catalog, source, source.name, limits=ImportLimits(max_file_bytes=1 << 20))[1] is False


def test_source_changed_during_copy_is_detected(tmp_path, monkeypatch):
    catalog = Catalog(tmp_path / "ws")
    catalog.start()
    source = write(tmp_path, {"w": torch.ones(64)})
    other = write(tmp_path, {"w": torch.ones(64), "extra": torch.ones(2)}, "other.safetensors")

    from layerdelta import checkpoints
    real = checkpoints.open_tensors
    calls = []

    def swap(path, **kw):
        calls.append(path)
        if len(calls) == 2:  # the verification of the copy sees different content
            return real(other, **kw)
        return real(path, **kw)

    monkeypatch.setattr(checkpoints, "open_tensors", swap)
    with pytest.raises(ImportFailure) as info:
        import_checkpoint(catalog, source, source.name)
    assert info.value.code == "file_changed" and catalog.list() == []


# ------------------------------------------------------------------ library: sharded

def make_shards(tmp_path, directory="hf"):
    folder = tmp_path / directory
    folder.mkdir()
    tensors = unet_like()
    names = sorted(tensors)
    split = {"model-00001-of-00002.safetensors": names[:4], "model-00002-of-00002.safetensors": names[4:]}
    for shard, members in split.items():
        save_file({n: tensors[n] for n in members}, folder / shard)
    index = folder / "model.safetensors.index.json"
    index.write_text(json.dumps({"metadata": {}, "weight_map": {n: s for s, ms in split.items() for n in ms}}))
    return folder, index, tensors


def test_sharded_import_copies_shards_and_hashes_the_file_set(tmp_path):
    catalog = Catalog(tmp_path / "ws")
    catalog.start()
    folder, index, tensors = make_shards(tmp_path)
    entry, existing = import_checkpoint(catalog, index, index.name)
    assert existing is False and entry["sharded"] and entry["shard_count"] == 2 and entry["tensor_count"] == 9
    assert entry["display_name"] == "model"
    listing = json.dumps(sorted([f.name, sha(f)] for f in folder.iterdir()), separators=(",", ":")).encode()
    expected = hashlib.sha256(b"layerdelta-fileset-v1\n" + listing).hexdigest()  # domain-separated from single-file ids
    assert entry["checkpoint_id"] == expected != hashlib.sha256(listing).hexdigest()
    stored = tmp_path / "ws" / "checkpoints" / expected
    assert sorted(p.name for p in stored.iterdir()) == sorted(p.name for p in folder.iterdir())
    for p in folder.iterdir():
        p.unlink()
    assert (stored / "model.safetensors.index.json").is_file()  # originals are no longer needed
    folder2, index2, _ = make_shards(tmp_path, "hf2")
    again, existing = import_checkpoint(catalog, index2, index2.name)
    assert existing is True and again["checkpoint_id"] == expected
    with pytest.raises(ValueError, match="Sharded"):
        checkpoint_path(tmp_path / "ws", expected, {"artifacts": {"model.safetensors": "a" * 64}})


def test_sharded_refusals(tmp_path):
    catalog = Catalog(tmp_path / "ws")
    catalog.start()
    folder, index, _ = make_shards(tmp_path)
    (folder / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(ImportFailure) as info:
        import_checkpoint(catalog, index, index.name)
    assert info.value.code == "shard_missing"
    folder, index, _ = make_shards(tmp_path, "evil")
    index.write_text(json.dumps({"weight_map": {"enc1.conv.bias": "../model.safetensors"}}))
    with pytest.raises(ImportFailure) as info:
        import_checkpoint(catalog, index, index.name)
    assert info.value.code == "index_invalid"
    folder, index, _ = make_shards(tmp_path, "links")
    shard = folder / "model-00001-of-00002.safetensors"
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(shard.read_bytes())  # identical content, so only the location is wrong
    shard.unlink()
    try:
        shard.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks need privileges on this machine")
    with pytest.raises(ImportFailure) as info:
        import_checkpoint(catalog, index, index.name)
    assert info.value.code == "shard_outside_directory"


# ------------------------------------------------------------------ catalog: restart, legacy, fixture

def test_catalog_survives_restart_and_adopts_legacy_files(tmp_path):
    schema = describe_model(ReferenceCNN())["state_schema"]
    fixture = {"artifacts": {"model.safetensors": "f" * 64}}
    workspace = tmp_path / "ws"
    (workspace / "reference").mkdir(parents=True)
    write(workspace / "reference", ReferenceCNN().state_dict())
    catalog = Catalog(workspace, schema)
    catalog.start(fixture)
    source = write(tmp_path, unet_like())
    entry, _ = import_checkpoint(catalog, source, source.name)
    # a checkpoint stored by the pre-catalog upload route, and an import that was running at "crash" time
    legacy = write(tmp_path, ReferenceCNN().state_dict(), "legacy.safetensors")
    (workspace / "checkpoints" / (sha(legacy) + ".safetensors")).write_bytes(legacy.read_bytes())
    catalog.create_import("human", "path", "m.safetensors", None)
    (workspace / "staging" / "import-stale").mkdir()
    restarted = Catalog(workspace, schema)
    restarted.start(fixture)
    listed = restarted.list()
    assert [c["origin"] for c in listed] == ["fixture", "imported", "imported"]
    assert listed[0]["checkpoint_id"] == "f" * 64 and listed[0]["adapter_compatible"] and listed[0]["tensor_count"] == 6
    assert restarted.get(entry["checkpoint_id"]) == entry
    adopted = restarted.get(sha(legacy))
    assert adopted["adapter_compatible"] is True and adopted["display_name"] == sha(legacy)[:12]
    assert restarted.list(compatible_only=True)[0]["origin"] == "fixture" and len(restarted.list(compatible_only=True)) == 2
    with restarted._db() as db:
        row = db.execute("SELECT status, error FROM imports").fetchone()
    assert row["status"] == "failed" and "interrupted" in row["error"]
    assert not (workspace / "staging" / "import-stale").exists()
    restarted.start(fixture)
    assert len(restarted.list()) == 3  # idempotent


def test_checkpoint_path_resolves_through_the_catalog(tmp_path):
    workspace = tmp_path / "ws"
    catalog = Catalog(workspace)
    catalog.start()
    source = write(tmp_path, unet_like())
    entry, _ = import_checkpoint(catalog, source, source.name)
    fixture = {"artifacts": {"model.safetensors": "a" * 64}}
    assert checkpoint_path(workspace, entry["checkpoint_id"], fixture) == workspace / "checkpoints" / (entry["checkpoint_id"] + ".safetensors")
    assert checkpoint_path(workspace, "a" * 64, fixture) == workspace / "reference" / "model.safetensors"
    unknown = "9" * 64
    assert checkpoint_path(workspace, unknown, fixture) == workspace / "checkpoints" / (unknown + ".safetensors")
    with pytest.raises(ValueError):
        checkpoint_path(workspace, "not-a-digest", fixture)
    assert checkpoint_path(tmp_path / "empty", unknown, fixture).name == unknown + ".safetensors"  # no catalog yet


# ------------------------------------------------------------------ HTTP

def test_http_import_by_path_catalog_and_permissions(tmp_path, env):
    app, client, human, agent, workspace, _ = env()
    source = write(tmp_path, unet_like())
    assert client.post(BASE + "/checkpoint-imports", headers=agent, json={"path": str(source)}).status_code == 403
    assert client.post(BASE + "/checkpoint-imports/upload?filename=m.safetensors", headers=agent, content=b"x").status_code == 403
    assert client.get(BASE + "/checkpoint-imports/imp_x", headers=agent).status_code == 403
    first = client.post(BASE + "/checkpoint-imports", headers=human, json={"path": str(source), "display_name": "U-Net toy"})
    assert first.status_code == 202 and first.json()["status"] in ("queued", "running", "succeeded")
    record = wait_import(client, human, first.json()["import_id"])
    assert record["status"] == "succeeded" and record["error"] is None and record["existing"] is False
    assert (record["stage"], record["stage_index"], record["stage_count"], record["unit"]) == ("publishing", 4, 4, "files")
    assert record["tensor_count"] == 9 and record["source"] == "path" and record["source_name"] == "model.safetensors"
    cid = record["checkpoint_id"]
    assert cid == sha(source)
    # catalog visible to human and agent; no filesystem paths leak
    for who in (human, agent):
        listing = client.get(BASE + "/checkpoints", headers=who).json()
        assert [c["origin"] for c in listing["items"]] == ["fixture", "imported"] and listing["next_cursor"] is None
        assert str(tmp_path) not in json.dumps(listing)
        entry = client.get(f"{BASE}/checkpoints/{cid}", headers=who).json()
        assert entry["display_name"] == "U-Net toy" and entry["tensor_count"] == 9 and entry["adapter_compatible"] is False
    assert client.get(BASE + "/checkpoints/" + "0" * 64, headers=human).status_code == 404
    assert client.get(BASE + "/checkpoint-imports/imp_missing", headers=human).status_code == 404
    paged = client.get(BASE + "/checkpoints?limit=1", headers=human).json()
    assert len(paged["items"]) == 1 and paged["next_cursor"]
    # a non-MNIST checkpoint is not offered for runs and cannot be used in a recipe
    usable = client.get(BASE + "/catalog", headers=agent).json()["checkpoints"]
    assert [c["checkpoint_id"] for c in usable] == ["a" * 64]
    edit = {"hypothesis": "h", "checkpoint_id": cid, "intervention": {"target_id": "features.block1.conv.kernel",
            "mode": "weight", "channel_index": 0, "operation": "zero"}}
    rejected = client.post(BASE + "/recipes", headers=agent, json=edit)
    assert rejected.status_code == 409 and rejected.json()["error"]["code"] == "dependency_mismatch"
    # identical bytes again: same id, flagged as existing
    again = start_import(client, human, source)
    assert again["checkpoint_id"] == cid and again["existing"] is True
    # validation errors at submit time
    assert client.post(BASE + "/checkpoint-imports", headers=human, json={"path": "relative/m.safetensors"}).status_code == 422
    bad = client.post(BASE + "/checkpoint-imports", headers=human, json={"path": str(source), "display_name": "x\ny"})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_display_name"
    assert client.post(BASE + "/checkpoint-imports", headers=human, json={"path": str(source), "extra": 1}).status_code == 422


def test_http_import_failures_report_stable_codes(tmp_path, env):
    app, client, human, agent, workspace, _ = env()
    truncated = tmp_path / "t.safetensors"
    truncated.write_bytes(write(tmp_path, {"w": torch.ones(8)}).read_bytes()[:-2])
    record = start_import(client, human, truncated)
    assert record["status"] == "failed" and record["error"]["code"] == "data_size" and record["checkpoint_id"] is None
    assert record["error"]["message"] and record["finished_at"]
    for name, code in (("model.pt", "unsupported_format"), ("model.bin", "unsupported_format")):
        path = tmp_path / name
        path.write_bytes(pickle.dumps({"a": 1}))
        assert start_import(client, human, path)["error"]["code"] == code
    assert start_import(client, human, tmp_path / "gone.safetensors")["error"]["code"] == "source_not_found"
    assert client.get(BASE + "/checkpoints", headers=human).json()["items"][-1]["origin"] == "fixture"


def test_http_upload_streams_to_staging_and_is_idempotent(tmp_path, env):
    app, client, human, agent, workspace, _ = env()
    source = write(tmp_path, unet_like())
    data = source.read_bytes()

    def chunks():
        for i in range(0, len(data), 1000):
            yield data[i:i + 1000]

    response = client.post(BASE + "/checkpoint-imports/upload?filename=up.safetensors&display_name=Uploaded",
                           headers={**human, "Content-Length": str(len(data)), "Content-Type": "application/octet-stream"},
                           content=chunks())
    assert response.status_code == 202, response.text
    record = wait_import(client, human, response.json()["import_id"])
    assert record["status"] == "succeeded" and record["source"] == "upload" and record["checkpoint_id"] == sha(source)
    assert client.get(f"{BASE}/checkpoints/{record['checkpoint_id']}", headers=agent).json()["display_name"] == "Uploaded"
    assert list((workspace / "staging").iterdir()) == []
    path_record = start_import(client, human, source)
    assert path_record["checkpoint_id"] == record["checkpoint_id"] and path_record["existing"] is True
    again = client.post(BASE + "/checkpoint-imports/upload?filename=up2.safetensors", content=data, headers=human)
    assert wait_import(client, human, again.json()["import_id"])["existing"] is True


def test_http_upload_refusals(tmp_path, env):
    limits = ImportLimits(max_file_bytes=2000)
    app, client, human, agent, workspace, _ = env(limits)
    data = write(tmp_path, {"w": torch.ones(8)}).read_bytes()
    post = lambda query, content, **h: client.post(f"{BASE}/checkpoint-imports/upload?{query}", content=content,  # noqa: E731
                                                  headers={**human, **h})
    too_big = post("filename=m.safetensors", bytes(5000))
    assert too_big.status_code == 413 and too_big.json()["error"]["code"] == "payload_too_large"
    assert post("filename=m.pt", data).json()["error"]["code"] == "unsupported_format"
    assert post("filename=m.txt", data).json()["error"]["code"] == "unsupported_format"
    assert post("filename=..%5Cm.safetensors", data).json()["error"]["code"] == "invalid_filename"
    assert post("filename=m.safetensors", b"").status_code in (411, 422)
    chunked = post("filename=m.safetensors", iter([data]))  # no Content-Length: refused before reading
    assert chunked.status_code == 413
    corrupt = post("filename=c.safetensors", data[:-3])
    assert corrupt.status_code == 202 and wait_import(client, human, corrupt.json()["import_id"])["error"]["code"] == "data_size"
    pickled = post("filename=p.safetensors", pickle.dumps({"w": [1.0] * 30}))
    assert wait_import(client, human, pickled.json()["import_id"])["error"]["code"] == "unsupported_format"
    assert list((workspace / "staging").iterdir()) == []
    assert [c["origin"] for c in client.get(BASE + "/checkpoints", headers=human).json()["items"]] == ["fixture"]


def test_http_sharded_import_by_path(tmp_path, env):
    app, client, human, agent, workspace, _ = env()
    folder, index, _ = make_shards(tmp_path)
    record = start_import(client, human, index)
    assert record["status"] == "succeeded" and record["total"] > 0
    entry = client.get(f"{BASE}/checkpoints/{record['checkpoint_id']}", headers=agent).json()
    assert entry["sharded"] and entry["shard_count"] == 2 and entry["adapter_compatible"] is False
    (folder / "model-00001-of-00002.safetensors").write_bytes(b"garbage")
    assert start_import(client, human, index)["error"]["code"] in ("file_size", "header_size", "header_json")


def test_legacy_upload_route_still_serves_mnist_checkpoints(tmp_path, env):
    app, client, human, agent, workspace, _ = env()
    data = save(ReferenceCNN().state_dict())
    response = client.post(BASE + "/checkpoints", headers=human, files={"file": ("my_cnn.safetensors", data)})
    assert response.status_code == 201
    digest = response.json()["checkpoint_id"]
    assert digest == response.json()["sha256"] == hashlib.sha256(data).hexdigest()
    entry = client.get(f"{BASE}/checkpoints/{digest}", headers=human).json()
    assert entry["display_name"] == "my_cnn" and entry["adapter_compatible"] is True and entry["origin"] == "imported"
    assert [c["checkpoint_id"] for c in client.get(BASE + "/catalog", headers=agent).json()["checkpoints"]] == ["a" * 64, digest]
    recipe = client.post(BASE + "/recipes", headers=agent, json={"hypothesis": "h", "checkpoint_id": digest, "intervention": {
        "target_id": "features.block1.conv.kernel", "mode": "weight", "channel_index": 0, "operation": "zero"}})
    assert recipe.status_code == 201
    bad = client.post(BASE + "/checkpoints", headers=human, files={"file": ("u.safetensors", save(unet_like()))})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_checkpoint"
    assert client.post(BASE + "/checkpoints", headers=agent, files={"file": ("x", data)}).status_code == 403


# ------------------------------------------------------------------ slow: large file, bounded memory

def serve(app):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started and time.monotonic() < deadline:
        time.sleep(.02)
    assert server.started
    return server, thread, f"http://127.0.0.1:{port}"


@pytest.mark.slow
def test_large_import_bounded_memory(tmp_path, monkeypatch):
    big, small = 256 << 20, 1024
    header = json.dumps({"big": {"dtype": "F32", "shape": [big], "data_offsets": [0, big * 4]},
                         "small": {"dtype": "F32", "shape": [small], "data_offsets": [big * 4, big * 4 + small * 4]}}).encode()
    path = tmp_path / "huge.safetensors"
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(header)) + header)
        stream.truncate(8 + len(header) + big * 4)
        stream.seek(0, os.SEEK_END)
        stream.write(torch.arange(small, dtype=torch.float32).numpy().tobytes())
    size = path.stat().st_size
    workspace = tmp_path / "ws"
    (workspace / "reference").mkdir(parents=True)
    (workspace / "reference" / "model.safetensors").touch()
    fixture = {"artifacts": {"model.safetensors": "a" * 64, "dataset.json": "b" * 64, "exploration.json": "c" * 64},
               "manifest_sha256": "d" * 64, "training": {"seed": 1}, "model": describe_model(ReferenceCNN())}
    monkeypatch.setattr(api, "load_fixture", lambda _: (fixture, {"dataset_revision_id": "v", "split_sha256": {}, "samples": []}, {}))
    creds = api.Credentials("demo")
    server, thread, url = serve(api.create_app(workspace, creds, quota=1, import_limits=ImportLimits(max_file_bytes=4 << 30)))
    process, peak, stop = psutil.Process(), [0], threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], process.memory_info().rss)
            time.sleep(.005)

    sampler = threading.Thread(target=sample)
    try:
        with httpx.Client(base_url=url, timeout=600) as client:
            csrf = client.post("/api/v1/session", json={"ticket": creds.ticket}).json()["csrf"]
            human = {"X-CSRF-Token": csrf}
            before = process.memory_info().rss
            sampler.start()

            def poll(import_id):
                while True:
                    record = client.get(f"{BASE}/checkpoint-imports/{import_id}", headers=human).json()
                    if record["status"] in ("succeeded", "failed"):
                        return record
                    time.sleep(.1)

            by_path = poll(client.post(BASE + "/checkpoint-imports", headers=human, json={"path": str(path)}).json()["import_id"])
            assert by_path["status"] == "succeeded", by_path

            def body():
                with path.open("rb") as stream:
                    while chunk := stream.read(1 << 20):
                        yield chunk

            up = client.post(BASE + "/checkpoint-imports/upload?filename=huge.safetensors",
                             headers={**human, "Content-Length": str(size)}, content=body())
            assert up.status_code == 202, up.text
            by_upload = poll(up.json()["import_id"])
            assert by_upload["status"] == "succeeded" and by_upload["existing"] is True
            assert by_upload["checkpoint_id"] == by_path["checkpoint_id"]
    finally:
        stop.set()
        if sampler.is_alive():
            sampler.join()
        server.should_exit = True
        thread.join(30)
    assert peak[0] - before < 200 << 20
    assert (workspace / "checkpoints" / (by_path["checkpoint_id"] + ".safetensors")).stat().st_size == size


def test_adoption_verifies_the_content_hash(tmp_path):
    workspace = tmp_path / "ws"
    catalog = Catalog(workspace)
    catalog.start()
    good = write(tmp_path, {"w": torch.ones(4)}, "good.safetensors")
    liar = write(tmp_path, {"w": torch.zeros(4)}, "liar.safetensors")
    (workspace / "checkpoints" / (sha(good) + ".safetensors")).write_bytes(good.read_bytes())
    (workspace / "checkpoints" / (sha(good).replace(sha(good)[0], "0" if sha(good)[0] != "0" else "1", 1) + ".safetensors")).write_bytes(liar.read_bytes())
    restarted = Catalog(workspace)
    restarted.start()
    assert [c["checkpoint_id"] for c in restarted.list()] == [sha(good)]
    assert len(list((workspace / "checkpoints").glob("*.safetensors"))) == 1
    quarantined = list((workspace / "quarantine").iterdir())
    assert len(quarantined) == 1 and sha(liar) in quarantined[0].name


def test_file_replaced_between_validation_and_copy_is_refused(tmp_path):
    catalog = Catalog(tmp_path / "ws")
    catalog.start()
    folder, index, _ = make_shards(tmp_path)
    shard = folder / "model-00001-of-00002.safetensors"
    swap = tmp_path / "swap.safetensors"
    swap.write_bytes(shard.read_bytes())  # same content, new file: only the identity differs

    def progress(stage, done, total, unit, **kw):
        if stage == "copying" and done == 0 and swap.exists():
            os.replace(swap, shard)

    with pytest.raises(ImportFailure) as info:
        import_checkpoint(catalog, index, index.name, progress=progress)
    assert info.value.code == "file_changed" and catalog.list() == []


def test_upload_disk_writes_run_in_the_threadpool(tmp_path, env, monkeypatch):
    app, client, human, agent, workspace, _ = env()
    calls = []
    real = api.run_in_threadpool

    async def spy(func, *args):
        calls.append(func.__name__)
        return await real(func, *args)

    monkeypatch.setattr(api, "run_in_threadpool", spy)
    data = write(tmp_path, {"w": torch.ones(64)}).read_bytes()
    response = client.post(BASE + "/checkpoint-imports/upload?filename=m.safetensors", headers=human, content=data)
    assert response.status_code == 202
    assert "sink" in calls and calls[-1] == "finish"
