"""v2 HTTP routes: envelope contract, capabilities, signed cursors, idempotent charged apply, blobs, client."""

import hmac
import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch
import uvicorn
from fastapi.testclient import TestClient
from safetensors.torch import save_file

from layerdelta import api, trace
from layerdelta.agent_demo import experiment
from layerdelta.checkpoints import Catalog
from layerdelta.client import AgentClient, ClientError
from layerdelta.model import ReferenceCNN, describe_model

BASE = "/api/v1/projects/demo"
REF = "a" * 64
ENVELOPE_KEYS = {"schema_version", "resolved", "result", "fingerprints", "side_effects", "truncated", "cursor"}
BIAS = {"tensor": "classifier.bias"}
PROGRAM = {"program": [{"op": "scale", "args": {"k": 2.0}, "select": BIAS}], "hypothesis": "double the bias"}


def make_workspace(tmp_path, monkeypatch):
    torch.manual_seed(2)
    workspace = tmp_path / "ws"
    (workspace / "reference").mkdir(parents=True)
    save_file(ReferenceCNN().state_dict(), workspace / "reference" / "model.safetensors")
    fixture = {"artifacts": {"model.safetensors": REF, "dataset.json": "b" * 64, "exploration.json": "c" * 64},
               "manifest_sha256": "d" * 64, "training": {"seed": 1}, "model": describe_model(ReferenceCNN())}
    manifest = {"dataset_revision_id": "mnist-v1", "split_sha256": {"holdout": "e" * 64}, "samples": []}
    monkeypatch.setattr(api, "load_fixture", lambda _: (fixture, manifest, {}))
    return workspace


@pytest.fixture
def env(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path, monkeypatch)
    opened = []

    def make(read=True, write=True, quota=3, blob_max=1 << 20):
        creds = api.Credentials("demo", weights_read=read, weights_write=write)
        app = api.create_app(workspace, creds, quota=quota, blob_max_bytes=blob_max)
        client = TestClient(app, base_url="http://127.0.0.1")
        client.__enter__()
        opened.append(client)
        csrf = client.post("/api/v1/session", json={"ticket": creds.ticket}).json()["csrf"]
        return client, {"Authorization": "Bearer " + creds.agent}, {"X-CSRF-Token": csrf}, creds, workspace

    yield make
    for client in opened:
        client.__exit__(None, None, None)


def post(client, headers, path, body=None, **kw):
    return client.post(BASE + path, headers=headers, json={} if body is None else body, **kw)


# ------------------------------------------------------------------ envelope, selectors, cursors

def test_every_v2_route_returns_the_envelope(env):
    client, agent, human, creds, _ = env()
    c = f"/checkpoints/{REF}"
    responses = {
        "describe": client.get(BASE + c + "/describe", headers=agent),
        "read-tensor": post(client, agent, c + "/read-tensor", {"selector": BIAS}),
        "stats": post(client, agent, c + "/stats", {"selector": BIAS}),
        "diff": post(client, agent, c + "/diff", {"other": REF}),
        "query": post(client, agent, c + "/query", {}),
        "plan": post(client, agent, c + "/plan", PROGRAM),
        "validate": post(client, agent, c + "/validate", PROGRAM),
        "apply": post(client, {**agent, "Idempotency-Key": "k1"}, c + "/apply", PROGRAM),
        "blobs": client.post(BASE + "/blobs", headers=agent, content=bytes(8)),
    }
    for name, response in responses.items():
        assert response.status_code in (200, 201), (name, response.text)
        body = response.json()
        assert set(body) == ENVELOPE_KEYS, name
        assert body["schema_version"] == 1 and isinstance(body["resolved"], dict) and isinstance(body["result"], dict)
        assert body["truncated"] is False and body["cursor"] is None
        assert {"checkpoint", "selection", "op_revision"} <= set(body["fingerprints"]) or "checkpoint_a" in body["fingerprints"]
        if name in ("describe", "read-tensor", "stats", "query", "plan", "validate", "diff"):
            assert body["side_effects"] == [], name  # pure tools always []
    assert responses["describe"].json()["fingerprints"]["checkpoint"] == REF
    assert responses["diff"].json()["fingerprints"] == {"checkpoint_a": REF, "checkpoint_b": REF, "selection": None, "op_revision": 1}
    kinds = [e["kind"] for e in responses["apply"].json()["side_effects"]]
    assert kinds == ["checkpoint_created", "quota_charged"]
    assert responses["blobs"].json()["side_effects"][0]["kind"] == "blob_stored"
    assert responses["plan"].json()["result"]["recipe"]["schema_version"] == 2


def test_resolved_selectors_round_trip_over_http(env):
    client, agent, human, creds, _ = env()
    c = f"/checkpoints/{REF}"
    sel = {"tensor": "features.block2.conv.weight", "index": {"0": [17, 3, 4], "2": {"start": 0, "stop": 2}}}
    first = post(client, agent, c + "/read-tensor", {"selector": sel, "max_elems": 50}).json()
    again = post(client, agent, c + "/read-tensor", {"selector": first["resolved"]["selector"], "max_elems": 50}).json()
    assert again["fingerprints"]["selection"] == first["fingerprints"]["selection"]
    assert again["result"]["values"] == first["result"]["values"] and again["resolved"] == first["resolved"]
    stats = post(client, agent, c + "/stats", {"selector": first["resolved"]["selector"], "reduce_over_axes": [1, 2, 3]}).json()
    assert stats["resolved"]["selector"] == first["resolved"]["selector"]
    assert [u["index"] for u in stats["result"]["units"]] == [{"0": 3}, {"0": 4}, {"0": 17}]


def test_cursors_are_signed(env):
    client, agent, human, creds, _ = env()
    c = f"/checkpoints/{REF}"
    page = client.get(BASE + c + "/describe", headers=agent, params={"limit": 2}).json()
    assert page["truncated"] is True and page["cursor"].startswith("c2.") and len(page["result"]["tensors"]) == 2
    second = client.get(BASE + c + "/describe", headers=agent, params={"limit": 2, "cursor": page["cursor"]}).json()
    assert second["result"]["offset"] == 2 and second["result"]["tensors"][0]["name"] != page["result"]["tensors"][0]["name"]
    forged = page["cursor"][:-1] + ("0" if page["cursor"][-1] != "0" else "1")
    for bad in (forged, "c1." + page["cursor"].split(".")[1], "c2.abc.def", "junk", page["cursor"].rsplit(".", 1)[0]):
        response = client.get(BASE + c + "/describe", headers=agent, params={"limit": 2, "cursor": bad})
        assert response.status_code == 422 and response.json()["error"]["code"] == "invalid_cursor"
        assert response.json()["error"]["field"] == "cursor"
    # a signed cursor still belongs to its request
    other = post(client, agent, c + "/query", {"limit": 1}).json()
    assert other["cursor"].startswith("c2.")
    reuse = post(client, agent, c + "/query", {"limit": 1, "name_glob": "*.bias", "cursor": other["cursor"]})
    assert reuse.status_code == 422 and reuse.json()["error"]["code"] == "invalid_cursor"


def test_errors_are_structured(env):
    client, agent, human, creds, _ = env()
    c = f"/checkpoints/{REF}"
    unknown = client.get(BASE + f"/checkpoints/{'9' * 64}/describe", headers=agent)
    assert unknown.status_code == 404 and unknown.json()["error"]["code"] == "not_found"
    bad = post(client, agent, c + "/read-tensor", {"selector": {"tensor": "classifier.bias", "index": {"0": [10]}}})
    error = bad.json()["error"]
    assert bad.status_code == 422 and error["code"] == "index_out_of_range" and error["field"] == "selector.index.0[0]"
    assert post(client, agent, c + "/read-tensor", {"selector": BIAS, "extra": 1}).status_code == 422
    assert post(client, agent, c + "/diff", {"other": "9" * 64}).status_code == 404
    both = post(client, agent, c + "/plan", {**PROGRAM, "recipe": {}})
    assert both.status_code == 422 and both.json()["error"]["code"] == "invalid_argument"
    invalid = post(client, agent, c + "/validate", {"program": [{"op": "scale", "args": {"k": 1e39}, "select": BIAS}], "hypothesis": "h"})
    assert invalid.status_code == 200 and invalid.json()["result"]["valid"] is False
    assert invalid.json()["result"]["error"]["code"] == "invalid_argument"


# ------------------------------------------------------------------ capabilities and the session record

def test_capabilities_are_enforced_and_recorded_per_session(env):
    client, agent, human, creds, workspace = env(read=True, write=False)
    c = f"/checkpoints/{REF}"
    assert post(client, {**agent, "Idempotency-Key": "k"}, c + "/apply", PROGRAM).status_code == 403
    assert client.post(BASE + "/blobs", headers=agent, content=bytes(4)).status_code == 403
    assert post(client, agent, c + "/plan", PROGRAM).status_code == 200
    assert client.get(BASE + "/agent-session", headers=agent).json()["capabilities"] == ["model:read", "weights:read", "weights:values"]
    catalog = Catalog(workspace)
    assert catalog.get_session(creds.principal) == {**catalog.get_session(creds.principal), "kind": "agent",
                                                    "capabilities": ["model:read", "weights:read", "weights:values"]}
    assert catalog.get_session("human")["capabilities"] == ["eval:run", "model:read", "weights:read", "weights:values", "weights:write"]
    # the human session needs no flag
    done = post(client, {**human, "Idempotency-Key": "h1"}, c + "/apply", PROGRAM)
    assert done.status_code == 200 and [e["kind"] for e in done.json()["side_effects"]] == ["checkpoint_created"]
    assert client.get(BASE + "/agent-session", headers=agent).json()["charged"] == 0  # humans are never charged


def test_model_read_only_token_sees_structure_but_no_values(env):
    client, agent, human, creds, _ = env(read=False, write=False)
    c = f"/checkpoints/{REF}"
    assert client.get(BASE + c + "/describe", headers=agent).status_code == 200
    for path, body in (("/read-tensor", {"selector": BIAS}), ("/stats", {"selector": BIAS}), ("/query", {}), ("/plan", PROGRAM),
                       ("/validate", PROGRAM), ("/diff", {"other": REF})):
        response = post(client, agent, c + path, body)
        assert response.status_code == 403 and response.json()["error"]["code"] == "forbidden", path


# ------------------------------------------------------------------ apply: child, idempotency, quota

def test_apply_registers_child_and_is_idempotent_and_charged_once(env):
    client, agent, human, creds, workspace = env(quota=3)
    c = f"/checkpoints/{REF}"
    headers = {**agent, "Idempotency-Key": "once"}
    first = post(client, headers, c + "/apply", PROGRAM)
    assert first.status_code == 200, first.text
    body = first.json()
    child = body["result"]["checkpoint"]
    assert (child["origin"], child["parent_id"], child["recipe_hash"]) == ("child", REF, body["result"]["recipe"]["recipe_hash"])
    assert child["checkpoint_id"] == body["result"]["manifest"]["child_sha256"] and body["result"]["replayed"] is False
    assert (workspace / "checkpoints" / (child["checkpoint_id"] + ".safetensors")).is_file()
    listed = [e["origin"] for e in client.get(BASE + "/checkpoints", headers=agent).json()["items"]]
    assert listed == ["fixture", "child"]
    assert client.get(BASE + "/agent-session", headers=agent).json()["charged"] == 1
    # the child is inspectable like any checkpoint, and the diff shows exactly the edited tensor
    diff = post(client, agent, c + "/diff", {"other": child["checkpoint_id"]}).json()["result"]
    assert diff["summary"]["changed"] == 1 and diff["tensors"][0]["name"] == "classifier.bias"
    # replay: same key, same body -> same child, no new charge, no new side effects
    again = post(client, headers, c + "/apply", PROGRAM).json()
    assert again["result"]["checkpoint"] == child and again["result"]["replayed"] is True and again["side_effects"] == []
    assert client.get(BASE + "/agent-session", headers=agent).json()["charged"] == 1
    # same key, different request
    conflict = post(client, headers, c + "/apply", {**PROGRAM, "hypothesis": "something else"})
    assert conflict.status_code == 409 and conflict.json()["error"]["code"] == "idempotency_conflict"
    # invalid and no-op programs cost nothing and release the key
    bad = post(client, {**agent, "Idempotency-Key": "bad"}, c + "/apply",
               {"program": [{"op": "scale", "args": {"k": 1e39}, "select": BIAS}], "hypothesis": "h"})
    noop = post(client, {**agent, "Idempotency-Key": "noop"}, c + "/apply",
                {"program": [{"op": "scale", "args": {"k": 1.0}, "select": BIAS}], "hypothesis": "h"})
    assert bad.status_code == 422 and noop.status_code == 422 and noop.json()["error"]["code"] == "no_op"
    assert client.get(BASE + "/agent-session", headers=agent).json()["charged"] == 1
    # recipe form: the plan's frozen recipe applies to the same child
    recipe = post(client, agent, c + "/plan", PROGRAM).json()["result"]["recipe"]
    by_recipe = post(client, {**agent, "Idempotency-Key": "recipe"}, c + "/apply", {"recipe": recipe}).json()
    assert by_recipe["result"]["checkpoint"]["checkpoint_id"] == child["checkpoint_id"]
    # quota: 3 charges in total, then 429, and the refusal is not charged either
    ok = post(client, {**agent, "Idempotency-Key": "third"}, c + "/apply",
              {"program": [{"op": "scale", "args": {"k": 3.0}, "select": BIAS}], "hypothesis": "h3"})
    assert ok.status_code == 200
    out = post(client, {**agent, "Idempotency-Key": "fourth"}, c + "/apply",
               {"program": [{"op": "scale", "args": {"k": 4.0}, "select": BIAS}], "hypothesis": "h4"})
    assert out.status_code == 429 and out.json()["error"]["code"] == "quota_exhausted"
    info = client.get(BASE + "/agent-session", headers=agent).json()
    assert (info["charged"], info["remaining"]) == (3, 0)
    # the edit row snapshots the capabilities the caller held
    with Catalog(workspace)._db() as db:
        row = db.execute("SELECT capabilities, charged, status FROM edits WHERE key=?", (trace.key_digest("once"),)).fetchone()
    assert json.loads(row["capabilities"]) == ["model:read", "weights:read", "weights:values", "weights:write"] and row["charged"] == 1


def test_concurrent_apply_with_one_key_makes_one_child_and_one_charge(env):
    client, agent, human, creds, workspace = env(quota=2)
    c = f"/checkpoints/{REF}"
    program = {"program": [{"op": "scale", "args": {"k": 2.5}, "select": BIAS}], "hypothesis": "race"}
    with ThreadPoolExecutor(6) as pool:
        results = list(pool.map(lambda _: post(client, {**agent, "Idempotency-Key": "race"}, c + "/apply", program), range(6)))
    assert [r.status_code for r in results] == [200] * 6, [r.text for r in results if r.status_code != 200]
    children = {r.json()["result"]["checkpoint"]["checkpoint_id"] for r in results}
    assert len(children) == 1
    assert sorted(r.json()["result"]["replayed"] for r in results) == [False] + [True] * 5
    assert sum(1 for r in results if r.json()["side_effects"]) == 1  # exactly one request did the work
    assert client.get(BASE + "/agent-session", headers=agent).json()["charged"] == 1
    assert len(list((workspace / "checkpoints").glob("*.safetensors"))) == 1
    assert list((workspace / "staging").iterdir()) == []


def test_failed_apply_releases_the_charge_and_key(env, monkeypatch):
    client, agent, human, creds, workspace = env(quota=1)
    from layerdelta import edit, v2
    real = edit.apply

    def boom(*args, **kw):
        raise RuntimeError("disk exploded")

    monkeypatch.setattr(v2.edit, "apply", boom)
    c = f"/checkpoints/{REF}"
    with pytest.raises(RuntimeError):
        post(client, {**agent, "Idempotency-Key": "k"}, c + "/apply", PROGRAM)
    assert client.get(BASE + "/agent-session", headers=agent).json()["charged"] == 0
    monkeypatch.setattr(v2.edit, "apply", real)
    assert post(client, {**agent, "Idempotency-Key": "k"}, c + "/apply", PROGRAM).status_code == 200


def test_apply_requires_an_idempotency_key(env):
    client, agent, human, creds, _ = env()
    assert post(client, agent, f"/checkpoints/{REF}/apply", PROGRAM).status_code == 422


# ------------------------------------------------------------------ blobs

def test_blob_upload_and_use_in_a_program(env):
    client, agent, human, creds, workspace = env(blob_max=1024)
    delta = np.linspace(0.5, 1.5, 10, dtype="<f4")
    uploaded = client.post(BASE + "/blobs", headers=agent, content=delta.tobytes())
    assert uploaded.status_code == 201
    blob = uploaded.json()["result"]
    assert blob["elems"] == 10 and blob["existing"] is False and blob["blob_id"].startswith("sha256:")
    assert (workspace / "blobs" / blob["blob_id"][7:]).read_bytes() == delta.tobytes()
    again = client.post(BASE + "/blobs", headers=agent, content=delta.tobytes()).json()
    assert again["result"]["existing"] is True and again["side_effects"] == []
    program = {"program": [{"op": "add", "args": {"blob": blob["blob_id"]}, "select": BIAS}], "hypothesis": "add a blob"}
    c = f"/checkpoints/{REF}"
    done = post(client, {**agent, "Idempotency-Key": "blob"}, c + "/apply", program).json()
    child = done["result"]["checkpoint"]["checkpoint_id"]
    parent_bias = np.array(post(client, agent, c + "/read-tensor", {"selector": BIAS}).json()["result"]["values"], dtype=np.float32)
    child_bias = np.array(post(client, agent, f"/checkpoints/{child}/read-tensor", {"selector": BIAS}).json()["result"]["values"], dtype=np.float32)
    assert np.array_equal(child_bias, parent_bias + delta)
    missing = {"program": [{"op": "add", "args": {"blob": "sha256:" + "0" * 64}, "select": BIAS}], "hypothesis": "h"}
    gone = post(client, agent, c + "/plan", missing)
    assert gone.status_code == 422 and gone.json()["error"]["code"] == "blob_missing"
    assert client.post(BASE + "/blobs", headers=agent, content=bytes(2048)).status_code == 413
    odd = client.post(BASE + "/blobs", headers=agent, content=bytes(7))
    assert odd.status_code == 422 and odd.json()["error"]["code"] == "blob_size"
    chunked = client.post(BASE + "/blobs", headers=agent, content=iter([bytes(8)]))
    assert chunked.status_code in (411, 413)
    assert list((workspace / "staging").iterdir()) == []


# ------------------------------------------------------------------ startup sweep, demo guard, client

def test_start_sweeps_partial_files_and_stale_edits(tmp_path):
    workspace = tmp_path / "ws"
    catalog = Catalog(workspace)
    catalog.start()
    (workspace / "staging" / ".apply-1.safetensors.partial-2").write_bytes(b"x")
    (workspace / "staging" / "apply-3.safetensors").write_bytes(b"x")
    (workspace / "staging" / "blob-4.part").write_bytes(b"x")
    state, _ = catalog.begin_edit("a", "k", "d", REF, "h" * 64, ["model:read"], charge=True, quota=5)
    assert state == "new"
    Catalog(workspace).start()
    assert list((workspace / "staging").iterdir()) == []
    assert catalog.get_edit("a", "k") is None  # the interrupted apply released its charge


def test_agent_demo_explains_a_missing_weights_read_capability():
    class Stub:
        def quota(self):
            return {"remaining": 1}

        def catalog(self):
            return {"demo": {"example_pairs": [{"source_sample_id": "s"}], "truth": 1},
                    "targets": [{"mode": "activation", "target_id": "t", "channel_count": 4}]}

        def samples(self, source_sample_id=None):
            return iter([{"sample_id": "s/normal", "condition": "normal"}, {"sample_id": "s/blurred", "condition": "blurred"}])

        def preview(self, request):
            return {"run_id": "r"}

        def preview_status(self, run_id):
            return {"status": "succeeded", "run_id": run_id, "result": {"target_id": "t", "channel_index": 3}}  # no activation

    with pytest.raises(ClientError) as info:
        experiment(Stub())
    assert info.value.code == "forbidden" and "weights:read" in str(info.value)


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


def test_agent_client_methods_map_one_to_one_onto_the_routes(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path, monkeypatch)
    creds = api.Credentials("demo", weights_read=True, weights_write=True)
    server, thread, url = serve(api.create_app(workspace, creds, quota=3))
    try:
        with AgentClient(url, "demo", creds.agent) as client:
            assert [c["origin"] for c in client.list_checkpoints()] == ["fixture"]
            assert client.get_checkpoint(REF)["checkpoint_id"] == REF
            names, cursor = [], None
            while True:
                page = client.describe(REF, limit=2, cursor=cursor)
                names += [t["name"] for t in page["result"]["tensors"]]
                cursor = page["cursor"]
                if not cursor:
                    break
            assert len(names) == 6 and names == sorted(names)
            read = client.read_tensor(REF, {"tensor": "features.block1.conv.weight", "index": {"0": [0]}}, max_elems=5)
            assert read["truncated"] is True and len(read["result"]["values"]) == 5
            more = client.read_tensor(REF, read["resolved"]["selector"], max_elems=5, cursor=read["cursor"])
            assert more["result"]["offset"] == 5 and "range" not in more["result"]
            assert client.stats(REF, BIAS, bins=4)["result"]["histogram"]["edges"]
            assert client.query(REF, name_glob="*.bias")["result"]["matched"] == 3
            card = client.graph(REF)  # nothing attached: the attach card
            assert card["result"]["graph"] is None and card["result"]["providers"][0]["name"] == "builtin"
            for call in (lambda: client.groups(REF), lambda: client.graph_node(REF, "n0"), lambda: client.graph(REF, view="modules")):
                with pytest.raises(ClientError) as info:
                    call()
                assert (info.value.code, info.value.status) == ("no_graph", 422)
            blob = client.upload_blob(np.ones(10, dtype="<f4").tobytes())["result"]["blob_id"]
            program = [{"op": "add", "args": {"blob": blob}, "select": BIAS}]
            assert client.validate(REF, program=program, hypothesis="h")["result"]["valid"] is True
            plan = client.plan(REF, program=program, hypothesis="h")
            assert plan["result"]["changed_elems_total"] == 10
            first = client.apply(REF, key="kk", recipe=plan["result"]["recipe"])
            second = client.apply(REF, key="kk", recipe=plan["result"]["recipe"])
            child = first["result"]["checkpoint"]["checkpoint_id"]
            assert second["result"]["checkpoint"]["checkpoint_id"] == child and second["result"]["replayed"] is True
            d = client.diff(child, REF, unit_axis=None)
            assert d["result"]["summary"]["changed"] == 1 and "changed_units" not in d["result"]["tensors"][0]
            assert client.diff(child, REF)["result"]["tensors"][0]["changed_units"]["total"] == 10
            assert client.quota()["charged"] == 1
            with pytest.raises(ClientError) as info:
                client.read_tensor(REF, {"tensor": "nope"})
            assert (info.value.code, info.value.status, info.value.field) == ("unknown_tensor", 422, "selector.tensor")
            with pytest.raises(ValueError):
                client.apply(REF, key="", program=program, hypothesis="h")
    finally:
        server.should_exit = True
        thread.join(30)


# ------------------------------------------------------------------ PR #22 review fixes

def apply_as(client, headers, key, body, c=f"/checkpoints/{REF}"):
    return post(client, {**headers, "Idempotency-Key": key}, c + "/apply", body).json()


def test_existing_children_are_not_recreated_and_effects_are_honest(env):
    client, agent, human, creds, workspace = env(quota=5)
    checkpoints = workspace / "checkpoints"
    # (b) the same recipe under two keys: one child, the second reports it as existing, both charged honestly
    first = apply_as(client, agent, "k1", PROGRAM)
    second = apply_as(client, agent, "k2", PROGRAM)
    child = first["result"]["checkpoint"]["checkpoint_id"]
    assert [e["kind"] for e in first["side_effects"]] == ["checkpoint_created", "quota_charged"] and first["result"]["existing"] is False
    assert [e["kind"] for e in second["side_effects"]] == ["checkpoint_existing", "quota_charged"]
    assert second["result"]["existing"] is True and second["result"]["checkpoint"] == first["result"]["checkpoint"]
    # (a) allow_noop on that child: the grandchild IS the child; no new file, no created effect, not its lineage
    noop = {"program": [{"op": "scale", "args": {"k": 1.0}, "select": BIAS}], "hypothesis": "noop", "allow_noop": True}
    r = apply_as(client, agent, "n1", noop, c=f"/checkpoints/{child}")
    assert r["result"]["existing"] is True and r["result"]["checkpoint"] == {"checkpoint_id": child}
    assert [e["kind"] for e in r["side_effects"]] == ["checkpoint_existing", "quota_charged"] and r["side_effects"][0]["id"] == child
    # (c) a different recipe that produces the same bytes: existing, and it does not claim the other lineage
    one = {"program": [{"op": "set", "args": {"value": 2.0}, "select": BIAS}], "hypothesis": "set"}
    two = {"program": [{"op": "set", "args": {"value": 1.0}, "select": BIAS}, {"op": "add", "args": {"value": 1.0}, "select": BIAS}],
           "hypothesis": "set then add"}
    a, b = apply_as(client, agent, "ka", one), apply_as(client, agent, "kb", two)
    same = a["result"]["checkpoint"]["checkpoint_id"]
    assert b["result"]["existing"] is True and b["result"]["checkpoint"] == {"checkpoint_id": same}
    assert client.get(BASE + f"/checkpoints/{same}", headers=agent).json()["recipe_hash"] == a["result"]["recipe"]["recipe_hash"]
    assert sorted(p.stem for p in checkpoints.glob("*.safetensors")) == sorted({child, same})  # no orphan copies
    assert [c["origin"] for c in client.get(BASE + "/checkpoints", headers=agent).json()["items"]] == ["fixture", "child", "child"]
    assert client.get(BASE + "/agent-session", headers=agent).json()["charged"] == 5


def test_two_keys_racing_on_one_recipe_create_the_child_once(env):
    client, agent, human, creds, workspace = env(quota=2)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda k: apply_as(client, agent, k, PROGRAM), ["r1", "r2"]))
    assert sorted(r["side_effects"][0]["kind"] for r in results) == ["checkpoint_created", "checkpoint_existing"]
    assert len({r["result"]["manifest"]["child_sha256"] for r in results}) == 1
    assert len(list((workspace / "checkpoints").glob("*.safetensors"))) == 1 and list((workspace / "staging").iterdir()) == []
    assert client.get(BASE + "/agent-session", headers=agent).json()["charged"] == 2


def test_weights_write_needs_weights_read(env):
    with pytest.raises(ValueError, match="weights_write requires weights_read"):
        api.Credentials("demo", weights_write=True)
    client, agent, human, creds, _ = env(read=True, write=True)
    creds.weights_read = False  # a token that somehow holds write without read must still be refused
    c = f"/checkpoints/{REF}"
    assert post(client, {**agent, "Idempotency-Key": "w"}, c + "/apply", PROGRAM).status_code == 403
    assert client.post(BASE + "/blobs", headers=agent, content=bytes(4)).status_code == 403


def test_cursor_key_is_not_derived_from_the_csrf_token(env):
    client, agent, human, creds, _ = env()
    cursor = client.get(BASE + f"/checkpoints/{REF}/describe", headers=agent, params={"limit": 2}).json()["cursor"]
    _, body, _ = cursor.split(".")
    old_key = b"layerdelta-v2-cursor:" + creds.csrf.encode()  # what a browser-held credential used to give away
    forged = f"c2.{body}." + hmac.digest(old_key, ("c1." + body).encode(), "sha256")[:16].hex()
    response = client.get(BASE + f"/checkpoints/{REF}/describe", headers=agent, params={"limit": 2, "cursor": forged})
    assert response.status_code == 422 and response.json()["error"]["code"] == "invalid_cursor"


def test_replay_is_answered_before_planning(env):
    client, agent, human, creds, workspace = env()
    blob = client.post(BASE + "/blobs", headers=agent, content=np.ones(10, dtype="<f4").tobytes()).json()["result"]["blob_id"]
    body = {"program": [{"op": "add", "args": {"blob": blob}, "select": BIAS}], "hypothesis": "blob"}
    first = apply_as(client, agent, "b1", body)
    (workspace / "blobs" / blob[7:]).unlink()  # the blob vanished: re-planning would now fail with blob_missing
    again = post(client, {**agent, "Idempotency-Key": "b1"}, f"/checkpoints/{REF}/apply", body)
    assert again.status_code == 200 and again.json()["result"]["replayed"] is True
    assert again.json()["result"]["checkpoint"] == first["result"]["checkpoint"]
    changed = {**body, "hypothesis": "different"}  # a conflicting key is also answered first
    conflict = post(client, {**agent, "Idempotency-Key": "b1"}, f"/checkpoints/{REF}/apply", changed)
    assert conflict.status_code == 409 and conflict.json()["error"]["code"] == "idempotency_conflict"
