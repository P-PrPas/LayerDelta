"""inspect-ops rev7 over HTTP: attach (human only), graph/groups/node routes, group selectors, reuse offers (v0.3-G4)."""

import time

import pytest
import torch
from safetensors.torch import save_file

from layerdelta.model import ReferenceCNN
from test_checkpoint_import import start_import
from test_v2_api import BASE, BIAS, PROGRAM, REF, env  # noqa: F401  (the env fixture)

C = f"/checkpoints/{REF}"


def get(client, headers, path, **kw):
    return client.get(BASE + path, headers=headers, **kw)


def wait_attach(client, human, attach_id, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = client.get(f"{BASE}/graph-attachments/{attach_id}", headers=human).json()
        if record["status"] in ("succeeded", "failed"):
            return record
        time.sleep(.05)
    pytest.fail("attach did not finish")


def attach(client, human, body, checkpoint=REF):
    response = client.post(f"{BASE}/checkpoints/{checkpoint}/graphs", headers=human, json=body)
    assert response.status_code == 202, response.text
    return wait_attach(client, human, response.json()["attach_id"])


def attached(env):
    client, agent, human, creds, workspace = env(quota=5)
    record = attach(client, human, {"provider": "builtin", "model": "reference_cnn"})
    assert record["status"] == "succeeded", record
    return client, agent, human, workspace, record


# ------------------------------------------------------------------ before anything is attached

def test_without_a_graph_describe_is_unchanged_and_the_attach_card_lists_providers(env):
    client, agent, human, creds, workspace = env()
    described = get(client, agent, C + "/describe").json()["result"]
    assert "graph" not in described and all(r["roles"] == [] and "binding" not in r for r in described["tensors"])
    card = get(client, agent, C + "/graph").json()
    assert card["result"]["graph"] is None and card["result"]["reusable"] == []
    providers = {p["name"]: p for p in card["result"]["providers"]}
    assert providers["builtin"]["enabled"] and "reference_cnn" in providers["builtin"]["models"]
    assert not providers["torch_export"]["enabled"] and providers["torch_export"]["reason"]
    assert not providers["onnx"]["enabled"]
    for path in ("/groups", "/graph/nodes/n0"):
        response = get(client, agent, C + path)
        assert response.status_code == 422 and response.json()["error"]["code"] == "no_graph"
    for call in ("read-tensor", "stats"):
        response = client.post(BASE + C + "/" + call, headers=agent, json={"selector": {"group": "g_x"}})
        assert response.status_code == 422 and response.json()["error"] == {
            "code": "no_graph", "message": "no graph is attached to this checkpoint", "field": "selector.group",
            "request_id": response.json()["error"]["request_id"]}
    assert get(client, agent, C + "/graph?view=modules").status_code == 422


# ------------------------------------------------------------------ attaching is a human action

def test_only_a_human_can_attach_and_the_job_reports_its_stages(env):
    client, agent, human, creds, workspace = env()
    body = {"provider": "builtin", "model": "reference_cnn"}
    assert client.post(BASE + C + "/graphs", headers=agent, json=body).status_code == 403
    assert client.get(BASE + "/graph-attachments/att_x", headers=agent).status_code == 403
    response = client.post(BASE + C + "/graphs", headers=human, json=body)
    first = response.json()
    assert response.status_code == 202 and first["status"] in ("queued", "running", "succeeded") and first["total"] == 3
    assert first["unit"] == "stages" and first["attach_id"].startswith("att_")
    done = wait_attach(client, human, first["attach_id"])
    assert done["status"] == "succeeded" and done["stage"] == "done" and done["done"] == 3 and done["error"] is None
    assert done["graph"]["provider"] == "builtin" and len(done["graph"]["ir_sha256"]) == 64
    assert client.get(BASE + "/graph-attachments/att_nope", headers=human).status_code == 404
    assert client.post(BASE + "/checkpoints/" + "f" * 64 + "/graphs", headers=human, json=body).status_code == 404


def test_attach_errors(env):
    client, agent, human, creds, workspace = env()
    failed = attach(client, human, {"provider": "builtin", "model": "no_such_model"})
    assert failed["status"] == "failed" and failed["error"]["code"] == "unknown_model"
    for body, code in (({"provider": "torch_export", "model": "x"}, "provider_disabled"), ({"provider": "onnx"}, "provider_disabled"),
                       ({}, "invalid_argument"), ({"provider": "builtin"}, "invalid_argument"),
                       ({"provider": "builtin", "model": "reference_cnn", "reuse": "a" * 64}, "invalid_argument"),
                       ({"reuse": "a" * 64}, "not_offered")):
        response = client.post(BASE + C + "/graphs", headers=human, json=body)
        assert response.status_code == 422 and response.json()["error"]["code"] == code, (body, response.text)
    assert client.post(BASE + C + "/graphs", headers=human, json={"provider": "builtin", "model": "m", "x": 1}).status_code == 422
    assert get(client, agent, C + "/graph").json()["result"]["graph"] is None  # nothing was attached by any of that


def test_a_graph_that_does_not_fit_the_checkpoint_is_refused_with_names(env, tmp_path):
    from layerdelta.providers import register_builtin
    from torch import nn
    register_builtin("test.wrong_shapes", lambda: nn.Sequential(nn.Linear(4, 4)), [((2, 4), "float32")], replace=True)
    client, agent, human, creds, workspace = env()
    record = attach(client, human, {"provider": "builtin", "model": "test.wrong_shapes"})
    assert record["status"] == "failed" and record["error"]["code"] == "graph_mismatch" and record["error"]["names"]
    assert get(client, agent, C + "/graph").json()["result"]["graph"] is None


# ------------------------------------------------------------------ after attaching

def test_describe_graph_groups_and_nodes_routes(env):
    client, agent, human, workspace, record = attached(env)
    ir_sha = record["graph"]["ir_sha256"]
    described = get(client, agent, C + "/describe").json()["result"]
    assert described["graph"]["ir_sha256"] == ir_sha and described["graph"]["providers"][0]["name"] == "builtin"
    rows = {r["name"]: r for r in described["tensors"]}
    assert rows["features.block1.conv.weight"]["binding"] == "bound" and rows["classifier.bias"]["roles"][0]["provenance"] == "derived"
    assert described["coverage"]["derived"] + described["coverage"]["unknown"] == sum(len(r["roles"]) for r in rows.values())
    page = get(client, agent, C + "/graph?limit=5").json()
    assert page["truncated"] and len(page["result"]["nodes"]) == 5 and page["result"]["graph"]["ir_sha256"] == ir_sha
    assert page["result"]["counts"]["groups"] > 0 and page["result"]["view"] == "nodes"
    more = get(client, agent, C + "/graph", params={"limit": 5, "cursor": page["cursor"]}).json()
    assert more["result"]["offset"] == 5 and more["result"]["nodes"][0]["id"] != page["result"]["nodes"][0]["id"]
    assert get(client, agent, C + "/graph", params={"limit": 5, "cursor": "c1.forged"}).status_code == 422
    modules = get(client, agent, C + "/graph?view=modules").json()["result"]["modules"]
    assert modules[0]["path"] == "" and modules[0]["node_count"] == page["result"]["counts"]["nodes"]
    conv = [n for n in get(client, agent, C + "/graph?limit=2000&op=*conv2d*").json()["result"]["nodes"]]
    assert conv and all(n["rule"] == n["op"] for n in conv)
    node = get(client, agent, C + f"/graph/nodes/{conv[0]['id']}").json()["result"]["node"]
    assert node["rule"] == conv[0]["op"] and node["parameters"] and node["groups"]
    assert get(client, agent, C + "/graph/nodes/n99999").json()["error"]["code"] == "unknown_node"
    listing = get(client, agent, C + "/groups").json()["result"]
    assert listing["matched"] == len(listing["groups"]) and {"label", "kind", "notes", "spans", "boundary", "members"} <= set(listing["groups"][0])
    closed = get(client, agent, C + "/groups?closed=false").json()["result"]["groups"]
    assert closed and all(g["boundary"] for g in closed)
    one = get(client, agent, C + "/groups?tensor=classifier.weight&limit=1").json()
    assert len(one["result"]["groups"]) == 1 and get(client, agent, C + "/groups?node=zz").json()["error"]["code"] == "unknown_node"


def test_group_selectors_over_http(env):
    client, agent, human, workspace, record = attached(env)
    groups = get(client, agent, C + "/groups?limit=500").json()["result"]["groups"]
    g = next(x for x in groups if x["closed"] and x["size"] > 1)
    selector = {"group": g["group_id"], "units": [0, 1]}
    read = client.post(BASE + C + "/read-tensor", headers=agent, json={"selector": selector, "max_elems": 5000}).json()
    assert read["result"]["group"]["unit_count"] == 2 and read["resolved"]["selector"]["expanded"]
    again = client.post(BASE + C + "/read-tensor", headers=agent, json={"selector": {k: read["resolved"]["selector"][k] for k in ("group", "units")},
                                                                         "max_elems": 5000}).json()
    assert again["fingerprints"]["selection"] == read["fingerprints"]["selection"]
    stats = client.post(BASE + C + "/stats", headers=agent, json={"selector": selector}).json()
    assert stats["result"]["members"] and stats["result"]["members"][0]["units"][0]["unit"] == 0
    query = client.post(BASE + C + "/query", headers=agent, json={"where": {"group": g["group_id"], "stat": "l2", "op": "ge", "value": 0}}).json()
    assert query["result"]["matches"][0]["member_used"]["reason"]
    filtered = client.post(BASE + C + "/read-tensor", headers=agent, json={"selector": {**selector, "members": {"roles": ["in_unit"]}}}).json()
    assert {m["role"] for m in filtered["result"]["members"]} == {"in_unit"}
    paged = client.post(BASE + C + "/read-tensor", headers=agent, json={"selector": selector, "max_elems": 3}).json()
    assert paged["truncated"] and paged["cursor"].startswith("c2.")  # signed like every other cursor
    forged = client.post(BASE + C + "/read-tensor", headers=agent, json={"selector": selector, "cursor": "c1.x"})
    assert forged.status_code == 422 and forged.json()["error"]["code"] == "invalid_cursor"
    bad = client.post(BASE + C + "/read-tensor", headers=agent, json={"selector": {"group": "g_" + "0" * 24}})
    assert bad.json()["error"]["code"] == "unknown_group"


def test_group_reads_need_weights_read_but_structure_does_not(env):
    client, agent, human, creds, workspace = env(read=False, write=False)
    attach(client, human, {"provider": "builtin", "model": "reference_cnn"})
    groups = get(client, agent, C + "/groups").json()["result"]["groups"]
    assert groups and get(client, agent, C + "/graph").status_code == 200 and get(client, agent, C + "/describe").status_code == 200
    for call in ("read-tensor", "stats"):
        denied = client.post(BASE + C + "/" + call, headers=agent, json={"selector": {"group": groups[0]["group_id"]}})
        assert denied.status_code == 403 and denied.json()["error"]["code"] == "forbidden"
    where = {"where": {"group": groups[0]["group_id"], "stat": "l2", "op": "ge", "value": 0}}
    assert client.post(BASE + C + "/query", headers=agent, json=where).status_code == 403


def test_a_child_inherits_the_graph_and_the_groups(env):
    client, agent, human, workspace, record = attached(env)
    parent = get(client, agent, C + "/groups?limit=500").json()["result"]["groups"]
    applied = client.post(BASE + C + "/apply", headers={**agent, "Idempotency-Key": "k1"}, json=PROGRAM).json()
    child = applied["result"]["checkpoint"]["checkpoint_id"]
    described = get(client, agent, f"/checkpoints/{child}/describe").json()["result"]
    assert described["graph"]["ir_sha256"] == record["graph"]["ir_sha256"]
    assert described["graph"]["table_signature"] == record["graph"]["table_signature"]
    inherited = get(client, agent, f"/checkpoints/{child}/groups?limit=500").json()["result"]["groups"]
    assert [g["group_id"] for g in inherited] == [g["group_id"] for g in parent]
    diff = client.post(BASE + f"/checkpoints/{child}/diff", headers=agent, json={
        "other": REF, "selector": {"group": parent[0]["group_id"]}}).json()
    assert diff["result"]["group"]["group_id"] == parent[0]["group_id"] and diff["fingerprints"]["checkpoint_b"] == REF


def test_the_same_graph_is_offered_never_attached_to_another_dtype(env, tmp_path):
    client, agent, human, workspace, record = attached(env)
    torch.manual_seed(3)
    half = {k: v.to(torch.bfloat16) if v.is_floating_point() else v for k, v in ReferenceCNN().state_dict().items()}
    path = tmp_path / "bf16.safetensors"
    save_file(half, path)
    sibling = start_import(client, human, path)
    assert sibling["status"] == "succeeded", sibling
    sid = sibling["checkpoint_id"]
    described = get(client, agent, f"/checkpoints/{sid}/describe").json()["result"]
    assert "graph" not in described  # a different dtype is a different layout: nothing is attached for it
    card = get(client, agent, f"/checkpoints/{sid}/graph").json()["result"]
    assert card["graph"] is None
    assert card["reusable"] == [{"ir_sha256": record["graph"]["ir_sha256"], "provider": "builtin", "checkpoints": [REF]}]
    assert get(client, agent, f"/checkpoints/{sid}/groups").status_code == 422  # still nothing attached
    refused = client.post(f"{BASE}/checkpoints/{sid}/graphs", headers=agent, json={"reuse": record["graph"]["ir_sha256"]})
    assert refused.status_code == 403  # an agent cannot accept the offer
    reused = attach(client, human, {"reuse": record["graph"]["ir_sha256"]}, checkpoint=sid)
    assert reused["status"] == "succeeded" and reused["graph"]["ir_sha256"] == record["graph"]["ir_sha256"]
    assert reused["graph"]["table_signature"] != record["graph"]["table_signature"]
    after = get(client, agent, f"/checkpoints/{sid}/describe").json()["result"]
    assert after["graph"]["ir_sha256"] == record["graph"]["ir_sha256"]
    ids = lambda cid: [g["group_id"] for g in get(client, agent, f"/checkpoints/{cid}/groups?limit=500").json()["result"]["groups"]]
    assert ids(sid) == ids(REF)  # group ids hash the graph and the members, not the dtype


def test_a_layout_that_differs_is_not_offered(env, tmp_path):
    client, agent, human, workspace, record = attached(env)
    state = {k: v.clone() for k, v in ReferenceCNN().state_dict().items()}
    state["classifier.extra"] = torch.zeros(3)
    path = tmp_path / "other.safetensors"
    save_file(state, path)
    other = start_import(client, human, path)
    card = get(client, agent, f"/checkpoints/{other['checkpoint_id']}/graph").json()["result"]
    assert card["reusable"] == []
    refused = client.post(f"{BASE}/checkpoints/{other['checkpoint_id']}/graphs", headers=human, json={"reuse": record["graph"]["ir_sha256"]})
    assert refused.status_code == 422 and refused.json()["error"]["code"] == "not_offered"


class Variant(ReferenceCNN):
    def forward(self, images):  # same parameters, a different graph (one more relu)
        return torch.relu(super().forward(images))


def test_attaching_a_graph_again_makes_it_the_current_one(env):
    from layerdelta.providers import register_builtin
    register_builtin("test.variant", Variant, (((1, 1, 28, 28), "float32"),), replace=True)
    client, agent, human, creds, workspace = env()
    a = attach(client, human, {"provider": "builtin", "model": "reference_cnn"})["graph"]["ir_sha256"]
    b = attach(client, human, {"provider": "builtin", "model": "test.variant"})["graph"]["ir_sha256"]
    assert a != b
    current = lambda: get(client, agent, C + "/describe").json()["result"]["graph"]["ir_sha256"]
    assert current() == b
    again = attach(client, human, {"provider": "builtin", "model": "reference_cnn"})
    assert again["status"] == "succeeded" and again["graph"]["ir_sha256"] == a and current() == a  # A, B, A: A is current
    assert attach(client, human, {"reuse": b})["graph"]["ir_sha256"] == b and current() == b  # reusing a stored graph counts too


def test_the_derivation_cache_survives_constant_eviction(env, monkeypatch):
    from layerdelta import v2
    monkeypatch.setattr(v2, "DERIVED_KEEP", 0)
    client, agent, human, creds, workspace = env()
    assert attach(client, human, {"provider": "builtin", "model": "reference_cnn"})["status"] == "succeeded"
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as pool:
        codes = list(pool.map(lambda _: get(client, agent, C + "/groups?limit=3").status_code, range(24)))
    assert set(codes) == {200}

