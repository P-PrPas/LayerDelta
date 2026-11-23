"""Group edits over HTTP (v0.3-G5): plan, validate and apply with group selectors, human-attached graph, agent token."""

import torch
from safetensors.torch import load_file

from layerdelta.model import ReferenceCNN
from test_graph_http import BASE, REF, C, attach, attached, env, get  # noqa: F401  (the env fixture)


def post(client, headers, path, body):
    return client.post(BASE + path, headers=headers, json=body)


def closed_group(client, agent):
    groups = get(client, agent, C + "/groups?limit=500&closed=true").json()["result"]["groups"]
    return next(g for g in groups if g["size"] > 1)


def permute_program(group):
    return {"program": [{"op": "permute_units", "args": {"perm": list(range(group["size"]))[::-1]},
                         "select": {"group": group["group_id"]}}], "hypothesis": "reverse the units"}


def forward(state):
    model = ReferenceCNN().eval()
    model.load_state_dict(state)
    with torch.no_grad():
        return model(torch.full((2, 1, 28, 28), .5) * torch.linspace(0, 1, 28))


def test_plan_validate_and_apply_a_permutation(env):
    client, agent, human, workspace, record = attached(env)
    group = closed_group(client, agent)
    body = permute_program(group)
    planned = post(client, agent, C + "/plan", body).json()
    assert planned["fingerprints"]["op_revision"] == 2 and planned["result"]["recipe"]["engine"]["layerdelta_ops_revision"] == 2
    moved = planned["result"]["moved_units"][0]
    assert moved["group"] == group["group_id"] and moved["total"] == group["size"] and moved["units"] and planned["result"]["warnings"] == []
    assert planned["result"]["groups"][0]["closed"] is True and all("via" in o for o in planned["result"]["ops"])
    checked = post(client, agent, C + "/validate", body).json()
    assert checked["result"]["valid"] is True and checked["fingerprints"]["op_revision"] == 2
    applied = client.post(BASE + C + "/apply", headers={**agent, "Idempotency-Key": "p1"}, json=body).json()
    child = applied["result"]["checkpoint"]["checkpoint_id"]
    assert [e["kind"] for e in applied["side_effects"]][0] == "checkpoint_created"
    parent_state = load_file(workspace / "reference" / "model.safetensors")
    child_state = load_file(workspace / "checkpoints" / f"{child}.safetensors")
    assert any(not torch.equal(parent_state[k], child_state[k]) for k in parent_state)
    assert torch.allclose(forward(parent_state), forward(child_state), atol=1e-5)  # a different file, the same function
    described = get(client, agent, f"/checkpoints/{child}/describe").json()["result"]["graph"]
    assert described["ir_sha256"] == record["graph"]["ir_sha256"]  # the child inherits the graph
    replay = client.post(BASE + C + "/apply", headers={**agent, "Idempotency-Key": "p2"},
                         json={"recipe": planned["result"]["recipe"]}).json()
    assert replay["result"]["checkpoint"]["checkpoint_id"] == child and replay["result"]["existing"] is True
    again = post(client, agent, f"/checkpoints/{child}/plan", {"program": [{"op": "permute_units", "args": {
        "perm": list(range(group["size"]))[::-1]}, "select": {"group": group["group_id"]}}], "hypothesis": "and back"})
    assert again.status_code == 200 and again.json()["result"]["changed_elems_total"] > 0


def test_an_open_group_refuses_a_permutation_but_allows_a_scale(env):
    client, agent, human, workspace, record = attached(env)
    groups = get(client, agent, C + "/groups?limit=500&closed=false").json()["result"]["groups"]
    head = next(g for g in groups if any(b["reason"] == "graph_output" for b in g["boundary"]))
    body = permute_program(head)
    refused = post(client, agent, C + "/plan", body)
    assert refused.status_code == 422 and refused.json()["error"]["code"] == "group_not_closed"
    checked = post(client, agent, C + "/validate", body).json()["result"]
    assert checked["valid"] is False and checked["error"]["code"] == "group_not_closed"
    scale = {"program": [{"op": "scale", "args": {"k": 2.0}, "select": {"group": head["group_id"], "units": [0]}}], "hypothesis": "h"}
    planned = post(client, agent, C + "/plan", scale).json()["result"]
    assert planned["warnings"][0]["code"] == "group_open" and planned["changed_elems_total"] > 0


def test_group_programs_need_a_graph_and_a_known_group(env):
    client, agent, human, creds, workspace = env()
    body = {"program": [{"op": "zero", "args": {}, "select": {"group": "g_" + "0" * 24}}], "hypothesis": "h"}
    refused = post(client, agent, C + "/plan", body)
    assert refused.status_code == 422 and refused.json()["error"]["code"] == "no_graph"
    assert post(client, agent, C + "/validate", body).json()["result"]["error"]["code"] == "no_graph"
    attach(client, human, {"provider": "builtin", "model": "reference_cnn"})
    unknown = post(client, agent, C + "/plan", body)
    assert unknown.status_code == 422 and unknown.json()["error"]["code"] == "unknown_group"


def test_a_recipe_with_a_group_replays_on_a_checkpoint_that_never_had_a_graph(env):
    client, agent, human, workspace, record = attached(env)
    group = closed_group(client, agent)
    recipe = post(client, agent, C + "/plan", permute_program(group)).json()["result"]["recipe"]
    assert recipe["program"][0]["via"]["ir_sha256"] == record["graph"]["ir_sha256"]
    for entry in recipe["program"]:  # graph-free: every op is tensor-local
        assert set(entry["select"]) == {"tensor", "shape", "index"}


def test_a_submitted_recipe_is_checked_against_the_graph(env):
    client, agent, human, workspace, record = attached(env)
    group = closed_group(client, agent)
    planned = post(client, agent, C + "/plan", permute_program(group)).json()["result"]
    assert [g["via_verified"] for g in planned["groups"]] == [True]
    recipe = planned["recipe"]
    echoed = post(client, agent, C + "/plan", {"recipe": recipe}).json()["result"]  # replaying the real recipe verifies again
    assert [g["via_verified"] for g in echoed["groups"]] == [True]
    import copy
    from layerdelta import edit
    fake = copy.deepcopy(recipe)
    fake["program"][0]["args"]["perm"].reverse()  # a gather the group edit does not expand to
    fake["recipe_hash"] = edit._finish(REF, fake["program"], recipe["hypothesis"], recipe["evaluation"], edit.ENGINE_V2)["recipe_hash"]
    refused = post(client, agent, C + "/plan", {"recipe": fake})
    assert refused.status_code == 422 and refused.json()["error"]["code"] == "noncanonical_recipe"
    assert refused.json()["error"]["field"] == "program[0].via"
    applied = client.post(BASE + C + "/apply", headers={**agent, "Idempotency-Key": "fake"}, json={"recipe": fake})
    assert applied.status_code == 422 and applied.json()["error"]["code"] == "noncanonical_recipe"
    bare = copy.deepcopy(recipe)
    del bare["program"][0]["via"]
    bare["recipe_hash"] = edit._finish(REF, bare["program"], recipe["hypothesis"], recipe["evaluation"], edit.ENGINE_V2)["recipe_hash"]
    assert post(client, agent, C + "/plan", {"recipe": bare}).json()["error"]["code"] == "noncanonical_recipe"
