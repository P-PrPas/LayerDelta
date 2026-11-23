"""Lineage and the apply-then-verify hint (v0.4-S6)."""

from test_v2_api import BASE, BIAS, PROGRAM, REF, env, post  # noqa: F401

C = f"/checkpoints/{REF}"


def test_get_checkpoint_walks_lineage_and_apply_says_what_to_diff(env):
    client, agent, human, creds, workspace = env(quota=5)
    first = post(client, {**agent, "Idempotency-Key": "a"}, C + "/apply", PROGRAM).json()
    child = first["result"]["checkpoint"]["checkpoint_id"]
    hint = first["result"]["diff_hint"]
    assert hint == {"checkpoint_id": REF, "other": child, "selectors": [first["result"]["recipe"]["program"][0]["select"]],
                    "selectors_truncated": False}
    # the hint is a ready diff request: the parent against the child, narrowed to what the program touched
    diff = post(client, agent, C + "/diff", {"other": hint["other"], "selector": hint["selectors"][0]})
    assert diff.status_code == 200 and diff.json()["result"] != {}
    whole = post(client, agent, C + "/diff", {"other": child}).json()["result"]
    assert [t["name"] for t in whole["tensors"]] == [BIAS["tensor"]]  # without a selector: only the changed tensor is listed
    # lineage, both directions; the list stays flat
    parent = client.get(BASE + C, headers=agent).json()
    assert parent["children"] == [child] and parent["parent_id"] is None
    kid = client.get(BASE + f"/checkpoints/{child}", headers=agent).json()
    assert kid["parent_id"] == REF and kid["children"] == [] and kid["recipe_hash"] == first["result"]["recipe"]["recipe_hash"]
    assert all("children" not in e for e in client.get(BASE + "/checkpoints", headers=agent).json()["items"])
    # a second program is a second child; a replay returns the same hint
    other = post(client, {**agent, "Idempotency-Key": "b"},
                 C + "/apply", {"program": [{"op": "scale", "args": {"k": 3.0}, "select": BIAS}], "hypothesis": "triple"}).json()
    kids = client.get(BASE + C, headers=agent).json()["children"]
    assert len(kids) == 2 and set(kids) == {child, other["result"]["checkpoint"]["checkpoint_id"]}
    replay = post(client, {**agent, "Idempotency-Key": "a"}, C + "/apply", PROGRAM).json()
    assert replay["result"]["replayed"] is True and replay["result"]["diff_hint"] == hint
