"""Activation probes (v0.5-E5): per-unit aggregates of graph-node outputs, the K rule, raw values behind weights:values, budgets and scoping.

The expected numbers come from running the reference CNN eagerly on the same preprocessed samples."""

import gc

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import load_file

from layerdelta import api, datasets, graph_run, preprocess
from layerdelta.model import ReferenceCNN
from test_evaluate import make_world, wait  # noqa: F401
from test_v2_api import BASE, REF, make_workspace  # noqa: F401

pytest.importorskip("PIL")


@pytest.fixture
def server(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path, monkeypatch)
    opened = []

    def make(budgets=None, min_aggregate=8, **caps):
        creds = api.Credentials("demo", **{"weights_read": True, "weights_write": True, "eval_run": True, **caps})
        client = TestClient(api.create_app(workspace, creds, quota=50, budgets=budgets or {}, min_aggregate=min_aggregate), base_url="http://127.0.0.1")
        client.__enter__()
        opened.append(client)
        csrf = client.post("/api/v1/session", json={"ticket": creds.ticket}).json()["csrf"]
        return client, {"Authorization": "Bearer " + creds.agent}, {"X-CSRF-Token": csrf}
    make.workspace = workspace
    yield make
    for client in opened:
        client.__exit__(None, None, None)


def probe(client, agent, key, checkpoint=REF, **body):
    return client.post(BASE + f"/checkpoints/{checkpoint}/probes", headers={**agent, "Idempotency-Key": key}, json=body)


def expected(workspace, dataset, node, split_by=None):
    """The same aggregates, from the eager model and the dataset's own preprocessing."""
    data = datasets.open_dataset(workspace, dataset)
    model = ReferenceCNN().eval()
    model.load_state_dict(load_file(workspace / "reference" / "model.safetensors"))
    program = torch.export.export(model, (torch.zeros(1, 1, 28, 28),))
    weights = dict(model.state_dict())
    groups = {}
    for row in data.samples("exploration"):
        image = preprocess.apply(data.record["preprocessing"], preprocess.decode((data.files / row["image"]).read_bytes()))["image"]
        _, seen = graph_run.run(program, weights, "", [image[None]], watch=[node])
        unit_major = seen[node][0].to(torch.float64).reshape(seen[node].shape[1], -1).numpy()
        groups.setdefault(row["label"] if split_by == "label" else "all", []).append(unit_major)
    out = {}
    for name, parts in groups.items():
        flat = np.concatenate(parts, axis=1)
        out[name] = {"n": flat.shape[1], "mean": flat.mean(axis=1), "std": flat.std(axis=1), "l2": np.sqrt((flat ** 2).sum(axis=1)),
                     "frac_positive": (flat > 0).mean(axis=1)}
    return out


def test_a_probe_aggregates_a_node_per_unit_and_matches_the_eager_model(server, tmp_path):
    client, agent, human = server()
    client, agent, human, dataset, child = make_world(client, agent, human, tmp_path)
    started = probe(client, agent, "p1", dataset_id=dataset, nodes=["n0"], split_by="label")
    assert started.status_code == 202, started.text
    done = wait(client, agent, f"/probes/{started.json()['probe_id']}")
    assert done["status"] == "succeeded", done
    node = done["nodes"][0]
    assert node["node_id"] == "n0" and node["unit_axis"] == 1 and node["units"] == 8 and done["skipped_small"] == [] and "values" not in done
    want = expected(server.workspace, dataset, "n0", split_by="label")
    assert set(node["groups"]) == set(want) and len(want) == 10
    for name, group in node["groups"].items():
        assert group["n"] == want[name]["n"]
        for stat in ("mean", "std", "l2", "frac_positive"):
            np.testing.assert_allclose(group[stat], want[name][stat], rtol=1e-6, atol=1e-9)
    info = client.get(BASE + "/agent-session", headers=agent).json()["budgets"]
    assert info["evals"]["used"] == 1 and info["eval_samples"]["used"] == 20

    again = probe(client, agent, "p1", dataset_id=dataset, nodes=["n0"], split_by="label").json()
    assert again["probe_id"] == started.json()["probe_id"] and again["replayed"] is True
    assert client.get(BASE + "/agent-session", headers=agent).json()["budgets"]["eval_samples"]["used"] == 20
    assert probe(client, agent, "p1", dataset_id=dataset, nodes=["n1"]).status_code == 409
    histogram = wait(client, agent, f"/probes/{probe(client, agent, 'p2', dataset_id=dataset, nodes=['n1'], bins=4).json()['probe_id']}")
    counts = histogram["nodes"][0]["groups"]["all"]["histogram"]["counts"]
    assert len(counts) == 8 and all(len(row) == 4 for row in counts) and sum(counts[0]) == 20 * 28 * 28  # every element of a unit lands in a bin


def test_raw_activations_need_weights_values_and_the_k_rule_applies_to_class_one(server, tmp_path):
    client, agent, human = server(weights_values=False)  # egress class 1
    client, agent, human, dataset, child = make_world(client, agent, human, tmp_path)
    assert probe(client, agent, "r1", dataset_id=dataset, nodes=["n0"], values=True).status_code == 403
    # the last node is the 10-unit output: one position per sample, so a label group of 2 samples gives 2 elements per unit (< K = 8)
    last = "n7"
    refused = probe(client, agent, "k2", dataset_id=dataset, nodes=[last], split_by="label")
    assert refused.status_code == 403 and refused.json()["error"]["code"] == "aggregate_too_small", refused.text
    few = probe(client, agent, "k3", dataset_id=dataset, nodes=[last], limit_samples=4)
    assert few.status_code == 403 and few.json()["error"]["code"] == "aggregate_too_small"
    ok = probe(client, agent, "k4", dataset_id=dataset, nodes=[last])  # 20 x 1 = 20 >= 8
    assert ok.status_code == 202
    done = wait(client, agent, f"/probes/{ok.json()['probe_id']}")
    assert done["status"] == "succeeded" and done["nodes"][0]["units"] == 10 and "values" not in done
    assert client.get(BASE + "/agent-session", headers=agent).json()["budgets"]["eval_samples"]["used"] == 20  # the refused probes charged nothing


def test_groups_below_k_are_neither_computed_nor_readable_and_histograms_need_k_per_bin(server, tmp_path):
    client, agent, human = server(weights_values=False, min_aggregate=60)  # n5 has 7 x 7 = 49 positions: a group of 1 sample is 49 (< 60), of 2 is 98
    client, agent, human, dataset, child = make_world(client, agent, human, tmp_path)
    rows = evalservice_subset(server.workspace, dataset, 13)
    sizes = {}
    for row in rows:
        sizes[row["label"]] = sizes.get(row["label"], 0) + 1
    assert set(sizes.values()) == {1, 2}  # a mixed case: some groups are below K, some are not
    done = wait(client, agent, f"/probes/{probe(client, agent, 'g1', dataset_id=dataset, nodes=['n5'], split_by='label', limit_samples=13).json()['probe_id']}")
    assert done["status"] == "succeeded", done
    kept = done["nodes"][0]["groups"]
    assert {g["group"] for g in done["skipped_small"]} == {label for label, n in sizes.items() if n == 1}
    assert set(kept) == {label for label, n in sizes.items() if n == 2} and all(g["n"] >= 60 for g in kept.values())
    assert not set(kept) & {g["group"] for g in done["skipped_small"]}  # what is listed as skipped is not in the result
    # a histogram needs K x bins elements per group: 60 x 4 = 240 > 98 for the groups of 2, so every group is below it and the probe is refused
    refused = probe(client, agent, "g2", dataset_id=dataset, nodes=["n5"], split_by="label", limit_samples=13, bins=4)
    assert refused.status_code == 403 and refused.json()["error"]["code"] == "aggregate_too_small", refused.text
    assert probe(client, agent, "g3", dataset_id=dataset, nodes=["n5"], bins=4).status_code == 202  # all 20 samples: 980 >= 240


def evalservice_subset(workspace, dataset, limit):
    from layerdelta import evalservice
    return evalservice.subset(datasets.open_dataset(workspace, dataset).samples("exploration"), limit)


def test_the_program_must_say_what_the_attached_graph_says(server):
    model = ReferenceCNN().eval()
    program = torch.export.export(model, (torch.zeros(1, 1, 28, 28),))
    graph_run.check_ops(program, {"n0": "aten.conv2d.default", "n7": "aten.linear.default"})
    with pytest.raises(graph_run.NodeUnsupported, match="n0 is aten.conv2d.default in the program but aten.linear.default"):
        graph_run.check_ops(program, {"n0": "aten.linear.default"})


def test_a_class_two_token_may_ask_for_raw_values(server, tmp_path):
    client, agent, human = server()
    client, agent, human, dataset, child = make_world(client, agent, human, tmp_path)
    started = probe(client, agent, "v1", dataset_id=dataset, nodes=["n7"], values=True, limit_samples=3)
    done = wait(client, agent, f"/probes/{started.json()['probe_id']}")
    assert done["status"] == "succeeded" and done["values"]["n7"]["shape"] == [1, 10] and len(done["values"]["n7"]["data"]) == 10


def test_unsupported_nodes_and_bad_requests_are_refused_before_any_charge(server, tmp_path):
    client, agent, human = server()
    client, agent, human, dataset, child = make_world(client, agent, human, tmp_path)
    cases = [({"nodes": ["n999"]}, "probe_unsupported"), ({"nodes": ["n0"], "unit_axis": 9}, "invalid_argument"), ({"nodes": ["x"]}, "invalid_argument"),
             ({"nodes": ["n0"], "split_by": "colour"}, "invalid_argument"), ({"nodes": ["n0"], "bins": 1000}, "invalid_argument")]
    for i, (body, code) in enumerate(cases):
        answer = probe(client, agent, f"u{i}", dataset_id=dataset, **body)
        assert answer.status_code == 422 and answer.json()["error"]["code"] == code, (body, answer.text)
    assert probe(client, agent, "u9", dataset_id="ds_" + "0" * 32, nodes=["n0"]).status_code == 404
    assert client.get(BASE + "/probes/pr_0123456789abcdef", headers=agent).status_code == 404
    assert client.get(BASE + "/agent-session", headers=agent).json()["budgets"]["eval_samples"]["used"] == 0


@pytest.mark.parametrize("caps, text", [({"weights_read": False, "eval_run": True}, "weights:read"), ({"weights_read": True, "eval_run": False}, "eval:run"),
                                        ({"weights_read": False, "weights_write": True, "eval_run": True, "profile": "structure"}, "profile")])
def test_a_probe_is_refused_without_weights_read_or_eval_run_or_in_the_structure_profile(tmp_path, monkeypatch, caps, text):
    workspace = make_workspace(tmp_path, monkeypatch)
    creds = api.Credentials("demo", **caps)
    with TestClient(api.create_app(workspace, creds), base_url="http://127.0.0.1") as client:
        header = {"Authorization": "Bearer " + creds.agent}
        refused = client.post(BASE + f"/checkpoints/{REF}/probes", headers={**header, "Idempotency-Key": "x"}, json={"dataset_id": "ds_x", "nodes": ["n0"]})
        assert refused.status_code == 403 and text in refused.json()["error"]["message"], refused.text
        reading = client.get(BASE + "/probes/pr_0123456789abcdef", headers=header).status_code
        assert reading == (404 if text == "eval:run" else 403)  # reading needs weights:read only; with it, an id that is not yours is unknown
    gc.collect()



# ---------------------------------------------------------------- activation edits in an evaluation (E5b)

def evaluate_with(client, agent, checkpoint, dataset, key, **body):
    return client.post(BASE + f"/checkpoints/{checkpoint}/evaluations", headers={**agent, "Idempotency-Key": key},
                       json={"dataset_id": dataset, "evaluator": "classification", **body})


def eager_predictions(workspace, dataset, change=None):
    data = datasets.open_dataset(workspace, dataset)
    model = ReferenceCNN().eval()
    model.load_state_dict(load_file(workspace / "reference" / "model.safetensors"))
    program = torch.export.export(model, (torch.zeros(1, 1, 28, 28),))
    out = []
    for row in data.samples("exploration"):
        image = preprocess.apply(data.record["preprocessing"], preprocess.decode((data.files / row["image"]).read_bytes()))["image"]
        logits, _ = graph_run.run(program, dict(model.state_dict()), "", [image[None]], change=change)
        out.append(int(logits[0].argmax()))
    return out


def test_an_activation_edit_runs_the_edited_graph_and_leaves_the_baseline_alone(server, tmp_path):
    client, agent, human = server()
    client, agent, human, dataset, child = make_world(client, agent, human, tmp_path)
    plain = evaluate_with(client, agent, REF, dataset, "i0", baseline=REF)  # REF against itself: nothing changes, and its run is cached
    plain_done = wait(client, agent, f"/evaluations/{plain.json()['evaluation_id']}")
    assert plain_done["status"] == "succeeded" and plain_done["counts"]["fixed"] + plain_done["counts"]["broken"] + plain_done["counts"]["changed"] == 0
    edit = {"node_id": "n1", "units": [0, 2, 5], "op": "zero"}
    started = evaluate_with(client, agent, REF, dataset, "i1", baseline=REF, intervention=edit)
    assert started.status_code == 202, started.text
    body = started.json()
    assert body["intervention"] == {"node_id": "n1", "unit_axis": 1, "units": [0, 2, 5], "op": "zero", "factor": None}
    assert body["baseline_cached"] is True and body["fingerprint"] != plain.json()["fingerprint"]  # the baseline run is reused, the edited run is its own
    done = wait(client, agent, f"/evaluations/{body['evaluation_id']}")
    assert done["status"] == "succeeded", done
    zeroed = lambda t: t.clone().index_fill_(1, torch.tensor([0, 2, 5]), 0.0)  # noqa: E731
    base = eager_predictions(server.workspace, dataset)
    edited = eager_predictions(server.workspace, dataset, change={"n1": zeroed})
    assert done["counts"]["fixed"] + done["counts"]["broken"] + done["counts"]["changed"] == sum(a != b for a, b in zip(base, edited))
    scaled = evaluate_with(client, agent, REF, dataset, "i2", baseline=REF, intervention={"node_id": "n1", "units": [1], "op": "scale", "factor": 3.0})
    assert wait(client, agent, f"/evaluations/{scaled.json()['evaluation_id']}")["status"] == "succeeded"
    events = client.get(BASE + "/trace", headers=agent, params={"limit": 500}).json()["result"]["events"]
    run = next(e for e in events if e.get("evaluation_id") == body["evaluation_id"])
    assert run["intervention_id"].startswith("iv_") and "n1" not in str(run)  # the trace names no node and no unit


def test_an_activation_edit_is_checked_and_needs_weights_read(server, tmp_path):
    client, agent, human = server()
    client, agent, human, dataset, child = make_world(client, agent, human, tmp_path)
    cases = [({"node_id": "n999", "units": [0], "op": "zero"}, "probe_unsupported"), ({"node_id": "n1", "units": [8], "op": "zero"}, "invalid_argument"),
             ({"node_id": "n1", "units": [], "op": "zero"}, "invalid_argument"), ({"node_id": "n1", "units": [0, 0], "op": "zero"}, "invalid_argument"),
             ({"node_id": "n1", "units": [0], "op": "scale"}, "invalid_argument"), ({"node_id": "n1", "units": [0], "op": "zero", "factor": 2}, "invalid_argument"),
             ({"node_id": "n1", "units": [0], "op": "set"}, "invalid_argument"), ({"node_id": "n1", "units": [0], "op": "scale", "factor": "big"}, "invalid_argument"),
             ({"node_id": "n1", "unit_axis": 7, "units": [0], "op": "zero"}, "invalid_argument"), ({"node_id": "n1", "op": "zero"}, "invalid_argument")]
    for i, (edit, code) in enumerate(cases):
        answer = evaluate_with(client, agent, REF, dataset, f"b{i}", baseline=None, intervention=edit)
        assert answer.status_code == 422 and answer.json()["error"]["code"] == code, (edit, answer.text)
    assert client.get(BASE + "/agent-session", headers=agent).json()["budgets"]["eval_samples"]["used"] == 0


def test_a_session_without_weights_read_cannot_edit_activations(server, tmp_path):
    client, agent, human = server(profile="structure", weights_read=False)
    client, agent, human, dataset, child = make_world(client, agent, human, tmp_path)
    refused = evaluate_with(client, agent, child, dataset, "s1", intervention={"node_id": "n1", "units": [0], "op": "zero"})
    assert refused.status_code == 403 and "weights:read" in refused.json()["error"]["message"]
    assert evaluate_with(client, agent, child, dataset, "s2").status_code == 202  # a plain evaluation is still the control arm's


# ---------------------------------------------------------------- group selectors (E5c)

def test_a_group_resolves_to_its_layer_output_for_a_probe_and_an_edit(server, tmp_path):
    from layerdelta import groupnode
    client, agent, human = server()
    client, agent, human, dataset, child = make_world(client, agent, human, tmp_path)
    groups = client.get(BASE + f"/checkpoints/{REF}/groups", headers=agent, params={"limit": 50}).json()["result"]["groups"]
    conv = next(g for g in groups if g["closed"] and any(m["tensor"] == "features.block1.conv.weight" and m["role"] == "out_unit" for m in g["members"]))
    by_group = wait(client, agent, f"/probes/{probe(client, agent, 'gr1', dataset_id=dataset, group=conv['group_id']).json()['probe_id']}")
    by_node = wait(client, agent, f"/probes/{probe(client, agent, 'gr2', dataset_id=dataset, nodes=['n0']).json()['probe_id']}")
    assert by_group["status"] == "succeeded" and by_group["nodes"][0]["node_id"] == "n0" and by_group["nodes"][0]["units"] == conv["size"]
    assert by_group["nodes"] == by_node["nodes"]  # the group IS the conv's output channels
    assert probe(client, agent, "gr3", dataset_id=dataset, group="g_none").status_code == 404
    both = probe(client, agent, "gr4", dataset_id=dataset, group=conv["group_id"], nodes=["n0"])
    assert both.status_code == 422 and both.json()["error"]["code"] == "invalid_argument"
    assert probe(client, agent, "gr5", dataset_id=dataset).status_code == 422
    edit = evaluate_with(client, agent, REF, dataset, "gr6", baseline=REF, intervention={"group": conv["group_id"], "units": [0, 1], "op": "zero"})
    assert edit.status_code == 202 and edit.json()["intervention"]["node_id"] == "n0" and edit.json()["intervention"]["unit_axis"] == 1, edit.text
    assert wait(client, agent, f"/evaluations/{edit.json()['evaluation_id']}")["status"] == "succeeded"
    missing = evaluate_with(client, agent, REF, dataset, "gr7", baseline=REF, intervention={"group": "g_none", "units": [0], "op": "zero"})
    assert missing.status_code == 404 and groupnode.PRODUCERS


def test_a_person_resolves_an_intervention_id_from_the_trace(server, tmp_path):
    client, agent, human = server()
    client, agent, human, dataset, child = make_world(client, agent, human, tmp_path)
    edit = evaluate_with(client, agent, REF, dataset, "iv1", baseline=REF, intervention={"node_id": "n1", "units": [4, 1, 6], "op": "scale", "factor": 0.5})
    wait(client, agent, f"/evaluations/{edit.json()['evaluation_id']}")
    event = next(e for e in client.get(BASE + "/trace", headers=agent, params={"limit": 500}).json()["result"]["events"] if e.get("intervention_id"))
    assert event["units"] == 3 and "n1" not in str(event)  # the event says how many units, never which
    answer = client.post(BASE + "/interventions/resolve", headers=human, json={"ids": [event["intervention_id"], "iv_0000000000000000"]}).json()["result"]
    assert answer["interventions"] == {event["intervention_id"]: {"node_id": "n1", "unit_axis": 1, "units": [1, 4, 6], "op": "scale", "factor": 0.5}}
    assert answer["unknown"] == ["iv_0000000000000000"]
    assert client.post(BASE + "/interventions/resolve", headers=agent, json={"ids": []}).status_code == 403
