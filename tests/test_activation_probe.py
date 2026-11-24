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

    def make(budgets=None, **caps):
        creds = api.Credentials("demo", **{"weights_read": True, "weights_write": True, "eval_run": True, **caps})
        client = TestClient(api.create_app(workspace, creds, quota=50, budgets=budgets or {}), base_url="http://127.0.0.1")
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
