"""Egress classes (v0.4-S4a, acceptance T6): what the backend releases to an agent token.

Class 0 metadata, class 1 plus aggregates (weights:read), class 2 plus raw values (weights:values, split out of weights:read).
"""

import json
import random

import pytest
from fastapi.testclient import TestClient
from safetensors.torch import load_file

from layerdelta import api
from test_v2_api import BASE, BIAS, PROGRAM, REF, make_workspace  # noqa: F401

C = f"/checkpoints/{REF}"


@pytest.fixture
def server(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path, monkeypatch)
    opened = []

    def make(**caps):
        creds = api.Credentials("demo", **{"weights_read": True, "weights_write": True, **caps})
        client = TestClient(api.create_app(workspace, creds, quota=500), base_url="http://127.0.0.1")
        client.__enter__()
        opened.append(client)
        csrf = client.post("/api/v1/session", json={"ticket": creds.ticket}).json()["csrf"]
        return client, {"Authorization": "Bearer " + creds.agent}, {"X-CSRF-Token": csrf}, creds

    make.workspace = workspace
    yield make
    for client in opened:
        client.__exit__(None, None, None)


def post(client, headers, path, body):
    return client.post(BASE + path, headers=headers, json=body)


def test_the_classes_and_the_capability_split():
    assert api.Credentials("p").egress_class == 0
    assert api.Credentials("p", weights_read=True).egress_class == 2  # v0.2 compatibility: a weights:read token keeps values
    assert api.Credentials("p", weights_read=True, weights_values=False).egress_class == 1
    assert "weights:values" not in api.Credentials("p", weights_read=True, weights_values=False).capabilities
    assert "weights:values" in api.Credentials("p", weights_read=True, weights_values=True).capabilities
    with pytest.raises(ValueError, match="weights_values requires weights_read"):
        api.Credentials("p", weights_values=True)


def test_class_one_gets_aggregates_and_is_refused_values(server):
    client, agent, human, creds = server(weights_values=False)
    info = client.get(BASE + "/agent-session", headers=agent).json()
    assert info["egress_class"] == 1 and "weights:values" not in info["capabilities"] and "weights:read" in info["capabilities"]
    for path, body in ((C + "/read-tensor", {"selector": BIAS}),
                       (C + "/diff", {"other": REF, "selector": BIAS, "values": True}),
                       (C + "/read-tensor", {"selector": {"group": "g_none"}})):
        refused = post(client, agent, path, body)
        assert refused.status_code == 403 and refused.json()["error"]["code"] == "forbidden", (path, body)
        assert "weights:values" in refused.json()["error"]["message"]
    for path, body in ((C + "/stats", {"selector": BIAS}), (C + "/query", {"limit": 3}), (C + "/diff", {"other": REF}),
                       (C + "/diff", {"other": REF, "selector": BIAS}), (C + "/plan", PROGRAM), (C + "/validate", PROGRAM)):
        assert post(client, agent, path, body).status_code == 200, (path, body)
    assert client.get(BASE + C + "/describe", headers=agent).status_code == 200
    assert post(client, human, C + "/read-tensor", {"selector": BIAS}).status_code == 200  # a person always has every class


def test_class_two_reads_values_and_class_zero_reads_nothing(server):
    client, agent, human, creds = server()
    assert client.get(BASE + "/agent-session", headers=agent).json()["egress_class"] == 2
    assert post(client, agent, C + "/read-tensor", {"selector": BIAS}).status_code == 200
    assert post(client, agent, C + "/diff", {"other": REF, "selector": BIAS, "values": True}).status_code == 200


def test_class_zero_agent_cannot_read_weights_at_all(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path, monkeypatch)
    creds = api.Credentials("demo")
    with TestClient(api.create_app(workspace, creds), base_url="http://127.0.0.1") as client:
        agent = {"Authorization": "Bearer " + creds.agent}
        assert client.get(BASE + "/agent-session", headers=agent).json()["egress_class"] == 0
        for path, body in ((C + "/read-tensor", {"selector": BIAS}), (C + "/stats", {"selector": BIAS})):
            assert client.post(BASE + path, headers=agent, json=body).status_code == 403
        assert client.get(BASE + C + "/describe", headers=agent).status_code == 200


# ------------------------------------------------------------------ the property: no element array leaves a class-1 session

def runs(state):
    """Every tensor as a flat list of its float32 values, to look for them in a result."""
    return {name: [float(x) for x in tensor.reshape(-1).tolist()] for name, tensor in state.items() if tensor.is_floating_point()}


def contains_elements(value, flats, minimum=3):
    """Is there a list of at least `minimum` numbers in `value` that is a contiguous run of some tensor's elements?"""
    if isinstance(value, dict):
        return any(contains_elements(v, flats, minimum) for v in value.values())
    if not isinstance(value, list):
        return False
    numbers = [v for v in value if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(numbers) >= minimum and len(numbers) == len(value):
        for flat in flats.values():
            for start in range(0, len(flat) - len(numbers) + 1):
                if all(abs(a - b) <= 1e-6 * max(1.0, abs(b)) for a, b in zip(numbers, flat[start:start + len(numbers)])):
                    return True
    return any(contains_elements(v, flats, minimum) for v in value)


def fuzz_calls(rng, names, shapes):
    """Seeded random calls over every pure tool that takes arguments (valid, borderline and invalid ones)."""
    for _ in range(120):
        name = rng.choice(names)
        shape = shapes[name]
        index = {str(a): rng.choice([[rng.randrange(n)], {"start": 0, "stop": rng.randrange(1, n + 1)}])
                 for a, n in enumerate(shape) if rng.random() < .5}
        selector = {"tensor": name, **({"index": index} if index else {})}
        kind = rng.choice(["stats", "stats_units", "diff", "diff_units", "query", "plan", "describe"])
        if kind == "stats":
            yield "POST", C + "/stats", {"selector": selector, "bins": rng.choice([None, 4, 16])}
        elif kind == "stats_units":
            axes = rng.sample(range(len(shape)), rng.randrange(0, len(shape) + 1))
            yield "POST", C + "/stats", {"selector": selector, "reduce_over_axes": axes, "limit": rng.choice([5, 64])}
        elif kind == "diff":
            yield "POST", C + "/diff", {"other": REF, "selector": selector, "include_range": rng.random() < .5}
        elif kind == "diff_units":
            yield "POST", C + "/diff", {"other": REF, "unit_axis": rng.choice([None, 0, 1]), "include_same": True}
        elif kind == "query":
            yield "POST", C + "/query", {"limit": rng.choice([1, 5, 50]), "name_glob": rng.choice([None, "*.weight", "*.bias"]),
                                         "where": rng.choice([None, {"stat": "mean", "axis": 0, "op": ">", "value": rng.uniform(-1, 1)}])}
        elif kind == "plan":
            op = rng.choice([{"op": "scale", "args": {"k": 2.0}}, {"op": "zero", "args": {}}, {"op": "add", "args": {"value": .5}}])
            yield "POST", C + "/plan", {"program": [{**op, "select": selector}], "hypothesis": "fuzz"}
        else:
            yield "GET", C + "/describe", {"limit": rng.choice([1, 3, 200])}


def test_no_element_array_in_any_class_one_result(server):
    client, agent, human, creds = server(weights_values=False)
    state = load_file(server.workspace / "reference" / "model.safetensors")
    flats = runs(state)
    shapes = {name: list(t.shape) for name, t in state.items() if t.is_floating_point() and t.ndim > 0}
    served = leaked = 0
    for method, path, body in fuzz_calls(random.Random(7), sorted(shapes), shapes):
        response = client.get(BASE + path, headers=agent, params=body) if method == "GET" else post(client, agent, path, body)
        assert response.status_code < 500, (path, body, response.text)
        served += response.status_code == 200
        leaked += contains_elements(response.json(), flats)
    assert served > 40 and leaked == 0
    # the detector sees what a class-2 session returns, so a passing run means something
    client2 = post(client, human, C + "/read-tensor", {"selector": {"tensor": "classifier.weight"}, "max_elems": 20}).json()
    assert contains_elements(client2, flats)


def test_known_limit_aggregates_over_single_elements_are_values(server):
    """Class 1 releases aggregates, not secrecy. A statistic over ONE element is that element, so per-unit `stats` of a tensor whose
    units have one element hands the values over unit by unit. The backend does not stop this yet (open design question on the
    S4a PR: a minimum aggregate size). This test records the behaviour so a fix has to change it on purpose."""
    client, agent, human, creds = server(weights_values=False)
    state = load_file(server.workspace / "reference" / "model.safetensors")
    units = post(client, agent, C + "/stats", {"selector": BIAS, "reduce_over_axes": []}).json()["result"]["units"]
    assert [u["mean"] for u in units] == pytest.approx(state["classifier.bias"].tolist())
