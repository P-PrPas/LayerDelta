"""The onnx provider (v0.3-G7): an .onnx file bound to a checkpoint by name, shape and bytes, with honest coverage.

Needs the optional `onnx` package (`pip install layerdelta[onnx]`); the module is skipped without it.
"""

import time
from collections import Counter

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import load_file, save_file

pytest.importorskip("onnx")

from layerdelta import api, coupling, onnx_provider  # noqa: E402
from layerdelta.graph import bind  # noqa: E402
from layerdelta.model import ReferenceCNN  # noqa: E402
from layerdelta.providers import GraphBuildError  # noqa: E402
from layerdelta.tensors import open_tensors  # noqa: E402
from test_corpus import ResNetLike  # noqa: E402
from test_v2_api import BASE, REF, make_workspace  # noqa: E402,F401

C = f"/checkpoints/{REF}"


def trained(model):
    """Random norm parameters and statistics, so no two initializers are equal (torch would deduplicate equal ones)."""
    torch.manual_seed(3)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.uniform_(.5, 1.5), m.bias.data.normal_(), m.running_mean.normal_(), m.running_var.uniform_(.5, 1.5)
    return model.eval()


@pytest.fixture
def export(tmp_path):
    def make(model=None, shape=(1, 3, 16, 16), fold=False, **kw):
        model = model or trained(ResNetLike())
        path = tmp_path / "model.onnx"
        torch.onnx.export(model, torch.zeros(shape), str(path), do_constant_folding=fold, dynamo=False, opset_version=17,
                          input_names=["x"], output_names=["y"], **kw)
        save_file({k: v.contiguous() for k, v in model.state_dict().items()}, tmp_path / "model.safetensors")
        return model, path, open_tensors(tmp_path / "model.safetensors")
    return make


def spec(path, **extra):
    return {"provider": "onnx", "path": str(path), **extra}


def reasons(ir):
    return Counter(n["kwargs"]["reason"]["str"] for n in ir["nodes"] if n["op"] == "onnx.Initializer")


def test_an_unfolded_export_binds_every_weight_by_name_shape_and_bytes(export):
    model, path, table = export()
    ir = onnx_provider.build(spec(path), table)
    bound = bind(ir, table)
    assert ir["provider"]["name"] == "onnx" and ir["provider"]["version"] == "onnx-ir8-opset17" and reasons(ir) == {}
    assert sorted(bound.unbound) == sorted(n for n in table.tensors if n.endswith("num_batches_tracked"))  # ONNX never reads them
    assert set(bound.bound) == set(table.tensors) - set(bound.unbound)
    conv = next(n for n in ir["nodes"] if n["op"] == "onnx.Conv")
    assert conv["module_path"] == "conv1" and conv["kwargs"]["group"] == 1 and len(conv["args"]) == 2  # a Conv without a bias has two inputs
    assert onnx_provider.build(spec(path), table)["ir_sha256"] == ir["ir_sha256"]


def test_derivation_covers_the_known_ops_and_reports_the_rest_as_opaque(export):
    _, path, table = export()
    ir = onnx_provider.build(spec(path), table)
    derived = coupling.derive(ir, table)
    assert derived.warnings == [] and derived.groups
    rules = {r["rule"] for r in derived.node_rules.values()}
    assert {"onnx.Conv", "onnx.BatchNormalization", "onnx.Gemm", "onnx.Add"} <= rules
    opaque = {n["op"] for n in ir["nodes"] if derived.node_rules[n["id"]]["opaque"]}
    assert opaque == {"onnx.ReduceMean"}  # an op without a rule is a boundary, not a guess
    assert any(g["closed"] and g["size"] > 1 for g in derived.to_dict()["groups"])


def test_folded_weights_are_reported_as_unbound_with_a_reason(export):
    model, path, table = export(fold=True)
    ir = onnx_provider.build(spec(path), table)
    assert "onnx.BatchNormalization" not in {n["op"] for n in ir["nodes"]} and reasons(ir)["not_in_checkpoint"] > 0
    bound = bind(ir, table)
    assert all(n in bound.unbound for n in table.tensors if ".bn" in n or n.startswith("bn") or "downsample.1" in n)
    assert coupling.derive(ir, table).coverage  # derivation still runs; coverage tells how much was understood


def test_a_changed_value_shape_or_dtype_is_never_bound(export, tmp_path):
    model, path, table = export()
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    state["conv1.weight"] = state["conv1.weight"] + 1
    state["fc.bias"] = torch.zeros(11)
    save_file(state, tmp_path / "edited.safetensors")
    ir = onnx_provider.build(spec(path), open_tensors(tmp_path / "edited.safetensors"))
    assert reasons(ir) == {"value_differs": 1, "shape_differs": 1}
    assert "conv1.weight" not in {p["name"] for p in ir["params"]} and "layer1.0.conv1.weight" in {p["name"] for p in ir["params"]}
    save_file({k: v.to(torch.float16) for k, v in state.items() if v.is_floating_point()}, tmp_path / "half.safetensors")
    half = onnx_provider.build(spec(path), open_tensors(tmp_path / "half.safetensors"))
    assert half["params"] == [] and set(reasons(half)) <= {"dtype_differs", "not_in_checkpoint", "shape_differs"} and reasons(half)["dtype_differs"] > 0


def test_dynamic_dimensions_take_the_spec_value_and_other_failures_have_codes(export, tmp_path):
    model, path, table = export(dynamic_axes={"x": {0: "batch"}})
    assert onnx_provider.build(spec(path), table)["inputs"][0]["shape"] == [1, 3, 16, 16]
    assert onnx_provider.build(spec(path, dynamic_dim=4), table)["inputs"][0]["shape"] == [4, 3, 16, 16]
    junk = tmp_path / "junk.onnx"
    junk.write_bytes(b"this is not a protobuf")
    with pytest.raises(GraphBuildError) as info:
        onnx_provider.build(spec(junk), table)
    assert info.value.code == "invalid_model"


def test_spec_validation_and_the_missing_package(export, tmp_path, monkeypatch):
    _, path, table = export()
    assert onnx_provider.check_spec(spec(path))["dynamic_dim"] == 1
    for bad in ({"path": "relative.onnx"}, {"path": str(tmp_path / "missing.onnx")}, {"path": str(tmp_path / "model.safetensors")},
                {"path": 5}, {"dynamic_dim": 0}, {"dynamic_dim": True}, {"dynamic_dim": 5000}, {"extra": 1}, {"provider": "builtin"}):
        with pytest.raises(GraphBuildError) as info:
            onnx_provider.check_spec({**spec(path), **bad})
        assert info.value.code == "invalid_spec", bad
    monkeypatch.setattr(onnx_provider, "_onnx", lambda: None)
    assert not onnx_provider.available() and "pip install" in onnx_provider.reason_unavailable()
    with pytest.raises(GraphBuildError) as info:
        onnx_provider.build(spec(path), table)
    assert info.value.code == "provider_disabled"


# ------------------------------------------------------------------ over HTTP

def test_a_human_attaches_an_onnx_model_and_an_agent_cannot(tmp_path, monkeypatch):
    workspace = make_workspace(tmp_path, monkeypatch)
    model = ReferenceCNN().eval()
    model.load_state_dict(load_file(workspace / "reference" / "model.safetensors"))
    path = tmp_path / "ref.onnx"
    torch.onnx.export(model, torch.zeros(1, 1, 28, 28), str(path), dynamo=False, opset_version=17, input_names=["x"])
    creds = api.Credentials("demo", weights_read=True, weights_write=True)
    with TestClient(api.create_app(workspace, creds), base_url="http://127.0.0.1") as client:
        csrf = client.post("/api/v1/session", json={"ticket": creds.ticket}).json()["csrf"]
        agent, human = {"Authorization": "Bearer " + creds.agent}, {"X-CSRF-Token": csrf}
        card = client.get(BASE + C + "/graph", headers=agent).json()["result"]["providers"]
        assert next(p for p in card if p["name"] == "onnx")["enabled"] is True
        body = {"provider": "onnx", "path": str(path)}
        assert client.post(BASE + C + "/graphs", headers=agent, json=body).status_code == 403
        bad = client.post(BASE + C + "/graphs", headers=human, json={"provider": "onnx", "path": "relative.onnx"})
        assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_spec"
        started = client.post(BASE + C + "/graphs", headers=human, json=body)
        assert started.status_code == 202
        deadline, done = time.monotonic() + 120, {}
        while time.monotonic() < deadline and done.get("status") not in ("succeeded", "failed"):
            time.sleep(.1)
            done = client.get(f"{BASE}/graph-attachments/{started.json()['attach_id']}", headers=human).json()
        assert done["status"] == "succeeded", done
        described = client.get(BASE + C + "/describe", headers=agent).json()["result"]
        assert described["graph"]["provider"] == "onnx" and described["coverage"]["derived"] > 0
        nodes = client.get(BASE + C + "/graph?op=onnx.Conv", headers=agent).json()["result"]
        assert nodes["graph"]["provider"] == "onnx"
