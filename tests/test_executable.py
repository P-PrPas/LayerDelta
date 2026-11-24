"""Executable artifacts (v0.5-E1, acceptance V4 and part of V9): a forward pass of an attached graph with a CHILD checkpoint's weights.

Library level: the graph store, `executable.capture` at attach, `executable.forward` in the isolated worker. The ONNX tests need the
optional `layerdelta[eval-onnx]` extra and are skipped without it.
"""

import importlib.util
import textwrap

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from layerdelta import executable, providers
from layerdelta.checkpoints import Catalog
from layerdelta.export_provider import TorchExportProvider
from layerdelta.graph import GraphStore
from layerdelta.model import ReferenceCNN
from layerdelta.tensors import open_tensors

ORT = importlib.util.find_spec("onnxruntime") is not None and importlib.util.find_spec("onnx") is not None

NET = textwrap.dedent('''
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 4, 3)
            self.bn = nn.BatchNorm2d(4)
            self.fc = nn.Linear(4 * 6 * 6, 3)
            self.register_buffer("scale", torch.tensor([2.0]), persistent=True)

        def forward(self, x):
            return self.fc(torch.relu(self.bn(self.conv(x))).flatten(1)) * self.scale

    def factory():
        return Net()
''')


class Net(nn.Module):  # the same model, in this process: the eager reference the worker's answer is compared with
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4 * 6 * 6, 3)
        self.register_buffer("scale", torch.tensor([2.0]), persistent=True)

    def forward(self, x):
        return self.fc(torch.relu(self.bn(self.conv(x))).flatten(1)) * self.scale


def randomised(model):
    torch.manual_seed(5)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.uniform_(.5, 1.5), m.bias.data.normal_(), m.running_mean.normal_(), m.running_var.uniform_(.5, 1.5)
    return model.eval()


def save(model, path):
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, path)
    return open_tensors(path)


def childof(model, path, factor=3.0, key="conv.weight"):
    """The same model with its conv weight scaled: what an apply produces."""
    state = {k: v.clone().contiguous() for k, v in model.state_dict().items()}
    state[key] *= factor
    save_file(state, path)
    return state


@pytest.fixture
def ws(tmp_path):
    catalog = Catalog(tmp_path / "ws")
    catalog.start()
    return tmp_path / "ws", GraphStore(catalog)


def eager(state, x, model_class=Net):
    model = model_class()
    model.load_state_dict(state)
    with torch.no_grad():
        return model.eval()(torch.from_numpy(x)).numpy()


def attach_user_net(tmp_path, ws, batch=2):
    workspace, store = ws
    code = tmp_path / "code"
    code.mkdir()
    (code / "usernet.py").write_text(NET, encoding="utf-8")
    model = randomised(Net())
    table = save(model, tmp_path / "parent.safetensors")
    spec = {"provider": "torch_export", "entry": "usernet:factory", "path": str(code), "example_inputs": [{"shape": [batch, 1, 8, 8], "dtype": "F32"}],
            "root": None}
    keep = tmp_path / "keep"
    keep.mkdir()
    ir = TorchExportProvider(enabled=True).build(spec, artifact_dir=keep)
    store.put(ir, table, provider="torch_export")
    return model, ir, executable.capture(store, ir, spec, table, workspace, keep)


def test_builtin_runs_with_a_childs_weights(tmp_path, ws):
    workspace, store = ws
    model = ReferenceCNN().eval()
    ir = providers.build_graph({"provider": "builtin", "model": "reference_cnn"})
    parent = save(model, tmp_path / "parent.safetensors")
    store.put(ir, parent, provider="builtin")
    record = executable.capture(store, ir, {"provider": "builtin", "model": "reference_cnn"}, parent, workspace)
    assert record["kind"] == "builtin" and record["artifact"] is None and record["batch"] == ir["inputs"][0]["shape"][0]
    shape = ir["inputs"][0]["shape"]
    x = np.random.default_rng(1).standard_normal((5, *shape[1:])).astype(np.float32)  # 5 samples, the program's batch is different: padding
    state = childof(model, tmp_path / "child.safetensors", key="features.block1.conv.weight")
    got = executable.forward(workspace, store, ir, tmp_path / "child.safetensors", [x])["outputs"][0]
    assert got.shape == (5, 10)
    np.testing.assert_allclose(got, eager(state, x, ReferenceCNN), rtol=1e-5, atol=1e-5)
    base = executable.forward(workspace, store, ir, tmp_path / "parent.safetensors", [x])["outputs"][0]
    assert not np.allclose(base, got)  # the child's weights are what ran, not the parent's
    again = executable.forward(workspace, store, ir, tmp_path / "child.safetensors", [x])["outputs"][0]
    assert again.tobytes() == got.tobytes()  # V9: two CPU runs are identical


def test_torch_export_runs_a_saved_program_with_a_childs_weights(tmp_path, ws):
    workspace, store = ws
    model, ir, record = attach_user_net(tmp_path, ws)
    assert record["kind"] == "torch_export" and record["artifact"].endswith(".pt2") and record["batch"] == 2
    assert (workspace / record["artifact"]).stem == record["sha256"]
    x = np.random.default_rng(2).standard_normal((5, 1, 8, 8)).astype(np.float32)
    state = childof(model, tmp_path / "child.safetensors")
    ran = executable.forward(workspace, store, ir, tmp_path / "child.safetensors", [x])
    np.testing.assert_allclose(ran["outputs"][0], eager(state, x), rtol=1e-4, atol=1e-4)
    assert ran["executable"]["sha256"] == record["sha256"] and ran["executable"]["fingerprint"] == executable.fingerprint(record)
    assert not np.allclose(ran["outputs"][0], eager(model.state_dict(), x), atol=1e-3)  # not the parent's constants and weights


def test_a_changed_program_file_is_refused_before_it_is_loaded(tmp_path, ws):
    workspace, store = ws
    model, ir, record = attach_user_net(tmp_path, ws)
    path = workspace / record["artifact"]
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    childof(model, tmp_path / "child.safetensors")
    with pytest.raises(executable.ForwardError) as raised:
        executable.forward(workspace, store, ir, tmp_path / "child.safetensors", [np.zeros((1, 1, 8, 8), np.float32)])
    assert raised.value.code == "executable_changed"


def test_a_checkpoint_that_does_not_fit_is_refused_before_any_process_starts(tmp_path, ws, monkeypatch):
    workspace, store = ws
    model, ir, record = attach_user_net(tmp_path, ws)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    state["fc.weight"] = torch.zeros(2, 4 * 6 * 6)  # reshaped
    save_file(state, tmp_path / "bad.safetensors")
    del state["conv.weight"]
    save_file(state, tmp_path / "missing.safetensors")
    monkeypatch.setattr(executable, "popen", lambda *a, **k: pytest.fail("a worker was started"))
    for name in ("bad", "missing"):
        with pytest.raises(executable.ForwardError) as raised:
            executable.forward(workspace, store, ir, tmp_path / f"{name}.safetensors", [np.zeros((1, 1, 8, 8), np.float32)])
        assert raised.value.code == "executable_mismatch", name


def test_inputs_are_checked_and_a_graph_without_an_executable_is_refused(tmp_path, ws):
    workspace, store = ws
    model, ir, record = attach_user_net(tmp_path, ws)
    childof(model, tmp_path / "child.safetensors")
    for bad in ([np.zeros((1, 1, 9, 8), np.float32)], [np.zeros((0, 1, 8, 8), np.float32)], []):
        with pytest.raises(executable.ForwardError) as raised:
            executable.forward(workspace, store, ir, tmp_path / "child.safetensors", bad)
        assert raised.value.code == "invalid_argument"
    other = dict(ir, ir_sha256="f" * 64)
    with pytest.raises(executable.ForwardError) as raised:
        executable.forward(workspace, store, other, tmp_path / "child.safetensors", [np.zeros((1, 1, 8, 8), np.float32)])
    assert raised.value.code == "executable_missing"


@pytest.mark.skipif(not ORT, reason="needs the eval-onnx extra")
def test_onnx_runs_with_a_childs_weights_and_refuses_what_it_cannot_bind(tmp_path, ws):
    from layerdelta import onnx_provider
    workspace, store = ws
    model = randomised(Net())
    onnx_path = tmp_path / "net.onnx"
    torch.onnx.export(model, torch.zeros(2, 1, 8, 8), str(onnx_path), do_constant_folding=False, dynamo=False, opset_version=17,
                      input_names=["x"], output_names=["y"], training=torch.onnx.TrainingMode.PRESERVE)
    table = save(model, tmp_path / "parent.safetensors")
    spec = {"provider": "onnx", "path": str(onnx_path)}
    ir = onnx_provider.build(spec, table)
    store.put(ir, table, provider="onnx")
    record = executable.capture(store, ir, spec, table, workspace)
    assert record["kind"] == "onnx" and record["artifact"].endswith(".onnx") and record["batch"] == 2
    x = np.random.default_rng(3).standard_normal((5, 1, 8, 8)).astype(np.float32)
    state = childof(model, tmp_path / "child.safetensors")
    got = executable.forward(workspace, store, ir, tmp_path / "child.safetensors", [x])["outputs"][0]
    np.testing.assert_allclose(got, eager(state, x), rtol=1e-4, atol=1e-4)
    assert not np.allclose(got, eager(model.state_dict(), x), atol=1e-3)

    folded = tmp_path / "folded.onnx"  # BatchNorm folded into the conv: its weights no longer carry checkpoint names
    torch.onnx.export(model, torch.zeros(2, 1, 8, 8), str(folded), do_constant_folding=True, dynamo=False, opset_version=17,
                      input_names=["x"], output_names=["y"])
    fspec = {"provider": "onnx", "path": str(folded)}
    fir = onnx_provider.build(fspec, table)
    store.put(fir, table, provider="onnx")
    executable.capture(store, fir, fspec, table, workspace)
    with pytest.raises(executable.ForwardError) as raised:
        executable.forward(workspace, store, fir, tmp_path / "child.safetensors", [x])
    assert raised.value.code == "executable_mismatch" and "do_constant_folding=False" in raised.value.message


def test_the_http_attach_records_the_executable(tmp_path, monkeypatch):
    import time
    from fastapi.testclient import TestClient
    from layerdelta import api
    from test_v2_api import BASE, REF, make_workspace
    workspace = make_workspace(tmp_path, monkeypatch)
    creds = api.Credentials("demo")
    with TestClient(api.create_app(workspace, creds), base_url="http://127.0.0.1") as client:
        csrf = client.post("/api/v1/session", json={"ticket": creds.ticket}).json()["csrf"]
        started = client.post(BASE + f"/checkpoints/{REF}/graphs", headers={"X-CSRF-Token": csrf},
                              json={"provider": "builtin", "model": "reference_cnn"})
        for _ in range(300):
            done = client.get(f"{BASE}/graph-attachments/{started.json()['attach_id']}", headers={"X-CSRF-Token": csrf}).json()
            if done["status"] in ("succeeded", "failed"):
                break
            time.sleep(.1)
        assert done["status"] == "succeeded", done
    store = GraphStore(Catalog(workspace))
    sha = done["graph"]["ir_sha256"]
    assert store.executable_for(sha)["kind"] == "builtin" and store.executable_for(sha)["model"] == "reference_cnn"
    assert not list((workspace / "staging").glob("attach-*"))  # nothing is left behind


def test_only_a_cast_that_loses_nothing_is_made():
    from layerdelta.forward_worker import _widening
    f16, bf16, f32, f64 = torch.float16, torch.bfloat16, torch.float32, torch.float64
    assert _widening(f32, f32) and _widening(f16, f32) and _widening(bf16, f32) and _widening(f32, f64)
    assert not _widening(f32, f16) and not _widening(f32, bf16) and not _widening(f16, bf16) and not _widening(bf16, f16)
    assert not _widening(torch.int64, f32) and _widening(torch.int64, torch.int64)
