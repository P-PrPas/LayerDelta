"""The torch_export provider (v0.3-G6): user model code in an isolated worker, JSON-only results, opt-in, human-only."""

import json
import os
import subprocess
import time

import pytest
import torch
from fastapi.testclient import TestClient
from safetensors.torch import save_file
from torch import nn

from layerdelta import api, export_provider
from layerdelta.coupling import derive
from layerdelta.export_provider import TorchExportProvider, check_spec
from layerdelta.graph import bind
from layerdelta.model import ReferenceCNN
from layerdelta.providers import GraphBuildError
from layerdelta.tensors import open_tensors
from test_v2_api import BASE, REF, make_workspace  # noqa: F401

C = f"/checkpoints/{REF}"

MODEL = '''
import json, os, subprocess, sys, time
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4 * 6 * 6, 3)

    def forward(self, x):
        return self.fc(torch.relu(self.bn(self.conv(x))).flatten(1))


class Wrapper(nn.Module):
    """The part the exporter cannot trace is `post`; the backbone is exportable."""

    def __init__(self):
        super().__init__()
        self.backbone = Net()
        self.post = nn.Linear(3, 2)

    def forward(self, x):
        y = self.backbone(x)
        return [self.post(r) for r in y if r.sum() > 0]  # data-dependent control flow


def factory():
    return Net()


def wrapper():
    return Wrapper()


def not_a_module():
    return 5


def raises():
    raise RuntimeError("boom from user code")


def crashes():
    os._exit(3)


def hangs():
    open(os.path.join(os.path.dirname(__file__), "pid.txt"), "w").write(str(os.getpid()))
    time.sleep(600)


def grandchild():
    here = os.path.dirname(__file__)
    code = "import os, time; f = open('held.log', 'w'); time.sleep(600)"
    child = subprocess.Popen([sys.executable, "-c", code], stdin=subprocess.DEVNULL)  # inherits the scratch cwd
    open(os.path.join(here, "gpid.txt"), "w").write(str(child.pid))
    time.sleep(600)


def growing():
    with open(os.path.join(_out(), "ir.json"), "w") as f:
        f.write("0" * (5 << 20))
        f.flush()
        time.sleep(600)


def chatty():
    for _ in range(200000):
        print("x" * 100)
    return Net()


def env_probe():
    open(os.path.join(os.path.dirname(__file__), "env.json"), "w").write(json.dumps({
        "env": sorted(os.environ), "cwd": os.getcwd(), "stdin": sys.stdin is None or sys.stdin.read() == ""}))
    return Net()


def _out():
    return sys.argv[sys.argv.index("--output") + 1]


def garbage():
    open(os.path.join(_out(), "ir.json"), "w").write("this is not json")
    os._exit(0)


def nan_json():
    open(os.path.join(_out(), "ir.json"), "w").write('{"x": NaN}')
    os._exit(0)


def huge():
    open(os.path.join(_out(), "ir.json"), "w").write("[" + "0," * 5000 + "0]")
    os._exit(0)


def liar():
    from layerdelta.providers import build_graph
    ir = build_graph({"provider": "builtin", "model": "reference_cnn"})  # a perfectly valid graph, labelled builtin
    from layerdelta.graph import ir_json
    open(os.path.join(_out(), "ir.json"), "w").write(ir_json(ir))
    os._exit(0)


def forged():
    from layerdelta.providers import build_graph
    ir = build_graph({"provider": "builtin", "model": "reference_cnn"})
    ir["nodes"][0]["op"] = "aten.something.else"  # content no longer matches its hash
    from layerdelta.graph import ir_json
    open(os.path.join(_out(), "ir.json"), "w").write(ir_json(ir))
    os._exit(0)
'''


@pytest.fixture
def project(tmp_path):
    code = tmp_path / "proj"
    code.mkdir()
    (code / "usermodel.py").write_text(MODEL, encoding="utf-8")
    return code


def spec(project, entry="usermodel:factory", **extra):
    return {"provider": "torch_export", "entry": entry, "path": str(project),
            "example_inputs": [{"shape": [1, 1, 8, 8], "dtype": "F32"}], **extra}


def checkpoint_of(tmp_path, model, name="user.safetensors"):
    path = tmp_path / name
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, path)
    return open_tensors(path)


def provider(**kw):
    return TorchExportProvider(enabled=True, **kw)


def alive(pid):
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


# ------------------------------------------------------------------ a user model, end to end

def test_a_user_model_becomes_a_graph_that_binds_and_derives(project, tmp_path):
    ir = provider().build(spec(project))
    assert ir["provider"]["name"] == "torch_export" and ir["provider"]["version"].startswith("torch-")
    assert {p["name"] for p in ir["params"]} >= {"conv.weight", "bn.running_mean", "fc.weight"}
    torch.manual_seed(1)
    table = checkpoint_of(tmp_path, _load(project, "factory"))
    bound = bind(ir, table)
    assert bound.names("unbound") == [] and "bn.num_batches_tracked" in bound.names("unused")
    result = derive(ir, table)
    assert result.coverage["unknown"] == 0 and any(g["closed"] and g["size"] == 4 for g in result.groups)
    again = provider().build(spec(project))  # a second worker, the same graph
    assert again["ir_sha256"] == ir["ir_sha256"]


def _load(project, name):
    import importlib.util
    module_spec = importlib.util.spec_from_file_location("usermodel_local", project / "usermodel.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return getattr(module, name)()


def test_root_exports_one_submodule_with_names_that_bind_to_the_full_checkpoint(project, tmp_path):
    with pytest.raises(GraphBuildError) as info:  # the whole wrapper cannot be exported: data-dependent control flow
        provider().build(spec(project, "usermodel:wrapper"))
    assert info.value.code == "export_failed"
    ir = provider().build(spec(project, "usermodel:wrapper", root="backbone"))
    assert all(p["name"].startswith("backbone.") for p in ir["params"])
    assert {n["module_path"] for n in ir["nodes"]} >= {"backbone.conv", "backbone.bn", "backbone.fc"}
    table = checkpoint_of(tmp_path, _load(project, "wrapper"))
    bound = bind(ir, table)
    assert set(bound.names("unbound")) == {"post.weight", "post.bias"}  # the part that was not exported
    assert any(g["closed"] and any(m["tensor"] == "backbone.conv.weight" for m in g["members"]) for g in derive(ir, table).groups)
    err = pytest.raises(GraphBuildError, lambda: provider().build(spec(project, "usermodel:wrapper", root="nope")))
    assert err.value.code == "export_failed" and "nope" in err.value.message


# ------------------------------------------------------------------ a failing worker leaves no graph

@pytest.mark.parametrize("entry, code", [
    ("usermodel:raises", "export_failed"), ("usermodel:not_a_module", "export_failed"), ("nomodule:factory", "export_failed"),
    ("usermodel:nothing", "export_failed"), ("usermodel:crashes", "worker_crashed"), ("usermodel:garbage", "invalid_graph"),
    ("usermodel:nan_json", "invalid_graph"), ("usermodel:forged", "invalid_graph"), ("usermodel:liar", "provider_mismatch")])
def test_failures_have_stable_codes_and_no_result(project, entry, code):
    with pytest.raises(GraphBuildError) as info:
        provider().build(spec(project, entry))
    assert info.value.code == code, info.value.message
    assert len(info.value.message) <= 600


def test_the_error_message_comes_from_the_user_code_but_is_bounded(project):
    with pytest.raises(GraphBuildError) as info:
        provider().build(spec(project, "usermodel:raises"))
    assert "boom from user code" in info.value.message


def test_a_hanging_worker_is_killed_at_the_timeout(project):
    started = time.monotonic()
    with pytest.raises(GraphBuildError) as info:
        provider(timeout=12).build(spec(project, "usermodel:hangs"))
    assert info.value.code == "worker_timeout" and time.monotonic() - started < 60
    pid = int((project / "pid.txt").read_text())
    deadline = time.monotonic() + 10
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(.1)
    assert not alive(pid)


def test_a_timeout_kills_grandchildren_and_still_reports_a_timeout(project):
    with pytest.raises(GraphBuildError) as info:
        provider(timeout=15).build(spec(project, "usermodel:grandchild"))
    assert info.value.code == "worker_timeout", info.value.message
    pid = int((project / "gpid.txt").read_text())
    deadline = time.monotonic() + 10
    while alive(pid) and time.monotonic() < deadline:
        time.sleep(.1)
    assert not alive(pid)


def test_a_result_that_keeps_growing_is_stopped_before_the_timeout(project):
    started = time.monotonic()
    with pytest.raises(GraphBuildError) as info:
        provider(timeout=60, result_max=1000).build(spec(project, "usermodel:growing"))
    assert info.value.code == "result_too_large" and time.monotonic() - started < 50


def test_output_and_result_sizes_are_capped(project, monkeypatch):
    monkeypatch.setattr(export_provider, "LOG_MAX", 1 << 20)
    with pytest.raises(GraphBuildError) as info:
        provider().build(spec(project, "usermodel:chatty"))
    assert info.value.code == "worker_output"
    with pytest.raises(GraphBuildError) as info:
        provider(result_max=1000).build(spec(project, "usermodel:huge"))
    assert info.value.code == "result_too_large"


def test_the_worker_does_not_inherit_secrets_or_stdin(project, monkeypatch):
    monkeypatch.setenv("LAYERDELTA_SECRET", "s3cret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "x")
    provider().build(spec(project, "usermodel:env_probe"))
    seen = json.loads((project / "env.json").read_text())
    assert not [k for k in seen["env"] if "SECRET" in k or k.startswith("LAYERDELTA")] and seen["stdin"] is True
    assert "PYTHONPATH" in seen["env"] and str(project) not in seen["cwd"] and "ld-export-" in seen["cwd"]


# ------------------------------------------------------------------ the spec and the opt-in

def test_a_disabled_provider_never_starts_anything(project):
    with pytest.raises(GraphBuildError) as info:
        TorchExportProvider().build(spec(project, "usermodel:crashes"))
    assert info.value.code == "provider_disabled" and not (project / "pid.txt").exists()


def test_spec_validation(project):
    good = spec(project)
    assert check_spec(good)["entry"] == "usermodel:factory"
    for change in ({"entry": "os.system"}, {"entry": "a b:c"}, {"entry": "../x:y"}, {"path": "relative/dir"}, {"path": str(project / "missing")},
                   {"path": 5}, {"root": "a b"}, {"root": 5}, {"example_inputs": []}, {"example_inputs": "x"},
                   {"example_inputs": [{"shape": [0], "dtype": "F32"}]}, {"example_inputs": [{"shape": [1], "dtype": "F33"}]},
                   {"example_inputs": [{"shape": [1] * 9, "dtype": "F32"}]}, {"example_inputs": [{"shape": [1 << 20, 1 << 20], "dtype": "F32"}]},
                   {"example_inputs": [{"shape": [1 << 14, 1 << 14], "dtype": "F32"}] * 2},
                   {"example_inputs": [{"shape": [1], "dtype": "F32", "x": 1}]}, {"extra": 1}, {"provider": "builtin"}):
        bad = {**good, **change}
        with pytest.raises(GraphBuildError) as info:
            check_spec(bad)
        assert info.value.code == "invalid_spec", change
    with pytest.raises(GraphBuildError):
        check_spec({k: v for k, v in good.items() if k != "entry"})
    with pytest.raises(GraphBuildError):
        check_spec("usermodel:factory")


# ------------------------------------------------------------------ over HTTP

@pytest.fixture
def server(tmp_path, monkeypatch, project):
    workspace = make_workspace(tmp_path, monkeypatch)
    opened = []

    def make(allow):
        creds = api.Credentials("demo", weights_read=True, weights_write=True)
        client = TestClient(api.create_app(workspace, creds, allow_user_code=allow, export_timeout=60), base_url="http://127.0.0.1")
        client.__enter__()
        opened.append(client)
        csrf = client.post("/api/v1/session", json={"ticket": creds.ticket}).json()["csrf"]
        return client, {"Authorization": "Bearer " + creds.agent}, {"X-CSRF-Token": csrf}

    yield make
    for client in opened:
        client.__exit__(None, None, None)


def wait_attach(client, human, attach_id):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        record = client.get(f"{BASE}/graph-attachments/{attach_id}", headers=human).json()
        if record["status"] in ("succeeded", "failed"):
            return record
        time.sleep(.1)
    pytest.fail("attach did not finish")


USER_REFERENCE = "from layerdelta.model import ReferenceCNN\n\n\ndef factory():\n    return ReferenceCNN()\n"


def test_user_code_is_off_unless_the_project_opts_in(server, project):
    client, agent, human = server(False)
    card = client.get(BASE + C + "/graph", headers=agent).json()["result"]["providers"]
    torch_export = next(p for p in card if p["name"] == "torch_export")
    assert torch_export["enabled"] is False and "--allow-user-code" in torch_export["reason"]
    body = {"provider": "torch_export", "entry": "usermodel:crashes", "path": str(project), "example_inputs": [{"shape": [1, 1, 28, 28], "dtype": "F32"}]}
    response = client.post(BASE + C + "/graphs", headers=human, json=body)
    assert response.status_code == 422 and response.json()["error"]["code"] == "provider_disabled"
    assert not (project / "pid.txt").exists()


def test_a_human_attaches_a_user_model_and_an_agent_cannot(server, project):
    (project / "ref.py").write_text(USER_REFERENCE, encoding="utf-8")
    client, agent, human = server(True)
    assert next(p for p in client.get(BASE + C + "/graph", headers=agent).json()["result"]["providers"]
                if p["name"] == "torch_export")["enabled"] is True
    body = {"provider": "torch_export", "entry": "ref:factory", "path": str(project), "example_inputs": [{"shape": [1, 1, 28, 28], "dtype": "F32"}]}
    assert client.post(BASE + C + "/graphs", headers=agent, json=body).status_code == 403
    started = client.post(BASE + C + "/graphs", headers=human, json=body)
    assert started.status_code == 202
    done = wait_attach(client, human, started.json()["attach_id"])
    assert done["status"] == "succeeded", done
    described = client.get(BASE + C + "/describe", headers=agent).json()["result"]
    assert described["graph"]["provider"] == "torch_export" and described["coverage"]["derived"] > 0
    groups = client.get(BASE + C + "/groups?limit=500&closed=true", headers=agent).json()["result"]["groups"]
    assert groups and any(g["size"] > 1 for g in groups)


def test_a_failing_worker_attaches_nothing(server, project):
    client, agent, human = server(True)
    for entry, code in (("usermodel:crashes", "worker_crashed"), ("usermodel:raises", "export_failed"), ("usermodel:liar", "provider_mismatch")):
        body = {"provider": "torch_export", "entry": entry, "path": str(project), "example_inputs": [{"shape": [1, 1, 28, 28], "dtype": "F32"}]}
        done = wait_attach(client, human, client.post(BASE + C + "/graphs", headers=human, json=body).json()["attach_id"])
        assert done["status"] == "failed" and done["error"]["code"] == code, done
    assert "graph" not in client.get(BASE + C + "/describe", headers=agent).json()["result"]
    assert client.get(BASE + C + "/graph", headers=agent).json()["result"]["graph"] is None
    bad = client.post(BASE + C + "/graphs", headers=human, json={"provider": "torch_export", "entry": "x", "path": str(project),
                                                                 "example_inputs": []})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "invalid_spec"
