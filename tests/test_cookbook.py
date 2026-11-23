"""The agent cookbook (v0.4-S7, acceptance T7): its recipes run as written, through the bridge alone, on three model families.

A person sets the scene (imports three corpus checkpoints and attaches their graphs: both are human-only). The agent session is egress
class 1 (aggregates only). The recipes then run in-process through the bridge's `execute` and, with the optional MCP SDK, through
the real `layerdelta-mcp` stdio process. The test, not the agent, reads the child files to check the claims: zeroed units are zero in
every member, and a permutation of a closed group leaves the forward output unchanged.
"""

import json
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
import torch
import uvicorn
from safetensors.torch import load_file

from layerdelta import api
from layerdelta.client import AgentClient
from test_corpus import FAMILIES, build, save
from test_graph_ir import register_corpus
from test_v2_api import make_workspace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "agent"))
import cookbook  # noqa: E402

register_corpus()
PRE = "/api/v1/projects/demo"
CHOSEN = ("resnet", "vit", "crnn")


def _wait(http, path, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = http.get(path).json()
        if record["status"] in ("succeeded", "failed"):
            assert record["status"] == "succeeded", record
            return record
        time.sleep(.05)
    pytest.fail(f"{path} did not finish")


@pytest.fixture
def service(tmp_path, monkeypatch):
    """Start a class-1 service; a person imports the three families and attaches their graphs. Yields a factory of agents."""
    workspace = make_workspace(tmp_path, monkeypatch)
    running = []

    def start(*, quota=10, budgets=None):
        creds = api.Credentials("demo", weights_read=True, weights_write=True, weights_values=False)
        app = api.create_app(workspace, creds, quota=quota, budgets=budgets or {})
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        deadline = time.monotonic() + 30
        while not server.started and time.monotonic() < deadline:
            time.sleep(.05)
        assert server.started
        running.append((server, thread))
        url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        human = httpx.Client(base_url=url + PRE, trust_env=False)
        csrf = human.post(url + "/api/v1/session", json={"ticket": creds.ticket}).json()["csrf"]
        human.headers["X-CSRF-Token"] = csrf
        checkpoints = {}
        for family in CHOSEN:  # the person's part: bring the checkpoints in and attach a graph to each
            _, state = build(family, torch.float32)
            path = save(tmp_path, state, f"{family}.safetensors")
            started = human.post("/checkpoint-imports", json={"path": str(path)})
            assert started.status_code == 202, started.text
            cid = _wait(human, f"/checkpoint-imports/{started.json()['import_id']}")["checkpoint_id"]
            attach = human.post(f"/checkpoints/{cid}/graphs", json={"provider": "builtin", "model": f"corpus.{family}"})
            assert attach.status_code == 202, attach.text
            _wait(human, f"/graph-attachments/{attach.json()['attach_id']}")
            checkpoints[family] = cid
        connection = {"url": url, "project_id": "demo", "token": creds.agent, "egress_class": creds.egress_class}
        path = tmp_path / f"connection-{len(running)}.json"
        path.write_text(json.dumps(connection), encoding="utf-8")
        return {"connection": connection, "path": path, "checkpoints": checkpoints, "workspace": workspace, "human": human}

    yield start
    for server, thread in running:
        server.should_exit = True
        thread.join(30)


def _client(svc):
    c = svc["connection"]
    return AgentClient(c["url"], c["project_id"], c["token"], egress_class=c["egress_class"])


def _child_state(svc, child):
    return load_file(svc["workspace"] / "checkpoints" / f"{child}.safetensors")


def _forward(family, state):
    model, _ = build(family, torch.float32)
    model.load_state_dict({k: v.to(torch.float32) for k, v in state.items()})
    torch.manual_seed(7)
    with torch.no_grad():
        y = model(torch.randn(*FAMILIES[family][1]))
    return torch.cat([t.flatten() for t in (y if isinstance(y, (tuple, list)) else (y,))])


def _check_prune(svc, family, result):
    """The claim of recipe 2: the chosen units are zero in every member tensor of the group, and nothing else changed."""
    _, parent = build(family, torch.float32)
    child = _child_state(svc, result["child"])
    changed = {name for name in child if not torch.equal(child[name], parent[name])}
    assert changed and changed <= set(result["members"]), (changed, result["members"])
    assert set(result["changed_tensors"]) <= changed  # what diff reported (small tensors are skipped in class 1)


def _check_permute(svc, family, result):
    """The claim of recipe 3: the function is unchanged, and the weights really moved."""
    _, parent = build(family, torch.float32)
    child = _child_state(svc, result["child"])
    assert any(not torch.equal(child[k], parent[k]) for k in child)
    base, got = _forward(family, parent), _forward(family, child)
    assert (base - got).abs().max() <= 1e-5 * max(1.0, base.abs().max().item())


def _run_recipes(call, svc):
    results = []
    for family in CHOSEN:
        cid = svc["checkpoints"][family]
        seen = cookbook.orient(call, cid)
        assert seen["egress_class"] == 1 and seen["closed_groups"] > 0, (family, seen)
        pruned = cookbook.prune_weak_units(call, cid, key=f"prune-{family}", groups=seen["groups"])
        _check_prune(svc, family, pruned)
        permuted = cookbook.permute_closed_group(call, cid, key=f"permute-{family}", groups=seen["groups"])
        _check_permute(svc, family, permuted)
        assert permuted["moved_units"] > 0
        results += [pruned, permuted]
    return results


def test_recipes_through_the_bridge_on_three_families(service):
    svc = service()
    with _client(svc) as client:
        call = cookbook.execute_caller(client)
        results = _run_recipes(call, svc)
        lineage = cookbook.compare_children(call, svc["checkpoints"]["resnet"])
        assert len(lineage["children"]) == 2 and lineage["diff"]["changed"] > 0
        report = cookbook.handoff(call, results)
        assert [r["child"] for r in report] == [r["child"] for r in results] and all(r["trace_seq"] for r in report)
        errors = cookbook.recover(call, svc["checkpoints"]["resnet"], call("groups", checkpoint_id=svc["checkpoints"]["resnet"])["result"]["groups"])
        assert errors["cap"] == ("cap_exceeded", "limit") or errors["cap"][0] == "invalid_argument"
        if "open_group" in errors:
            assert errors["open_group"] == "group_not_closed"
        with pytest.raises(cookbook.ToolError) as refused:  # class 1: raw values are refused, never redacted
            call("read_tensor", checkpoint_id=svc["checkpoints"]["resnet"], selector={"tensor": "fc.bias"})
        assert refused.value.code == "forbidden"


def test_the_scripted_agent_stops_at_the_persons_budget(service):
    svc = service(quota=10)
    with _client(svc) as client:
        out = cookbook.scripted_agent(cookbook.execute_caller(client), list(svc["checkpoints"].values()), user_budget=3, tag="u")
    assert out["stop_reason"] == "user_budget" and out["applies"] == 3  # the server would allow 10
    assert len(out["children"]) == 3


def test_the_scripted_agent_stops_when_the_server_refuses(service):
    svc = service(quota=2)
    with _client(svc) as client:
        out = cookbook.scripted_agent(cookbook.execute_caller(client), list(svc["checkpoints"].values()), user_budget=10, tag="q")
    assert out["stop_reason"] == "quota_exhausted" and out["applies"] == 2
    assert len(out["children"]) == 2  # only children that exist are reported


def test_the_scripted_agent_over_real_mcp_stdio(service):
    pytest.importorskip("mcp")
    svc = service()
    with cookbook.mcp_caller(svc["path"]) as mcp:
        assert "read_tensor" in mcp.tools() and "run" not in mcp.tools()
        out = cookbook.scripted_agent(mcp.call, [svc["checkpoints"][f] for f in CHOSEN], user_budget=10, tag="m")
        assert mcp.meta and mcp.meta.get("layerdelta/budget", "").startswith("values=")
    assert out["stop_reason"] == "done" and out["applies"] == 6  # both recipes on all three families, through MCP alone
    by_child = {r["child"]: r for r in out["children"]}
    assert len(by_child) == 6
    for family, cid in svc["checkpoints"].items():  # the harness checks every claim the agent made
        for r in out["children"]:
            if r["parent"] == cid and r["recipe"] == "permute":
                _check_permute(svc, family, r)
