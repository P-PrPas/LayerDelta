"""The Antigravity session driver and its audit (v0.5-E7b, the part both isolation options need). A fake `agy` (tests/fake_agy.py) speaks the stream-json the real
one emitted in the E7s spike, so no host and no model are involved."""

import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "bench"))
import agy_session as ag  # noqa: E402

FAKE = [sys.executable, str(Path(__file__).with_name("fake_agy.py"))]


def tool(step, name, **parameters):
    return {"event": "step_update", "step_update": {"conversation_id": "c1", "step_index": step, "state": "DONE", "step_type": "tool", "tool_name": name,
                                                    "duration_seconds": 0.1, "tool_info": {"name": name, "parameters": parameters, "output": "ok"}}}


def reply(step, out=10, think=5, total=100):
    return {"event": "step_update", "step_update": {"conversation_id": "c1", "step_index": step, "state": "DONE", "step_type": "agent_response", "text_delta": "x",
                                                    "usage": {"input_tokens": total - out - think, "output_tokens": out, "thinking_tokens": think, "cache_read_tokens": 7,
                                                              "total_tokens": total}}}


INIT = {"event": "init", "init": {"model": "gemini-4-hydrogen-high", "cwd": ".", "tools": ["run_command", "read_file", "search_web"], "permission_mode": "request-review"}}
DONE = {"event": "result", "result": {"conversation_id": "c1", "status": "SUCCESS", "response": "done"}}


@pytest.fixture
def host(tmp_path):
    store = tmp_path / "agy-store"
    (store / "conversations").mkdir(parents=True)
    (store / "conversations" / "other-arm.json").write_text('{"secret": "the other arms edits"}')
    (store / "brain").mkdir()
    (store / "brain" / "old.md").write_text("what an earlier session learned")
    (store / "history.jsonl").write_text("an earlier prompt")
    (store / "cache").mkdir()
    (store / "cache" / "conversation_metadata.json").write_text("titles")
    (store / "cache" / "default_project_id.txt").write_text("project")
    (store / "settings.json").write_text("{}")
    (store / "installation_id").write_text("id")
    session = tmp_path / "session"
    session.mkdir()
    env = {"FAKE_AGY_MCP": str(tmp_path / "mcp.json"), "FAKE_AGY_STORE": str(store), "FAKE_AGY_RECORD": str(tmp_path / "record.json"),
           "FAKE_AGY_SCRIPT": str(tmp_path / "script.json")}

    class Host:
        pass
    h = Host()
    h.store, h.session, h.tmp, h.env = store, session, tmp_path, env
    h.script = lambda events: (tmp_path / "script.json").write_text(json.dumps(events))
    h.record = lambda: json.loads((tmp_path / "record.json").read_text())
    h.run = lambda arm="R", **kw: ag.run_session(arm, session, "do the task", results=tmp_path / "results", agy=FAKE, store=store, **{"env": env, **kw})
    return h


def test_a_session_is_counted_the_way_the_design_budgets_it(host):
    host.script([INIT, tool(1, "run_command", CommandLine="python analyse.py"), reply(2, out=40, think=60, total=500), tool(3, "view_file", AbsolutePath=str(host.session / "model.py")),
                 reply(4, out=10, think=0, total=300), DONE])
    record = host.run()
    assert record["steps"] == 2 and record["tool_calls"] == 2  # steps are the tool steps
    assert record["completion_tokens"] == 110 and record["total_tokens"] == 800 and record["cache_read_tokens"] == 14  # output + thinking; total as the host counts
    assert record["status"] == "SUCCESS" and record["stopped"] is None and record["violations"] == [] and record["tools_offered"] == 3
    argv = host.record()["argv"]
    assert argv[0] == "-p" and "--model" in argv and argv[argv.index("--model") + 1] == "gemini-4-hydrogen-high" and argv[argv.index("--effort") + 1] == "high"
    assert "--dangerously-skip-permissions" not in argv  # never unless the owner's decision is passed in
    assert host.record()["cwd"] == str(host.session)
    assert (host.tmp / "results" / "transcript.ndjson").read_text().count("\n") == 6 and (host.tmp / "results" / "session.json").is_file()


def test_what_earlier_sessions_left_is_out_of_reach_during_the_session_and_back_after(host):
    host.script([INIT, DONE])
    host.run()
    during = host.record()["store_entries"]
    assert "brain" not in during and "history.jsonl" not in during and "bench-backup" in during  # the knowledge and history of earlier sessions are not in the store
    assert "settings.json" in during and "installation_id" in during and "cache" in during  # auth and settings stay
    assert (host.store / "conversations" / "other-arm.json").is_file() and (host.store / "brain" / "old.md").read_text() == "what an earlier session learned"
    assert (host.store / "history.jsonl").read_text() == "an earlier prompt" and (host.store / "cache" / "conversation_metadata.json").read_text() == "titles"
    assert (host.store / "cache" / "default_project_id.txt").is_file()
    assert not (host.store / "brain" / "note.md").exists() and not (host.store / "conversations" / "this-session.json").exists()  # this session's own are gone
    assert (host.tmp / "results" / "conversations" / "this-session.json").is_file()  # its conversation is kept with its results
    assert not (host.store / "bench-backup").exists()


def test_a_crash_leaves_the_store_restorable_and_a_leftover_backup_blocks_the_next_session(host):
    (host.store / "bench-backup").mkdir()
    with pytest.raises(ag.StoreBusy):
        host.run()
    (host.store / "bench-backup").rmdir()
    host.script([INIT, DONE])
    broken = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))  # noqa: E731
    original = ag.subprocess.Popen
    ag.subprocess.Popen = broken
    try:
        with pytest.raises(RuntimeError):
            host.run()
    finally:
        ag.subprocess.Popen = original
    assert (host.store / "brain" / "old.md").is_file() and (host.store / "conversations" / "other-arm.json").is_file()  # restored though the session never started
    assert not (host.store / "bench-backup").exists()


def test_a_place_the_quarantine_does_not_know_raises(host):
    host.script([INIT, DONE])
    with pytest.raises(ag.StoreChanged) as raised:
        host.run(env={**host.env, "FAKE_AGY_MYSTERY": "1"})
    assert raised.value.names == ["mystery"] and raised.value.record["store_unexpected"] == ["mystery"]
    assert (host.store / "brain" / "old.md").is_file() and not (host.store / "bench-backup").exists()  # restored all the same
    host.run()  # an ordinary session (presence locks, brain notes) raises nothing


def test_the_agent_gets_a_scrubbed_environment(host, monkeypatch):
    for name in ("LAYERDELTA_TICKET", "GITHUB_TOKEN", "ANTHROPIC_API_KEY", "HARNESS_HUMAN_CSRF"):
        monkeypatch.setenv(name, "secret-value")
    monkeypatch.setenv("AGY_ADC_AUTH", "1")
    host.script([INIT, DONE])
    host.run(env={**host.env, "HTTPS_PROXY": "http://127.0.0.1:9"})
    names = set(host.record()["env"])
    assert not names & {"LAYERDELTA_TICKET", "GITHUB_TOKEN", "ANTHROPIC_API_KEY", "HARNESS_HUMAN_CSRF"}  # the driver's secrets stop at the driver
    assert "PATH" in {n.upper() for n in names} and "AGY_ADC_AUTH" in names and "HTTPS_PROXY" in names  # what it needs, and what the caller chose to add
    assert ag.scrubbed_env().keys() <= {k for k in os.environ if k in ag.ENV_KEEP or k.upper().startswith("AGY_")}


def test_the_bridge_is_added_for_arm_l_and_removed_after_and_arm_r_refuses_it(host):
    host.script([INIT, DONE])
    host.run("L", mcp=["layerdelta-mcp", "--connection", "c.json"])
    assert host.record()["mcp"] == {"layerdelta": ["layerdelta-mcp", "--connection", "c.json"]}  # present while the session ran
    assert not json.loads((host.tmp / "mcp.json").read_text())  # gone after
    (host.tmp / "mcp.json").write_text(json.dumps({"layerdelta": ["x"]}))
    with pytest.raises(ag.McpConflict):
        host.run("R")
    with pytest.raises(ag.McpConflict):
        host.run("L", mcp=["layerdelta-mcp"])  # already configured: someone else's session may be running
    (host.tmp / "mcp.json").write_text("{}")
    with pytest.raises(ag.McpConflict):
        host.run("L")  # no command given


def test_permissions_are_skipped_only_when_asked(host):
    host.script([INIT, DONE])
    host.run(skip_permissions=True)
    assert "--dangerously-skip-permissions" in host.record()["argv"]


def test_the_step_limit_stops_the_session_and_kills_it(host):
    host.script([INIT] + [{**tool(i, "run_command", CommandLine="echo hi"), "_sleep": 0.2} for i in range(1, 60)] + [DONE])
    started = time.monotonic()
    record = host.run(limits=ag.Limits(wall_seconds=60, steps=3, completion_tokens=10**9))
    assert record["stopped"] == "steps" and 3 < record["steps"] < 20 and time.monotonic() - started < 30


def test_the_completion_token_limit_counts_output_and_thinking(host):
    host.script([INIT] + [{**reply(i, out=40, think=60), "_sleep": 0.2} for i in range(1, 40)] + [DONE])
    record = host.run(limits=ag.Limits(wall_seconds=60, steps=10**6, completion_tokens=350))
    assert record["stopped"] == "tokens" and 350 < record["completion_tokens"] < 3000


def test_the_wall_limit_stops_a_silent_session(host):
    host.script([INIT, {**tool(1, "run_command", CommandLine="sleep"), "_sleep": 60}, DONE])
    started = time.monotonic()
    record = host.run(limits=ag.Limits(wall_seconds=2, steps=10, completion_tokens=10**9))
    assert record["stopped"] == "wall" and time.monotonic() - started < 40


# ---------------------------------------------------------------- the audit

def kinds(calls, session, **kw):
    return sorted((v["kind"], v["severity"]) for v in ag.audit(calls, session, **kw))


def test_the_audit_flags_web_tools_reads_outside_and_the_agents_own_store(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    store = tmp_path / ".gemini" / "antigravity-cli"
    calls = [{"step": 1, "name": "search_web", "parameters": {"query": "mnist"}}, {"step": 2, "name": "browser_get_dom", "parameters": {}},
             {"step": 3, "name": "read_url_content", "parameters": {"Url": "https://example.com"}},
             {"step": 4, "name": "read_file", "parameters": {"Path": str(tmp_path / "elsewhere" / "data.npz")}},
             {"step": 5, "name": "list_dir", "parameters": {"DirectoryPath": str(store / "conversations")}},
             {"step": 6, "name": "write_to_file", "parameters": {"TargetFile": str(tmp_path / "elsewhere" / "x.py"), "CodeContent": "print(1)"}},
             {"step": 7, "name": "read_file", "parameters": {"Path": "model.py"}},                                    # relative: inside
             {"step": 8, "name": "view_file", "parameters": {"AbsolutePath": str(session / "sub" / "exploration.npz")}}]  # inside
    got = ag.audit(calls, session, store=store)
    assert [(v["kind"], v["step"]) for v in got] == [("web_tool", 1), ("web_tool", 2), ("web_tool", 3), ("read_outside", 4), ("protected_path", 5), ("write_outside", 6)]
    assert all(v["severity"] == "violation" for v in got)


def test_the_audit_flags_downloads_training_and_protected_paths_in_commands_and_code(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    workspace, archives = tmp_path / "workspace", tmp_path / "mnist"
    calls = [{"step": 1, "name": "run_command", "parameters": {"CommandLine": "curl.exe -sI https://example.com"}},
             {"step": 2, "name": "run_command", "parameters": {"CommandLine": "python -c \"import torchvision.datasets as d; d.MNIST('.', download=True)\""}},
             {"step": 3, "name": "write_to_file", "parameters": {"TargetFile": str(session / "fit.py"), "CodeContent": "loss.backward()\nopt = torch.optim.SGD(m.parameters(), 0.1)"}},
             {"step": 4, "name": "run_command", "parameters": {"CommandLine": f"copy {archives}\\t10k-images-idx3-ubyte.gz ."}},
             {"step": 5, "name": "run_command", "parameters": {"CommandLine": f"python {session}\\analyse.py"}},                    # inside: fine
             {"step": 6, "name": "run_command", "parameters": {"CommandLine": "C:\\Python314\\python.exe analyse.py"}},                  # a program elsewhere: review
             {"step": 7, "name": "run_command", "parameters": {"CommandLine": "type ..\\other\\final.safetensors"}}]
    got = ag.audit(calls, session, protected=[workspace, archives], store=tmp_path / ".gemini" / "antigravity-cli")
    by_step = {}
    for v in got:
        by_step.setdefault(v["step"], []).append((v["kind"], v["severity"]))
    assert by_step[1] == [("download", "violation")] and ("download", "violation") in by_step[2]
    assert ("training", "violation") in by_step[3] and ("protected_path", "violation") in by_step[4]
    assert 5 not in by_step and by_step[6] == [("path_outside", "review")] and by_step[7] == [("path_outside", "review")]


def test_code_written_now_and_run_later_cannot_hide_its_paths(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    workspace = tmp_path / "workspace"
    calls = [{"step": 1, "name": "write_to_file", "parameters": {"TargetFile": str(session / "peek.py"),
                                                                  "CodeContent": f"import numpy as np; np.load(r'{workspace}/mnist/t10k.npz'); open(r'{tmp_path}/notes.txt')"}},
             {"step": 2, "name": "replace_file_content", "parameters": {"TargetFile": str(session / "peek.py"), "ReplacementContent": "x = 1"}}]
    got = ag.audit(calls, session, protected=[workspace], store=tmp_path / ".gemini" / "antigravity-cli")
    assert sorted((v["kind"], v["severity"]) for v in got) == [("path_outside", "review"), ("protected_path", "violation")]


def test_a_clean_transcript_has_no_findings(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    calls = [{"step": 1, "name": "view_file", "parameters": {"AbsolutePath": str(session / "TASK.md")}},
             {"step": 2, "name": "run_command", "parameters": {"CommandLine": "python -c \"import numpy as np; print(np.load('exploration.npz')['labels'][:5])\""}},
             {"step": 3, "name": "write_to_file", "parameters": {"TargetFile": "final.py", "CodeContent": "import torch\nstate['classifier.weight'][3] *= 0"}}]
    assert ag.audit(calls, session, store=tmp_path / ".gemini" / "antigravity-cli") == []


def test_the_real_stream_shape_from_the_spike_is_parsed():
    """The events `agy` printed in the E7s spike for a one-word prompt, as data (the transcript is read line by line with json.loads)."""
    usage = {"input_tokens": 16147, "output_tokens": 33, "thinking_tokens": 28, "cache_read_tokens": 0, "total_tokens": 16180}
    events = [{"event": "init", "conversation_id": "a9bef7e3", "init": {"model": "gemini-4-hydrogen-high", "cwd": "C:/x", "tools": ["run_command", "view_file"],
                                                                       "permission_mode": "request-review"}},
              {"event": "step_update", "step_update": {"conversation_id": "a9bef7e3", "step_index": 0, "state": "DONE", "step_type": "user_input"}},
              {"event": "step_update", "step_update": {"conversation_id": "a9bef7e3", "step_index": 1, "state": "ACTIVE", "step_type": "agent_response", "text_delta": "OK"}},
              {"event": "step_update", "step_update": {"conversation_id": "a9bef7e3", "step_index": 1, "state": "DONE", "step_type": "agent_response", "text_delta": " ",
                                                       "duration_seconds": 3.6, "usage": usage}},
              {"event": "result", "result": {"conversation_id": "a9bef7e3", "status": "SUCCESS", "response": "OK", "duration_seconds": 7.2, "num_turns": 1, "usage": usage}}]
    stream = ag.Stream()
    for event in events:
        stream.feed(json.loads(json.dumps(event)))
    assert (stream.steps, stream.completion_tokens, stream.total_tokens, stream.input_tokens) == (0, 61, 16180, 16147)
    assert stream.conversation_id == "a9bef7e3" and stream.result["status"] == "SUCCESS" and stream.tools == ["run_command", "view_file"]
