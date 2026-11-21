"""A successful worker exit must still match its declared result digests."""

import json
from queue import Queue
import subprocess
import sys
from threading import Thread
import time

import psutil
import pytest

from layerdelta.store import Store, StoreError
from layerdelta.process import popen


def test_child_handle_owns_real_python_interpreter_and_preserves_venv():
    script = "import json,os,sys;print(json.dumps([os.getpid(),sys.prefix]))"
    with popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True) as child:
        pid, prefix = json.loads(child.communicate(timeout=10)[0])
        assert pid == child.pid and prefix == sys.prefix


@pytest.mark.parametrize("fault", ["staged-byte", "omitted", "size", "skipped-report", "export-child"])
def test_supervisor_refuses_damaged_staging_before_catalog_commit(tmp_path, fault):
    worker = tmp_path / "fault_worker.py"
    worker.write_text("""
import hashlib, json, sys
from pathlib import Path
request = json.loads(Path(sys.argv[sys.argv.index('--request')+1]).read_bytes())
out = Path(sys.argv[sys.argv.index('--output')+1])
out.mkdir()
fault = request['payload']['fault']
if fault == 'export-child':
    (out/'child.safetensors').write_bytes(b'original')
    report = {'child_sha256': hashlib.sha256(b'original').hexdigest()}
    (out/'child.safetensors').write_bytes(b'modified')
else:
    files = ('predictions.json','comparison.json')
    for f in files:
        (out/f).write_bytes(b'{}')
    report = {'artifacts':{f:{'sha256':hashlib.sha256(b'{}').hexdigest(),'bytes':2} for f in files}}
    if fault == 'staged-byte':
        (out/'predictions.json').write_bytes(b'[]')
    if fault == 'omitted':
        del report['artifacts']['predictions.json']
    if fault == 'size':
        report['artifacts']['comparison.json']['bytes'] += 1
    if fault == 'skipped-report':
        report = {}
(out/'report.json').write_text(json.dumps(report))
""", encoding="utf-8")
    store = Store(tmp_path / "workspace", child_command=[sys.executable, str(worker)])
    store.start()
    try:
        kind = "checkpoint_export" if fault == "export-child" else "baseline"
        run = store.submit("human", "/jobs", "key", {"fault":fault}, kind)["run_id"]
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            job = store.get(run, "human")
            if job["status"] in ("succeeded", "failed"):
                break
            time.sleep(.03)
        assert job["status"] == "failed"
        assert job["error"]["code"] == "artifact_unavailable"
        with pytest.raises(StoreError):
            store.artifact(run, "human", "report.json")
        assert not (store.workspace / "runs" / run).exists()
    finally:
        store.close()


def test_abrupt_supervisor_exit_releases_running_worker_and_recovers_queue(tmp_path):
    workspace = tmp_path / "workspace"
    worker = tmp_path / "worker.py"
    worker.write_text("""
import json, os, sys, time
from pathlib import Path
root = Path(sys.argv[sys.argv.index('--workspace')+1])
request = json.loads(Path(sys.argv[sys.argv.index('--request')+1]).read_bytes())
if request['payload'].get('slow'):
    (root/'worker.pid').write_text(str(os.getpid()))
    time.sleep(60)
output = Path(sys.argv[sys.argv.index('--output')+1])
output.mkdir()
(output/'report.json').write_text('{}')
""", encoding="utf-8")
    owner = """
from layerdelta.store import Store
import json, os, sys, time
s = Store(sys.argv[1], child_command=[sys.executable, sys.argv[2]])
s.start()
a = s.submit('human','/previews','a',{'slow':True},'preview')
b = s.submit('human','/previews','b',{},'preview')
print(json.dumps([a['run_id'],b['run_id'],os.getpid()]),flush=True)
time.sleep(60)
"""
    with subprocess.Popen([sys.executable,"-c",owner,str(workspace),str(worker)],stdout=subprocess.PIPE,
                          text=True) as parent:
        try:
            lines = Queue()
            Thread(target=lambda: lines.put(parent.stdout.readline()), daemon=True).start()
            first, second, supervisor_pid = json.loads(lines.get(timeout=15))
            deadline = time.monotonic()+15
            while not (workspace/"worker.pid").exists() and time.monotonic()<deadline:
                time.sleep(.03)
            child = psutil.Process(int((workspace/"worker.pid").read_text()))
        finally:
            psutil.Process(supervisor_pid).kill()
            parent.wait(timeout=5)
    recovered = Store(workspace,child_command=[sys.executable,str(worker)])
    recovered.start()
    try:
        assert recovered.get(first, "human")["error"]["code"] == "interrupted"
        deadline = time.monotonic()+15
        while recovered.get(second,"human")["status"] != "succeeded" and time.monotonic()<deadline:
            time.sleep(.03)
        assert recovered.get(second,"human")["status"] == "succeeded"
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    finally:
        recovered.close()
