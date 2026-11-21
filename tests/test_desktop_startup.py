import subprocess
import sys

import pytest

from spikes.desktop.shell import wait_for_worker


def test_worker_that_never_announces_readiness_times_out():
    with subprocess.Popen(
        [sys.executable, "-I", "-c", "import sys; sys.stdin.buffer.read(1)"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    ) as worker:
        try:
            with pytest.raises(RuntimeError, match="startup timed out"):
                wait_for_worker(worker, timeout=0.1)
        finally:
            worker.stdin.close()
            worker.wait(timeout=5)
