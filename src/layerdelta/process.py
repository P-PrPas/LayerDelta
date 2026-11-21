"""Launch the current Python with a handle to its actual interpreter on Windows."""

import os
import subprocess
import sys


def popen(command, **kwargs):
    if os.name == "nt" and command[0] == sys.executable and sys.prefix != sys.base_prefix:
        # CPython multiprocessing uses this convention to bypass the venv redirector
        # while preserving venv site-packages. kill()/wait() then own the real PID.
        env = kwargs.get("env")
        kwargs["env"] = dict(os.environ if env is None else env, __PYVENV_LAUNCHER__=sys.executable)
        command = [sys._base_executable, *command[1:]]
    return subprocess.Popen(command, **kwargs)
