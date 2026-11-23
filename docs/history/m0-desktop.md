# M0 · Windows desktop packaging spike

The spike chose **pywebview + Microsoft Edge WebView2** for Windows x64, and packages the shell as a PyInstaller onedir build. The program starts from an executable, then creates a Python worker that listens on a loopback port assigned by the OS. The React page and the workspace data come from the same origin. The two processes coordinate their lifecycle through a pipe.

This approach was chosen because the renderer can use the built JavaScript directly, users can run the program from a directory that was copied elsewhere, and the separation of the Python runtime could be tested for real. The inference runtime is distributed as separate CPU or CUDA profiles before M2, so that a CPU machine only installs the dependencies it needs. The final choice of how to install PyTorch depends on its size and on the results of tests on a clean Windows machine in M3/M4.

## Evidence from testing on the development machine

| Item | What was checked and the result |
|---|---|
| Build | Windows 11 x64 build 26200, Python 3.12.5, PyInstaller 6.22.3, pywebview 6.2.1, React 19.3.1, esbuild 0.28.2, WebView2 Runtime installed on the machine 140.0.3485.66 |
| Size | 34,269,391 bytes for the onedir shell, including the bundled Python, the native folder picker and the React assets, on disk. This size excludes the system WebView2 Runtime and the PyTorch profile. |
| Installed separately from the checkout | The smoke test copied the whole onedir package to a temporary directory and opened it from a workspace with a Thai name containing spaces. The program's PATH was set to only Windows and System32. |
| Program the user opens | Two paths were tested from the same packaged executable: the worker bundled with the shell, and a worker started with the project venv executable, given by absolute path |
| Display | The actual folder name text was checked in the DOM, after React fetched the data from the Python worker through a hidden WebView2 window |
| Time until visible | From before the worker is spawned until the workspace name can be read. Four consecutive runs: bundled 3.295 / 1.754 / 2.485 / 2.517 seconds; external 2.228 / 2.084 / 2.134 / 1.853 seconds. |
| Shutdown | All eight runs above closed the window. The launcher closed the pipe so the worker shut down, and the exit code 0 was confirmed within 5 seconds. |

After a timeout was added for the worker, the 34,269,921-byte package was tested again on the same machine. The bundled mode took 2.564 seconds and the external mode took 2.058 seconds. Both read the Thai workspace name from the DOM, and the worker exited with code 0.

## Reproducing

On a build machine with a Python launcher that can select 3.12, Node.js with npm, and Windows x64 with the WebView2 Runtime installed, run the following from the repository root:

```powershell
.\spikes\desktop\build.ps1
py -3.12 spikes/desktop/smoke.py
```

If a project venv exists, test the worker with the same executable prepared for inference:

```powershell
py -3.12 spikes/desktop/smoke.py --runtime .venv/Scripts/python.exe
```

If the build machine restricts running PowerShell scripts, run the commands in [build.ps1](../../spikes/desktop/build.ps1) in order, from the spike's folder. The CI commands use the same smoke script to test both the bundled worker and the external worker, and they require a Windows session that can create a hidden WebView2 window.

For manual testing, open `spikes/desktop/dist/LayerDeltaSpike/LayerDeltaSpike.exe` and choose a folder through the native dialog. If the selection is cancelled, the launcher exits before creating a worker. The smoke script supplies the path through `--workspace`.

## Limits and next checkpoints

- This test used a temporary copy on a development machine where WebView2 was already installed. The way to check or install the prerequisites on a clean Windows machine still has to be confirmed. Selecting and cancelling the native dialog must also be tested by hand before G1 passes.
- During the first experiment, a hidden WebView2 window once took more than 20 seconds to fire its startup event. The staging rounds recorded in the table all passed the eight runs. In M2, record the phase, the exit code and the timeout separately, to diagnose slow startup on users' machines.
- The work in the spike's worker is reading the workspace name. This timing covers the shell, the worker and the React renderer. Model loading and CPU/CUDA inference rely on the evidence from the reference probe and the runner in M0/M1. A package that includes the real workflow must be measured again in M3/M4.
- When the application service is connected in M2, the server must check the session, the project scope and the actor against the API contract. The launcher waits at most 20 seconds for a ready signal from the worker. If that time is exceeded, it closes the pipe, waits no more than another 5 seconds for the worker, and then forces it to stop.
