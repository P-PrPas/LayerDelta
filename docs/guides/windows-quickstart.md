# First experiment on Windows x64

Extract the complete release-candidate ZIP to a writable directory. Keep the directory in place after installing the runtime. The installer downloads CPython, the selected PyTorch wheel and its pinned dependencies. Initial workspace creation also needs access to the MNIST mirror.

1. Run **Install CPU.cmd** for a CPU installation, or **Install CUDA.cmd** for the NVIDIA profile. To keep both, extract the ZIP into two directories. The installer verifies package files, installs WebView2 when required, checks the runtime and creates a desktop shortcut.
2. Open the LayerDelta shortcut and choose an empty workspace folder. The first launch downloads four digest-verified archives, trains the two-block reference CNN and records its exploration baseline. Reopening the folder uses the saved reference.
3. In **Model explorer**, choose the device, a source, condition, layer and channel. Select **Preview channel** to see the digit, activation map and kernel statistics from the worker. Switch between normal and blurred to inspect the paired images.
4. Open **Experiment editor**. Choose the block 2 kernel target, channel 17, and the zero operation. Enter a hypothesis and run the intervention. Inspect both the blurred true-digit slice and the normal guardrail in the comparison. Select a changed sample to inspect its source.
5. Open **History & artifacts** to revisit a run, save the recipe and report, or confirm an export from a successful weight run. To reproduce the result elsewhere, first copy the frozen reference fixture and its verified archives into the second workspace, then import the recipe. The new workspace must resolve the same parent and manifest digests.
6. **Final evaluation** records a human confirmation against the frozen recipe hash and evaluates the holdout split. Record any holdout access when reporting the result.

## Run an Agent from the installed package

Close the Desktop session for this workspace. Open PowerShell in the extracted package directory and run:

```powershell
$project = Join-Path $env:LOCALAPPDATA 'LayerDelta-demo'
$connection = Join-Path $env:TEMP ("LayerDelta-" + [guid]::NewGuid() + ".json")
Write-Host "Connection file for the second terminal: $connection"
.\runtime\Scripts\layerdelta.exe --workspace $project --agent-token-file $connection
```

In a second PowerShell window opened in the same package directory, substitute the path printed above:

```powershell
.\runtime\Scripts\python.exe -m layerdelta.agent_demo --connection "<connection file path>" --budget 1 --device cpu --mode weight
```

The example requests both previews, proposes one recipe and reports the changed predictions. The connection file authorizes this Agent session. Stopping the server with Ctrl+C closes the session. Use a fresh connection filename on the next launch.
