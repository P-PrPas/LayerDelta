# LayerDelta

LayerDelta is a local Windows workbench for measuring how a change to one CNN channel affects individual predictions. Select a normal/blurred MNIST pair, inspect its feature maps, change an activation or kernel, then compare the images that improved and regressed. The desktop app and typed Agent client run the same experiment engine on CPU or NVIDIA CUDA.

For a packaged installation, follow the [Windows quickstart](docs/windows-quickstart.md). LayerDelta is under active development and has no public release yet. See the [roadmap](docs/Roadmap.md) for milestones and status, and the [internal build notes](docs/release-notes.md) for tested hardware and artifact provenance. LayerDelta code is distributed under [Apache-2.0](LICENSE).

![Run comparison showing improved accuracy on blurred nines alongside a regression on normal images](docs/images/run-comparison.png)

## Build the reference fixture on Windows

Use CPython 3.12 x64. The recorded build used Python 3.12.5. The first run downloads the MNIST archives and verifies their SHA-256 digests. Allow several GB of disk space for a CUDA environment.

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -e ".[test]"
.venv/Scripts/python -m layerdelta.fixture --data-dir .layerdelta/mnist --output .layerdelta/reference
.venv/Scripts/python -m layerdelta.probe --data-dir .layerdelta/mnist --fixture .layerdelta/reference --device cpu
.venv/Scripts/python -m pytest
```

For NVIDIA, use `https://download.pytorch.org/whl/cu124` in the torch installation command in a fresh environment, then run the same probe with `--device cuda`. Its JSON includes the actual tensor device, source IDs, predicted labels and ten logits per image. The reference was trained on CPU even when using the CUDA-enabled wheel. [PyTorch wheel profiles](https://pytorch.org/get-started/previous-versions/)

The build writes a Safetensors checkpoint, dataset manifest, model architecture and exploration results into a **new directory**. To test reproducibility, run the fixture command a second time with a different output directory and compare the file digests in the two copies of `fixture.json`. Hardware and numerical-library changes can produce a different trained checkpoint. Use the same checkpoint bytes when comparing runtime behavior.

See [the M0 reference measurements](docs/m0-reference.md), [desktop packaging spike](docs/m0-desktop.md), and [the v0.1 implementation plan](docs/Implementation_Plan_v0.1.md). Work toward model-agnostic weight-space tooling follows the [target architecture](docs/Architecture_Target.md) and the [v0.2 task board](docs/Implementation_Plan_v0.2.md).

## Run a channel experiment

Use a new output directory for each execution. On Windows PowerShell, after building the fixture above:

```powershell
$inputs = @('--fixture', '.layerdelta/reference', '--data-dir', '.layerdelta/mnist', '--cache', '.layerdelta/cache', '--device', 'cpu')
.venv/Scripts/layerdelta-run @inputs --output .layerdelta/baseline
.venv/Scripts/layerdelta-run @inputs --output .layerdelta/activation --target features.block2.relu.output --channel 0 --operation scale --scale 0.5 --hypothesis "Reducing this channel may affect blurred nines."
.venv/Scripts/layerdelta-run @inputs --output .layerdelta/weight --target features.block2.conv.kernel --channel 0 --operation zero --hypothesis "Removing this kernel may change blurred nines."
.venv/Scripts/layerdelta-run @inputs --output .layerdelta/replay-weight --recipe .layerdelta/weight/recipe.json
```

Each run starts a worker and writes ten logits per image to `predictions.json`. Open `comparison.json` for changed sample IDs, primary and normal-guardrail denominators, and confusion matrices. `report.json` records the actual device, parent digest, baseline identity and timings. Change the device to `cuda` to request the GPU explicitly.
