# LayerDelta

LayerDelta investigates how changing one CNN channel affects predictions for individual images. The M0 reference fixture trains a two-block CNN on MNIST and selects a normal/blurred image pair from measured exploration errors. It includes a CLI probe for the same checkpoint on CPU and NVIDIA CUDA.

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

See [the M0 reference measurements](docs/m0-reference.md), [desktop packaging spike](docs/m0-desktop.md), and [the v0.1 implementation plan](docs/Implementation_Plan_v0.1.md).
