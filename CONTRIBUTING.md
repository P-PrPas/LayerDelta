# Contributing to LayerDelta

Thank you for helping with LayerDelta. This guide covers the development setup, how to run the tests and the desktop build, and the conventions for pull requests. For what the project is and where it is heading, start with the [README](README.md) and the [roadmap](docs/roadmap/Roadmap.md).

## Where things live

| Path | Contents |
|---|---|
| `src/layerdelta/` | Python package: experiment engine, adapters, evaluator, API, store and CLI entry points |
| `tests/` | pytest suite. `tests/fixtures/` holds small controlled fixtures. |
| `desktop/` | Desktop app: React/TypeScript renderer, Python launcher and `build.ps1` |
| `examples/` | Agent client example and a sample recipe |
| `release/` | Packaging scripts for the Windows release ZIP (maintainers only) |
| `tools/` | Verification scripts for measurement milestones |
| `spikes/` | Experiments that are not part of the product (for example the desktop packaging spike) |
| `docs/` | All documentation. See [docs/README.md](docs/README.md) for the index. |
| `QUESTION.md` | Open questions waiting for a decision from the project owner |

## Development setup (Windows)

Requirements:

- CPython 3.12 x64 (the project requires `>=3.12,<3.13`; the recorded builds use 3.12.5).
- Node.js 22 and npm, only if you work on the desktop renderer.
- NVIDIA driver and CUDA-enabled PyTorch, only if you test on a GPU. CPU work needs no GPU.

Create a virtual environment and install the CPU build of PyTorch first, then the package with its test extra:

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -e ".[test]"
```

For NVIDIA, install the CUDA wheel instead, for example `--index-url https://download.pytorch.org/whl/cu124`, in a fresh environment. The list of wheel profiles is in the [PyTorch previous versions page](https://pytorch.org/get-started/previous-versions/).

Generated state lives under `.layerdelta/` in the repository root. It is git-ignored, together with the virtual environment, caches and build output.

## Build the reference fixture

The first run downloads the MNIST archives and checks their SHA-256 digests. The fixture trains the two-block reference CNN, then writes a Safetensors checkpoint, a dataset manifest and the exploration baseline into the **output directory**, which must be new:

```powershell
.venv/Scripts/python -m layerdelta.fixture --data-dir .layerdelta/mnist --output .layerdelta/reference
.venv/Scripts/python -m layerdelta.probe --data-dir .layerdelta/mnist --fixture .layerdelta/reference --device cpu
```

To check reproducibility, run the fixture command again with another output directory and compare the file digests in both copies of `fixture.json`. Changes in hardware or numerical libraries can produce a different checkpoint, so compare runtime behavior using the same checkpoint bytes. The probe JSON includes the actual tensor device, source IDs, predicted labels and ten logits per image. Use `--device cuda` on a GPU machine.

## Run a channel experiment from the CLI

Each run needs a new output directory. After building the fixture:

```powershell
$inputs = @('--fixture', '.layerdelta/reference', '--data-dir', '.layerdelta/mnist', '--cache', '.layerdelta/cache', '--device', 'cpu')
.venv/Scripts/layerdelta-run @inputs --output .layerdelta/baseline
.venv/Scripts/layerdelta-run @inputs --output .layerdelta/activation --target features.block2.relu.output --channel 0 --operation scale --scale 0.5 --hypothesis "Reducing this channel may affect blurred nines."
.venv/Scripts/layerdelta-run @inputs --output .layerdelta/weight --target features.block2.conv.kernel --channel 0 --operation zero --hypothesis "Removing this kernel may change blurred nines."
.venv/Scripts/layerdelta-run @inputs --output .layerdelta/replay-weight --recipe .layerdelta/weight/recipe.json
```

Each run starts a worker and writes ten logits per image to `predictions.json`. `comparison.json` lists the changed sample IDs, the primary and normal-guardrail denominators, and the confusion matrices. `report.json` records the actual device, the parent digest, the baseline identity and timings.

## Running the tests

The test suite is run with pytest from the repository root:

```powershell
.venv/Scripts/python -m pytest -q
```

By default, tests marked `slow` (large-file tests) are skipped. Run them explicitly with:

```powershell
.venv/Scripts/python -m pytest -q -m slow
```

Some tests need a writable temporary directory. If `%TEMP%` is not writable on your machine, pass a base directory you control:

```powershell
.venv/Scripts/python -m pytest -q --basetemp="<a writable folder>"
```

**Git worktrees.** The shared virtual environment is installed in editable mode and points at the main checkout. In a worktree, the tests would import the wrong code unless you set `PYTHONPATH` to the `src` folder of the worktree you are testing:

```powershell
$env:PYTHONPATH = "<worktree>\src"
.venv/Scripts/python -m pytest -q
```

CI (`.github/workflows/fixture.yml`) runs the non-slow and slow test suites on `windows-latest` with the CPU runtime, builds the reference fixture and probe on MNIST, and builds the desktop app. `.github/workflows/windows-release.yml` builds the release ZIP and installs it on Windows for checking. Read the workflow files for the exact steps.

## Desktop renderer

The desktop renderer is in `desktop/`. Install the locked dependencies and build the bundle:

```powershell
cd desktop
npm ci
npm run build
```

`desktop/build.ps1` runs the same two steps (`npm ci --no-audit --no-fund`, then `npm run build`). The built bundle is written to `desktop/dist`, which is git-ignored.

## Pull requests

- Keep pull requests small and coherent: one change, one reason. Split refactors from behavior changes.
- Use a title prefix that names the work item, for example `v0.3-G3: ...` for a task in the v0.3 plan. For documentation-only work, use a plain descriptive title.
- CI must be green before merging. Do not merge on red CI.
- Do not create tags, releases or published artifacts. Release readiness is parked until the project owner asks for it (see the [roadmap](docs/roadmap/Roadmap.md)).
- Keep documentation in English. When a change affects behavior described in `docs/`, update the relevant doc in the same pull request.
- Do not commit local state: `.layerdelta/`, virtual environments, caches, build output, bearer-token connection files or logs. The `.gitignore` covers the usual cases.

## Documentation

- Start at [docs/README.md](docs/README.md) to find the right document.
- Guides are for users and integrators (`docs/guides/`). Reference documents are the contracts for the API and formats (`docs/reference/`). Architecture and roadmap documents are in `docs/architecture/` and `docs/roadmap/`. The design specs for the desktop screens are in `docs/design/`. The historical records of earlier milestones are in `docs/history/`; new work should not be added there.
- Open questions for the project owner are tracked in [QUESTION.md](QUESTION.md).
