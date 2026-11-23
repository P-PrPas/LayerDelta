# Beta validation on Windows x64

## Scripted agent pilot

Two sessions ran the same deterministic selection policy against the same reference checkpoint on each device. Each session had one experiment submission. Both saw the frozen exploration confusion and source `mnist/t10k/00530/blurred`. The candidate post-ReLU channels in block 2 were 0 and 23. The policy selected the channel with the highest mean activation and used candidate order to break ties. The aggregate arm had no per-channel measurements and used equal priors. The informed arm measured both means on the sample. Both predicted that scaling the selected channel by 0.5 would increase accuracy for blurred images with true label 9.

| Arm | Candidate means (0, 23) on CPU | Selected channel | Valid recipes / submissions | Direction correct |
|---|---|---:|---:|---:|
| Aggregate | Equal priors | 0 | 1/1 | 0/1 |
| Channel | 0.254213, 0.143282 | 0 | 1/1 | 0/1 |

Both arms spent their entire session budget and selected the same intervention on CPU and CUDA. The resulting recipe hash was `43a4a931d8bbecb70dafa972506c38b2f23695cc90ea5c7fd24ba0068f34c98f`. Counts matched across devices. This single paired task had no difference between the arms. It provides one observation of a deterministic policy's behavior.

The protocol designated the channel arm for final evaluation before inspecting either arm's outcome. The harness explicitly confirmed its frozen hash through the human HTTP session in a disposable workspace. Both audit events recorded holdout split digest `1a9c49c1d1c78b0c7580daf87721240a06845b85e2942109fc5da5d65d4cf8d2`. The scoped agent subsequently received HTTP 404 for the holdout report.

| Split and slice | N | Correct before | Correct after | Delta (pp) | Fixed | Regressed | Changed wrong |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exploration, blurred true 9 | 504 | 425 | 424 | −0.1984 | 2 | 3 | 1 |
| Exploration, all normal | 4,999 | 4,811 | 4,808 | −0.0600 | 2 | 5 | 0 |
| Holdout, blurred true 9 | 505 | 428 | 428 | 0.0000 | 0 | 0 | 1 |
| Holdout, all normal | 5,001 | 4,797 | 4,793 | −0.0800 | 6 | 10 | 1 |

## Execution measurements

The Windows 11 host ran PyTorch 2.6.0+cu124 on its CPU and RTX 3050 Laptop GPU. The following durations span run submission through HTTP comparison retrieval on an already prepared workspace. The informed arm ran second and reused the baseline cache. Each cell is one observation. Result bytes cover the prediction, comparison and recipe files in the worker report. Peak VRAM measures PyTorch tensor allocations.

| Device / arm | Baseline cache | Submit to comparison (s) | Worker (s) | Peak RAM (MiB) | Peak VRAM (MiB) | Result bytes |
|---|---|---:|---:|---:|---:|---:|
| CPU aggregate | Cold | 7.93 | 3.91 | 608.1 | 0 | 3,666,974 |
| CPU channel | Warm | 6.77 | 2.50 | 605.8 | 0 | 3,666,974 |
| CUDA aggregate | Cold | 6.14 | 1.85 | 987.3 | 40.4 | 3,666,859 |
| CUDA channel | Warm | 6.56 | 2.00 | 989.3 | 40.4 | 3,666,859 |

| Device | Aggregate run | Channel run | Confirmed holdout run |
|---|---|---|---|
| CPU | `run_9b7c12c138a74db49d425fa5ef91e435` | `run_ea0d03880ea84f8cb4594065f36d689f` | `run_bc9400aaee96439fa0f76b88d64763b9` |
| CUDA | `run_101dccd9fbcf446c8fb92943cb5c16a6` | `run_5efbfb5d452a46b39fb59f9c4cec0977` | `run_ec69076fe8fc412889eb47d9a1c595f5` |

The fixture and archives must already exist under the source workspace. Each invocation copies them into an isolated workspace and writes a new result file containing per-arm evidence, timings and the holdout audit event. CUDA runs require the validated CUDA installation.

```powershell
.venv/Scripts/python tools/verify_m3.py --source .layerdelta --output .layerdelta/pilot-cpu.json --device cpu --confirm-scripted-holdout
.venv/Scripts/python tools/verify_m3.py --source .layerdelta --output .layerdelta/pilot-cuda.json --device cuda --confirm-scripted-holdout
```

## Failure and privacy checks

- HTTP tests use the production FastAPI application, session exchange, SQLite supervisor and a subprocess that injects a worker OOM. Selecting an unavailable GPU returns an error before the quota charge. An accepted OOM consumes one charge. A retry with a new key succeeds and the client can retrieve the result. Concurrent identical submissions yield one job. Requests with an occupied key and different content return a conflict.
- Tests exercise CSRF, local origin and host restrictions, project scope, denied agent export, holdout confirmation and audit, sample/preview lookup, job history, report and change endpoints. The live pilot checks the private result on a real CUDA and CPU holdout.
- Process tests kill the supervisor while its actual evaluation child is alive and another job is queued. The next supervisor records the interruption, runs the queued job, and the old evaluation PID exits. Separate tests inject disk-write failure, request cancellation at the publish boundary, corrupt result files and simulate a crash between directory rename and SQLite commit.
- The loader and paired evaluator suites exercise corrupt header/shape/dtype/bytes, non-finite tensors, corpus/manifest drift, missing or duplicate prediction rows, empty slices, and label or split mismatches.

```powershell
.venv/Scripts/python -m pytest -q tests/test_m3_http.py tests/test_store.py tests/test_publication_integrity.py tests/test_checkpoint.py tests/test_evaluation.py
```

## Human pilot

Human participants measured to date: **0**. Workbench usability, task completion rate and human time-to-conclusion remain open. The scripted confirmation measured the product's authorization and evaluation behavior.

The timed participant protocol specifies balanced task order, a common fixture, a scoring key and the observations needed before drawing a usability conclusion. [Participant protocol](human-pilot.md)

**Decision (owner):** the human pilot is deferred and is a hard gate for announcing v0.1, not for merging M3. Until participants have been timed, no release note or README may claim workbench usability or a time-to-conclusion advantage over scripts.
