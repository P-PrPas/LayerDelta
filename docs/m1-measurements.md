# M1 runner measurements

On Windows 11 x64 build 26200, both CPU and NVIDIA RTX 3050 Laptop GPU (4 GiB, driver 566.07) completed the exploration sequence on the M0 parent checkpoint `cf9e1bb4c660949dd574521c7436884e8f5d4abd0ae6f8679c4a7011f361ce69`. The process used Python 3.12.5, PyTorch 2.6.0+cu124, deterministic float32, one intra-op thread, 14 inter-op threads and batches of 128. TF32 was disabled. The CPU identified as Intel64 Family 6 Model 154 Stepping 3.

## Paired evidence

Each intervention zeroed channel 17 in block 2 of its own working model. The activation recipe targeted the post-ReLU output. The weight recipe targeted the convolution kernel and preserved its bias. The frozen recipe hashes were identical across devices. Counts below were identical on each device.

| Run | Overall correct / 9,998 | Blurred true 9 correct / 504 | Normal correct / 4,999 | Fixed | Regressed | Changed wrong |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 9,409 | 425 | 4,811 | 0 | 0 | 0 |
| Activation | 9,430 | 412 | 4,814 | 66 | 45 | 25 |
| Weight | 9,429 | 411 | 4,816 | 64 | 44 | 22 |

Both edits improved overall exploration accuracy and reduced accuracy for blurred nines. The primary slice fell by 2.58 percentage points for activation and 2.78 for weight. This measurement used exploration data exclusively.

## Cross-device calibration and regression

The first byte of SHA-256 of each UTF-8 source ID assigned values 0–63 to calibration and 64–255 to regression. Keeping image variants together yielded 2,564 calibration rows and 7,434 regression rows. Across all ten baseline logits on calibration, the maximum absolute CPU/CUDA difference was 0.00000762939453125. Doubling it and rounding up to the next power of two gave the frozen absolute tolerance **0.0000152587890625**, with zero relative tolerance.

| Run | Max absolute logit difference on regression | Samples beyond tolerance | Differing predicted-class IDs on all 9,998 rows |
|---|---:|---:|---|
| Baseline | 0.0000095367431640625 | 0 | `[]` |
| Activation | 0.0000095367431640625 | 0 | `[]` |
| Weight | 0.000011444091796875 | 0 | `[]` |

## Isolation and timing

On each device the acceptance harness injected an exception after the first edited forward, then killed a separate weight worker after its first edited forward. Neither worker published a result directory. The parent SHA-256 stayed fixed. Loading the parent in a new worker with an empty cache reproduced the original full prediction file byte for byte, including logits. Repeating the baseline with the warm cache also reproduced its predictions.

| Device | Run | Baseline cache | Wall (s) | Inference (s) | Peak working set (MiB) | Peak allocated GPU tensors (MiB) | Result files (bytes) |
|---|---|---|---:|---:|---:|---:|---:|
| CPU | Baseline | Cold | 8.074 | 1.378 | 529.5 | 0 | 3,677,891 |
| CPU | Activation | Warm | 6.304 | 1.172 | 564.0 | 0 | 3,701,807 |
| CPU | Weight | Warm | 6.448 | 1.086 | 563.3 | 0 | 3,700,898 |
| CPU | Baseline replay | Warm | 4.370 | 0 | 529.3 | 0 | 3,677,877 |
| CPU | Recovery baseline | Cold | 9.462 | 1.086 | 530.4 | 0 | 3,677,892 |
| CUDA | Baseline | Cold | 5.437 | 0.335 | 849.3 | 38.6 | 3,678,220 |
| CUDA | Activation | Warm | 6.528 | 0.449 | 946.8 | 40.4 | 3,701,888 |
| CUDA | Weight | Warm | 6.683 | 0.423 | 983.4 | 38.6 | 3,701,096 |
| CUDA | Baseline replay | Warm | 6.099 | 0 | 615.2 | 0 | 3,678,198 |
| CUDA | Recovery baseline | Cold | 5.967 | 0.330 | 850.2 | 38.6 | 3,678,221 |

Each table entry is one observation. Wall time spans CLI launch to exit and includes fixture validation, imports, cache I/O and artifact serialization. Cold and warm describe the application baseline cache. GPU memory measures allocations tracked by PyTorch. The acceptance harness allows 180 seconds per CLI run. Previews should be limited to one sample and one channel, at most 28×28 values per displayed map in this architecture.

## Reproduce

Prepare the M0 fixture and CUDA-enabled environment as described in the README. Use a new output directory. The harness leaves logits and sample IDs in the local workspace. `measurements.json` records run IDs, fingerprints, bytes, errors and every class disagreement.

```powershell
$out = ".layerdelta/m1-acceptance"
.venv/Scripts/python tools/verify_m1.py measure --output $out
.venv/Scripts/python tools/verify_m1.py isolation --output $out
.venv/Scripts/python tools/verify_m1.py regression --output $out
```

To qualify a different runtime, run `calibrate` after `measure`. It writes a new calibration record once. Supply that record through `regression --tolerance-file` for that qualification. Routine reference regression uses the frozen tolerance above.
