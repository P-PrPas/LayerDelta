# Windows release candidate validation

The release check used a Windows 11 x64 build 26200 host with an NVIDIA GeForce RTX 3050 Laptop GPU, 4 GiB VRAM and driver 566.07. Each profile was installed independently from the extracted candidate ZIP using its installer. The CPU installation used the CPU-only PyTorch wheel on this same host. The host already had WebView2 and the NVIDIA driver installed.

| Artifact | SHA-256 |
|---|---|
| Windows x64 ZIP, 0.1.0rc1 | `092ac8cd1c1ab09bc5b85f7ac3a1f1e293b1d42bf410ec07d21127dcb0a27313` |
| Python wheel, 0.1.0rc1 | `2fcd41fa47d6c7b0dd6459d21fad3348858b85087ffd122d7e47271ad655457c` |

For both profiles, all 18 installed LayerDelta package files and the license, package metadata and console entry points matched the corresponding wheel bytes. The test driver ran from an OS temporary directory with `PYTHONPATH` unset and Python isolated mode. It confirmed that LayerDelta imported from the installed profile's site-packages.

## Headless first launch

The installed `layerdelta.exe` started from a new workspace for each profile. The CPU run downloaded all four MNIST archives. The CUDA run started with the verified archives cached. Both trained their own reference fixture. After loopback health became ready, the installed Agent example requested the paired previews and submitted an activation experiment with a budget of one.

| PyTorch profile | Actual device | Launch to ready | Launch through Agent result and shutdown | First published run | Changed rows |
|---|---|---:|---:|---|---:|
| 2.6.0+cpu | cpu | 41.83 s | 63.95 s | `run_a20a2524aa7c48d7870b4c93a0b9242f` | 136 |
| 2.6.0+cu124 | cuda:0 | 40.61 s | 70.03 s | `run_03e8a13f57004ccc8a06f3efc203f6bc` | 136 |

These are single observations with the two acceptance runs overlapping. Both profiles reproduced the frozen M0 parent SHA-256 `cf9e1bb4c660949dd574521c7436884e8f5d4abd0ae6f8679c4a7011f361ce69`.

## Replay and child verification

The same installed Python ran a copy of [verify_m2.py](../../tools/verify_m2.py) outside the checkout, selecting the corresponding device. The script creates separate author and replay workspaces. It tests scoped Agent and human sessions over HTTP, retrieves all changed rows, confirms holdout, exports a weight child, reimports the frozen recipe, and uploads the child through the checkpoint endpoint. The last comparison uses a baseline run of the uploaded child. The byte digest of its per-sample predictions covers sample identity, prediction, and all ten logits. Tensor verification checks the exact changed output slice and every unaffected parameter, including bias.

| RC profile | Fresh-workspace replay run | Uploaded child baseline run | Predictions SHA-256 for both the source weight run and child baseline |
|---|---|---|---|
| CPU | `run_19a297ccaa914b30ad0f6524f98f593e` | `run_bbf1e14c2a244b5c84a047508c8bad32` | `7fd966354e1b211f1acb1065acc3ff5166ef85faec5a767c60e59677f1a26c84` |
| CUDA | `run_ba9df726931e47059fce269b862ab368` | `run_583e3f240e0a455a812346de2d0f6d7c` | `d1333163dc5e678cf1ccedee801057244d74255a674937367c130a4d9dae9c07` |

Both profiles published 9,998 exploration predictions, matched all 130 changes between Agent and human sessions, and evaluated 10,002 rows after test-session holdout confirmation. Each export produced child SHA-256 `e0bb841b89085c0da83c5f71fda4ff3e21e9eac16d453387f78164c10aa27ce8`. Re-import in the fresh workspace preserved the original recipe hash on each profile.
