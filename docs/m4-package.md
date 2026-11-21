# Windows release candidate

The tested `LayerDelta-0.1.0rc1-windows-x64.zip` is 33,393,352 bytes. Its SHA-256 is `092ac8cd1c1ab09bc5b85f7ac3a1f1e293b1d42bf410ec07d21127dcb0a27313`. The included `layerdelta-0.1.0rc1-py3-none-any.whl` has SHA-256 `2fcd41fa47d6c7b0dd6459d21fad3348858b85087ffd122d7e47271ad655457c`. Each CI build produces its own checksum alongside the ZIP.

## Installation

The ZIP contains the WebView2 desktop shell, built React renderer, wheel, Windows runtime profiles, example recipe, headless example, user guides and third-party notices. The installer checks the package manifest before running the bundled uv executable. It installs managed CPython 3.12.5, checks the 28 dependency wheels against the profile hashes, installs LayerDelta and creates a shortcut. Runtime wheels, MNIST archives and model weights are downloaded or generated at installation or first launch. [Windows quickstart](windows-quickstart.md)

The local verification used two new ZIP extractions with separate managed Python installations and separate workspaces. The installation and desktop smoke used a PATH containing only Windows and System32. The host already had WebView2 Evergreen 140.0.3485.66. The WebView2 bootstrap branch remains a separate test case for a Windows machine with the runtime absent. [Microsoft WebView2 distribution](https://learn.microsoft.com/microsoft-edge/webview2/concepts/distribution)

| Host component | Observed |
|---|---|
| OS and CPU | Windows 11 x64 build 26200, Intel Core i7-12700H |
| CPU profile | CPython 3.12.5, PyTorch 2.6.0+cpu |
| NVIDIA profile | CPython 3.12.5, PyTorch 2.6.0+cu124, RTX 3050 Laptop GPU, 4 GiB, driver 566.07 |

Both shortcut targets opened WebView2, downloaded the archives into empty workspaces, produced parent SHA-256 `cf9e1bb4c660949dd574521c7436884e8f5d4abd0ae6f8679c4a7011f361ce69` and displayed the requested device alongside the blurred 00530 image and its activation map. On each device the primary slice contained 504 blurred nines. The activation edit fixed 22, regressed 3 and changed 8 wrong predictions to another wrong class. The weight edit yielded counts 19, 3 and 8 respectively. Both child exports reached a downloadable file and both launchers exited with code 0.

| Device | Desktop activation run | Desktop weight run | Confirmed child export |
|---|---|---|---|
| CPU | `run_203962f8d7cb46f597dda6b3d7f2d9d2` | `run_a2f4cd46352944c6806422f949996dc2` | `run_298decb4f4cd4ec49293134f76af1569` |
| CUDA 0 | `run_bd251c27353042a387118f811508f803` | `run_4d7d2936e9804374ba7829f96b27b0dc` | `run_6d97c4b10486422d8f64bb673042dc1e` |

The smoke passed workspace paths directly and used the installed shortcuts' targets. Interactive use of the native folder picker remains a manual check.

## Reproduction

Run the following from a Windows checkout with CPython 3.12, Node 22 and WebView2. The packaging job performs a second installation and desktop-to-headless replay on a newly allocated Windows CI runner. Its uploaded ZIP carries the manifest and separate archive checksum. [Release workflow](../.github/workflows/windows-release.yml)

```powershell
./release/build.ps1
Get-FileHash dist/LayerDelta-0.1.0rc1-windows-x64.zip
Expand-Archive dist/LayerDelta-0.1.0rc1-windows-x64.zip .layerdelta/installed-cpu
$app = (Get-ChildItem .layerdelta/installed-cpu -Directory).FullName
& "$app/install.ps1" -ProfileName cpu
& "$app/LayerDelta.exe" --workspace .layerdelta/package-demo --smoke cpu
```

The hidden-window smoke uses the executable's normal startup, session exchange and real React controls. It selects a device, opens a reference image, renders its activation map, submits activation and kernel edits, reads their comparisons, opens history, confirms a child export and checks the download link. The command prints the run IDs and recipe hashes. Use the weight run ID in the following command for a full cross-workspace replay. Repeat in a separate extraction with `-ProfileName cuda` and `--smoke cuda` on the NVIDIA host.

```powershell
& "$app/runtime/Scripts/python.exe" -I desktop/verify_parity.py .layerdelta/package-demo <weight-run-id>
```
