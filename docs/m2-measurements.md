# M2 HTTP workflow

The reference Agent searches for the normal and blurred variants of a source image, requests a post-ReLU preview on the selected channel for both, checks its remaining submissions and tests one frozen recipe. It prints the recipe and run IDs, measured metrics and paginated prediction changes. Each accepted run consumes one submission. The default user budget is one.

## Run the example

Start the server in a Windows terminal and keep it open. The workspace contains the M0 `reference` and `mnist` directories. The server creates a new connection file for this session.

```powershell
.venv/Scripts/python -m layerdelta.server --workspace .layerdelta --agent-token-file .layerdelta/agent-session.json
```

In a second terminal, choose the device and the intervention mode. The bundled example uses a deterministic callback in the local Python process. The callback receives target ID, channel, true label and each image's sample ID, mean, maximum and zero fraction.

```powershell
.venv/Scripts/python examples/agent/demo.py --connection .layerdelta/agent-session.json --device cpu --mode weight --budget 1
```

## Windows alpha acceptance

The [acceptance script](../tools/verify_m2.py) copies the frozen fixture into two fresh workspaces and launches real loopback servers. It exchanges the desktop launch ticket for a human session cookie and CSRF token. The Agent uses its own scoped bearer credential. Desktop renderer interaction is covered separately.

- Both actors ran the same frozen weight recipe on each selected device. Every slice, confusion matrix, change category, sample ID and execution fingerprint matched. The Agent followed the pagination cursor to retrieve the entire changed set.
- A human-session request confirmed the frozen hash and evaluated 10,002 holdout rows on each device. An export from the published exploration weight run supplied a Safetensors child and lineage. All tensors and bias values matched the parent except the authorized output-channel slice, which was exactly zero.
- In the second workspace the Agent imported the recipe and replayed the full exploration run. A human uploaded the exported child through the checkpoint endpoint and submitted a baseline against that child. Its per-sample prediction file, including all ten logits per image, matched the original weight result byte for byte on the source device.

The fixture's source `mnist/t10k/00530` had mean post-ReLU activation 0.333156 (normal) and 0.274059 (blurred) at block 2, channel 17 on both devices. Zeroing the corresponding kernel fixed 64 predictions, regressed 44 and changed the wrong class for 22. Overall correct increased from 9,409 to 9,429 out of 9,998. Correct on blurred true nines fell from 425 to 411 out of 504.

| Actual device | Agent weight run | Human weight run | Re-imported child predictions SHA-256 |
|---|---|---|---|
| cpu | `run_f102952dea064142a6a139a96f850d4d` | `run_9d2506e67e624d68bbdc7af58dfd6d45` | `7fd966354e1b211f1acb1065acc3ff5166ef85faec5a767c60e59677f1a26c84` |
| cuda:0 | `run_66336765628746439d5623506a9a0ade` | `run_619f31de31674d87a86eff5431c890f9` | `d1333163dc5e678cf1ccedee801057244d74255a674937367c130a4d9dae9c07` |

Both exports produced child checkpoint SHA-256 `e0bb841b89085c0da83c5f71fda4ff3e21e9eac16d453387f78164c10aa27ce8`. The two device-specific hypotheses include observed means to eight significant digits. Each frozen recipe retained its original hash on import.

Run the checks with a CUDA-capable environment or select CPU with `--devices cpu`. The output directory must be new. `acceptance.json` records both actors' run IDs, holdout, replay and child run IDs, digests and actual devices.

```powershell
.venv/Scripts/python tools/verify_m2.py --source .layerdelta --output .layerdelta/m2-acceptance
```
