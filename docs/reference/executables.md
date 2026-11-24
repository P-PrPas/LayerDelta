# Executables: running an attached graph with a checkpoint's weights (v0.5-E1)

Library level, no HTTP route yet (the `evaluate` tool is E4). Design: [v0.5 design §1](../architecture/v0.5-evaluators-benchmark.md). Code: `executable.py` (record, capture, `forward`), `forward_worker.py` (the isolated worker), `graph.py` (`GraphStore.put_executable` / `executable_for`, table `executables`).

A graph describes a model; an evaluation runs it with the weights of a **child** checkpoint. Each provider leaves one executable per graph, recorded at attach time (a human action):

| Provider | Executable | Stored as |
|---|---|---|
| `builtin` | the registered module of this code base | the registry key (`model`); nothing to store. Only models registered in `src/` (`reference_cnn`) exist in the worker process |
| `torch_export` | the `ExportedProgram`, saved by the export worker in the same run that built the graph | `executables/<sha256>.pt2`; record has `sha256`, `bytes`, `torch_version`, `root_prefix` |
| `onnx` | the attached `.onnx` file, copied and proven to be the bytes the graph was built from (its hash is recomputed into the graph's `spec_sha256`) | `executables/<sha256>.onnx` |

If the export worker cannot save the program (an unserialisable custom op), the attach still succeeds; the record says why and an evaluation answers `executable_missing`. A graph attached before v0.5 has no record: attach it again.

## `forward(workspace, store, ir, checkpoint_path, inputs, timeout=120, threads=1)`

Returns `{"outputs": [array per graph output], "executable": {kind, fingerprint, sha256}}`. `inputs` are arrays with samples on the first axis.

- **Weights are substituted by checkpoint name** from the graph's `params` (a tied name may be stored under another member). A parameter the checkpoint lacks or reshapes is `executable_mismatch`, decided **before any process starts**. Persistent buffers (BatchNorm statistics) are bound the same way. Lifted tensor constants and non-persistent buffers come from the artifact and are part of its hash.
- **Dtype.** A checkpoint tensor is cast to the program's dtype only when nothing is lost (the same dtype, or a float into a wider float); anything else (F32 into an F16 program, F16 and BF16 into each other) is `executable_mismatch`, so a weight is never rounded silently.
- **Outputs are batch-first** (a contract): every graph output has the batch as its first dimension (OCR: `[N, T, A]`); `forward` returns them samples first, padding removed.
- **Batch size.** The graph is static-shape, so the program runs at the exported batch size `B` (the first dimension of its first input). Samples are padded with zeros to `B` and the outputs trimmed, so any number of samples works. Every output must have `B` as its first dimension; a model that mixes samples in eval mode (batch-wide statistics) is out of scope.
- **Isolation.** Every forward pass runs in a separate process (`forward_worker`): scrubbed environment, own working directory, no stdin, timeout and tree kill, caps on log and output size (`OUTPUT_MAX`, `INPUT_MAX`), one worker at a time. The API process never runs a model. `threads` defaults to 1 so two CPU runs are bit-identical (V9).
- **ONNX** needs the optional extra `layerdelta[eval-onnx]` (`onnxruntime`), loaded in the worker only. A file with float tensors of the checkpoint that are not bound to it (BatchNorm folded into the convolution, renamed or pre-transposed weights) is refused with `executable_mismatch`: an edit to such a tensor would not change the output. Export with `do_constant_folding=False` and `training=PRESERVE`.

## Loading a `.pt2` unpickles, and what protects it

In torch 2.6 `torch.export.load` calls `torch.load(..., weights_only=False)` on the program's state dict and constants, so loading is **not** pickle-free. The protection is the trust chain, not the format:

1. only the export worker writes the `.pt2`, during the human-triggered attach of code the person already chose to run (Q1 option 3);
2. the file is loaded only in the forward worker, from the very bytes the worker hashed against the record (`executable_changed` on a difference, or on a different torch version); the file is read once, so it cannot change between the check and the load;
3. an agent can cause the *load* of an artifact a person made, never supply or change one.

The residual risk equals the attach's own trust. Documented fallback, not built: replaying the graph IR with an in-tree interpreter of ATen ops avoids pickle but covers only the ops we implement.

## Errors

`executable_mismatch`, `executable_changed`, `executable_missing`, `forward_failed` (422; the catalog entries are in [errors.md](errors.md)), plus the worker's `worker_timeout`, `worker_output` and `invalid_argument` for inputs of the wrong shape or count.
