# GraphIR, graph providers, binding and the graph store (v0.3-G2)

Library level, no HTTP. Design context: [v0.3-graph-semantics.md](../architecture/v0.3-graph-semantics.md) section 1, 3.1, 3.2 and 7.1, [ADR 0001](../architecture/adr/0001-depgraph-oracle.md). Code: `graph.py` (IR, binding, store; torch-free, generic), `providers.py` (provider seam, builtin provider, `export_to_ir`), `builtin_models.py` (our own registered models).

## GraphIR (`ir_version: 1`)

Canonical JSON (`ir_json`: sorted keys, no spaces, no NaN) and `ir_sha256` = sha256 of the canonical JSON of everything else. `finalize_ir` seals, `verify_ir` checks schema, references and hash (`invalid_graph`, `unsupported_ir_version`, `ir_hash_mismatch`).

```json
{"ir_version": 1,
 "provider": {"name": "builtin", "version": "torch-2.6.0", "spec_sha256": "…"},
 "inputs": [{"value": "v0", "shape": [2,3,16,16], "dtype": "F32"}],
 "params": [{"name": "stem.conv.weight", "value": "p3", "kind": "parameter" | "buffer", "shape": [8,3,3,3], "dtype": "F32"}],
 "ties": [["embed.weight", "head.weight"]],
 "nodes": [{"id": "n7", "op": "aten.conv2d.default", "args": ["v0", "p3", null, [2,2], [1,1], [1,1], 1], "kwargs": {},
            "outputs": [{"value": "v8", "kind": "tensor", "shape": [2,8,16,16], "dtype": "F32"}],
            "module_path": "stem.conv", "module_type": "Conv2d"}],
 "outputs": ["v40"]}
```

- **Values**: `v<k>` for inputs and node outputs, `p<k>` for parameters and buffers. A bare string in `args`/`kwargs` is always a value reference (it must be defined earlier). Literals: `null`, booleans, ints, finite floats; wrappers `{"str": …}`, `{"enum": "torch.float32"}`, `{"float": "inf" | "-inf" | "nan"}`. Lists are positional. A node has one value; a node that returns a tuple has `kind: "tuple"` with `items`, and `getitem` nodes extract.
- **params** carry `shape` and `dtype` (binding needs them) and exist only for state tensors the graph actually receives: a parameter `forward` never reads is absent. Buffers such as BatchNorm `num_batches_tracked` are present with no users. Two placeholders that alias one tensor become one entry.
- **ties** are declared by the provider from the module itself (shared parameter storage), not guessed from bytes.
- **module_path / module_type** come from `nn_module_stack` (innermost module; the class name, not a qualified path, so it does not depend on how a test module was imported). Root-level ops have path `""`. A path is kept only if it names a module of the exported model (`named_modules()`); when `forward`'s receiver is not literally `self`, torch 2.6 reports Dynamo source strings such as `getattr(L['s'].layer1, '0').conv`, and those become `null` (with a `null` module type) so they never reach `ir_sha256`.
- Static shapes only. Node order and ids are deterministic, so building twice (or in another process) gives the same `ir_sha256` (tested, including a subprocess).

## Providers

`GraphProvider.build(spec) -> GraphIR`. Three providers exist; `builtin` first: spec `{"provider": "builtin", "model": "<registry name>"}`. The registry is a process-local dict filled by code (`register_builtin(name, factory, example_inputs)`); a spec can name only a registry key (`^[a-z][a-z0-9_.-]{0,63}$`: no module paths, no colons), so no arbitrary import is reachable. `reference_cnn` (the MNIST adapter) registers itself; the T6 corpus families register from `tests/`. Unknown names, malformed specs and unknown providers are structured errors (`unknown_model`, `invalid_spec`, `unknown_provider`). `export_to_ir(module, example_inputs, provider)` is the generic `torch.export` to IR conversion (CPU, default torch 2.6 export, signature-based parameter binding) that the isolated `torch_export` provider (below) reuses inside its worker; the builtin provider never runs user code.

### `torch_export` (user models, opt-in)

Spec: `{"provider": "torch_export", "entry": "pkg.mod:factory", "path": "<absolute dir put on sys.path>", "example_inputs": [{"shape": [1, 3, 224, 224], "dtype": "F32"}], "root": null | "backbone"}`. `factory()` takes no arguments and returns an `nn.Module`. `root` exports one submodule (the workaround for post-processing the exporter cannot trace); parameter names then carry the `root.` prefix, so the graph binds to the full checkpoint.

Gating (owner decision Q1, option 3): off unless the server starts with `--allow-user-code` (`create_app(allow_user_code=True)`); attaching is human-only (agent tokens get 403); `--export-timeout` (default 120 s) bounds the worker. The provider list reports `torch_export` as unavailable, with a reason, while it is off.

What the isolation does: the export runs in a separate process with its own working directory, a scrubbed environment (no LayerDelta credentials or tokens in its environment; only OS identity and path variables survive; it can still read files the OS user can read), no stdin, a hard timeout (the whole process tree is killed, grandchildren included), caps on log, result and total input size, and one export at a time. The result is canonical JSON only, re-verified with `verify_ir` and labelled with the provider name by the dispatcher, never by the worker. Nothing the user code returns is unpickled, imported or executed by the server. A crash, hang, exception or malformed result leaves no graph attached.

What it does not do: it is not a sandbox. The worker runs as the same OS user and can read what that user can read, so use it only for code you already trust to run.

### `onnx` (an `.onnx` file, no code runs)

Spec: `{"provider": "onnx", "path": "<absolute .onnx file>", "dynamic_dim": 1}` (`dynamic_dim` optional, 1 to 4096: the value given to every dynamic input dimension, because the IR is static-shape). Attached like any provider (`POST .../graphs`, human-only). It needs the optional `onnx` package (`pip install "layerdelta[onnx]"`, pinned `onnx==1.17.0`; a protobuf parser, no runtime, no code from the file is executed); without it the provider card says so. The file is read as data only, up to 512 MiB, and external tensor data is never read.

Unlike the other providers it builds against the checkpoint's table, because an ONNX file carries its own weights and they may not be the checkpoint's. An initializer becomes a graph parameter only if its name is a checkpoint tensor, the shape is equal, the dtype is equal and the sha256 of its bytes equals the checkpoint tensor's. Every other initializer the graph reads becomes an `onnx.Initializer` node with `kwargs {name, reason}`, and the checkpoint tensor it might have been stays `unbound`. Reasons: `not_in_checkpoint` (BatchNorm folded into Conv, renamed or pre-transposed constants such as `onnx::MatMul_12`), `shape_differs`, `dtype_differs`, `value_differs` (the checkpoint was edited after export, or the file holds other weights), `external_data`, `unreadable`. List them with `GET .../graph?op=onnx.Initializer`.

Ops are named `onnx.<OpType>` (`<domain>.<OpType>` outside the default domain). Inputs are the positional `args` (an omitted optional input is `null`), attributes are `kwargs` (strings `{"str": ...}`, tensors up to 64 elements as lists, larger tensors and subgraphs as a `{"str": ...}` stub). A node whose outputs are used beyond the first returns a tuple, with `_operator.getitem` nodes. `module_path` is the common prefix of the node's bound weights (ONNX has no module stack), `module_type` is `null`. `provider.version` is `onnx-ir<ir_version>-opset<default opset>`, `spec_sha256` covers the spec, the file bytes and the checkpoint's table signature.

Coupling rules (`coupling_onnx.py`, default-domain opsets 11 to 21; any other opset makes every op opaque with a warning) map ONNX ops onto the torch rules: `Conv`, `ConvTranspose`, `Gemm` (only `transA=0, transB=1`, which is `nn.Linear`), `MatMul`, `BatchNormalization`, `LayerNormalization`, `Gather` on axis 0 of a parameter (embedding), `Add/Sub/Mul/Div/Max/Min`, `Concat`, `Flatten/Reshape/Squeeze/Unsqueeze/Transpose`, pools, and common element-wise ops. Everything else (`ReduceMean`, `Slice`, `Split`, `Resize`, `LSTM`, packed `MatMul` weights) is an opaque boundary and shows up in coverage as such. Known gaps: a torch export deduplicates equal-valued initializers into `Identity` nodes (only for untrained, identical tensors), and pre-transposed `MatMul` weights are not matched to the checkpoint by content.

## Binding

`bind(ir, table) -> BoundGraph`. Every IR parameter must exist in the table with the same shape and the same dtype family (an F32 graph binds to a BF16 checkpoint); otherwise `GraphMismatch` (`graph_mismatch`) lists the offending names and reasons, and nothing attaches. Per table tensor:

| status | meaning | example |
|---|---|---|
| `bound` | the graph uses it; `uses` lists `{node, op, arg, path, module_path, module_type}` (`arg` is the argument position or kwarg name, `path` the index inside a list argument) | conv weights |
| `unused` | in the graph, no consumer | `num_batches_tracked` |
| `unbound` | in the table, not in the graph | a parameter `forward` never reads (`tok_embed.weight`) |

A fused op is decoded by position: the single `aten.lstm.input` node lists all weights in one list argument, so each LSTM tensor has one use with `arg: 2` and `path: [k]`; the flat order is layer-major, then direction (forward, reverse), then `weight_ih, weight_hh, bias_ih, bias_hh` (checked in the tests against `num_layers` and `bidirectional` read from the node). Tied names share the uses of the name the graph carries, and a tie member may be absent from the checkpoint (safetensors stores tied weights once); a stored member then stands in for the parameter and is checked for shape and dtype family exactly like it (`graph_mismatch`, labelled `member (tied to param)`). Roles and coupling are G3.

## Graph store

`GraphStore(catalog)` over the catalog's SQLite file (table `graphs(ir_sha256, provider, spec_sha256, table_signature, ir_json, created_at)`). `table_signature(table)` = sha256 of the sorted `[name, dtype, shape]` list (values and file layout do not matter). `put(ir, table, provider=...)` proves the binding first, then attaches the graph to that signature (idempotent); `for_table(table)` and `graph_for(table)` find graphs by signature, so every checkpoint with the same layout inherits its parent's graph, including each child written by `apply`. The provider recorded is the one the dispatcher built the graph with (`providers.attach(store, spec, table)` passes the spec's provider), never read from the graph: a graph that names a different provider than the caller is refused (`provider_mismatch`), so a worker cannot label its graph as another provider's. A different dtype is a different signature and inherits nothing (the graph can still be attached to it explicitly). One signature may carry several graphs (one per provider); `graph_for` returns the newest, or the one of a given provider. Rows are verified on load.
