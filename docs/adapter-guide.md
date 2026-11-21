# Adding an adapter

The release accepts checkpoints for `mnist-two-conv`, revision 1. To try another set of trained weights with this topology, upload a Safetensors file through Model explorer. The loader checks every tensor name, shape, dtype, byte extent and finite value before constructing a fresh CPU model.

Adding a different topology requires a source change. The adapter is compiled into the application so a checkpoint can supply tensors only.

1. Define the model and its forward pass in [model.py](../src/layerdelta/model.py). Give changed semantics a new architecture or adapter revision.
2. Register each editable tensor with `target_catalog`. Specify its mode, module path, zero-based channel count and axis. A kernel edit also needs a fully qualified parameter name. The activation target must identify the output immediately after ReLU. Keep the convolution bias outside the kernel slice.
3. Update the state schema and allocation cap in [checkpoint.py](../src/layerdelta/checkpoint.py). Load a fresh instance with strict keys and reject nonfinite values. Include kernel shape and output shape in the fixture so the UI can draw the true spatial dimensions.
4. Freeze the adapter revision, dataset dependencies and target mapping into the recipe via [domain.py](../src/layerdelta/domain.py). A new architecture also needs its own fixture and explicit dispatch in the worker. Update the preview's mapping between activation and weight targets in [engine.py](../src/layerdelta/engine.py).
5. Exercise the first and last channel of each target on both devices. Check scale 1, zero versus scale 0, and a biased convolution where activation and weight edits produce different outputs. Compare parent bytes after success, failure and cancellation. Export a child and reload it through the ordinary loader. Compare every sample's prediction to the source weight run.

The [frozen weight recipe](../examples/recipes/zero-block2-channel17.json) uses the documented reference checkpoint and all blurred true nines for the primary slice. A locally trained checkpoint with a different digest needs a new recipe. The import endpoint reports the mismatched dependency.
