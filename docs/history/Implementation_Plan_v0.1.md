# LayerDelta v0.1 — Implementation and Acceptance Plan

> **Status:** M0–M4 complete (internal). The pilot and publication steps are parked under *Release readiness* in the [Roadmap](../roadmap/Roadmap.md).

Start from a repository that contains the design documents. The goal of the release is for users on Windows x64 to open the example CNN, inspect a channel, try a single-point edit, see which images become correct or incorrect, and hand the recipe to an Agent for a repeat run. Correct and auditable measurement is the acceptance criterion. A positive change in accuracy is a hypothesis that can be tested.

Read the original requirements together with the [PRD and G1–G10](LayerDelta_PRD_v0.1.md), the [Architecture](../architecture/Architecture.md), the [HTTP API](../reference/api.md), the [Agent guide](../guides/agent-integration.md) and the [open-source proposal](LayerDelta_Open_Source_Proposal.md).

## Scope confirmed after reading the documents

| Decision | Effect on the implementation |
|---|---|
| MNIST and one architecture with two Conv blocks | One built-in adapter. Other checkpoints can be loaded when their state schema matches. General models, multi-point edits, in-app retraining and MCP are moved to after v0.1. |
| Verify the evidence before proposing a recipe | The typed HTTP client must be able to search for paired normal/blurred images and request a preview for each channel. These two capabilities are added to the existing tools table. The headless example must complete the whole workflow using the real client. |
| Windows x64, CPU and NVIDIA CUDA | Both devices are acceptance targets, from preview, baseline, intervention and holdout through to loading the child back and checking its results. The device actually used is recorded every time. The user chooses to switch to CPU when the GPU is not working. |
| Measure correct fixes and side effects | Pick the demonstration number pair from the baseline on the exploration split. Measure the accuracy of blurred images whose **true label** is the target class. Compare with the accuracy on normal images, and show the results across classes. Keep a separate count of cases that were wrong and became wrong in a different class. |
| The human controls confirmation and checkpoints | The Agent explores within the project/session and the given budget. The user confirms the frozen recipe before the holdout is opened, and confirms the export of a child from a successful weight run. |

## Order of work and hand-off points

The main path is **a fixture with a verifiable identity → a runner that can verify its answers → a Desktop and an Agent that use the same service → fault and pilot testing → installation from the real release.** Experimental packaging work and GPU machine surveys can start alongside the fixture. Once the sample schema passes M1, Desktop and the typed client can be developed in parallel. Both sides must be tested against the real backend before M2 is complete.

### M0 · Pin the example and the acceptance environment

| Task | How to do it | Evidence before hand-off |
|---|---|---|
| 0.1 Create the reference model | Define the topology with two Conv blocks and bias. Name the layers and the hook after the ReLU. Identify the kernel parameter, channel axis, shape, dtype and the read-only points. Write a script that prepares the checkpoint, with the seed, training config and provenance. | Adapter specification, and a manifest that points to the parent Safetensors by SHA-256, plus examples of the first and last channel of each target. |
| 0.2 Create the dataset manifest | Split into train/exploration/holdout using source IDs before creating normal/blurred variants. Assign the ID of each variant. Store the source digest, the preprocessing revision, the blur parameters and N per class. | Uniqueness and split disjointness can be checked. Paired images from the same source can be found. Rebuilding the manifest gives the same members and digests. |
| 0.3 Choose the demonstration task | Use a first baseline on the exploration split. Check the confusion counts, how clear the images are, and how many examples the user can inspect. Choose the number pair and the blur level from the actual results. Pin the task and record the reason before the quickstart is developed. | A class/condition table, the sample IDs used in the demonstration, the definition of the primary slice, and a normal guardrail. The runner in M1 confirms the reference results again. |
| 0.4 Prepare Windows CPU/CUDA | Identify a real CPU machine and a real NVIDIA machine. Record the GPU model, VRAM, driver, OS, and the Python/PyTorch build. Try loading a checkpoint and running a forward pass from a child process on both. | A table of runtimes that can be tested, with the commands to reproduce it. If no GPU machine can be found, continue the work on CPU and keep CUDA as pending acceptance. |
| 0.5 Try packaging | Try a desktop shell that opens the React renderer, starts and stops the Python backend, selects a workspace, and finds the runtime on a clean machine. Test the CPU install path without NVIDIA, and the CUDA install on the target machine. | Record the shell choice and the way the runtime is distributed, with evidence for startup, install size and limits. Use this result to choose the packaging method before M2 starts. |

M0 is complete when the fixture can be opened, its origin and mapping can be checked, and there is a data set to compute the M1 answers. It also includes small examples whose expected results can be calculated by hand. Evidence of usage rights and of how artifacts may be distributed is collected from this stage, so that the delivery method can be chosen in time for M4.

### M1 · Build the runner from the CLI until results can be proven

1. **Domain and loader.** Create schemas for the adapter, dataset, evaluation, recipe and run. Keep the canonical recipe hash separate from the execution fingerprint, so that one recipe can move between workspaces and devices. Before loading Safetensors, check the header and file size, the key set, shapes, dtypes and byte counts. Check that values are finite, and do a strict load into a new model instance. Use a golden recipe to confirm the hash, and report dependency mismatches field by field.
2. **Device and baseline.** The CLI accepts the selected device. Check its availability in the worker. Use eval/inference mode. Record the predictions and logits used to compare across devices, per sample ID. Store the runtime/build, the GPU and driver where relevant, dtype, precision/determinism settings, threads and batch size in the execution metadata. The fingerprint uses the settings that affect predictions, together with the checkpoint, adapter, manifest and evaluator. The baseline cache and the preview cache must be separated for CPU and CUDA, and for any settings that affect results.
3. **Two kinds of intervention.** Each run uses its own process and builds a working model from the original parent. For activation, attach a hook to the output after the ReLU, only for the selected channel, and remove the hook at the end of every case. For weight, modify the kernel slice of the output channel and keep the bias unchanged. Support zero and a finite scale, using the proposed range [0, 2]. The fixture checks identity at 1, that zero equals scale 0, the mapping of the first and last channels, and a bias case where the two modes give different results.
4. **Paired evaluator.** Pair the baseline and the experiment results by sample ID. Check that label, split and condition match. Reject items that are missing, duplicated or extra. Compute N, accuracy, delta in percentage points, the confusion matrix, and the counts of fixed, regressed and changed-wrong cases. Use the true label and the condition to define the primary slice. Pin the members of the selected slice from the baseline before the experiment. An empty group gives N = 0 and a null metric. Report the overall result as a micro-average over the selected sample-condition rows.
5. **CLI PoC and cross-device checks.** Run baseline → activation → weight → baseline separately on CPU and CUDA. Store JSON per image and the hash of the parent. Test failure and cancel during an experiment, then open a fresh baseline. On the same fixture, compare CPU and CUDA logits, measure the difference, and list the sample IDs of every image where the predicted class differs. Set the tolerance using a calibration set and lock it before regression testing. Keep the logits for diagnostics on the local machine only.
6. **Benchmark.** Record the whole-run time separately from inference time, the batch size, peak RAM/VRAM, the artifact size, and cold/warm cache, by machine. Use these to set the timeouts and preview sizes in the next milestone. Making the GPU faster than the CPU on a small CNN is not an acceptance criterion.

**M1 pass point:** a developer calls the CLI and gets the recipe, the baseline ID and per-image results, which can be checked against the fixture. G2–G5 pass on the supported runtime. There is a cross-device report, kept separate from the measurement of the intervention. If the mapping, the denominator or the isolation is wrong, fix the runner before connecting the UI.

### M2 · Connect the runner to the workflow that people and Agents actually use

| Step | What is built | How to verify through the real entry point |
|---|---|---|
| 2.1 Application and persistence | Shared application services for catalog, preview, propose, submit, status, cancel, compare and export. SQLite stores state and permissions. The filesystem stores immutable results. Start one supervisor per workspace, and run one child at a time. Record the run, quota charge and idempotency key in a single transaction. | After a lost response, submitting again returns the same run ID, and the budget is charged once. Jobs that are not yet published have a status that can be polled. |
| 2.2 HTTP contract | FastAPI/Pydantic following the existing endpoint catalog. Add the device capability and the requested/actual runtime to the relevant schemas. Preview returns the sample/source IDs, target, hook, kernel statistics, the mean/max values and the zero fraction of the activation. Define the data types, units, dimension reduction and schema version so that the API and the client match. | A contract fixture can open the normal and blurred images of the same source and the same channel, on both CPU and CUDA. Summary values match a direct computation on the tensor. The cache key includes the channel and the device. |
| 2.3 Desktop | Connect the screens Project/Model Explorer → Experiment Editor → Run Comparison → History & Artifacts → Final Evaluation. Choose the runtime before starting. Show the actual device, N, the images that were fixed or made worse, and before/after per image. Support keyboard use, loading, empty and error states. | Open from the icon. Select an image and channel, run an activation experiment, then create a separate weight recipe. Record run IDs. Open the history, and export the recipe and report from published results. |
| 2.4 Typed client and headless CLI | Start the backend on loopback through the CLI. Use a token limited to a project/session. Build typed tools from the same schema as HTTP. Add sample search and preview to the client as the real tasks need them. The example uses a deterministic stub to run inspect → pick the paired images → preview → check the budget → propose → run/status → compare/get_changes, with bounded polling and an idempotency key per submission. | The example works without signing up for an LM provider. A hypothesis cites real sample IDs, channels and statistics. It stops when it reaches the user's budget or the server's quota. The payload sent to a simulated provider contains only the summary fields the user has allowed. Images and maps stay on the local machine by default. |
| 2.5 Replay, holdout and child | Import a recipe into a new workspace and verify it by digest. The human confirms the frozen recipe through the UI before a holdout run. Record an audit entry, and keep the baselines for each split/device separate. Use the existing weight-patch implementation to build a Safetensors child. Record the parent/child SHA, the recipe hash and the source run ID. | Open the child through the normal loader, and run it with the intervention turned off on the source runtime. The prediction for each image matches the source run, and the tensor diff is limited to the approved slice. The recipe and report for both modes can be opened. |

**M2 pass point:** the desktop and headless entry points together cover the workflow from opening the demo to replaying it. G1, G6 and G7 pass at the alpha level, using the same recipe through the UI and the Agent. The changed image IDs, counts and metrics match on the same runtime. This is repeated on CPU and CUDA. The API, architecture, PRD and Agent guide documents are corrected to match the schema actually in use, especially for preview and device selection.

### M3 · Beta, recovery and pilot

| Test set | Cases and results to check |
|---|---|
| Publish, cancel and restart | Write staging on the same filesystem. Check the digests before publishing, then commit the catalog together with the terminal state. Simulate a full disk, a worker crash, a timeout, a crash after publishing but before committing, and a cancel that races with success. There must be one terminal state. The API reads only committed results. A restart keeps the queued jobs, marks interrupted running jobs as failed/interrupted, and the workspace can be recovered. |
| Permissions and budget | Keep the human session and the Agent token separate. Check permissions before reading any object or artifact, every time. Test another project, the wrong scope, holdout access through the catalog, sample, preview, history, cache, report and URL, Agent export, and CSRF in the human UI. Test concurrent submission, the same key with a different payload, a full queue, an exhausted quota, retry after failed or cancelled, and stopping at the user's budget even when the server still has quota. |
| Data correctness | Test a recipe with a wrong schema, a hash mismatch, a tensor with the wrong shape or with non-finite values, missing or duplicated samples, source IDs crossing splits, N = 0, and a lost manifest of results. The baseline cache must drop the bad results. Return error codes that the UI and the Agent can use to make decisions. |
| GPU and resource recovery | Start the app on a machine without CUDA and use the CPU. When the selected GPU is unusable, return the reason and wait for a new choice. Simulate GPU out-of-memory, cancel and worker exit, then run the next job. Check the parent hash, the actual device, and the cache values after switching compute devices. |
| Clean install and performance | Test the Desktop and headless app on Windows CPU and NVIDIA according to the support matrix, including a path on a machine with no development environment. Measure the quickstart, the time from submit to comparison, peak RAM/VRAM and disk use, against the values frozen from the PoC. |
| Pilot | Give the same task to people using the workbench and to a reference script. Swap the order. Measure the time to a conclusion and the checks on the results. Give the same Agent model two conditions with equal budget: one sees aggregated results, the other sees per-channel data. Measure valid recipes and the direction of predicted results before they are seen. Test a candidate that was frozen against the holdout, with a normal guardrail and a holdout opening history. |

**M3 pass point:** G1–G9 results from the real entry points. A pilot report that gives the number of samples, the errors and the limitations. Results where accuracy stays the same or drops are reported as pilot results.

### M4 · Open-source release

1. Build the Desktop package and the way to install the headless CLI, following the packaging spike. Pin dependency profiles that have been verified. Record the OS, GPU, driver and runtime actually tested, along with the steps to check the device when the app starts.
2. Prepare a quickstart from a clean machine, a numeric Agent example, a guide for adding an adapter, sample schemas and recipes, release notes, and issue templates that let contributors attach fingerprints and run IDs.
3. The project owner chooses the code license. Apache-2.0 is the proposal. Check the data rights for the checkpoints separately. Prepare steps for users to download or build the artifacts from permitted sources, with digest checks. If the files cannot be distributed directly, use this approach.
4. Test the release candidate from the package that will actually be released, on both CPU and CUDA. Open the demo, replay it across workspaces, and re-import the child. Keep the G1–G10 results, and record the package version and the test machines before declaring v0.1.

## Acceptance evidence table

| Gate | Evidence kept | Checkpoint |
|---|---|---|
| G1 Quickstart | Record of a clean install and the comparison from the Desktop/headless | M2 alpha, M3, M4 real package |
| G2 Mapping | The first and last target/channel match the tensor/hook in the fixture, for every target | M1 |
| G3 Semantics | Scale 1, zero, scale 0 and the bias case, in both modes | M1 |
| G4 Isolation | Per-image baseline matches the parent hash before and after success, failure and cancel | M1, M3 |
| G5 Metrics | Golden counts, confusion matrix, slice membership, delta and empty slice | M1 |
| G6 UI/API parity | The same recipe and runtime give the same counts, metrics and changed IDs | M2, M3 |
| G7 Export/replay | The recipe hash stays the same across workspaces. The loaded child gives per-image results matching the source run, with tensor-diff evidence | M2, M4 |
| G8 Failure paths | Error codes, quota, restart, disk failure and an unavailable device pass the fault suite | M3 |
| G9 Holdout | Split-disjoint manifest, audit of the frozen recipe, and test results for Agent permissions on every channel | M3 |
| G10 Release | Windows packages and install steps, documentation, provenance, license, and the list of hardware/runtimes that passed | M4 |

## First tasks when implementation starts

Start with 0.1 and 0.2, to get the first checkpoint, target mapping and sample manifest. Work on 0.4–0.5 in parallel, to prove the worker path on Windows. Then run the baseline on the exploration split, to choose the demonstration task in 0.3. The first delivery is the fixture with its fingerprints and a set of paired example images, which the runner, the UI and the Agent will all refer to.
