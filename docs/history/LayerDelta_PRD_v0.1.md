# LayerDelta — Product Requirements Document

> **Status:** v0.1 scope, delivered as an internal build. The product direction has since broadened to model-agnostic weight-space tooling; current milestones and status are in the [Roadmap](../roadmap/Roadmap.md). Examples below that use 8→3 predate the frozen fixture, whose demonstration slice is blurred true 9 (most frequent error 9→4).

M2 alpha exposes a two-block MNIST workbench through the Windows desktop launcher and a loopback headless service. The explorer requests the original image, one post-ReLU channel map, and kernel statistics for a paired exploration sample. Each run records the actual device. Final evaluation and weight-child export require a human session and explicit confirmation. The request and response payloads and their units are documented in the [HTTP contract](../reference/api.md).

| Item | Details |
|---|---|
| App name / repo and CLI | LayerDelta / `layerdelta` |
| Version | v0.1 · Draft for development planning and acceptance |
| Date | 22 November 2026 |
| Product type | Local-first open-source workbench for experimenting with deep learning models |
| Main user groups | Students, researchers and ML engineers who need to analyze model behavior |
| Source document | [LayerDelta Open Source Proposal](LayerDelta_Open_Source_Proposal.md) |

## 1. Problem and product value

When a CNN reads the digit 8 as 3 more often on blurred images, the experimenter can open a confusion matrix to see the error pattern. But testing which output channel contributes to those errors requires writing code in several steps: identifying the layer and the hook point, running the same paired data, computing results per group, and keeping evidence of what each run changed.

LayerDelta lets the user pick a target that the system can identify, state a hypothesis, run one change, and see how the predictions change in the target group and in other groups. The Desktop UI and the Agent API use the same validation rules, experiment recipes and measurement results. A recipient can therefore check the settings and run the experiment again, provided they have matching artifacts.

**Main result of v0.1:** the user starts from the example CNN and passes the recipe, together with the baseline results and the intervention, to a person or an Agent who can run one complete cycle again. A gain in accuracy is a possible outcome of an experiment. Reports of improved, worse and unchanged results must all be auditable to the same standard.

## 2. Users and the work they need to do

| User | Main task | Evidence that the task is done |
|---|---|---|
| Students and instructors | Look at a filter and the activation for the same channel, then try changing its value | Can explain where the change was made and point to the predictions that changed |
| Researchers and ML engineers | Inspect failure slices and find effects across groups | Compare results on the same paired examples, with the sample counts and the adverse effects that occurred |
| Agent developers | Let an Agent explore targets and test hypotheses within a budget | Use the structured API from inspect to compare, and can audit how much budget the Agent used |
| Open-source contributors | Reproduce results or prepare support for a new model | Open the examples, follow the guide, and understand the contract of the model adapter |

## 3. Scope and priorities

P0 is the functionality that must pass before v0.1 is opened. P1 is value-adding work that can be prioritized after the main workflow passes acceptance.

| Priority | Scope |
|---|---|
| P0 | One PyTorch CNN architecture classifying digits 0–9, one reference checkpoint, and normal/blurred images that can be regenerated |
| P0 | Local desktop app, layer diagram, list of output channels, weight statistics and activation map for the selected image |
| P0 | Activation intervention and filter weight edit, zero/scale, one target at a time |
| P0 | Baseline, before/after, per-group results, run history, JSON recipes, export of measurements and of the child checkpoint from a weight edit |
| P0 | Inspect/propose/run/compare API that uses the same backend as the UI, with an Agent budget limit |
| P1 | Charts comparing multiple runs, advanced history search, and helpers for exploring several channels |
| After v0.1 | General models and custom code, arbitrary tensor slices, multi-target edits, retraining, Transformers, OCR recognizers, training-time intervention, and automatic publication of checkpoints |

### Proposal for the initial specification

Use **MNIST and a small CNN with two Conv blocks** as the starting fixture, so that there are digits 0–9 and several output channels to explore. The layers that the UI exposes for editing are defined by the adapter of the model bundled with the program. The real topology is chosen and the checkpoint is frozen during the PoC. The proposed scale values are limited to the range [0, 2], inclusive at both ends. A value of 1 is used to check identity, and 0 gives the same result as zero at the same position. The real architecture, the Agent budget figures and Apache-2.0 are still proposals.

The Desktop packaging and headless mode are P0 for v0.1. MCP is an integration layer built on top of the HTTP tool contract. It must pass acceptance, but it does not block the release. The target OS and the desktop shell are to be confirmed by the packaging spike in M0–M1.

## 4. Acceptance workflow

1. The user opens the example project. The system shows the architecture, the checkpoint fingerprint and the dataset version.
2. The user selects a layer and a channel, then views the kernel shape, the weight statistics and the activation map for the chosen image. The system shows where the hook is, for example the output after the ReLU of a block.
3. The user runs the baseline on the exploration split, then looks at a slice of blurred images and at the examples the baseline reads as 8 → 3.
4. The user creates a recipe, for example multiplying the activation of the selected channel by 0.5, and records the hypothesis that the change will alter the number of 8 → 3 errors on blurred images.
5. The user reviews the paired comparison and checks the results on normal images, on other classes, and on the examples that were fixed or became wrong.
6. The user creates a separate weight-edit recipe that multiplies the kernel of the same channel, then checks the evaluation results again.
7. The user exports the recipe and metrics so that an Agent or a colleague can run them again. Then the user chooses which recipe to evaluate against the holdout, and, if wanted, approves the export of a child checkpoint through the UI.

## 5. Functional requirements

| ID | P0 function | Behavior and acceptance criteria |
|---|---|---|
| FR-01 | Open the example project | The quickstart opens the reference CNN up to the inspect page, which shows the model ID, architecture revision, checkpoint hash and dataset revision, matching the manifest |
| FR-02 | Load a checkpoint | Accepts a tensor-only checkpoint for a supported architecture. Checks the file size, the list of tensors, shapes, dtypes and the schema before inference. A file that does not match the spec gives a message naming the point of failure. Existing projects keep working. |
| FR-03 | Explore the model | Shows the sequence of layers and output channels from the adapter, with indexes starting at 0, tensor shapes, kernel statistics, and an activation map linked to a sample ID. Points that can only be explored are marked read-only. |
| FR-04 | Baseline | Uses the source checkpoint, eval mode, the preprocessing and the sample set from the manifest. Stores predictions per sample ID together with the metrics. Reusing an earlier result requires all the fingerprints of the evaluation conditions to match. |
| FR-05 | Define an experiment | The form and the API accept the hypothesis, target ID, mode, operation, value and evaluation config, all under the same schema. A successful propose creates a recipe in a ready-to-run state, with no measurements yet. |
| FR-06 | Activation intervention | Supports zero or scale on the output channel at the declared hook. Applies to every sample in the run. The fixture criterion is that the selected channel changes as the recipe says, and that the other channels at the same point keep their original values. |
| FR-07 | Weight edit | Zero or scale applied only to the kernel slice of the selected output channel. Shows the tensor name, the slice and the statistics before and after. Checks that the values are finite after the transformation, and records the hash of the parent checkpoint. |
| FR-08 | Manage runs | Shows the status and the observable progress. Supports cancel and retry. A subsequent run on the same baseline must give the same results as the earlier run, whether it succeeded, failed or was cancelled. The source checkpoint hash is checked as part of this. |
| FR-09 | Compare | Shows baseline and intervention, and the delta, on the same group. Includes the sample count, the confusion matrices and a list of the examples whose predictions changed. The UI and the API give the same results from one run. |
| FR-10 | Recipes and history | Opens history from a run ID, and can export and import recipes as JSON. If the fingerprint or the schema does not match, the system names the dependency that must be available before the recipe can be run. |
| FR-11 | Export results | Exports the recipe and the metrics as JSON for both modes. Shows the number of samples, the data scope, and the provenance needed to interpret them. |
| FR-12 | Export a checkpoint | The user confirms it from a successful weight run. The system creates a child artifact that records the parent hash, the recipe ID and the child hash, without overwriting the existing name. Loading the child back and evaluating it without the patch must give the same results as the source weight run. |
| FR-13 | Agent API | The Agent reads the target catalog, proposes recipes, runs them and reads the comparison, under the credential and budget it was given. The backend checks permissions for export and final evaluation separately from the permission to run exploration. |
| FR-14 | Final evaluation | The user confirms the frozen recipe before it starts. The system records that the holdout was opened, together with the recipe hash, and returns results that can be compared across overall, normal and blurred, with N for each. |

### Meaning of an edit

| Mode | Target | How it works | Artifact |
|---|---|---|---|
| Activation | Output tensor at the hook point the adapter declares | Changes the channel slice by index, and keeps the shape and dtype through the whole forward pass | Recipe and run result |
| Weight | Kernel of the Conv output channel that the adapter maps | Changes the slice of the weight tensor in the working model. In v0.1 the scope covers kernels only. | Recipe, run result and child checkpoint when the user confirms |

Whether the position is before or after the activation function is part of the target's identity. For example, multiplying the kernel before the bias and ReLU is a different experiment from multiplying the feature map after the ReLU. The UI must show this position both when the target is selected and when the results are read.

## 6. Data and evaluation method

### Dataset and split

- Split into train, exploration and final holdout using the source sample IDs before the transformed images are created. Store the membership lists and hashes in the manifest.
- Create normal and blurred images from the source images using a transform with a fixed revision and fixed parameters. IDs must distinguish the source image from the condition.
- Every comparison pairs the baseline and the intervention by sample ID, split and condition. It shows the N of each group, and also the weighting definition for the overall result when two conditions are combined.
- A slice such as "baseline reads 8 as 3" has its members pinned from the original baseline for the whole comparison. The primary holdout slice uses a condition defined in advance, for example label 8 and blurred.
- By default, the exploration screens block access to the holdout through both the Desktop UI and the Agent API. A recipe that will be reported as a confirmed result must be recorded before the holdout is opened. Repeated looks at the holdout must be reported when the results are summarized.

### Metrics and display

| Measure | Definition used |
|---|---|
| Accuracy | Number of correct predictions divided by N for the same group. Shows baseline, intervention and the difference in percentage points. |
| Confusion matrix | Rows are the ground truth and columns are the predictions. Both use the class order 0–9. |
| Per-class results and image condition | Accuracy on the samples whose label matches the class, split into normal/blurred according to the given filter, with N shown. |
| Prediction changed | Number of sample IDs whose predicted class changed, split into wrong→right, right→wrong and wrong→wrong in a different class. |
| Empty group | Shows N = 0 and a null metric, with the reason, so that export and UI interpret it the same way. |

For results from small groups, the reader sees the actual counts alongside the percentages. Conclusions are reported as the effect of the intervention as defined on that evaluation set. A claim that a weight edit is useful in practice requires results from actually editing the weights, measured on the holdout, and checked alongside the results on normal images.

## 7. Data contract and API

### What must be stored in a recipe and a run result

| Data | Key fields |
|---|---|
| Recipe identity | schema version, recipe ID, hypothesis, creator type (human or agent), hash of the frozen recipe |
| Model provenance | model ID, architecture revision, adapter version, parent checkpoint SHA-256 |
| Data provenance | dataset ID/hash, split, sample manifest hash, preprocessing revision and parameters, and the seed when randomness is involved |
| Intervention | target ID, mode, channel index, hook position or tensor slice, operation, and scale when relevant |
| Evaluation | evaluator version, metric IDs, slice IDs and the runtime metadata needed to check reproducibility |
| Result | run ID, baseline run ID, recipe hash, status, metrics, sample counts, prediction differences, and the error code if the run failed |

A frozen recipe can be reused, and changing a value creates a new revision. Each run gets a new run ID, even when it uses the same recipe. To check reproducibility, refer to the recipe hash together with the relevant fingerprints and the measurements.

| Action | Minimum behavior | Rejection cases to test |
|---|---|---|
| Inspect | Returns the model, the confirmed structure, target IDs and the allowed operations | Model or target not in the catalog |
| Propose | Validates the payload and returns a recipe ID or field errors | Channel out of range, wrong mode, scale outside the range or not finite |
| Run | Checks permissions, fingerprint and budget, then returns a run ID | Dependency mismatch, quota exhausted, holdout access without permission |
| Run status / Cancel | Returns the status and progress, and the cancel command, subject to permissions | Run does not exist, run already finished, run belongs to another project |
| Compare | Returns the paired comparison result from a completed run | Run not yet finished, or evaluation fingerprints do not match |
| Export checkpoint | Creates the artifact after the user confirms | Activation run, unsuccessful run, duplicate destination, Agent credential |

The detailed HTTP paths belong to the API specification. The semantics in the table above are shared between the Desktop UI and the Agent API.

## 8. State, recovery and quality requirements

- **Lifecycle:** uses the states queued, running, succeeded, failed and cancelled. A retry creates a new run ID and references the earlier run. When the system restarts, a run that was cut off in the middle is marked as interrupted, under the failed state, and can be retried.
- **Isolation:** the MVP uses one evaluation worker by default and stores requests in a queue. The working model is separate from the source checkpoint. The system must remove the hook and discard the working state on every path through a run, including worker failure.
- **Artifact integrity:** writing the recipe, the metrics and the checkpoint must commit atomically at the boundary of the artifact the user can open. If a write fails, a half-written result must never be shown as a success.
- **Reproducibility:** on the same reference runtime, the fixture identity must produce identical class predictions and confusion counts for every sample. The runtime version and the device are recorded so that results can be compared across machines. If floating-point tolerances are needed, they are defined in the PoC before the release criteria are locked.
- **Agent budget:** the maximum number of runs and the timeout per run are set from the local config. Quota is reserved atomically when the backend accepts a run. Repeated requests are covered by an idempotency key per session, and the remaining budget is shown. The proposed default is 10 runs per session, for testing the UX before adjusting it based on PoC results.
- **Local-first:** the UI and API bind to loopback by default. Running the model and evaluating results does not require connecting to an LLM provider. Data and checkpoints stay on the machine. The Agent example states which data is sent to a provider when such a connection is enabled.
- **Usability:** there are status screens for loading, failed runs, empty groups, and results that can be read across several groups at once. The user can review results from the keyboard, and can distinguish improvements from regressions by text or by symbol.
- **Resource use:** measure the baseline run time, the display time, peak RAM and the size of artifacts on the reference fixture from the PoC. Each milestone states the dataset and runtime used for measurement before a performance budget is set.

## 9. Main screens

| Screen | Data and interactions |
|---|---|
| Project | Opens the demo, shows the checkpoint and dataset fingerprints, and starts a baseline |
| Model Explorer | Select a layer, channel and image. Shows the shape, kernel statistics, activation map and hook position |
| Experiment Editor | Sets the hypothesis, mode, target and operation. Shows a preview of the scope of the edit and the errors per field |
| Run Comparison | Shows status, overall results, the confusion matrix, per-group results, and the images with their paired predictions |
| History & Artifacts | Opens older recipes and results, enters recipes from JSON, exports results, and requests creation of a child checkpoint |
| Final Evaluation | Shows the selected recipe and the holdout scope, asks for confirmation before starting, and then shows the results next to the baseline results |

## 10. Acceptance criteria before release

| Gate | Acceptance evidence |
|---|---|
| G1 · Quickstart | From a clean machine on the target OS, install and open the desktop app, run the demo until a comparison is shown. Also start the backend from the CLI and run the Agent without opening a window. |
| G2 · Mapping | The fixture checks the hook channel and kernel index for every target opened in the catalog, including the first and the last target. |
| G3 · Semantics | Zero and scale are tested in both modes. Scale = 1 gives predictions equal to the baseline. A fixture shows that the two modes give different results when there is a bias. |
| G4 · Isolation | The sequence baseline → edit → failure/cancel → baseline gives the same baseline predictions, and the source hash is unchanged. |
| G5 · Metrics | A fixture with known answers gives accuracy, confusion, delta and changed items that match the pre-computed values, including the empty-group case. |
| G6 · UI/API parity | The same recipe run through the Desktop UI and through the Agent API, with the same fingerprints, gives the same prediction counts, metrics and sample IDs. |
| G7 · Export/replay | A recipe can be exported and opened in a new LayerDelta instance. The child checkpoint loads and gives results that match the weight run. The tensor diff is checked against the parent. |
| G8 · Error paths | Checks malformed recipes, unsupported models, hash mismatches, exhausted quota, unauthorized export and disk write failure. |
| G9 · Holdout | Checks that source IDs do not cross splits, and that the Agent cannot see the holdout during exploration. Confirms the provenance of the final run results. |
| G10 · Release package | Has a desktop package for the target OS and install steps for the headless CLI, with the dependencies and their versions. Also includes the first experiment guide, the Agent example, the adapter guide, the origin of the artifacts, and the decision about distribution rights. |

## 11. Evaluating value after the PoC

| Hypothesis | How to measure and interpret |
|---|---|
| LayerDelta helps people analyze faster | Have users do the same task with LayerDelta and with a reference script, swapping the order. Measure the time to a correct conclusion, the tasks completed, and the checks for adverse effects across groups. |
| Intervention data helps an Agent form hypotheses | Use the same LM model, the same tasks, and equal total tool and time budgets. Compare a group that sees aggregated results with a group that sees per-channel results. Measure how accurately each group predicts the direction of an intervention's effect before seeing it, and the share of recipes that run successfully. |
| A weight edit is useful for the blurred task | Select a recipe from exploration and lock the candidate before the holdout is opened. Compare it with the original checkpoint and with a suitable reference method for the task. Report the normal/blurred results with N, and the number of examples that improved and got worse. |

Record baseline values for time and success rate during the pilot before setting numeric targets. The correctness of the tool is established by the release gates in section 10. The three research questions above are used to judge the direction after the pilot, including the case where the results are zero or negative.

## 12. Delivery plan

| Milestone | Main work | Exit criteria |
|---|---|---|
| M0 · Reference specification | Define the CNN, layer IDs, checkpoint, split manifests and preprocessing | The reference fixture opens and the fingerprints used in the PoC are locked |
| M1 · Runner PoC | Build the runner, the adapter mapping, the baseline, zero/scale and the paired comparison | Both modes run end to end on one machine, the JSON results are produced, and the identity and isolation checks pass |
| M2 · Product alpha | Connect the desktop shell, headless CLI, API, queue, recipe replay and export | The workflow from opening the demo to replay works through the Desktop UI and the Agent API |
| M3 · Beta validation | Test the holdout, failure paths, quickstart and a pilot with sample users | G1–G9 pass, and the issues are recorded together with the pilot measurements |
| M4 · Open-source v0.1 | Prepare the release package, guides and issue templates | G10 passes, and the project owner confirms the license and the right to distribute the artifacts |

## 13. Risks and decisions

| Risk | Response |
|---|---|
| A displayed graph suggests that every operation is supported | Use the mapping from the adapter, and show the real hook positions and tensor slices |
| Users treat a channel as if it were a class | Show all classes and the normal/blurred pairs, together with the improved and worsened examples |
| Tuning until the result wins on the exploration split alone | Fix the split by source ID. Record the number of runs and the recipes chosen before the holdout is opened. |
| Results change with the runtime or the data source | Record the fingerprints and use the reference runtime for the replay criterion |
| The Agent runs so often that its results are hard to follow | Limit the quota and allow one target per recipe. Keep provenance and show the remaining budget. |
| The right to distribute checkpoints or data is unclear | The project owner checks the rights for each artifact separately, and a permitted source is prepared before release |

| Open question | Proposed default | Decision point |
|---|---|---|
| License | Apache-2.0 for the code | Before the public repository is opened |
| Reference CNN and dataset | CNN with two Conv blocks on MNIST | M0, after the suitability and origin of the artifacts are confirmed |
| Blur severity and split size | Fixed parameters that produce a sufficient failure slice | M0–M1, based on the baseline results and sample counts |
| How to choose the LM | Support callers through one API. Use one LM model per comparison experiment. | Before the Agent pilot |
| Quota, timeout and performance targets | Use the config values and the benchmark results from the PoC | Before beta |

After v0.1, the proposed order is to add a small Transformer to test the adapter contract, then add an OCR recognizer for cropped text with Character Error Rate, and use the ablation results to form hypotheses for LLM-Guided Dynamic Dropout as a separate training project. The effect of the new dropout rule is measured in that project's own training protocol.
