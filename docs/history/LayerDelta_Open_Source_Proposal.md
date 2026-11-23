# LayerDelta
## Draft Open-Source Project Proposal v0.1

**LayerDelta** is a tool that lets people and AI agents explore deep learning models, try changing weights or activations, and measure the before and after results on the same data. It starts with a local-first CNN and is designed so that support for other architectures can be added later.

## 1. Problem and goal

When a model keeps reading digits incorrectly, an engineer may have a confusion matrix and examples of the failed images. But answering "which part of the model contributes to this error?" requires writing code to find the layer, capture activations, apply an intervention, run again, and compare the results. The same tool should let people run experiments through a screen, and let agents run experiments through an API, using the same specification and the same measurements.

The intended result is that the user states a hypothesis, chooses an intervention, checks the side effects, reverts the change, and passes the experiment to someone else who can reproduce it. Improving the model is a hypothesis that has to be proven case by case.

## 2. Main users

| User | What they want to do |
|---|---|
| Students and instructors | See how layers, channels and filters relate to each other, and experiment with a small model |
| Researchers and ML engineers | Analyze failure slices, test hypotheses, and record interventions that can be reproduced |
| Agent developers | Let an agent see the structure the tool can identify, read measurements, propose experiments and check results within a set budget |
| Contributors | Add loaders, structure identifiers or evaluation sets for new models and tasks |

## 3. User experience

1. Open the example project, which contains the architecture, a checkpoint, a labeled dataset and an evaluator.
2. View the model diagram, select a layer, and open its list of output channels. Check the shape and statistics of the weight tensor, or view the activation map for an example image.
3. Run the baseline and select a group of interest, such as blurred images or the images of an 8 that are read as a 3.
4. State a hypothesis, for example "channel 17 helps separate 8 from 3 in blurred images", and try turning off or multiplying that channel's activation by 0.5.
5. View the images and predictions before and after, together with the results on blurred images, normal images and other classes. Show the number of examples and the errors that increased or decreased.
6. If you want to test editing the weights, create a second experiment, for example multiplying the kernel of that output channel. Show where the edit was made and the values before and after, then run the evaluation again.
7. Save the experiment as a recipe that can be reopened, or export a new checkpoint while keeping the original.

The result of turning off an activation is evidence that the intervention at that position changes the model's output on the data used for testing. Concluding that editing the weights will help requires actually running the weight edit and checking it on held-out data.

## 4. Types of intervention

| Type | What is changed | Example | What is exported |
|---|---|---|---|
| Activation intervention | Values produced during the forward pass | Turn off a whole channel, multiply by 0.5, or add a constant | A recipe that can be run again |
| Weight edit | Parameters the model uses to compute | Multiply the kernel of a filter, or change a bias, per parameter | A patch recipe and a new checkpoint |
| Future training intervention | The rules the trainer uses | Let an LLM choose the random channel-dropout rule for the next round | The rules and the training history |

For the MVP, support zero and scale in both modes first. Addition, subtraction, division and free selection of tensor slices can be added later. The screen must state the intervention point clearly, for example before or after the activation function. Editing only the kernel can give a different result from multiplying the whole activation when a bias or other processing layer is involved.

## 5. MVP scope

| Part | Proposal for v0.1 |
|---|---|
| Model | A PyTorch CNN that classifies digits, with one architecture and one example checkpoint |
| Data | Digit images, with normal and blurred images that can be regenerated, and an exploration split kept separate from the final evaluation split |
| Screens | A web UI opened on the user's machine, showing layers, channels, tensor shapes and before/after results |
| Editing | Activation intervention and filter weight edits at the supported positions, with values restored during the experiment |
| Evaluation | Overall accuracy, confusion matrix, results by class and image condition, and the examples whose predictions changed |
| Reproducibility | JSON experiment recipes, run history, the source checkpoint ID, and exportable measurements |
| Agent | An API that calls inspect, propose, run and compare through the same backend as the UI |

Opening general files, inferring the computation graph from weight tensors, retraining the whole model, and an Agent publishing checkpoints are all out of scope for v0.1. They are in later versions, in order of priority.

## 6. Proposed architecture

- **Model adapter.** Connects an architecture with a checkpoint and states how the targets shown on screen relate to a tensor slice or a hook point. It returns only what is actually known. If only the tensor name and shape can be shown, the adapter must say so.
- **Experiment engine.** Checks the type of the target and the parameters, applies the temporary change inside a run, and removes the hook or restores the state when the run finishes or an error occurs.
- **Evaluator.** Computes the baseline and the intervention with the same preprocessing, dataset and evaluation method.
- **Artifact store.** Keeps the source checkpoint read-only, along with the experiment recipes, the measurements and any exported checkpoints. Exported checkpoints link back to their parent checkpoint.
- **UI and agent API.** Both call the same set of experiment commands, so an experiment can move from a person to an agent and back.

A `.pt` file may contain only a state dictionary or some other object. The tensor shapes make it possible to explore the parameters, but drawing the graph and running inference require a matching architecture. For the MVP, use an architecture bundled with the program and accept weights as tensor-only data that can be checked. Check names, sizes, data types and the file size before processing. Avoid running any code that comes with the user's file.

## 7. Experiment contract between people and agents

Every experiment records the model ID and the checkpoint hash, the adapter version, the dataset ID and hash, the preprocessing, the seed if randomness is involved, the target, the type of edit, the edit value, and the metrics required. A conceptual example command is: "On the blurred exploration images, multiply the activation of channel 17 after the last block by 0.5, and compare with the baseline."

- **Inspect** returns only the supported targets, with their layer names and the types of edit allowed.
- **Propose** accepts a hypothesis and checkable parameters. A proposal has a status of waiting to run.
- **Run** checks that the target exists, that the value is within the configured range, and that the experiment quota has not been used up. It then creates a run that is separate from other runs.
- **Compare** returns the totals, the per-group results, the number of samples, and the IDs of samples whose predictions changed.
- **Export** is a separate action that the user approves. Agents experiment temporarily within a budget, and use the data and checkpoint on the local machine by default.

## 8. Success criteria and evaluation plan

| Question | How it is measured |
|---|---|
| Does the tool work end to end? | Open the example model, view the channels, run the baseline, try both modes, revert, run the same recipe again, and open an exported checkpoint |
| Can the evaluation be trusted? | An identity edit gives the same result as the baseline. One run does not change the results of the next run. Export changes only the tensors that were meant to change. |
| Do people work better with it? | Have users do the same task with LayerDelta and with a basic script. Measure the time, the correctness of the conclusions, and the ability to check the effects on other groups. |
| Does an agent make better use of internal data? | Use the same LM model. Compare a group that sees aggregated results with a group that sees per-channel results, and have each predict the outcome of experiments it has not yet seen, with the same total tool budget. |
| Is editing the model useful? | Measure on the held-out set, against the original checkpoint and against a baseline suited to the task, such as preprocessing or fine-tuning. Report the side effects on normal images as well. |

For a future OCR experiment, use a recognizer on cropped text images, and measure Character Error Rate separately by blur and lighting. Check results on held-out text before adding a detector for end-to-end testing.

## 9. Development plan and extensions

1. **Proof of concept.** Build a Python runner that opens the example CNN, intervenes on one channel, edits a filter, compares both results, and outputs JSON.
2. **v0.1 for people and agents.** Add the UI, the API, recipe saving and a starter guide for the example model. The pass criterion is that people and agents run the same recipe and get equivalent results under the same conditions.
3. **Second model family.** Add a small Transformer to test whether the experiment mechanism can be reused. Specifically, define the mapping of attention heads in the adapter.
4. **Applied work.** Add one OCR recognizer and an evaluation set for text. Add files and architectures as users or contributors bring test cases.
5. **Research extension.** Connect experiment recipes to LLM-Guided Dynamic Dropout. The agent reads statistics and ablation results, then proposes a random mask rule for training. The effect of adjusting activations during inference should be used to form hypotheses for this work, and re-evaluated during training.

## 10. Opening the project and contributions

- Open the core, the UI, the API and the example CNN so they can run locally. Apache-2.0 is proposed as the default code license, as a proposal for the project owner to decide before publication.
- Check the rights to the checkpoints and datasets separately from the code license. For anything not distributed with the project, document its origin and how to load it.
- Prepare a README with demo images, a guide to running a first experiment, one agent example, and a guide to building a model adapter.
- Open issue templates for bug reports, requests for new architectures, and sharing experiment results. Contributors should state at least the model version and the experiment recipe.
- Keep the original checkpoint so that results can be traced back. Avoid overwriting when exporting, and show the impact on the groups the model already handles well.

## 11. Risks and open decisions

| Issue | How this draft handles it |
|---|---|
| Users interpret a channel as representing a single class | Always show results across classes and across several data groups |
| Repeated tuning fits the small dataset | Keep the exploration split separate from the final evaluation split, and record the history of attempts |
| Supporting too many models before the workflow is complete | Finish the CNN first, then use a Transformer to test the adapter design |
| An agent spends a large budget or edits many places, making results hard to follow | Limit the number of runs, and start with one change at a time |
| A graph looks complete but the mapping is wrong | Test the mapping per architecture with a checkable fixture, and show a status when the structural information is insufficient |

The example CNN architecture and the license are still proposals. The value that v0.1 must prove is that people and agents can use the same experiment to explain the effect of a change, measure its impact, and reproduce the work from start to finish.

## Earlier concept documents

- LLM_Guided_Dynamic_Dropout.md (private concept note, not in this repository)
