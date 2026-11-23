# LayerDelta — Agent weight inspection, experiment trace, and benchmark

> **Status:** parked research spec. The weight-access split and trace are now v0.2 tasks T4/T5, and the benchmark is scheduled for v0.5 across several model families. See the [Roadmap](../roadmap/Roadmap.md).

Supplementary document for coding agents · proposed requirements for updating the repository.

## Goal

Let an agent open a single checkpoint and explore the weights of the CNN by layer and channel, in order to gather information before choosing a weight edit. Then measure the effect of the edit on the same sample set as the baseline, and make it possible to review afterwards what information the agent used and which experiments it ran. The research goal is to test whether seeing the weights helps an agent choose edits that give better results under an equal budget.

Follow the contracts in [Agent Integration](../guides/agent-integration.md), the [API](../reference/api.md) and the [Architecture](../architecture/Architecture.md). Check file names and the actual implementation in the repo before making changes.

## 1. Let the agent explore weights

- Add a typed tool named `preview_weights` as a proposed name in the existing HTTP wrapper. Use the catalog from `inspect` to choose the checkpoint, the target and the output channel. The wrapper uses the existing preview job and status contract, with an added structured result for the Conv kernel slice.
- Accept a checkpoint ID, a target ID, an output channel and an input-channel range, following the adapter's mapping. Return the parameter name, shape, dtype, checkpoint fingerprint, tensor axes, the index range actually returned, the statistics of the slice, and the kernel values as numbers in the order of the given axes. The next range can be requested when the slice is larger than the cap.
- For Conv2d, use the existing mapping `[C_out, C_in/groups, K_h, K_w]`. Select the output channel first, then select the input-channel slice within the group. The returned values must come from the specified checkpoint. The same data is used to draw a heatmap in the UI, so that people and the agent refer to the same kernel location.
- The backend checks the project/session, the target, the indexes and the response cap. It takes the data from the catalog rather than from a file path supplied by the agent. Weight reads use the checkpoint directly. If the existing preview requires a sample ID, keep it compatible with existing callers, and add a contract that supports weight preview without choosing an image.
- Open the backend from the Desktop or the headless CLI. Use the same wrapper with a local or frontier LM. Weight and image data stay on the local machine by default. They can be sent to a frontier provider when the user allows it, within the scope defined. Complete the HTTP integration before adding an MCP bridge that calls the existing API.
- Keep the existing experiment flow `inspect → propose → run → status → compare`. Use `get_changes` to view examples whose predictions changed, and `get_budget` to view the budget. In this flow, `compare` means the prediction results of the baseline and the intervention on the same sample set. For this benchmark use weight mode, and use an operation the adapter supports, such as scale or zero, with one intervention per recipe.

## 2. Record tool use and experiments

Use SQLite and the existing recipe/run data. Add an event for each tool call that the backend receives. Each event stores the session ID, the time and order, the operation name, the request ID, the status or error code, the checkpoint ID and fingerprint, the target/channel or range that was read, and the recipe, run and baseline IDs where relevant. In the benchmark, link the session to the benchmark case and the arm, so that the conditions can be audited afterwards.

- A preview event records which range was requested and whether it succeeded. It refers to the checkpoint and the slice returned, without copying the whole tensor into the log.
- A mutation event links the proposed recipe, the job accepted into the queue, and the run result. It makes clear whether the job was proposed, ordered to run, succeeded, failed or was cancelled. A resend with the same idempotency key may appear in the request history, but it must be linked to the original run and charged to the budget only once.
- Remove tokens, authentication headers, images and full prompts from the log. Use the recipe and the report as the source of detail for operations and metrics.
- The MVP uses a timeline per session, which can be opened from a tool call to the recipe and the run result. Use the same schema for both arms. A cross-project dashboard and a model registry are out of scope for this work.

## 3. Benchmark: the effect of seeing weights

Use an experiment with two arms that differ only in whether they may call the weight preview.

| Condition | Sees weights | Controlled |
|---|---|---|
| Shared tools | Catalog, weight edits, running, viewing metrics and prediction changes | Same for both arms |
| Weight access | Reads statistics and kernel values within the defined scope | Weight preview is turned off, along with any other path to raw weights or weight statistics |
| Starting point and budget | Same architecture and checkpoint, same frontier model and config, same task and split, same run budget and agent budget | Same conditions for both arms |

1. Start with the OCR CNN that reads 8 as 3 on blurred images. Set the main metric as the rate of predicting 3 on blurred images whose true label is 8, and use the accuracy on normal images as a guardrail. Set the guardrail threshold, the run budget, the token budget and the number of sessions in advance.
2. Give both arms the same samples and feedback from exploration. Enforce the preview permission at the backend, and restrict indirect tools in the benchmark runner. Pair sessions by task and starting point. Use several independent session pairs, and alternate the order of runs to reduce order effects.
3. Have each session submit one final recipe before the holdout results are revealed. Freeze the recipe hash, so that an evaluator runs the holdout through the human permission. Use the same images for the baseline and for every recipe being compared.
4. Report results per session, with the number of samples, the difference in the main metric, the accuracy on normal images, the number of cases fixed and the number of new errors. Also report the number of runs, tokens and time. Summarize the difference between the arms as a paired comparison, and report the uncertainty. Use the trace to check what the experimental arm looked at before it chose where to edit.
5. Interpret the results only within the scope of the CNN, the checkpoint, the OCR task and the frontier model tested. This experiment measures the added value of seeing weights for edits restricted to a limited set of operations. To measure the value of LayerDelta as a whole, run a separate benchmark against the user's normal workflow.

## Order of work and acceptance criteria

1. Connect the weight preview to the existing wrapper and adapter. Check that a deterministic agent client can open a slice of a single checkpoint, that the values and axes match the source tensor, and that an incorrect target or a request beyond the scope is rejected.
2. Add the session trace and the timeline. Check that the path from a weight preview to the recipe and the run result can be traced backwards. Check that a retry does not create a second run or charge the quota twice, and that no token or full tensor appears in the log.
3. Build the benchmark runner and report. Check that both arms can perform and evaluate edits equally. Check that the control arm cannot read weights, and that agents on both sides cannot reach the holdout. Then run the whole cycle end to end on the fixture before using a real frontier model.

The work described in this document adds the capabilities and the evaluation tools. Whether agents actually perform better must come from a benchmark that is run for real.
