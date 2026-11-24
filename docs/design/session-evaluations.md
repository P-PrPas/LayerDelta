# Session viewer: evaluations and probes (v0.5 E9 design spec)

Owner: Senior UX/UI Designer · Status: **proposed, rev 1** · Extends: [session-trace-viewer.md](session-trace-viewer.md) (accepted, rev 2), which still covers everything this spec does not change · Data: the v0.5 tools in [v0.5-evaluators-benchmark.md](../architecture/v0.5-evaluators-benchmark.md) §3–§4 (`evaluate`, `get_evaluation`, `evaluation_changes`, `probe`) and the backend gaps in §9 · Roadmap: v0.5 task E9 ([board](../roadmap/Implementation_Plan_v0.5.md)), acceptance "UX/UI sign-off" · Mockup: [session-evaluations-mockup.html](session-evaluations-mockup.html) · Implementer: Frontend Developer, after E4

## 1. Purpose and non-goals

From v0.5 an agent can measure its own edits: it runs `evaluate` on a child, reads the metric deltas, looks at which samples changed, and probes activations. A person supervising that session needs to answer five more questions, in addition to the five in the trace viewer spec:

1. **What did the agent measure?** Which checkpoint, against which baseline, on which dataset, split and evaluator, with how many samples, and was an activation intervention applied?
2. **What did it find?** The metric deltas with their intervals, and whether each one is a real change or noise.
3. **Which samples changed?** The fixed and the broken samples, seen as images. The agent never sees these pixels; the person can.
4. **Is this measurement tied to an edit?** The chain read → plan → apply → evaluate, drawn from server links only.
5. **What did it cost, and what was refused?** The `evals` and `eval_samples` budgets, and the refusals: holdout, `executable_mismatch`, `probe_unsupported`, `no_baseline`, `aggregate_too_small`.

**Non-goals (v0.5).**
- No evaluation editing or launching from the viewer. Editing an evaluation is v0.6 (design §0). The viewer has no *Run evaluation* button.
- No final evaluation UI changes. The human-only holdout flow stays where it is; the viewer only shows that a final evaluation happened in the human session.
- No per-pixel prediction overlays for segmentation, and no per-sample activation maps. Both need a new forward pass (§9, deferred).
- No new judgements. The viewer prints the server's interval and category, and never computes its own significance, category or "best edit".

## 2. Principles

| Reference | What works there | Principle here |
|---|---|---|
| Experiment trackers (W&B, MLflow run compare) | A run shows its config next to its metrics | An evaluation card shows *what was measured* (dataset, split, evaluator, baseline, N, intervention) above *what it found* |
| Forest plots in clinical reports | The interval is drawn against a zero line; the reader sees "crosses zero" without reading numbers | Each metric delta is a point and an interval on an axis with a zero line, plus the numbers in text |
| Code review diffs | Regressions are what reviewers look for first | Broken samples come before fixed samples |
| Image-dataset browsers (FiftyOne, Know Your Data) | Thumbnails with labels in a dense grid, a detail pane on click | A sample grid with truth, before and after on every tile, and a larger view in the inspector |
| The trace viewer (rev 2) | Ids verbatim, meaning lines, server links only | Unchanged: every new field is shown verbatim, and the chain comes from `links` |

## 3. Information architecture

No new rail item and no new pane. E9 adds to the existing Sessions viewer (Sessions 280 | Timeline 1fr | Inspector 400):

- **Timeline:** the Run lane is renamed **Measure** and holds the v0.5 tools (§4.1).
- **A fourth tab, *Evaluations*,** next to Timeline, Applies and Proposals: one card per evaluation (§4.3).
- **An evaluation detail** that replaces the tab body when a card or a timeline row is opened, with two sub-views: *Metrics* and *Samples* (§4.4, §4.5). The Back control and `Esc` return to the list with the scroll position kept.
- **The inspector** gains sections for evaluation and probe events (§4.6).
- **The context bar** gains the `evals` and `eval samples` meters (§4.8).
- **The Applies cards** gain a one-line result per linked evaluation (§4.3).
- **URL:** `?session=&seq=` as before, plus `&eval=<evaluation_id>&view=metrics|samples&cat=broken|fixed|changed`. Reloading restores the detail.

## 4. Screens and behaviour

### 4.1 Timeline: the Measure lane

| Lane | Tools | Colour |
|---|---|---|
| Measure (was Run) | `evaluate`, `get_evaluation`, `evaluation_changes`, `probe`; legacy `propose`, `run`, `status`, `compare`, `changes`, `cancel` | violet |

The other lanes are unchanged. The filter chip `Runs` becomes `Measure`.

- **Target for `evaluate`:** `child-name vs parent-name · dataset/split · N samples`, e.g. `mnist-c1 vs mnist-base · mnist-v1/exploration · 2,000`. With `baseline: null`: `mnist-c1 · absolute · mnist-v1/exploration · 2,000`. Names resolve as in the trace viewer; dataset ids are shown as the registered dataset id (an id, not free text).
- **Target for `probe`:** `checkpoint · node or group · N samples`, e.g. `mnist-base · conv2 (group g_conv2) · 64`.
- **Flags** (added to the rev 2 set):
  - charges: `evals 1`, `eval_samples 4,000` (the first non-zero dimension is shown, the rest in the tooltip, as rev 2);
  - `cached baseline` when the baseline was a cache hit and not charged;
  - `intervention` when `evaluate` carried one. It always shows, because a measurement with an intervention measures something other than the checkpoint;
  - `raw` (alert colour) on a class-2 probe that returned raw activations.
- **Polling bursts:** `get_evaluation` and `evaluation_changes` calls on the same `evaluation_id` collapse into one row under their `evaluate` row: *"6 result reads · #45–#52"*, using the read-burst rule of rev 2. A failed one never collapses.
- **Provenance highlight:** selecting an `evaluate` row highlights the apply it links to (`links.apply_seq`, §9 gap E3) and that apply's own plan and read links, so the whole read → plan → apply → evaluate chain lights up. Selecting an apply also lights up the evaluations that link to it. Links come from the server only, as in rev 2.

### 4.2 Status of an evaluation job

`evaluate` and `probe` return 202 and run as jobs. The trace event records the submission; the job's state comes from `GET /evaluations/{id}` (and `GET /probes/{id}`, §9 gap E6), read as the human session.

| State | Row and card | Polling |
|---|---|---|
| `queued` | `202` plus a grey `queued` chip | Every 2 s while the row or card is on screen and the window is visible |
| `running` | violet `running` chip with elapsed time | Same |
| `succeeded` | the primary metric delta inline on the card (§4.3); the row stays `202` | Stops |
| `failed` | alert `failed · <error_code>` chip; the meaning line explains the code (§4.7) | Stops |
| `cancelled` | grey `cancelled` | Stops |

The trace stays the source of order; the job state is decoration and is never shown as an HTTP status.

### 4.3 Evaluations tab (B, fourth tab)

One card per `evaluate` event in the loaded pages, newest at the bottom like the timeline. A card:

```
#44  evaluate · 12:06:02                                      succeeded · 41 s
mnist-c1 → vs mnist-base · mnist-v1 / exploration · classification v1 · 2,000 samples
from apply #41 · recipe 7be2… · fingerprint 3c9a…            cached baseline
intervention: none

accuracy        97.80% → 98.35%    +0.55 pp  [+0.10, +1.00]   ├────●──┼──────┤  Improved
confusion 8→3    1.20% →  0.40%    −0.80 pp  [−1.30, −0.35]   Improved
                                                              (lower is better)
Fixed 31 · Broken 20 · Changed, still wrong 4
[Metrics]  [Samples: 20 broken]  [Compare child with parent ↗]  [Show in timeline]
```

- **Header:** seq, `evaluate`, time, job state and duration.
- **What was measured** (one or two lines, mono for ids): checkpoint → baseline, dataset and split, evaluator name and version, N, the apply it came from, the recipe and the fingerprint (short ids with tooltips and copy). `intervention:` is always printed, `none` included.
- **What it found:** the primary metric and at most two secondary metrics, each as a delta row (§5 *DeltaRow*). The rest are in the detail.
- **Counts:** `Fixed n · Broken n · Changed, still wrong n`, from the server's counts. Each count is a link to the Samples view filtered to that category.
- **Actions:** *Metrics*, *Samples* (the label names the broken count, because that is what a reviewer opens first), *Compare child with parent ↗* (tensor browser, rev 2), *Show in timeline*.
- **Applies cards** gain one line per linked evaluation: *"Measured #44: accuracy +0.55 pp [+0.10, +1.00] · Improved"*, which opens the card.
- **Intervention evaluations** carry the label *"Measurement only. No checkpoint was created."* and print the intervention: `conv2 units 5, 7 · zero` (node or group name resolved; units as a compact list).
- **Absolute evaluations** (`baseline: null`) show metric values without deltas or intervals and say *"No baseline: absolute metrics only."*

### 4.4 Evaluation detail: Metrics

A full-height view of one evaluation.

- **Measured block:** the card's *what was measured* lines, plus the evaluator spec (decoder args, metric args) as a key-value list, and the dataset's preprocessing steps in order (`resize 28 · grayscale · scale · mean/std`). Both are server data shown verbatim.
- **Metric table:** every metric the server returns, primary first and then in server order. Columns: metric · baseline · checkpoint · delta · 95% interval · interval plot · reading. The interval plot shares one axis per unit across rows, so widths compare; the zero line is drawn in every row.
- **Reading** (one word or two, from the interval and the metric's direction, §9 gap E2):
  - interval entirely on the better side of 0 → **Improved** (mint);
  - entirely on the worse side → **Worse** (alert);
  - contains 0 → **No clear change** (muted);
  - no interval (absolute evaluation, or N too small for the server to give one) → nothing.

  The word is always printed; colour is never the only signal. The viewer never says "significant".
- **Per-class rows** (per-class recall, per-class IoU) are behind a *Per class (10)* disclosure, sorted by the absolute delta, largest first. Class names come from the dataset (file-derived text, plain text).
- **Confusion pair** (classification): `truth → confused_with` shows both class names and the rate.
- **Method note** (small, always shown): *"Deltas are paired on the same samples. Intervals are 95% paired bootstrap, 2,000 resamples, fixed seed."*

### 4.5 Evaluation detail: Samples (fixed and broken, with pixels)

The view a supervisor uses to check that a metric gain is not hiding damage.

- **Pixel notice** (a slim bar at the top, always shown): *"Images are shown to you only. The agent saw sample ids, labels and conditions, never pixels."*
- **Category tabs:** `Broken 20` (default) · `Fixed 31` · `Changed, still wrong 4`. Categories and counts are the server's (§9 gap E4); the viewer never re-derives them.
- **Filters:** truth class, condition (e.g. `clean`, `blur`), and for OCR a minimum edit distance. Sort: by severity (the per-sample score change, §9 gap E4) or by `sample_id`.
- **Grid:** tiles of 96 px images by default, 160 px with the *Large* toggle, in pages of 60 following `evaluation_changes` paging. A tile is a button; selecting it opens the sample in the inspector (§4.6).
- **Tile content by task** (truth, before and after on every tile, in text, never colour only):

| Task | Image area | Text under the image |
|---|---|---|
| classification | the image | `truth 8 · 3 → 8` (before → after); the class names when the dataset has them |
| segmentation | the image with the **truth** mask as a 45% overlay, toggleable (`m`) | `mIoU 0.71 → 0.52` and the class with the largest IoU drop: `disc −0.30` |
| detection | the image with truth boxes (dashed, mint) and the kept boxes after (solid, amber); `b` switches the solid boxes between before and after | `TP 3 → 2 · FP 0 → 1` with IoU 0.5 matching as the server reports |
| ocr | the text-line image, full width of the tile | `truth 4071`, `before 4071`, `after 4011` with the changed characters underlined and boxed, and `CER 0.00 → 0.25` |

- **Image rendering:** images draw at integer scale with `image-rendering: pixelated` when the source is 64 px or smaller, so a 28 px digit stays a digit. Larger images are scaled smoothly to fit. The image is the stored source file (§9 gap E1), not the preprocessed tensor; a *Model input* toggle (P2, gap E1b) shows what the model saw after resize and pad.
- **Loading and failure per tile:** a grey placeholder at the right aspect, then the image. A failed image shows *"Image unavailable"* with the HTTP code, and the text stays, because the text is the evidence.
- **Empty category:** *"No broken samples. 31 fixed, 4 changed but still wrong."*

### 4.6 Inspector (C): evaluation, probe and sample

**Evaluation event** (`evaluate`): the rev 2 verbatim fields, plus the new ids (`evaluation_id`, `dataset_id`, `split`, split hash, `evaluation_fingerprint`, `baseline_id`, `intervention`, `budget_charged` with `evals` and `eval_samples`). Then *What this means* (§4.7), the job state, the primary delta row, and the actions *Open evaluation*, *Samples*, *Compare child with parent ↗*, *Copy fingerprint*.

**Result reads** (`get_evaluation`, `evaluation_changes`): fields and *"The agent read the results of #44."* with a link to #44. For `evaluation_changes`, the page cursor position: *"Page 2 (rows 61–120)"*, so a reviewer knows how much of the change list the agent looked at.

**Probe event** (`probe`):
- the target: node (G4 node id and its module path) or group (`g_conv2`, resolved to its `out_unit` node), `limit_samples`, the split-by key (`label` or `condition`), and the K that applied in class 1;
- the result (gap E6), read as the human: a per-unit table with mean, std, l2 and fraction positive, a small histogram per unit when the agent asked for one, and the units the server reported as `skipped_small`, listed as skipped rather than hidden;
- *Open in Architecture ↗* at the node or group (G8), and *Open in Tensors ↗* at the node's weight when it has one;
- in a class-2 session with raw activations returned: the flag `raw` and the line *"Raw activation values were released to this session (egress class 2)."* The viewer does not display the raw tensor itself.

**Sample** (a tile from §4.5): the image at the largest integer scale that fits 368 px; the toggles of the tile (mask, before/after boxes); `sample_id` with copy; truth, condition, before and after in full; the per-sample scores before and after; and the line *"Shown to you only."*

### 4.7 Meaning lines (added to the rev 2 table)

| Status and code | Meaning line |
|---|---|
| 403 `forbidden` on `split: holdout` | *"Blocked: the holdout split is reserved for the human final evaluation. Nothing was run."* |
| 403 `forbidden`, `eval:run` missing | *"Blocked: this token lacks `eval:run`. Nothing was run."* |
| 403 `forbidden`, probe without `weights:read` | *"Blocked: probes need `weights:read`. Nothing was run."* |
| 429 `budget_exhausted`, field `evals` | *"The evaluation budget is used up. Nothing was run."* |
| 429 `budget_exhausted`, field `eval_samples` | *"The sample budget would be exceeded by this evaluation (N needed, R remaining). Nothing was run."* (N and R only when the server reports them) |
| 422 `executable_mismatch` | *"The checkpoint does not fit the model's executable (a missing tensor, a shape or dtype difference, or a changed artifact). Nothing was run."* plus the server's hint verbatim when present |
| 422 `probe_unsupported` | *"This node cannot be probed with this model's provider."* plus the server's reason verbatim |
| 422 `no_baseline` | *"This checkpoint has no parent to compare with, and no baseline was given."* |
| 422 `aggregate_too_small` | *"Too few samples for this probe under egress class 1 (each statistic needs at least K values). Nothing was released."* |
| job `failed` | *"The evaluation job failed: `<error_code>`."* plus the server message verbatim |
| 202 `evaluate` with a replayed key | *"The earlier evaluation was returned. No new charge."* (rev 2 replay rule) |

### 4.8 Budgets in the context bar

`summary.budget` gains `evals` and `eval_samples` (design V7). They are shown as meters next to the v0.4 ones, with the same rules (amber at ≥ 80%, alert at 100% with a link to the 429 event, *Edit* through the existing budget dialog, lowering confirmed once).

The context bar now has up to six meters. To fit at 1280 px: meters for dimensions **without a limit** fold into one `Usage` button whose popover lists them (`Values 4 · Blob 0 B`); meters with a limit always stay visible. Below 1280 px the rev 2 responsive rule applies.

## 5. Components

| Component | Notes |
|---|---|
| **DeltaRow** | metric name · baseline → checkpoint · signed delta with unit (`pp` for rates, absolute otherwise) · `[lo, hi]` · IntervalPlot · reading word. Mono numbers, right-aligned, fixed decimals per unit: rates 2 dp in %, pp 2 dp, counts with thousands separators |
| **IntervalPlot** | 120 × 14 px. A zero tick, the interval as a 2 px bar, the point as a 6 px dot. The better direction is always to the right, so for lower-is-better metrics the axis is mirrored and labelled `better →` once per table. Colour from the reading; the reading word is always printed next to it |
| **EvaluationCard** | §4.3 |
| **SampleTile** | §4.5. 96 or 160 px image area, two text lines, a 2 px category edge on the left (alert for broken, mint for fixed, muted for changed). `role="button"` with an `aria-label` that reads the whole record: *"Sample 1f3a…, truth 8, before 3, after 8, fixed"* |
| **BoxOverlay** | SVG over the image: truth dashed mint, prediction solid amber, 1.5 px, labels only on hover or in the inspector |
| **MaskOverlay** | the truth index mask drawn with a fixed 8-colour categorical palette at 45%; the ignore index is transparent |
| **TextDiff** | OCR: per-character alignment from the server's before/after strings, changed characters underlined and boxed, never colour only |
| **JobChip** | queued · running · succeeded · failed · cancelled (§4.2) |

## 6. Wording rules (added to rev 2 §6)

- "Measured", "evaluated" and "probed". Never "tested", "proved" or "validated".
- "Improved", "Worse" and "No clear change" are the only readings. Never "significant", "better model" or "regression detected".
- "Fixed" and "Broken" are the server's categories. "Changed, still wrong" is the third.
- "Images are shown to you only" wherever pixels appear.
- Class names, condition names and OCR strings are dataset or model text: plain text everywhere, as rev 2 says for agent-originated text.

## 7. Keyboard and accessibility

- From rev 2: `j`/`k`, `Enter`, `[`/`]`, `e`, `l`, `/`. New: `v` cycles Timeline → Applies → Proposals → Evaluations; in the detail, `1`/`2` switch Metrics/Samples, `b`/`f`/`c` choose Broken/Fixed/Changed, arrow keys move in the sample grid, `m` toggles the mask overlay, `Esc` goes back.
- The sample grid is a `role="grid"`; each tile reads its whole record (§5). Images have `alt` text built from the record, because the image itself is not described.
- The interval plot is `aria-hidden`; the numbers and the reading word carry the meaning.

## 8. States

| State | What shows |
|---|---|
| No evaluations in the session | Evaluations tab: *"This session has not run an evaluation."* |
| Evaluation running | Card with the job chip; Metrics and Samples disabled with *"Results appear when the job finishes."* |
| Failed | Card with the failure chip and the meaning line; no metrics |
| Absolute evaluation | Values only, *"No baseline: absolute metrics only."* |
| Changes still paging | *"Showing 60 of 320 broken · Load more"* |
| Results route unavailable (older server) | The trace rows still show; the card says *"Results are not available from this server."* |

## 9. Backend gaps (requests to the AI Scientist / API owner)

P1: the viewer cannot answer one of the §1 questions without it. P2: it works in a degraded way.

| # | Field / route | Why | Priority |
|---|---|---|---|
| E1 | `GET /datasets/{dataset_id}/samples/{sample_id}/image`, **human only** (403 for any agent token). Returns the stored source file after re-checking its SHA-256 against the manifest, `Content-Type: image/png` or `image/bmp`, `Cache-Control: private, max-age=…`, `X-Content-Type-Options: nosniff`. The server does not decode it | Pixels for fixed and broken samples (§4.5). Added to the V3/V5 route walk as an agent-refused route | P1 |
| E1b | The same route with `?view=model`: the preprocessed input rendered as PNG by the evaluation worker | *Model input* toggle | P2 |
| E2 | In `GET /evaluations/{id}`, per metric: `{name, primary, unit: rate\|count\|score, better: higher\|lower, baseline, value, delta, ci: [lo, hi] \| null}`, and `per_class` rows the same way | DeltaRow, reading words, mirrored axis for CER/WER (§4.4) | P1 |
| E3 | Evaluate and probe trace events: `evaluation_id` / `probe_id`, `dataset_id`, `split`, split hash, fingerprint, `baseline_id`, `baseline_cached`, intervention as ids and numbers (`node_id` or group id, unit count, op, factor), and `links.apply_seq` to the apply that created the checkpoint. Result-read events carry `evaluation_id` and the page offset | Measure lane targets, chain highlight, inspector (§4.1, §4.6) | P1 (most is in design §3.2; `links.apply_seq`, `baseline_cached` and the page offset are the additions) |
| E4 | In `evaluation_changes` rows: `category: fixed\|broken\|changed` for every task, and a per-sample `score_before` / `score_after` (sample accuracy 0/1, sample mIoU, AP@0.5 or CER) with a sort option by score change | Category tabs and severity sort for all four tasks (§4.5) | P1 (category), P2 (score) |
| E5 | `GET /datasets/{dataset_id}/samples/{sample_id}/label`, **human only**: the index-PNG mask, the boxes as JSON, or the text line, after the hash check | Truth overlays for segmentation and detection (§4.5) | P1 for segmentation and detection |
| E6 | `GET /probes/{probe_id}` → status and the aggregates exactly as returned to the agent (including `skipped_small`) | Probe inspector (§4.6) | P1 |
| E7 | Job fields on `GET /evaluations/{id}`: `status`, `created_at`, `started_at`, `finished_at`, `error_code`, `message` | Job chips and durations (§4.2) | P1 |
| E8 | `summary.budget.evals` and `summary.budget.eval_samples` | Meters (§4.8); likely already part of V7 | P1 |

**Deferred (not requested for v0.5):** predicted segmentation masks and per-sample activation maps. Both need a new forward pass per sample, which belongs to the v0.6 evaluation UI with its own human-side budget.

All new routes are human only and add nothing to the agent surface. The trace stays ids and numbers only (T5).

## 10. Risks

- **Untrusted image bytes in the WebView.** Gap E1 serves stored PNG or BMP files without decoding them on the server, so the WebView's image decoder parses untrusted bytes. The renderer is sandboxed, the format list is closed and the hash is checked, and the alternative (a server-side decode) would put a second decoder in the API process, which the design forbids. If this is not acceptable, E1b (worker-rendered PNG) becomes the only path.
- **Reading intervals as verdicts.** A supervisor may read "Improved" as "approve". The reading words describe the interval only, and the method note is always visible. The benchmark's conclusions are E8's, not the viewer's.
- **Small N.** With `limit_samples` small, intervals are wide and readings stay "No clear change". N is printed on every card so the reason is visible.
- **Pixel leakage by screenshot.** The person can screenshot images. That is the person's data and right; the notice makes clear that the agent did not see them.
- **Context bar crowding.** Mitigated by folding unlimited meters (§4.8).
