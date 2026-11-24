# Evaluators (v0.5-E3)

Task evaluators turn a model's raw outputs into per-sample records, judge each record, and compute the task's metrics. This is a library with no HTTP route; the `evaluate` tool (E4) calls it.
- **Design:** [v0.5 design §3](../architecture/v0.5-evaluators-benchmark.md).
- **Code:** `src/layerdelta/evaluators/`.
- **Golden oracles:** `tests/test_evaluators.py`. Every expected number in it is computed by hand in a comment.

Evaluators use numpy only and import nothing with side effects, because they run in the evaluation worker.

## The contract (what E4 calls)

| Call | Returns |
|---|---|
| `make(spec)` | An evaluator. `spec` is `{task, version?, ...settings}`. Unknown settings are refused (`invalid_argument`, `field` names the setting). |
| `canonical_spec()` | The spec with every default filled in, hashed into the evaluation fingerprint. |
| `records(outputs, samples)` | One JSON-able record per sample. `outputs` are the graph's outputs, batch-first, with N samples. `samples` are N dicts: the manifest row plus what preprocessing produced. A wrong layout is refused as `evaluator_mismatch`, never guessed. |
| `correct(record, sample)` | A boolean. With the baseline's, it gives the change category: `fixed` (incorrect → correct), `broken` (correct → incorrect), `changed`. |
| `score(record, sample)`, `score_better` | The per-sample score and its direction, for sorting changes by severity. |
| `metrics(records, samples)` | A list of `{name, primary, unit, better, value, per_class?}`. Exactly one metric is primary. `per_class` rows are `{class, value, n}`. `value` is null when undefined, for example no sample of a class. |
| `metric_parts(records, samples)`, `metrics_from_parts(sums)` | Per-sample sufficient statistics that **sum**. `metrics(r, s)` equals `metrics_from_parts` of the parts summed over `s`, so the paired bootstrap sums parts over resampled indices. Detection returns None, because mAP does not decompose: resample it through `metrics`. |
| `resampler(records, samples)` | `run(index)`, the metrics of the resample that takes samples `index` (repeats allowed), built once per evaluation. Every evaluator has it: parts are summed when they exist; detection does its matching once and repeats each image's rows, so 2000 resamples of mAP stay affordable. `metrics` equals `run` on every sample once. |

E4 fills `baseline`, `delta` (always `value − baseline`) and `ci` around each metric.

## Tasks

| Task | Output contract | Decoder | Record | Correct | Score | Metrics (primary first) |
|---|---|---|---|---|---|---|
| `classification` | logits `[N, C]` | argmax | `{pred}` | `pred == label` | 1 or 0 (higher) | `accuracy`, or the confusion-pair rate when `confusion` is set; `accuracy:<condition>` for each condition; per-class recall |
| `segmentation` | logits `[N, C, H, W]` | argmax per pixel, resized to the mask with nearest-exact | per-class pixel counts `{inter, pred, truth}` | sample mIoU ≥ `correct_miou` (0.5) | sample mIoU (higher) | `miou`, `dice`, `pixel_accuracy` |
| `detection` | candidates `[N, K, 4 + 1 + C]`: box, objectness, class probabilities | score = objectness × class probability; `score_threshold`; class-wise NMS (`nms_iou`); top `max_detections`; boxes mapped to source coordinates | `{boxes: [[x1, y1, x2, y2, score, class], ...]}` in source coordinates | at `conf_threshold`, no FP and no FN | F1 at `conf_threshold` and `match_iou` (higher) | `map50`, `map50_95` (COCO 101-point) |
| `ocr` | logits `[N, T, A]` | CTC greedy (collapse repeats, drop the blank) | `{text}` | exact match | CER of the sample (lower) | `cer`, `wer`, `exact_match` |

### Settings and defaults

- **Every task:** `output` (0) is the index of the graph output to read.
- **`classification`:**
  - `classes` (inferred when null);
  - `confusion` `{truth, confused_with, condition?}`: the v0.1 metric, for example 8 read as 3 on blurred images.
- **`segmentation`:** `classes` (required), `ignore_index` (255; padding uses it), `correct_miou` (0.5).
- **`detection`:**
  - `classes` (required);
  - `class_ids`: the category_id of each class, by default the class index itself;
  - `box_format`: `xyxy` or `cxcywh`;
  - `box_scale`: `absolute` model-input pixels or `relative`;
  - `score_threshold` (0.001), `nms_iou` (0.5), `max_detections` (100), `conf_threshold` (0.25), `match_iou` (0.5).
- **`ocr`:** `alphabet` (required: the A − 1 non-blank symbols in index order), `blank` (0).

### Definitions that decide results

- **Segmentation.**
  - mIoU and Dice are **dataset-level**: per-class intersection and union are summed over samples, then averaged over the classes present.
  - A class absent from both truth and prediction is left out of a mean.
  - A sample with nothing to score (all pixels ignored, or no class present) is correct, with score 1.0.
- **Detection matching** is one rule shared by AP, F1 and correctness:
  - per image and class, detections are visited by descending score (ties: the order given);
  - each takes the unmatched truth box with the highest IoU at or above the threshold, one to one;
  - IoU uses continuous coordinates, with no +1;
  - an inverted box (x2 < x1) has zero area and matches nothing.
- **Detection AP.**
  - AP is the mean of the interpolated precision at 101 recall points.
  - A class with no truth box is left out of mAP. Its per-class value is null.
  - An image with neither truth nor detections has F1 1.0 and is correct.
  - `map50_95` averages the thresholds 0.50, 0.55, …, 0.95.
- **OCR.**
  - CER and WER are summed edit distances over summed reference lengths, so they can exceed 1.
  - A sample's CER counts an empty reference as length 1.
  - WER splits on whitespace.

## Oracles

`test_evaluators.py` checks every task against hand-computed values. It also checks the summing property on random resamples, and that a wrong layout is refused.

`test_map_matches_pycocotools_on_random_cases` (marker `oracle`, extra `layerdelta[oracle]`, workflow *Dev oracles*) compares `map50` and `map50_95` with pycocotools 2.0.11 on 20 random cases, to 1e-9. As in [ADR 0001](../architecture/adr/0001-depgraph-oracle.md), pycocotools is a development oracle only: nothing in `src/` imports it.
