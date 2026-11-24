"""Task evaluators (v0.5-E3, acceptance V2): golden oracles on tiny hand-checked cases, and the contract E4 relies on.

Every expected number below is computed by hand in the comment next to it, not by the code under test.
"""

import json

import numpy as np
import pytest

from layerdelta import evaluators
from layerdelta.evaluators import EvaluatorError, make
from layerdelta.evaluators.detection import average_precision, iou, match, nms
from layerdelta.evaluators.ocr import edit_distance
from layerdelta.evaluators.segmentation import nearest_exact

IDENTITY = {"source": [10, 10], "scale": [1.0, 1.0], "offset": [0.0, 0.0], "output": [10, 10]}


def by_name(metrics):
    return {m["name"]: m for m in metrics}


def onehot_logits(ids, width):
    out = np.zeros((*np.shape(ids), width), dtype=np.float32)
    np.put_along_axis(out, np.asarray(ids)[..., None], 1.0, axis=-1)
    return out


# ---------------------------------------------------------------- the contract

@pytest.mark.parametrize("task", sorted(evaluators.EVALUATORS))
def test_specs_are_canonical_and_strict(task):
    required = {"segmentation": {"classes": 2}, "detection": {"classes": 2}, "ocr": {"alphabet": "ab"}}.get(task, {})
    e = make({"task": task, **required})
    spec = e.canonical_spec()
    assert spec["task"] == task and spec["version"] == 1 and json.loads(json.dumps(spec)) == spec
    assert make(spec).canonical_spec() == spec  # a canonical spec round-trips
    with pytest.raises(EvaluatorError) as unknown:
        make({"task": task, **required, "surprise": 1})
    assert unknown.value.code == "invalid_argument" and unknown.value.field == "surprise"
    assert e.score_better in ("higher", "lower")


def test_unknown_task_and_missing_settings_are_refused():
    for spec, field in (({"task": "regression"}, "task"), ({"task": "segmentation"}, "classes"), ({"task": "ocr"}, "alphabet"),
                        ({"task": "ocr", "alphabet": "aa"}, "alphabet"), ({"task": "detection", "classes": 2, "box_format": "yolo"}, "box_format")):
        with pytest.raises(EvaluatorError) as refused:
            make(spec)
        assert refused.value.field == field


def test_a_wrong_output_layout_is_refused_not_guessed():
    e = make({"task": "classification"})
    with pytest.raises(EvaluatorError) as refused:
        e.records([np.zeros((3, 2, 2))], [{"label": 0}] * 3)
    assert refused.value.code == "evaluator_mismatch"
    with pytest.raises(EvaluatorError):
        e.records([np.zeros((2, 4))], [{"label": 0}] * 3)  # N differs from the samples
    with pytest.raises(EvaluatorError):
        make({"task": "ocr", "alphabet": "abc"}).records([np.zeros((1, 5, 3))], [{"text": ""}])  # 3 symbols + blank = 4


@pytest.mark.parametrize("seed", range(5))
def test_parts_sum_to_the_metrics_of_any_resample(seed):
    """The paired bootstrap sums per-sample parts over resampled indices; that must equal recomputing on the resampled set."""
    rng = np.random.default_rng(seed)
    n = 12
    cases = [
        (make({"task": "classification", "confusion": {"truth": 1, "confused_with": 2, "condition": "blur"}}),
         [rng.normal(size=(n, 3))], [{"label": int(rng.integers(3)), "condition": str(rng.choice(["clean", "blur"]))} for _ in range(n)]),
        (make({"task": "segmentation", "classes": 3}), [rng.normal(size=(n, 3, 4, 4))],
         [{"mask": rng.integers(0, 3, size=(4, 4))} for _ in range(n)]),
        (make({"task": "ocr", "alphabet": "ab "}), [rng.normal(size=(n, 6, 4))],
         [{"text": "".join(rng.choice(list("ab "), size=int(rng.integers(1, 5))))} for _ in range(n)]),
    ]
    for e, outputs, samples in cases:
        records = e.records(outputs, samples)
        assert json.loads(json.dumps(records)) == records
        index = rng.integers(0, n, size=n)
        parts = e.metric_parts(records, samples)
        summed = e.metrics_from_parts({k: v[index].sum(axis=0) for k, v in parts.items()})
        assert summed == e.metrics([records[i] for i in index], [samples[i] for i in index]), e.task


# ---------------------------------------------------------------- classification

def test_classification_golden():
    # labels 0 1 1 2 | predictions 0 1 2 2 -> accuracy 3/4; recall: class 0 1/1, class 1 1/2, class 2 1/1
    # confusion 1 -> 2 within "blur": samples with label 1 in blur are #2 only (#1 is clean); #2 predicted 2 -> rate 1/1
    samples = [{"label": 0, "condition": "clean"}, {"label": 1, "condition": "clean"}, {"label": 1, "condition": "blur"},
               {"label": 2, "condition": "blur"}]
    e = make({"task": "classification", "classes": 3, "confusion": {"truth": 1, "confused_with": 2, "condition": "blur"}})
    records = e.records([onehot_logits([0, 1, 2, 2], 3)], samples)
    assert [r["pred"] for r in records] == [0, 1, 2, 2]
    m = by_name(e.metrics(records, samples))
    assert m["confusion:1->2:blur"]["value"] == 1.0 and m["confusion:1->2:blur"]["primary"] and m["confusion:1->2:blur"]["better"] == "lower"
    assert m["accuracy"]["value"] == 0.75 and not m["accuracy"]["primary"]
    assert [c["value"] for c in m["accuracy"]["per_class"]] == [1.0, 0.5, 1.0]
    assert m["accuracy:clean"]["value"] == 1.0 and m["accuracy:blur"]["value"] == 0.5
    assert [e.correct(r, s) for r, s in zip(records, samples)] == [True, True, False, True]
    assert e.score(records[2], samples[2]) == 0.0 and e.score_better == "higher"


def test_classification_without_confusion_has_accuracy_primary_and_empty_classes_are_none():
    e = make({"task": "classification", "classes": 3})
    samples = [{"label": 0}, {"label": 0}]
    m = by_name(e.metrics(e.records([onehot_logits([0, 1], 3)], samples), samples))
    assert m["accuracy"]["primary"] and m["accuracy"]["value"] == 0.5
    assert [c["value"] for c in m["accuracy"]["per_class"]] == [0.5, None, None]


# ---------------------------------------------------------------- segmentation

def test_nearest_exact_samples_pixel_centres():
    # 2 -> 4: centres (i + .5) * 2/4 = .25 .75 1.25 1.75 -> rows 0 0 1 1;  4 -> 2: (i + .5) * 2 = 1, 3 -> rows 1 3
    small = np.array([[1, 2], [3, 4]])
    assert nearest_exact(small, 4, 4).tolist() == [[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]
    big = np.arange(16).reshape(4, 4)
    assert nearest_exact(big, 2, 2).tolist() == [[5, 7], [13, 15]]


def test_segmentation_golden():
    # sample A: truth [[0,0],[1,1]], pred [[0,1],[1,1]] -> class 0: I1 U2 (.5); class 1: I2 U3 (2/3); sample mIoU 7/12
    # sample B: truth [[1,255],[1,1]] (one ignored pixel), pred all 1 -> class 1: I3 U3; class 0 absent -> sample mIoU 1.0
    # dataset: class 0 I1 U2 -> .5; class 1 I5 U6 -> 5/6; mIoU (.5 + 5/6)/2 = 2/3
    # Dice: class 0 2*1/(1+2) = 2/3; class 1 2*5/(6+5) = 10/11; mean (2/3 + 10/11)/2
    # pixel accuracy: correct 1 + 2 + 3 = 6 of 4 + 3 valid pixels = 6/7
    samples = [{"mask": np.array([[0, 0], [1, 1]])}, {"mask": np.array([[1, 255], [1, 1]])}]
    pred = np.array([[[0, 1], [1, 1]], [[1, 1], [1, 1]]])
    e = make({"task": "segmentation", "classes": 2})
    records = e.records([onehot_logits(pred, 2).transpose(0, 3, 1, 2)], samples)
    assert records[0] == {"inter": [1, 2], "pred": [1, 3], "truth": [2, 2]}
    assert records[1] == {"inter": [0, 3], "pred": [0, 3], "truth": [0, 3]}
    assert e.score(records[0], samples[0]) == pytest.approx(7 / 12) and e.score(records[1], samples[1]) == 1.0
    assert [e.correct(r, s) for r, s in zip(records, samples)] == [True, True]
    m = by_name(e.metrics(records, samples))
    assert m["miou"]["value"] == pytest.approx(2 / 3) and m["miou"]["primary"]
    assert [c["value"] for c in m["miou"]["per_class"]] == pytest.approx([0.5, 5 / 6])
    assert m["dice"]["value"] == pytest.approx((2 / 3 + 10 / 11) / 2)
    assert m["pixel_accuracy"]["value"] == pytest.approx(6 / 7)


def test_segmentation_edge_cases_and_resize():
    e = make({"task": "segmentation", "classes": 2, "correct_miou": 0.9})
    nothing = {"mask": np.full((2, 2), 255)}  # all ignored: nothing to score -> correct, 1.0
    record = e.records([np.zeros((1, 2, 2, 2))], [nothing])[0]
    assert e.score(record, nothing) == 1.0 and e.correct(record, nothing)
    # a 1x1 prediction of class 1 against a 2x2 mask: every mask pixel samples it
    one = {"mask": np.array([[1, 1], [0, 1]])}
    record = e.records([onehot_logits(np.array([[[1]]]), 2).transpose(0, 3, 1, 2)], [one])[0]
    assert record == {"inter": [0, 3], "pred": [0, 4], "truth": [1, 3]}
    assert not e.correct(record, one)  # class 0 IoU 0, class 1 IoU 3/4 -> 3/8 < .9
    with pytest.raises(EvaluatorError):
        e.records([np.zeros((1, 2, 2, 2))], [{"mask": np.array([[0, 7], [0, 0]])}])  # 7 is neither a class nor the ignore index


# ---------------------------------------------------------------- detection

def test_iou_nms_and_matching():
    a = np.array([[0, 0, 2, 2]], dtype=float)
    b = np.array([[1, 1, 3, 3], [0, 0, 2, 2], [5, 5, 6, 6]], dtype=float)
    assert iou(a, b)[0].tolist() == pytest.approx([1 / 7, 1.0, 0.0])  # overlap 1, union 4 + 4 - 1
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [0, 0, 10, 10]], dtype=float)
    # #1 overlaps #0 by 81/100 > .5 and is the same class -> suppressed; #2 is the same box but another class -> kept
    assert nms(boxes, np.array([.9, .8, .7]), np.array([0, 0, 1]), 0.5).tolist() == [0, 2]
    # greedy, highest IoU: the .9 detection takes truth #1 (IoU 1) not #0 (IoU 1/7); the .8 one then gets nothing at .5
    pred = np.array([[0, 0, 2, 2, .9, 0], [0, 0, 2, 2, .8, 0]], dtype=float)
    truth = np.array([[1, 1, 3, 3, 0], [0, 0, 2, 2, 0]], dtype=float)
    assert match(pred, truth, 0.5).tolist() == [True, False]
    assert match(pred, truth, 0.1).tolist() == [True, True]  # at .1 the second takes truth #0 (IoU 1/7 >= .1)
    assert match(pred[::-1], truth, 0.5).tolist() == [False, True]  # given in another order: still the .9 one matches first


def test_average_precision_golden():
    # 2 positives; detections by score: TP, FP, TP -> precision 1, 1/2, 2/3; recall 1/2, 1/2, 1
    # envelope 1, 2/3, 2/3; recall points 0.00..0.50 (51) -> 1; 0.51..1.00 (50) -> 2/3 -> AP = (51 + 50 * 2/3) / 101
    assert average_precision(np.array([.9, .8, .7]), np.array([True, False, True]), 2) == pytest.approx((51 + 100 / 3) / 101)
    assert average_precision(np.array([]), np.array([], dtype=bool), 3) == 0.0
    assert average_precision(np.array([.5]), np.array([True]), 1) == 1.0


def test_detection_decode_maps_boxes_back_to_the_source_image():
    # model input 10x10 made from a 20x20 source scaled by .5 and padded by (2, 0); relative cxcywh (.5, .5, .2, .2) -> xyxy 4,4,6,6 in
    # the input -> minus offset (2, 0) -> 2,4,4,6 -> / .5 -> 4,8,8,12 in the source
    geometry = {"source": [20, 20], "scale": [0.5, 0.5], "offset": [2.0, 0.0], "output": [10, 10]}
    raw = np.array([[[.5, .5, .2, .2, .9, .2, .8], [.5, .5, .2, .2, .0001, .5, .5]]])  # the second is under score_threshold
    e = make({"task": "detection", "classes": 2, "box_format": "cxcywh", "box_scale": "relative"})
    (record,) = e.records([raw], [{"geometry": geometry, "boxes": []}])
    assert record["boxes"] == [pytest.approx([4, 8, 8, 12, 0.72, 1])]  # score .9 * .8, class 1


def test_detection_golden_map_f1_and_correctness():
    # one class, image 1: truth boxes T1 [0,0,4,4], T2 [6,6,10,10]; detections .9 on T1, .8 nowhere, .7 on T2 -> AP50 = (51 + 100/3) / 101
    # image 2: no truth, no detection -> F1 1.0, correct
    e = make({"task": "detection", "classes": 1, "conf_threshold": 0.75})
    raw = np.zeros((2, 3, 6))
    raw[0] = [[0, 0, 4, 4, .9, 1], [20, 20, 24, 24, .8, 1], [6, 6, 10, 10, .7, 1]]
    raw[1, :, 4] = 0.0  # all below score_threshold
    samples = [{"geometry": {**IDENTITY, "source": [30, 30], "output": [30, 30]},
                "boxes": [{"bbox": [0, 0, 4, 4], "category_id": 0}, {"bbox": [6, 6, 4, 4], "category_id": 0}]},
               {"geometry": IDENTITY, "boxes": []}]
    records = e.records([raw], samples)
    assert len(records[0]["boxes"]) == 3 and records[1]["boxes"] == []
    m = by_name(e.metrics(records, samples))
    assert m["map50"]["value"] == pytest.approx((51 + 100 / 3) / 101) and m["map50"]["primary"]
    # every match here has IoU 1, so AP is the same at every threshold 0.5..0.95
    assert m["map50_95"]["value"] == pytest.approx((51 + 100 / 3) / 101)
    # at conf .75: detections .9 (TP) and .8 (FP); T2 missed -> TP 1, FP 1, FN 1 -> F1 2/(2 + 2) = .5, not correct
    assert e.score(records[0], samples[0]) == 0.5 and not e.correct(records[0], samples[0])
    assert e.score(records[1], samples[1]) == 1.0 and e.correct(records[1], samples[1])


def test_detection_iou_thresholds_and_classes_without_truth():
    # a detection at IoU 0.64 with its truth box ([0,0,10,10] vs [0,0,8,8]: 64/100) counts at 0.50..0.60 (3 of 10 thresholds) only
    e = make({"task": "detection", "classes": 2, "class_ids": [7, 9]})
    raw = np.array([[[0, 0, 8, 8, 1.0, 1.0, 0.0]]])
    sample = {"geometry": {**IDENTITY, "source": [20, 20], "output": [20, 20]}, "boxes": [{"bbox": [0, 0, 10, 10], "category_id": 7}]}
    m = by_name(e.metrics(e.records([raw], [sample]), [sample]))
    assert m["map50"]["value"] == 1.0 and m["map50_95"]["value"] == pytest.approx(0.3)
    assert m["map50"]["per_class"][1] == {"class": 1, "value": None, "n": 0}  # category 9 has no truth: left out of the mean
    with pytest.raises(EvaluatorError):
        e.metrics([{"boxes": []}], [{"boxes": [{"bbox": [0, 0, 1, 1], "category_id": 3}]}])


# ---------------------------------------------------------------- OCR

def test_edit_distance_golden():
    assert edit_distance("4071", "4011") == 1 and edit_distance("", "abc") == 3 and edit_distance("kitten", "sitting") == 3
    assert edit_distance("ab cd".split(), "ab ce".split()) == 1


def test_ocr_golden():
    # alphabet "0123456789", blank 0: index i>0 is digit i-1. Steps 5 5 0 1 8 8 2 -> collapse 5,1,8,2 (the blank splits nothing here)
    # -> "4071"; steps 5 1 1 0 1 2 -> 5,1,1,2 (the blank separates the two 1s) -> "4001"
    e = make({"task": "ocr", "alphabet": "0123456789"})
    steps = [[5, 5, 0, 1, 8, 8, 2], [5, 1, 1, 0, 1, 2, 0]]
    samples = [{"text": "4071"}, {"text": "4071"}]
    records = e.records([onehot_logits(steps, 11)], samples)
    assert [r["text"] for r in records] == ["4071", "4001"]
    assert [e.correct(r, s) for r, s in zip(records, samples)] == [True, False]
    assert e.score(records[1], samples[1]) == 0.25 and e.score_better == "lower"  # one substitution in four characters
    # CER = (0 + 1) / (4 + 4); WER = (0 + 1) / (1 + 1); exact match 1/2
    m = by_name(e.metrics(records, samples))
    assert m["cer"]["value"] == 0.125 and m["cer"]["primary"] and m["cer"]["better"] == "lower"
    assert m["wer"]["value"] == 0.5 and m["exact_match"]["value"] == 0.5


def test_ocr_blank_position_and_cer_above_one():
    e = make({"task": "ocr", "alphabet": "ab", "blank": 2})  # indices 0 -> a, 1 -> b, 2 -> blank
    assert e.decode([0, 2, 0, 1, 1]) == "aab"
    sample = {"text": "a"}
    record = e.records([onehot_logits([[0, 2, 0, 2, 0]], 3)], [sample])[0]
    assert record["text"] == "aaa" and e.score(record, sample) == 2.0  # two insertions on one character: CER 2
    assert by_name(e.metrics([record], [sample]))["cer"]["value"] == 2.0


# ---------------------------------------------------------------- dev-only oracle (ADR 0001 pattern)

@pytest.mark.oracle
@pytest.mark.parametrize("seed", range(20))
def test_map_matches_pycocotools_on_random_cases(seed):
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    rng = np.random.default_rng(seed)
    e = make({"task": "detection", "classes": 3})
    images, truth, records, samples = [], [], [], []
    for image in range(20):
        boxes = []
        for _ in range(int(rng.integers(0, 4))):
            x, y = rng.uniform(0, 80, size=2)
            w, h = rng.uniform(5, 20, size=2)
            boxes.append({"bbox": [float(x), float(y), float(w), float(h)], "category_id": int(rng.integers(3))})
        dets = []
        for box in boxes:
            if rng.random() < 0.8:
                x, y, w, h = box["bbox"]
                jitter = rng.normal(0, 2, size=4)
                x1, x2 = sorted((x + jitter[0], x + w + jitter[2]))  # an inverted box has zero area here; pycocotools would give it a
                y1, y2 = sorted((y + jitter[1], y + h + jitter[3]))  # negative one, so the generator never makes one
                dets.append([x1, y1, x2, y2, float(rng.uniform(.3, 1)), box["category_id"]])
        for _ in range(int(rng.integers(0, 3))):
            x, y = rng.uniform(0, 80, size=2)
            dets.append([x, y, x + 10, y + 10, float(rng.uniform(.05, .6)), int(rng.integers(3))])
        images.append({"id": image, "width": 100, "height": 100})
        truth += [{"id": len(truth) + k + 1, "image_id": image, "area": b["bbox"][2] * b["bbox"][3], "iscrowd": 0, **b}
                  for k, b in enumerate(boxes)]
        records.append({"boxes": dets})
        samples.append({"boxes": boxes})
    gt = COCO()
    gt.dataset = {"images": images, "annotations": truth, "categories": [{"id": c} for c in range(3)]}
    gt.createIndex()
    dt = gt.loadRes([{"image_id": i, "bbox": [d[0], d[1], d[2] - d[0], d[3] - d[1]], "score": d[4], "category_id": int(d[5])}
                     for i, r in enumerate(records) for d in r["boxes"]]) if any(r["boxes"] for r in records) else None
    run = COCOeval(gt, dt, "bbox")
    run.params.areaRng, run.params.areaRngLbl = [[0, 1e10]], ["all"]
    run.evaluate()
    run.accumulate()
    precision = run.eval["precision"][:, :, :, 0, -1]  # [T, R, K]
    expected_50_95 = precision[precision > -1].mean()
    expected_50 = precision[0][precision[0] > -1].mean()
    m = by_name(e.metrics(records, samples))
    assert m["map50"]["value"] == pytest.approx(expected_50, abs=1e-9)
    assert m["map50_95"]["value"] == pytest.approx(expected_50_95, abs=1e-9)
