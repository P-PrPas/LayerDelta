"""Measure M1 CLI runs, isolation and cross-device logits on the frozen exploration split."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time


# Frozen before the first regression on the Windows cu124 reference runtime.
REFERENCE_ABSOLUTE_TOLERANCE = 2 ** -16


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args, device, case, cache):
    cmd = [sys.executable, "-m", "layerdelta.runner", "--fixture", str(args.fixture),
           "--data-dir", str(args.data_dir), "--cache", str(cache), "--device", device,
           "--output", str(args.output / f"{device}-{case}")]
    targets = {"activation": "features.block2.relu.output", "weight": "features.block2.conv.kernel"}
    if case in targets:
        cmd += ["--target", targets[case], "--channel", "17", "--operation", "zero",
                "--hypothesis", "Zero channel 17 to measure blurred 9 accuracy"]
    return cmd


def run_case(args, device, case, cache):
    start = time.perf_counter()
    subprocess.run(command(args, device, case, cache), check=True, timeout=180)
    (args.output / f"{device}-{case}.wall").write_text(str(time.perf_counter() - start))


def fault_worker(recipe, marker, kind, worker_args):
    # Only this disposable process installs a hook. The real runner loads the model,
    # applies the frozen patch, and computes an actual batch before the fault occurs.
    from layerdelta import engine, runner

    infer = engine.infer

    def interrupt(model, images, rows, device, batch_size, spec=None, **kwargs):
        if spec is None:
            return infer(model, images, rows, device, batch_size, spec, **kwargs)

        def after_batch(*_):
            marker.touch()
            if kind == "cancel":
                time.sleep(120)
            raise RuntimeError("M1 acceptance injected post-forward failure")

        hook = model.register_forward_hook(after_batch)
        try:
            return infer(model, images, rows, device, batch_size, spec, **kwargs)
        finally:
            hook.remove()

    engine.infer = interrupt
    sys.argv = [sys.argv[0], "--worker", "--recipe", str(recipe), *worker_args]
    runner.main()


def isolation(args, device):
    baseline = args.output / f"{device}-baseline"
    parent = args.fixture / "model.safetensors"
    original = read(baseline / "report.json")["parent_sha256"]
    for kind in ("fail", "cancel"):
        marker = args.output / f"{device}-{kind}.marker"
        cmd = command(args, device, kind, args.output / "cache")
        recipe = args.output / f"{device}-weight" / "recipe.json"
        with subprocess.Popen([sys.executable, __file__, "_fault", str(recipe), str(marker),
                               kind, *cmd[3:]], stderr=subprocess.PIPE, text=True) as proc:
            try:
                deadline = time.monotonic() + 120
                while not marker.exists() and proc.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                if not marker.exists():
                    raise RuntimeError(f"{kind} never reached forward on {device}")
                if kind == "cancel":
                    proc.kill()
                _, stderr = proc.communicate(timeout=10)
                if proc.returncode == 0 or (kind == "fail" and "injected post-forward failure" not in stderr):
                    raise RuntimeError(f"Unexpected {kind} result: {proc.returncode}, {stderr}")
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
        if (args.output / f"{device}-{kind}").exists() or sha(parent) != original:
            raise RuntimeError("Fault published a partial result or altered the parent")

    run_case(args, device, "recovery", args.output / f"{device}-empty-cache")
    fresh = args.output / f"{device}-recovery"
    if sha(baseline / "predictions.json") != sha(fresh / "predictions.json"):
        raise RuntimeError("Cold post-fault baseline differs from original")


def diagnostics(root, case, calibration_only=False):
    cpu_rows = read(root / f"cpu-{case}" / "predictions.json")
    gpu_rows = read(root / f"cuda-{case}" / "predictions.json")
    gpu = {row["sample_id"]: row for row in gpu_rows}
    if len(cpu_rows) != len(gpu) or len(gpu_rows) != len(gpu) or {r["sample_id"] for r in cpu_rows} != gpu.keys():
        raise ValueError("CPU and CUDA sample IDs disagree")
    for cpu in cpu_rows:
        # Keep both image variants on the same side of the calibration boundary.
        calibration = hashlib.sha256(cpu["source_sample_id"].encode()).digest()[0] < 64
        if calibration_only and not calibration:
            continue
        other = gpu[cpu["sample_id"]]
        if any(cpu[k] != other[k] for k in ("label", "condition", "source_sample_id", "split")):
            raise ValueError("Mismatched cross-device sample metadata")
        differences = [abs(a - b) for a, b in zip(cpu["logits"], other["logits"], strict=True)]
        if len(differences) != 10 or not all(map(math.isfinite, differences)):
            raise ValueError("Invalid logits")
        yield calibration, cpu["sample_id"], max(differences), cpu["prediction"] != other["prediction"]


def summarize(args, tolerance):
    summary = {"absolute_logit_tolerance": tolerance, "runs": {}, "cross_device": {}}
    for device in ("cpu", "cuda"):
        baseline = read(args.output / f"{device}-baseline" / "report.json")
        baseline_digest = baseline["artifacts"]["predictions.json"]["sha256"]
        for case in ("baseline", "activation", "weight", "warm", "recovery"):
            directory = args.output / f"{device}-{case}"
            report = read(directory / "report.json")
            state = "cold" if case in ("baseline", "recovery") else "warm"
            if report["cache"] != state or report["runtime"]["actual_device"].split(":")[0] != device:
                raise ValueError("Unexpected cache state or runtime")
            if report["baseline_id"] != baseline["baseline_id"]:
                raise ValueError("Incompatible evaluation fingerprints on the same device")
            if case in ("warm", "recovery") and sha(directory / "predictions.json") != baseline_digest:
                raise ValueError("Baseline predictions changed after intervention or recovery")
            summary["runs"][directory.name] = {
                "wall_seconds": float(directory.with_suffix(".wall").read_text()),
                "artifact_bytes": sum(p.stat().st_size for p in directory.iterdir()),
                **{k: report[k] for k in ("cache", "inference_seconds", "peak_ram_bytes", "peak_vram_bytes",
                                          "run_id", "baseline_id", "recipe_hash", "sample_count")},
            }
    for case in ("baseline", "activation", "weight"):
        calibration_n = regression_n = 0
        calibration_max = regression_max = 0.0
        differing_predictions = []
        outside_regression_tolerance = []
        for calibration, sample_id, error, changed in diagnostics(args.output, case):
            if calibration:
                calibration_n += 1
                calibration_max = max(calibration_max, error)
            else:
                regression_n += 1
                regression_max = max(regression_max, error)
                if error > tolerance:
                    outside_regression_tolerance.append(sample_id)
            if changed:
                differing_predictions.append(sample_id)
        summary["cross_device"][case] = dict(
            calibration_rows=calibration_n, regression_rows=regression_n,
            calibration_max=calibration_max, regression_max=regression_max,
            differing_prediction_ids=differing_predictions, outside_regression_tolerance=outside_regression_tolerance,
        )
    (args.output / "measurements.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if any(c["outside_regression_tolerance"] for c in summary["cross_device"].values()):
        raise RuntimeError("Cross-device regression exceeded the frozen tolerance; see measurements.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["measure", "isolation", "calibrate", "regression"])
    parser.add_argument("--fixture", type=Path, default=Path(".layerdelta/reference"))
    parser.add_argument("--data-dir", type=Path, default=Path(".layerdelta/mnist"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance-file", type=Path, help="Explicit calibration record for a new runtime")
    args = parser.parse_args()
    if args.phase == "measure":
        args.output.mkdir(parents=True, exist_ok=False)
        for device in ("cpu", "cuda"):
            for case in ("baseline", "activation", "weight", "warm"):
                run_case(args, device, case, args.output / "cache")
    elif args.phase == "isolation":
        for device in ("cpu", "cuda"):
            isolation(args, device)
    elif args.phase == "calibrate":
        peak = max(error for _, _, error, _ in diagnostics(args.output, "baseline", calibration_only=True))
        # A power-of-two envelope with a factor-of-two margin over the observed maximum.
        tolerance = 2 ** math.ceil(math.log2(max(peak * 2, 2 ** -23)))
        with (args.output / "calibration.json").open("x", encoding="utf-8") as stream:
            json.dump({"baseline_calibration_peak": peak, "absolute_logit_tolerance": tolerance}, stream)
    else:
        tolerance = (read(args.tolerance_file)["absolute_logit_tolerance"]
                     if args.tolerance_file else REFERENCE_ABSOLUTE_TOLERANCE)
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Expected a finite nonnegative logit tolerance")
        summarize(args, tolerance)


if __name__ == "__main__":
    if sys.argv[1:2] == ["_fault"]:
        fault_worker(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5:])
    else:
        main()
