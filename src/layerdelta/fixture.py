"""Build the canonical, CPU-trained reference fixture from verified MNIST archives."""

import argparse
import hashlib
import json
import platform
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

from .data import build_manifest, download_mnist, load_split
from .model import ReferenceCNN, describe_model


TRAINING = {
    "revision": 1,
    "seed": 20261122,
    "epochs": 1,
    "batch_size": 128,
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "loss": "cross_entropy",
    "dtype": "torch.float32",
    "device": "cpu",
    "threads": 1,
}


def _write_json(path: Path, value: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def train(images: torch.Tensor, labels: torch.Tensor) -> ReferenceCNN:
    """Train only on the normal train split; use an independent generator for ordering."""
    torch.manual_seed(TRAINING["seed"])
    model = ReferenceCNN().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING["learning_rate"])
    order = torch.Generator(device="cpu").manual_seed(TRAINING["seed"])
    for _ in range(TRAINING["epochs"]):
        shuffled = torch.randperm(len(labels), generator=order)
        for indices in shuffled.split(TRAINING["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            scores = model(images[indices])
            loss = torch.nn.functional.cross_entropy(scores, labels[indices])
            loss.backward()
            optimizer.step()
    return model.eval()


def exploration_evidence(rows: list[dict], predictions: list[int]) -> dict:
    """Rank blurred confusions by frequency, breaking ties by ascending label pair."""
    matrices = {condition: [[0] * 10 for _ in range(10)] for condition in ("normal", "blurred")}
    source_pairs = {}
    for row, prediction in zip(rows, predictions, strict=True):
        if row["split"] != "exploration":
            raise ValueError("Demo selection requires exploration samples")
        label, condition = row["label"], row["condition"]
        matrices[condition][label][prediction] += 1
        pair = source_pairs.setdefault(row["source_sample_id"], {})
        if condition in pair:
            raise ValueError("Duplicate condition for exploration source")
        pair[condition] = {"sample_id": row["sample_id"], "label": label, "prediction": prediction}

    counts = matrices["blurred"]
    pairs = ((label, pred) for label in range(10) for pred in range(10) if label != pred)
    label, predicted = max(pairs, key=lambda pair: counts[pair[0]][pair[1]])
    if counts[label][predicted] == 0:
        raise ValueError("No blurred exploration errors available for the reference demo")

    examples = []
    for source_id, variants in source_pairs.items():
        if set(variants) != {"normal", "blurred"} or variants["normal"]["label"] != variants["blurred"]["label"]:
            raise ValueError(f"Missing or mismatched exploration pair for {source_id}")
        blurred = variants["blurred"]
        if (blurred["label"], blurred["prediction"]) == (label, predicted):
            examples.append({"source_sample_id": source_id, **variants})
    examples.sort(key=lambda pair: (pair["normal"]["prediction"] != label, pair["source_sample_id"]))
    return {
        "split": "exploration",
        "conditions": {
            condition: {
                "N": sum(map(sum, matrix)),
                "correct": sum(matrix[i][i] for i in range(10)),
                "confusion_truth_by_prediction": matrix,
            }
            for condition, matrix in matrices.items()
        },
        "demo": {
            "selection_rule": "most frequent off-diagonal blurred pair, ties by ascending (truth, prediction)",
            "truth": label,
            "baseline_prediction": predicted,
            "pair_count": counts[label][predicted],
            "primary_slice": {"condition": "blurred", "truth": label, "N": sum(counts[label])},
            "guardrail_slice": {"condition": "normal", "N": sum(map(sum, matrices["normal"]))},
            "example_pairs": examples[:5],
        },
    }


def build_fixture(output: Path, data_dir: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Choose a new fixture directory: {output}")
    if data_dir.resolve().is_relative_to(output.resolve()):
        raise ValueError("Choose a cache directory outside the fixture output")
    output.parent.mkdir(parents=True, exist_ok=True)
    download_mnist(data_dir)
    manifest = build_manifest(data_dir)
    torch.set_num_threads(TRAINING["threads"])
    torch.use_deterministic_algorithms(True)
    train_x, train_y, _ = load_split(data_dir, manifest, "train")
    model = train(train_x, train_y)
    train_count = len(train_y)
    del train_x, train_y

    images, _, rows = load_split(data_dir, manifest, "exploration")
    with torch.inference_mode():
        predictions = [
            prediction
            for batch in images.split(TRAINING["batch_size"])
            for prediction in model(batch).argmax(1).tolist()
        ]
    evidence = exploration_evidence(rows, predictions)

    with tempfile.TemporaryDirectory(dir=output.parent) as tmpdir:
        staged = Path(tmpdir) / "fixture"
        staged.mkdir()
        save_file(model.state_dict(), str(staged / "model.safetensors"))
        _write_json(staged / "dataset.json", manifest)
        _write_json(staged / "exploration.json", evidence)
        fixture = {
            "schema_version": 1,
            "model": describe_model(model),
            "training": {**TRAINING, "sample_count": train_count},
            "runtime": {
                "torch": str(torch.__version__),
                "python": platform.python_version(),
                "os": platform.system(),
                "cpu_capability": torch.backends.cpu.get_cpu_capability(),
                "mkldnn": torch.backends.mkldnn.enabled,
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            },
            "manifest_sha256": manifest["manifest_sha256"],
            "artifacts": {
                name: file_sha256(staged / name)
                for name in ("model.safetensors", "dataset.json", "exploration.json")
            },
        }
        _write_json(staged / "fixture.json", fixture)
        staged.rename(output)
    return fixture


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="New fixture directory")
    parser.add_argument("--data-dir", type=Path, default=Path(".cache/mnist"), help="MNIST archive cache")
    args = parser.parse_args()
    fixture = build_fixture(args.output, args.data_dir)
    print(json.dumps({"output": str(args.output), **fixture["artifacts"]}, indent=2))


if __name__ == "__main__":
    main()
