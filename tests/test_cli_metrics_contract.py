import gzip
import json
import pickle
import struct

import numpy as np
import pytest
from typer.testing import CliRunner

from micro_dl.cli import app
from micro_dl.run import load_checkpoint


@pytest.mark.parametrize("experiment", ["mnist", "cifar100"])
def test_train_epoch_metric_contains_validation_accuracy(tmp_path, monkeypatch, experiment):
    cache = tmp_path / "data" / experiment
    cache.mkdir(parents=True)
    labels = np.array([0, 1, 0, 1], dtype=np.uint8)
    if experiment == "mnist":
        images = np.zeros((4, 1, 2, 2), dtype=np.uint8)
        for prefix in ("train", "t10k"):
            with gzip.open(cache / f"{prefix}-images-idx3-ubyte.gz", "wb") as handle:
                handle.write(struct.pack(">IIII", 2051, 4, 2, 2) + images.tobytes())
            with gzip.open(cache / f"{prefix}-labels-idx1-ubyte.gz", "wb") as handle:
                handle.write(struct.pack(">II", 2049, 4) + labels.tobytes())
    else:
        images = np.zeros((4, 3 * 32 * 32), dtype=np.uint8)
        for split in ("train", "test"):
            with (cache / split).open("wb") as handle:
                pickle.dump({b"data": images, b"fine_labels": labels.tolist()}, handle)
    monkeypatch.setattr("micro_dl.cli._require_clean_git", lambda: None)
    monkeypatch.setattr("micro_dl.cli._git", lambda *args: "abc1234")

    result = CliRunner().invoke(app, ["train", "--experiment", experiment, "--backend", "numpy", "--epochs", "1",
                                             "--batch-size", "4", "--data-dir", str(tmp_path / "data"),
                                             "--run-root", str(tmp_path / "runs")])

    assert result.exit_code == 0, result.output
    metric_path = next((tmp_path / "runs").glob(f"{experiment}/*/metrics.jsonl"))
    epoch = [json.loads(line) for line in metric_path.read_text().splitlines() if '"type":"epoch"' in line][0]
    assert 0.0 <= epoch["validation_accuracy"] <= 1.0


def test_classification_best_checkpoint_records_maximum_accuracy(tmp_path, monkeypatch):
    cache = tmp_path / "data" / "mnist"
    cache.mkdir(parents=True)
    images = np.zeros((4, 1, 2, 2), dtype=np.uint8)
    labels = np.array([0, 1, 0, 1], dtype=np.uint8)
    for prefix in ("train", "t10k"):
        with gzip.open(cache / f"{prefix}-images-idx3-ubyte.gz", "wb") as handle:
            handle.write(struct.pack(">IIII", 2051, 4, 2, 2) + images.tobytes())
        with gzip.open(cache / f"{prefix}-labels-idx1-ubyte.gz", "wb") as handle:
            handle.write(struct.pack(">II", 2049, 4) + labels.tobytes())
    monkeypatch.setattr("micro_dl.cli._require_clean_git", lambda: None)
    monkeypatch.setattr("micro_dl.cli._git", lambda *args: "abc1234")

    result = CliRunner().invoke(
        app,
        ["train", "--experiment", "mnist", "--backend", "numpy", "--epochs", "2",
         "--batch-size", "4", "--data-dir", str(tmp_path / "data"), "--run-root", str(tmp_path / "runs")],
    )

    assert result.exit_code == 0, result.output
    run_dir = next((tmp_path / "runs").glob("mnist/*"))
    epochs = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines() if '"type":"epoch"' in line]
    checkpoint = load_checkpoint(run_dir / "checkpoints" / "best.zip")
    assert checkpoint["manifest"]["selection_metric"] == "validation_accuracy"
    assert checkpoint["manifest"]["best_score"] == max(item["validation_accuracy"] for item in epochs)
