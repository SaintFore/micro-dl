import gzip
import json
import pickle
import struct

import numpy as np
import pytest
from typer.testing import CliRunner

from micro_dl.cli import app


@pytest.mark.parametrize("experiment", ["mnist", "cifar100"])
def test_profile_reports_training_phase_costs_as_json(tmp_path, experiment):
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
    output = tmp_path / "profile.json"

    result = CliRunner().invoke(
        app,
        ["profile", "--experiment", experiment, "--backend", "numpy", "--steps", "1",
         "--batch-size", "4", "--data-dir", str(tmp_path / "data"), "--output", str(output)],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert json.loads(output.read_text()) == report
    assert report["experiment"] == experiment
    assert report["backend"] == "numpy"
    assert report["steps"] == 1
    assert report["total_ms"] >= 0.0
    assert set(report["phases"]) == {"data", "forward", "backward", "optimizer"}
    for phase in report["phases"].values():
        assert phase["total_ms"] >= 0.0
        assert phase["ms_per_step"] >= 0.0
        assert 0.0 <= phase["percent"] <= 100.0


def test_profile_rejects_incomplete_cache_with_clear_message(tmp_path):
    result = CliRunner().invoke(
        app,
        ["profile", "--experiment", "mnist", "--backend", "numpy", "--steps", "1",
         "--data-dir", str(tmp_path / "missing")],
    )

    assert result.exit_code != 0
    assert "cache is incomplete" in result.output


def test_profile_rejects_non_positive_steps():
    result = CliRunner().invoke(app, ["profile", "--experiment", "mnist", "--steps", "0"])

    assert result.exit_code != 0
    assert "x>=1" in result.output
