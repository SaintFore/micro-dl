import zipfile

import numpy as np
import pytest

from micro_dl.run import RunConfig, load_checkpoint, save_checkpoint


def test_checkpoint_round_trip_uses_split_safe_members(tmp_path):
    destination = tmp_path / "last.zip"
    config = RunConfig("mnist", backend="numpy")
    save_checkpoint(
        destination,
        config_toml=config.to_toml(),
        model_state={"weight": np.array([1.0, 2.0], dtype=np.float32)},
        optimizer_state={"lr": 0.1, "moments": [np.array([3.0])]},
        scheduler_state={"epoch": 2},
        rng_state={"numpy": np.array([4, 5], dtype=np.uint32)},
        metadata={"git_commit": "abc123", "epoch": 2, "global_step": 7},
    )
    loaded = load_checkpoint(destination)
    np.testing.assert_array_equal(loaded["model"]["weight"], [1.0, 2.0])
    np.testing.assert_array_equal(loaded["optimizer"]["moments"][0], [3.0])
    np.testing.assert_array_equal(loaded["rng"]["numpy"], [4, 5])
    config_path = tmp_path / "config.toml"
    config_path.write_text(loaded["config_toml"])
    assert RunConfig.from_toml(config_path) == config
    with zipfile.ZipFile(destination) as archive:
        assert set(archive.namelist()) == {"manifest.json", "config.toml", "model.npz", "optimizer.json", "optimizer.npz", "scheduler.json", "rng.json", "rng.npz"}


def test_checkpoint_rejects_tampered_member(tmp_path):
    destination = tmp_path / "last.zip"
    save_checkpoint(destination, config_toml="experiment = \"mnist\"\n", model_state={}, optimizer_state={})
    replacement = tmp_path / "bad.zip"
    with zipfile.ZipFile(destination) as source, zipfile.ZipFile(replacement, "w") as target:
        for name in source.namelist():
            target.writestr(name, b"bad" if name == "config.toml" else source.read(name))
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(replacement)
