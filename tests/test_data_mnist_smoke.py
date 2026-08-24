import struct
import gzip

import numpy as np
import pytest

from micro_dl.data import DataLoader, MNIST, Normalize
from micro_dl.experiments import build_mnist_experiment
from micro_dl import Tensor


def test_mnist_idx_parsing_batching_and_deterministic_shuffle(tmp_path):
    images = np.arange(4 * 2 * 2, dtype=np.uint8).reshape(4, 2, 2)
    labels = np.array([0, 1, 2, 3], dtype=np.uint8)
    images_path = tmp_path / "images.idx"
    labels_path = tmp_path / "labels.idx"
    images_path.write_bytes(struct.pack(">IIII", 2051, 4, 2, 2) + images.tobytes())
    labels_path.write_bytes(struct.pack(">II", 2049, 4) + labels.tobytes())
    dataset = MNIST.from_idx(images_path, labels_path, transform=Normalize(0.0, 1.0))

    batches = list(DataLoader(dataset, batch_size=3, shuffle=True, seed=7))
    assert [batch[0].shape for batch in batches] == [(3, 1, 2, 2), (1, 1, 2, 2)]
    assert sum(len(targets) for _, targets in batches) == 4
    order = np.concatenate([targets for _, targets in batches]).tolist()
    expected = np.arange(4); np.random.default_rng(7).shuffle(expected)
    assert order == expected.tolist()
    second = list(DataLoader(dataset, batch_size=4, shuffle=True, seed=7))[0][1].tolist()
    assert second == expected.tolist()

    cache = MNIST.cache_paths(tmp_path / "cache")
    assert cache[0].parent == tmp_path / "cache"
    assert not cache[0].exists()
    cache[0].parent.mkdir()
    cache[0].write_bytes(gzip.compress(images_path.read_bytes()))
    cache[1].write_bytes(gzip.compress(labels_path.read_bytes()))
    assert len(MNIST.from_cache(cache[0].parent)) == 4


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_mnist_experiment_runs_one_training_batch(backend):
    images = np.zeros((4, 1, 2, 2), dtype=np.uint8)
    labels = np.array([0, 1, 0, 1], dtype=np.uint8)
    dataset = MNIST(images, labels)
    if backend == "cupy":
        try:
            from micro_dl import Tensor
            Tensor([0.0], backend="cupy")
        except RuntimeError:
            pytest.skip("CuPy Backend unavailable")
    trainer, loader = build_mnist_experiment(dataset, batch_size=4, backend=backend)
    before = [parameter.to_numpy() for parameter in trainer.model.parameters()]

    result = trainer.fit(loader, epochs=1)

    assert len(result.epochs) == 1
    assert np.isfinite(result.epochs[0].train_loss)
    assert any(
        not np.allclose(old, parameter.to_numpy())
        for old, parameter in zip(before, trainer.model.parameters())
    )


def test_canonical_mnist_experiment_uses_readable_cnn():
    images = np.zeros((4, 1, 28, 28), dtype=np.uint8)
    labels = np.arange(4, dtype=np.uint8)

    trainer, _ = build_mnist_experiment(MNIST(images, labels), batch_size=4)

    assert tuple(trainer.model._modules) == ("features", "classifier")
    assert trainer.model(Tensor(images.astype(np.float32))).shape == (4, 10)
