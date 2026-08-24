import pickle

import numpy as np
import pytest

from micro_dl import Tensor
from micro_dl.data import CIFAR100, Compose, DataLoader, Normalize, RandomCrop, RandomHorizontalFlip
from micro_dl.experiments import CIFAR100ResNet, assemble_cifar100_experiment, build_cifar100_experiment
from micro_dl.optim import AdamW, CosineAnnealingLR


def test_cifar100_pickle_parsing_and_transforms(tmp_path):
    data = np.arange(2 * 3 * 32 * 32, dtype=np.uint8).reshape(2, -1)
    path = tmp_path / "train"
    with path.open("wb") as handle:
        pickle.dump({b"data": data, b"fine_labels": [2, 7]}, handle)
    transform = Compose([RandomHorizontalFlip(1.0), Normalize(0.0, 1.0)])
    dataset = CIFAR100.from_file(path, transform=transform)

    image, label = dataset[0]
    assert image.shape == (3, 32, 32)
    assert label == 2
    np.testing.assert_allclose(image[..., 0], data[0].reshape(3, 32, 32)[..., -1] / 255.0)
    inputs, targets = next(iter(DataLoader(dataset, batch_size=2)))
    assert inputs.shape == (2, 3, 32, 32)
    assert targets.tolist() == [2, 7]
    assert CIFAR100.cache_paths(tmp_path / "cache")[0].parent == tmp_path / "cache"
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "train").write_bytes(path.read_bytes())
    assert len(CIFAR100.from_cache(cache)) == 2


def test_random_crop_uses_padding_and_preserves_chw_shape():
    image = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)

    cropped = RandomCrop(4, padding=2, seed=7)(image)

    assert cropped.shape == image.shape
    assert np.count_nonzero(cropped == 0) > 0


def test_dataloader_state_restores_shuffle_and_augmentation_sequence():
    images = np.arange(8 * 3 * 4 * 4, dtype=np.uint8).reshape(8, 3, 4, 4)
    transform = Compose([RandomCrop(4, padding=2, seed=7), RandomHorizontalFlip(0.5, seed=9)])
    loader = DataLoader(CIFAR100(images, np.arange(8), transform=transform), batch_size=4, shuffle=True, seed=3)
    next(iter(loader))
    saved = loader.state_dict()
    expected_inputs, expected_targets = next(iter(loader))

    restored_transform = Compose([RandomCrop(4, padding=2, seed=7), RandomHorizontalFlip(0.5, seed=9)])
    restored = DataLoader(CIFAR100(images, np.arange(8), transform=restored_transform), batch_size=4, shuffle=True, seed=3)
    restored.load_state_dict(saved)
    actual_inputs, actual_targets = next(iter(restored))

    np.testing.assert_array_equal(actual_inputs.to_numpy(), expected_inputs.to_numpy())
    np.testing.assert_array_equal(actual_targets, expected_targets)


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_cifar100_cnn_experiment_runs_short_training_smoke(backend):
    if backend == "cupy":
        try: Tensor([0.0], backend="cupy")
        except RuntimeError: pytest.skip("CuPy Backend unavailable")
    images = np.zeros((2, 3, 4, 4), dtype=np.uint8)
    images[1] = 255
    dataset = CIFAR100(images, [0, 1])
    trainer, loader = build_cifar100_experiment(dataset, batch_size=2, backend=backend)
    before = [parameter.to_numpy() for parameter in trainer.model.parameters()]

    result = trainer.fit(loader, epochs=1)

    assert len(result.epochs) == 1
    assert np.isfinite(result.epochs[0].train_loss)
    assert any(not np.allclose(old, parameter.to_numpy()) for old, parameter in zip(before, trainer.model.parameters()))


def test_cifar100_resnet_hierarchy_produces_logits_and_gradients():
    model = CIFAR100ResNet(base_channels=2, backend="numpy")
    inputs = Tensor(np.ones((2, 3, 8, 8), dtype=np.float32), requires_grad=True)

    logits = model(inputs)

    assert logits.shape == (2, 100)
    logits.sum().backward()
    assert inputs.grad_to_numpy().shape == inputs.shape


def test_canonical_cifar100_assembler_owns_legacy_equivalent_recipe(tmp_path):
    cache = tmp_path / "cifar100"
    cache.mkdir()
    payload = {b"data": np.zeros((4, 3 * 32 * 32), dtype=np.uint8), b"fine_labels": [0, 1, 2, 3]}
    for split in ("train", "test"):
        with (cache / split).open("wb") as handle:
            pickle.dump(payload, handle)

    trainer, train_loader, validation_loader = assemble_cifar100_experiment(
        cache, backend="numpy", batch_size=2, epochs=30, seed=7,
    )

    assert isinstance(trainer.model, CIFAR100ResNet)
    assert isinstance(trainer.optimizer, AdamW)
    assert trainer.optimizer.decoupled_weight_decay == 5e-4
    assert trainer.loss.label_smoothing == 0.1
    assert trainer.max_grad_norm == 5.0
    assert isinstance(trainer.scheduler, CosineAnnealingLR)
    assert trainer.scheduler.total_epochs == 30
    assert [type(item).__name__ for item in train_loader.dataset.transform.transforms] == [
        "RandomCrop", "RandomHorizontalFlip", "Normalize",
    ]
    assert type(validation_loader.dataset.transform).__name__ == "Normalize"
    assert train_loader.shuffle is True
    assert validation_loader.shuffle is False
