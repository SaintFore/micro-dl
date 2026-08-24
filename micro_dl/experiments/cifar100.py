"""Canonical CIFAR-100 ResNet-like Experiment assembly."""

from micro_dl.data import CIFAR100, Compose, DataLoader, Normalize, RandomCrop, RandomHorizontalFlip
from micro_dl.nn import BatchNorm2d, Conv2d, CrossEntropyLoss, Linear, Module, ResidualBlock, Sequential
from micro_dl.optim import AdamW, CosineAnnealingLR
from micro_dl.training import Trainer


class CIFAR100ResNet(Module):
    def __init__(self, *, classes=100, base_channels=32, backend="numpy"):
        super().__init__()
        width = base_channels
        self.stem = Sequential(
            Conv2d(3, width, 3, padding=1, backend=backend),
            BatchNorm2d(width, backend=backend),
        )
        self.stage1 = Sequential(ResidualBlock(width, backend=backend), ResidualBlock(width, backend=backend))
        self.stage2 = Sequential(ResidualBlock(width, width * 2, stride=2, backend=backend), ResidualBlock(width * 2, backend=backend))
        self.stage3 = Sequential(ResidualBlock(width * 2, width * 4, stride=2, backend=backend), ResidualBlock(width * 4, backend=backend))
        self.stage4 = ResidualBlock(width * 4, width * 8, stride=2, backend=backend)
        self.classifier = Linear(width * 8, classes, backend=backend)

    def forward(self, value):
        value = self.stem(value).relu()
        value = self.stage4(self.stage3(self.stage2(self.stage1(value))))
        return self.classifier(value.mean(axis=(2, 3)))


def build_cifar100_experiment(dataset, *, backend="numpy", batch_size=64, epochs=30, base_channels=32):
    model = CIFAR100ResNet(base_channels=base_channels, backend=backend)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, backend=backend)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    trainer = Trainer(
        model,
        CrossEntropyLoss(label_smoothing=0.1),
        optimizer,
        scheduler=CosineAnnealingLR(optimizer, total_epochs=epochs),
        max_grad_norm=5.0,
        selection_metric="validation_accuracy",
    )
    return trainer, loader


def assemble_cifar100_experiment(cache_dir, *, backend="numpy", batch_size=64, epochs=30, seed=0):
    mean = [[[0.5071]], [[0.4867]], [[0.4408]]]
    std = [[[0.2675]], [[0.2565]], [[0.2761]]]
    normalize = Normalize(mean, std)
    augmentation = Compose([
        RandomCrop(32, padding=4, seed=seed),
        RandomHorizontalFlip(0.5, seed=seed),
        normalize,
    ])
    train_set = CIFAR100.from_cache(cache_dir, transform=augmentation)
    validation_set = CIFAR100.from_cache(cache_dir, train=False, transform=normalize)
    trainer, train_loader = build_cifar100_experiment(
        train_set, backend=backend, batch_size=batch_size, epochs=epochs,
    )
    validation_loader = DataLoader(validation_set, batch_size=batch_size, backend=backend)
    return trainer, train_loader, validation_loader
