"""Readable MNIST MLP assembly."""

from micro_dl.data import DataLoader
from micro_dl.nn import BatchNorm2d, Conv2d, CrossEntropyLoss, Linear, Module, ReLU, Sequential
from micro_dl.optim import Adam
from micro_dl.training import Trainer


class MNISTCNN(Module):
    def __init__(self, *, backend="numpy"):
        super().__init__()
        self.features = Sequential(
            Conv2d(1, 16, 3, padding=1, backend=backend),
            BatchNorm2d(16, backend=backend),
            ReLU(),
            Conv2d(16, 32, 3, stride=2, padding=1, backend=backend),
            BatchNorm2d(32, backend=backend),
            ReLU(),
            Conv2d(32, 64, 3, stride=2, padding=1, backend=backend),
            BatchNorm2d(64, backend=backend),
            ReLU(),
        )
        self.classifier = Linear(64, 10, backend=backend)

    def forward(self, value):
        value = self.features(value)
        return self.classifier(value.mean(axis=(2, 3)))


def build_mnist_experiment(dataset, *, backend="numpy", batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, backend=backend)
    model = MNISTCNN(backend=backend)
    trainer = Trainer(
        model,
        CrossEntropyLoss(),
        Adam(model.parameters(), lr=1e-3),
        selection_metric="validation_accuracy",
    )
    return trainer, loader
