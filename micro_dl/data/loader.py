"""Dataset protocol and re-iterable mini-batch loading."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from micro_dl.tensor import Tensor


class Dataset(Protocol):
    def __len__(self) -> int: ...
    def __getitem__(self, index: int): ...


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 32, *, shuffle: bool = False, seed: int = 0, backend: str = "numpy"):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.backend = backend
        self._epoch = 0

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.default_rng(self.seed + self._epoch).shuffle(indices)
        self._epoch += 1
        for start in range(0, len(indices), self.batch_size):
            samples = [self.dataset[int(index)] for index in indices[start : start + self.batch_size]]
            inputs, targets = zip(*samples)
            yield Tensor(np.stack(inputs), backend=self.backend), np.asarray(targets, dtype=np.int64)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def state_dict(self):
        transform = getattr(self.dataset, "transform", None)
        return {
            "epoch": self._epoch,
            "transform": transform.state_dict() if hasattr(transform, "state_dict") else None,
        }

    def load_state_dict(self, state):
        self._epoch = int(state["epoch"])
        transform = getattr(self.dataset, "transform", None)
        if state.get("transform") is not None:
            if not hasattr(transform, "load_state_dict"):
                raise ValueError("dataset transform cannot restore state")
            transform.load_state_dict(state["transform"])
