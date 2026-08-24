"""CIFAR-100 Python-format parsing."""

import pickle
from pathlib import Path

import numpy as np


class CIFAR100:
    def __init__(self, images, labels, *, transform=None):
        if len(images) != len(labels):
            raise ValueError("images and labels must have equal length")
        self.images = np.asarray(images)
        self.labels = np.asarray(labels)
        self.transform = transform

    @classmethod
    def from_file(cls, path, *, transform=None):
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle, encoding="bytes")
        data = payload.get(b"data", payload.get("data"))
        labels = payload.get(b"fine_labels", payload.get("fine_labels"))
        if data is None or labels is None:
            raise ValueError("invalid CIFAR-100 file")
        images = np.asarray(data, dtype=np.uint8).reshape(-1, 3, 32, 32)
        return cls(images, labels, transform=transform)

    @property
    def input_shape(self): return tuple(self.images.shape[1:])
    @staticmethod
    def cache_paths(cache_dir):
        root = Path(cache_dir)
        return root / "train", root / "test", root / "meta"
    @classmethod
    def from_cache(cls, cache_dir, *, train=True, transform=None):
        train_path, test_path, _ = cls.cache_paths(cache_dir)
        selected = train_path if train else test_path
        if not selected.exists():
            raise FileNotFoundError("CIFAR-100 cache is incomplete")
        return cls.from_file(selected, transform=transform)
    def __len__(self): return len(self.labels)
    def __getitem__(self, index):
        image = self.images[index].astype(np.float32) / 255.0
        if self.transform is not None: image = self.transform(image)
        return image, int(self.labels[index])
