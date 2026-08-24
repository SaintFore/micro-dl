"""IDX-format MNIST parsing with explicit local caching boundaries."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np


class MNIST:
    def __init__(self, images, labels, *, transform=None):
        if len(images) != len(labels):
            raise ValueError("images and labels must have equal length")
        self.images = np.asarray(images)
        self.labels = np.asarray(labels)
        self.transform = transform

    @classmethod
    def from_idx(cls, images_path, labels_path, *, transform=None):
        images_raw = _read(images_path)
        labels_raw = _read(labels_path)
        magic, count, rows, columns = struct.unpack(">IIII", images_raw[:16])
        label_magic, label_count = struct.unpack(">II", labels_raw[:8])
        if magic != 2051 or label_magic != 2049 or count != label_count:
            raise ValueError("invalid MNIST IDX files")
        images = np.frombuffer(images_raw, dtype=np.uint8, offset=16).reshape(count, 1, rows, columns)
        labels = np.frombuffer(labels_raw, dtype=np.uint8, offset=8)
        return cls(images, labels, transform=transform)

    def __len__(self): return len(self.labels)
    @property
    def input_shape(self): return tuple(self[0][0].shape)
    def __getitem__(self, index):
        image = self.images[index].astype(np.float32) / 255.0
        if self.transform is not None: image = self.transform(image)
        return image, int(self.labels[index])

    @staticmethod
    def cache_paths(cache_dir, *, train: bool = True):
        """Return explicit local cache targets without performing network access."""
        root = Path(cache_dir)
        prefix = "train" if train else "t10k"
        return root / f"{prefix}-images-idx3-ubyte.gz", root / f"{prefix}-labels-idx1-ubyte.gz"

    @classmethod
    def from_cache(cls, cache_dir, *, train=True, transform=None):
        images, labels = cls.cache_paths(cache_dir, train=train)
        if not images.exists() or not labels.exists():
            raise FileNotFoundError("MNIST cache is incomplete")
        return cls.from_idx(images, labels, transform=transform)


def _read(path):
    path = Path(path)
    return gzip.open(path, "rb").read() if path.suffix == ".gz" else path.read_bytes()
