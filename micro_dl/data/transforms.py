"""Composable host-side input transforms."""

import numpy as np
import copy


class Compose:
    def __init__(self, transforms): self.transforms = tuple(transforms)
    def __call__(self, value):
        for transform in self.transforms:
            value = transform(value)
        return value
    def state_dict(self):
        return [transform.state_dict() if hasattr(transform, "state_dict") else {} for transform in self.transforms]
    def load_state_dict(self, state):
        if len(state) != len(self.transforms): raise ValueError("transform state count does not match")
        for transform, item in zip(self.transforms, state):
            if hasattr(transform, "load_state_dict"): transform.load_state_dict(item)


class Normalize:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
    def __call__(self, value): return (np.asarray(value, dtype=np.float32) - self.mean) / self.std


class RandomHorizontalFlip:
    def __init__(self, probability=0.5, seed=0):
        self.probability = probability
        self._rng = np.random.default_rng(seed)
    def __call__(self, value):
        return np.flip(value, axis=-1).copy() if self._rng.random() < self.probability else value
    def state_dict(self): return copy.deepcopy(self._rng.bit_generator.state)
    def load_state_dict(self, state): self._rng.bit_generator.state = copy.deepcopy(state)


class RandomCrop:
    def __init__(self, size: int, *, padding: int = 0, seed: int = 0):
        self.size = int(size)
        self.padding = int(padding)
        self._rng = np.random.default_rng(seed)
        if self.size <= 0 or self.padding < 0:
            raise ValueError("crop size must be positive and padding non-negative")

    def __call__(self, value):
        image = np.asarray(value)
        if self.padding:
            image = np.pad(image, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)))
        height, width = image.shape[-2:]
        if self.size > height or self.size > width:
            raise ValueError("crop size exceeds padded image")
        row = int(self._rng.integers(0, height - self.size + 1))
        column = int(self._rng.integers(0, width - self.size + 1))
        return image[:, row:row + self.size, column:column + self.size].copy()
    def state_dict(self): return copy.deepcopy(self._rng.bit_generator.state)
    def load_state_dict(self, state): self._rng.bit_generator.state = copy.deepcopy(state)
