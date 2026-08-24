"""NumPy CPU Backend."""

from __future__ import annotations

from typing import Any

import numpy as np

from micro_dl.backends.base import Backend


class NumpyBackend(Backend):
    name = "numpy"

    def __init__(self) -> None:
        self._rng = np.random.default_rng(0)

    def array(self, value: Any, *, dtype: Any = None, copy: bool = True) -> np.ndarray:
        if type(value).__module__.split(".", 1)[0] == "cupy":
            value = value.get()
        if dtype is None and not hasattr(value, "dtype"):
            dtype = np.float32
        return np.array(value, dtype=dtype, copy=copy)

    def owns(self, value: Any) -> bool:
        return isinstance(value, np.ndarray)

    def to_numpy(self, storage: np.ndarray) -> np.ndarray:
        return storage.copy()

    def ones_like(self, storage: np.ndarray) -> np.ndarray:
        return np.ones_like(storage)

    def zeros_like(self, storage): return np.zeros_like(storage)
    def add(self, left, right): return np.add(left, right)
    def subtract(self, left, right): return np.subtract(left, right)
    def multiply(self, left, right): return np.multiply(left, right)
    def divide(self, left, right): return np.divide(left, right)
    def power(self, value, exponent): return np.power(value, exponent)
    def negative(self, value): return np.negative(value)
    def matmul(self, left, right): return np.matmul(left, right)
    def sum(self, value, axis=None, keepdims=False): return np.asarray(np.sum(value, axis=axis, keepdims=keepdims))
    def max(self, value, axis=None, keepdims=False): return np.asarray(np.max(value, axis=axis, keepdims=keepdims))
    def argmax(self, value, axis=None): return np.asarray(np.argmax(value, axis=axis))
    def reshape(self, value, shape): return np.reshape(value, shape)
    def transpose(self, value, axes=None): return np.transpose(value, axes)
    def broadcast_to(self, value, shape): return np.broadcast_to(value, shape)
    def exp(self, value): return np.exp(value)
    def log(self, value): return np.log(value)
    def maximum(self, left, right): return np.maximum(left, right)
    def random_normal(self, shape, scale):
        return self._rng.normal(0.0, scale, size=shape).astype(np.float32)
    def one_hot(self, indices, classes):
        return np.eye(classes, dtype=np.float32)[np.asarray(indices, dtype=np.int64)]
    def manual_seed(self, seed):
        self._rng = np.random.default_rng(seed)
    def rng_state(self): return self._rng.bit_generator.state
    def load_rng_state(self, state): self._rng.bit_generator.state = state
    def synchronize(self): pass
    def random_uniform(self, shape): return self._rng.random(shape).astype(np.float32)
    def zeros(self, shape, dtype=None): return np.zeros(shape, dtype=dtype or np.float32)
    def pad(self, value, padding): return np.pad(value, padding)
    def conv2d_forward(self, value, weight, bias, stride, padding):
        from micro_dl.backends.convolution import forward
        return forward(np, value, weight, bias, stride, padding)
    def conv2d_backward(self, value, weight, grad, stride, padding):
        from micro_dl.backends.convolution import backward
        return backward(np, value, weight, grad, stride, padding)
    def greater(self, left, right): return np.greater(left, right)
    def ndim(self, value): return value.ndim
    def shape(self, value): return value.shape
