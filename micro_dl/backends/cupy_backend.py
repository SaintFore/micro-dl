"""Optional CuPy CUDA Backend."""

from __future__ import annotations

from typing import Any

from micro_dl.backends.base import Backend


class CupyBackend(Backend):
    name = "cupy"

    def __init__(self) -> None:
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError
        except (ImportError, Exception) as exc:
            raise RuntimeError(
                "CuPy Backend is unavailable; install the CUDA extra and provide a CUDA device"
            ) from exc
        self._cp = cp
        self._rng = cp.random.default_rng(0)

    def array(self, value: Any, *, dtype: Any = None, copy: bool = True):
        if dtype is None and not hasattr(value, "dtype"):
            dtype = self._cp.float32
        storage = self._cp.asarray(value, dtype=dtype)
        return storage.copy() if copy else storage

    def owns(self, value: Any) -> bool:
        return isinstance(value, self._cp.ndarray)

    def to_numpy(self, storage: Any):
        return self._cp.asnumpy(storage).copy()

    def ones_like(self, storage: Any):
        return self._cp.ones_like(storage)

    def zeros_like(self, storage): return self._cp.zeros_like(storage)
    def add(self, left, right): return self._cp.add(left, right)
    def subtract(self, left, right): return self._cp.subtract(left, right)
    def multiply(self, left, right): return self._cp.multiply(left, right)
    def divide(self, left, right): return self._cp.divide(left, right)
    def power(self, value, exponent): return self._cp.power(value, exponent)
    def negative(self, value): return self._cp.negative(value)
    def matmul(self, left, right): return self._cp.matmul(left, right)
    def sum(self, value, axis=None, keepdims=False): return self._cp.sum(value, axis=axis, keepdims=keepdims)
    def max(self, value, axis=None, keepdims=False): return self._cp.max(value, axis=axis, keepdims=keepdims)
    def argmax(self, value, axis=None): return self._cp.argmax(value, axis=axis)
    def reshape(self, value, shape): return self._cp.reshape(value, shape)
    def transpose(self, value, axes=None): return self._cp.transpose(value, axes)
    def broadcast_to(self, value, shape): return self._cp.broadcast_to(value, shape)
    def exp(self, value): return self._cp.exp(value)
    def log(self, value): return self._cp.log(value)
    def maximum(self, left, right): return self._cp.maximum(left, right)
    def random_normal(self, shape, scale):
        return self._rng.standard_normal(shape, dtype=self._cp.float32) * scale
    def one_hot(self, indices, classes):
        return self._cp.eye(classes, dtype=self._cp.float32)[indices.astype(self._cp.int64)]
    def manual_seed(self, seed):
        self._rng = self._cp.random.default_rng(seed)
    def rng_state(self):
        bit_generator = self._rng.bit_generator
        return {"seed": int(bit_generator._seed), "state": self._cp.asnumpy(bit_generator._state)}
    def load_rng_state(self, state):
        self._rng = self._cp.random.default_rng(state["seed"])
        target = self._rng.bit_generator._state
        source = self._cp.asarray(state["state"])
        if target.shape != source.shape:
            raise ValueError("CuPy RNG state shape does not match this runtime")
        self._cp.copyto(target, source)
    def synchronize(self): self._cp.cuda.get_current_stream().synchronize()
    def random_uniform(self, shape): return self._rng.random(shape, dtype=self._cp.float32)
    def zeros(self, shape, dtype=None): return self._cp.zeros(shape, dtype=dtype or self._cp.float32)
    def pad(self, value, padding): return self._cp.pad(value, padding)
    def conv2d_forward(self, value, weight, bias, stride, padding):
        from micro_dl.backends.convolution import forward
        return forward(self._cp, value, weight, bias, stride, padding)
    def conv2d_backward(self, value, weight, grad, stride, padding):
        from micro_dl.backends.convolution import backward
        return backward(self._cp, value, weight, grad, stride, padding)
    def greater(self, left, right): return self._cp.greater(left, right)
    def ndim(self, value): return value.ndim
    def shape(self, value): return value.shape
