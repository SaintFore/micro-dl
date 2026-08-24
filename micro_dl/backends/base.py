"""Narrow capabilities shared by array Runtime implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Backend(ABC):
    """Internal capability seam used by Tensor and Operators."""

    name: str

    @abstractmethod
    def array(self, value: Any, *, dtype: Any = None, copy: bool = True) -> Any:
        """Create Runtime Storage from a public input value."""

    @abstractmethod
    def owns(self, value: Any) -> bool:
        """Return whether value is Storage owned by this Backend."""

    @abstractmethod
    def to_numpy(self, storage: Any) -> Any:
        """Return an independent NumPy observation of Storage."""

    @abstractmethod
    def ones_like(self, storage: Any) -> Any:
        """Create unit Storage matching an existing value."""

    @abstractmethod
    def zeros_like(self, storage: Any) -> Any: ...

    @abstractmethod
    def add(self, left: Any, right: Any) -> Any: ...

    @abstractmethod
    def subtract(self, left: Any, right: Any) -> Any: ...

    @abstractmethod
    def multiply(self, left: Any, right: Any) -> Any: ...

    @abstractmethod
    def divide(self, left: Any, right: Any) -> Any: ...

    @abstractmethod
    def power(self, value: Any, exponent: Any) -> Any: ...

    @abstractmethod
    def negative(self, value: Any) -> Any: ...

    @abstractmethod
    def matmul(self, left: Any, right: Any) -> Any: ...

    @abstractmethod
    def sum(self, value: Any, axis: Any = None, keepdims: bool = False) -> Any: ...

    @abstractmethod
    def max(self, value: Any, axis: Any = None, keepdims: bool = False) -> Any: ...

    @abstractmethod
    def argmax(self, value: Any, axis: Any = None) -> Any: ...

    @abstractmethod
    def reshape(self, value: Any, shape: tuple[int, ...]) -> Any: ...

    @abstractmethod
    def transpose(self, value: Any, axes: Any = None) -> Any: ...

    @abstractmethod
    def broadcast_to(self, value: Any, shape: tuple[int, ...]) -> Any: ...

    @abstractmethod
    def exp(self, value: Any) -> Any: ...

    @abstractmethod
    def log(self, value: Any) -> Any: ...

    @abstractmethod
    def maximum(self, left: Any, right: Any) -> Any: ...

    @abstractmethod
    def random_normal(self, shape: tuple[int, ...], scale: float) -> Any: ...

    @abstractmethod
    def one_hot(self, indices: Any, classes: int) -> Any: ...

    @abstractmethod
    def manual_seed(self, seed: int) -> None: ...

    @abstractmethod
    def rng_state(self) -> Any: ...

    @abstractmethod
    def load_rng_state(self, state: Any) -> None: ...

    @abstractmethod
    def synchronize(self) -> None: ...

    @abstractmethod
    def random_uniform(self, shape: tuple[int, ...]) -> Any: ...

    @abstractmethod
    def zeros(self, shape: tuple[int, ...], dtype: Any = None) -> Any: ...

    @abstractmethod
    def pad(self, value: Any, padding: tuple[tuple[int, int], ...]) -> Any: ...

    @abstractmethod
    def conv2d_forward(self, value: Any, weight: Any, bias: Any, stride: int, padding: int) -> Any: ...

    @abstractmethod
    def conv2d_backward(self, value: Any, weight: Any, grad: Any, stride: int, padding: int) -> tuple[Any, Any, Any]: ...

    @abstractmethod
    def greater(self, left: Any, right: Any) -> Any: ...

    @abstractmethod
    def ndim(self, value: Any) -> int: ...

    @abstractmethod
    def shape(self, value: Any) -> tuple[int, ...]: ...
