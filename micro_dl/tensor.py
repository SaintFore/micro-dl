"""Tensor Core: multidimensional values, Storage ownership, and graph identity."""

from __future__ import annotations

from typing import Any

from micro_dl.autograd.function import GradFn
from micro_dl.backends.context import get_default_backend, resolve_backend

class Tensor:
    """A multidimensional value whose private Storage belongs to one Backend."""

    def __init__(
        self,
        data: Any,
        *,
        requires_grad: bool = False,
        backend: str | None = None,
        dtype: Any = None,
    ) -> None:
        if isinstance(data, Tensor):
            backend_name = backend or data.backend
            source = data.to_numpy() if backend_name != data.backend else data._storage
        else:
            backend_name = backend or get_default_backend()
            source = data

        implementation = resolve_backend(backend_name)
        self._backend = implementation
        self._storage = implementation.array(source, dtype=dtype, copy=True)
        self.requires_grad = requires_grad
        self.grad: Any | None = None
        self._grad_fn: GradFn | None = None
        self._graph_consumed = False

    @classmethod
    def _from_storage(
        cls,
        storage: Any,
        *,
        backend: str,
        requires_grad: bool = False,
        grad_fn: GradFn | None = None,
    ) -> Tensor:
        implementation = resolve_backend(backend)
        if not implementation.owns(storage):
            raise TypeError(f"Storage does not belong to {backend!r} Backend")

        obj = cls.__new__(cls)
        obj._backend = implementation
        obj._storage = storage
        obj.requires_grad = requires_grad
        obj.grad = None
        obj._grad_fn = grad_fn
        obj._graph_consumed = False
        return obj

    @property
    def backend(self) -> str:
        return self._backend.name

    @property
    def shape(self) -> tuple[int, ...]:
        return self._storage.shape

    @property
    def ndim(self) -> int:
        return self._storage.ndim

    @property
    def size(self) -> int:
        return self._storage.size

    @property
    def dtype(self):
        return self._storage.dtype

    def item(self):
        if self.size != 1:
            raise ValueError("item() 只支持单元素 Tensor")
        return self._storage.item()

    def argmax(self, axis: int | None = None) -> Tensor:
        return Tensor._from_storage(
            self._backend.argmax(self._storage, axis=axis),
            backend=self.backend,
        )

    def to_numpy(self):
        return self._backend.to_numpy(self._storage)

    def to_backend(self, backend: str) -> Tensor:
        """Copy this Tensor to a Backend while preserving autograd connectivity."""
        if backend == self.backend:
            return self
        target = resolve_backend(backend)
        storage = target.array(self.to_numpy(), copy=True)
        tracks_grad = self.requires_grad
        grad_fn = _BackendTransfer(self, target) if tracks_grad else None
        return Tensor._from_storage(
            storage,
            backend=backend,
            requires_grad=tracks_grad,
            grad_fn=grad_fn,
        )

    def backward(self, grad: Any | None = None) -> None:
        from micro_dl.autograd.engine import backward

        backward(self, grad)

    def zero_grad(self) -> None:
        self.grad = None

    def grad_to_numpy(self):
        """Return an independent CPU observation of this Tensor's gradient."""
        if self.grad is None:
            return None
        return self._backend.to_numpy(self.grad)

    def copy_(self, value: Any) -> Tensor:
        """Replace this Tensor's value without changing its identity or Backend."""
        if isinstance(value, Tensor):
            if value.backend != self.backend:
                raise ValueError("copy_ requires the same Backend")
            value = value._storage
        replacement = self._backend.array(value, dtype=self.dtype, copy=True)
        if replacement.shape != self.shape:
            raise ValueError(
                f"copy shape {replacement.shape} does not match Tensor shape {self.shape}"
            )
        self._storage = replacement
        return self

    def grad_tensor(self) -> Tensor | None:
        """Return the current gradient as a detached Tensor on the same Backend."""
        if self.grad is None:
            return None
        return Tensor._from_storage(
            self._backend.array(self.grad, copy=True),
            backend=self.backend,
        )

    def scale_grad_(self, factor: float) -> None:
        """Scale an accumulated gradient without exposing private Storage."""
        if self.grad is not None:
            self.grad = self._backend.multiply(self.grad, factor)

    def grad_squared_sum(self) -> float:
        if self.grad is None:
            return 0.0
        squared = self._backend.multiply(self.grad, self.grad)
        return float(self._backend.sum(squared).item())

    @classmethod
    def random_normal(cls, shape, scale, *, backend=None, requires_grad=False):
        backend_name = backend or get_default_backend()
        implementation = resolve_backend(backend_name)
        return cls._from_storage(
            implementation.random_normal(tuple(shape), scale),
            backend=backend_name,
            requires_grad=requires_grad,
        )

    def __add__(self, other):
        from micro_dl.ops import add
        return add(self, other)

    __radd__ = __add__

    def __sub__(self, other):
        from micro_dl.ops import subtract
        return subtract(self, other)

    def __rsub__(self, other):
        return Tensor(other, backend=self.backend) - self

    def __mul__(self, other):
        from micro_dl.ops import multiply
        return multiply(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        from micro_dl.ops import divide
        return divide(self, other)

    def __pow__(self, exponent):
        from micro_dl.ops import power
        return power(self, exponent)

    def __neg__(self):
        from micro_dl.ops import negative
        return negative(self)

    def __matmul__(self, other):
        from micro_dl.ops import matmul
        return matmul(self, other)

    def sum(self, axis=None, keepdims=False):
        from micro_dl.ops import reduce_sum
        return reduce_sum(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            count = self.size
        elif isinstance(axis, int):
            count = self.shape[axis]
        else:
            count = 1
            for item in axis:
                count *= self.shape[item]
        return self.sum(axis=axis, keepdims=keepdims) / count

    def reshape(self, *shape):
        from micro_dl.ops import reshape
        normalized = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        return reshape(self, tuple(normalized))

    def transpose(self, *axes):
        from micro_dl.ops import transpose
        normalized = axes or None
        return transpose(self, normalized)

    @property
    def T(self):
        return self.transpose()

    def exp(self):
        from micro_dl.ops import exp
        return exp(self)

    def log(self):
        from micro_dl.ops import log
        return log(self)

    def relu(self):
        from micro_dl.ops import relu
        return relu(self)


class _BackendTransfer(GradFn):
    def __init__(self, parent: Tensor, target_backend: Any) -> None:
        super().__init__(parent)
        self.target_backend = target_backend

    def backward(self, grad_output: Any):
        parent = self.parents[0]
        observed = self.target_backend.to_numpy(grad_output)
        return (parent._backend.array(observed, copy=True),)
