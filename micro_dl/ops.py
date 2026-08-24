"""Backend-neutral Tensor Operators and their local gradient mathematics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from micro_dl.autograd.function import GradFn
from micro_dl.autograd.grad_mode import is_grad_enabled


class OperatorGradFn(GradFn):
    def __init__(self, parents, rule: Callable[[Any], tuple[Any | None, ...]]):
        super().__init__(*parents)
        self._rule = rule

    def backward(self, grad_output: Any):
        return self._rule(grad_output)


def _tensor(value, reference):
    from micro_dl.tensor import Tensor

    if isinstance(value, Tensor):
        if value.backend != reference.backend:
            raise ValueError("mixed Backend Operator inputs are not allowed")
        return value
    return Tensor(value, backend=reference.backend)


def _result(storage, parents, rule):
    from micro_dl.tensor import Tensor

    backend = parents[0]._backend
    if not backend.owns(storage):
        storage = backend.array(storage, copy=True)
    requires_grad = is_grad_enabled() and any(parent.requires_grad for parent in parents)
    grad_fn = OperatorGradFn(parents, rule) if requires_grad else None
    return Tensor._from_storage(
        storage,
        backend=parents[0].backend,
        requires_grad=requires_grad,
        grad_fn=grad_fn,
    )


def _unbroadcast(backend, grad, shape):
    while backend.ndim(grad) > len(shape):
        grad = backend.sum(grad, axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and backend.shape(grad)[axis] != 1:
            grad = backend.sum(grad, axis=axis, keepdims=True)
    return grad


def add(left, right):
    right = _tensor(right, left)
    b = left._backend
    return _result(b.add(left._storage, right._storage), (left, right), lambda g: (_unbroadcast(b, g, left.shape), _unbroadcast(b, g, right.shape)))


def subtract(left, right):
    right = _tensor(right, left)
    b = left._backend
    return _result(b.subtract(left._storage, right._storage), (left, right), lambda g: (_unbroadcast(b, g, left.shape), _unbroadcast(b, b.negative(g), right.shape)))


def multiply(left, right):
    right = _tensor(right, left)
    b = left._backend
    return _result(b.multiply(left._storage, right._storage), (left, right), lambda g: (_unbroadcast(b, b.multiply(g, right._storage), left.shape), _unbroadcast(b, b.multiply(g, left._storage), right.shape)))


def divide(left, right):
    right = _tensor(right, left)
    b = left._backend
    def rule(g):
        left_grad = b.divide(g, right._storage)
        denominator = b.power(right._storage, 2)
        right_grad = b.negative(b.divide(b.multiply(g, left._storage), denominator))
        return _unbroadcast(b, left_grad, left.shape), _unbroadcast(b, right_grad, right.shape)
    return _result(b.divide(left._storage, right._storage), (left, right), rule)


def power(value, exponent):
    b = value._backend
    storage = b.power(value._storage, exponent)
    return _result(storage, (value,), lambda g: (b.multiply(g, b.multiply(exponent, b.power(value._storage, exponent - 1))),))


def negative(value):
    b = value._backend
    return _result(b.negative(value._storage), (value,), lambda g: (b.negative(g),))


def matmul(left, right):
    right = _tensor(right, left)
    b = left._backend
    return _result(b.matmul(left._storage, right._storage), (left, right), lambda g: (b.matmul(g, b.transpose(right._storage)), b.matmul(b.transpose(left._storage), g)))


def reduce_sum(value, axis=None, keepdims=False):
    b = value._backend
    def rule(g):
        expanded = g
        if axis is not None and not keepdims:
            axes = (axis,) if isinstance(axis, int) else axis
            shape = list(value.shape)
            for item in axes:
                shape[item] = 1
            expanded = b.reshape(g, tuple(shape))
        return (b.broadcast_to(expanded, value.shape),)
    return _result(b.sum(value._storage, axis=axis, keepdims=keepdims), (value,), rule)


def reshape(value, shape):
    b = value._backend
    original = value.shape
    return _result(b.reshape(value._storage, shape), (value,), lambda g: (b.reshape(g, original),))


def transpose(value, axes=None):
    b = value._backend
    inverse = None if axes is None else tuple(sorted(range(len(axes)), key=axes.__getitem__))
    return _result(b.transpose(value._storage, axes), (value,), lambda g: (b.transpose(g, inverse),))


def exp(value):
    b = value._backend
    storage = b.exp(value._storage)
    return _result(storage, (value,), lambda g: (b.multiply(g, storage),))


def log(value):
    b = value._backend
    return _result(b.log(value._storage), (value,), lambda g: (b.divide(g, value._storage),))


def relu(value):
    b = value._backend
    storage = b.maximum(value._storage, 0)
    mask = b.greater(value._storage, 0)
    return _result(storage, (value,), lambda g: (b.multiply(g, mask),))


def cross_entropy(logits, targets, *, reduction="mean", label_smoothing=0.0):
    b = logits._backend
    if hasattr(targets, "backend"):
        if targets.backend != logits.backend:
            raise ValueError("mixed Backend loss inputs are not allowed")
        target_storage = targets._storage
    else:
        target_storage = b.array(targets, copy=True)
    one_hot = b.one_hot(target_storage, logits.shape[1])
    if label_smoothing:
        one_hot = b.add(b.multiply(one_hot, 1.0 - label_smoothing), label_smoothing / logits.shape[1])
    maximum = b.max(logits._storage, axis=1, keepdims=True)
    shifted = b.subtract(logits._storage, maximum)
    exponentials = b.exp(shifted)
    log_normalizer = b.log(b.sum(exponentials, axis=1, keepdims=True))
    log_probabilities = b.subtract(shifted, log_normalizer)
    probabilities = b.exp(log_probabilities)
    losses = b.negative(b.sum(b.multiply(one_hot, log_probabilities), axis=1))
    if reduction == "mean":
        result = b.divide(b.sum(losses), logits.shape[0])
    elif reduction == "sum":
        result = b.sum(losses)
    else:
        result = losses

    def rule(grad_output):
        gradient = b.subtract(probabilities, one_hot)
        if reduction == "mean":
            gradient = b.divide(gradient, logits.shape[0])
        if reduction == "none":
            grad_output = b.reshape(grad_output, (grad_output.shape[0], 1))
        return (b.multiply(gradient, grad_output),)

    return _result(result, (logits,), rule)


def dropout(value, probability):
    b = value._backend
    mask = b.greater(b.random_uniform(value.shape), probability)
    scale = 1.0 / (1.0 - probability)
    return _result(b.multiply(value._storage, b.multiply(mask, scale)), (value,), lambda g: (b.multiply(g, b.multiply(mask, scale)),))


def conv2d(value, weight, bias=None, *, stride=1, padding=0):
    if weight.backend != value.backend or (bias is not None and bias.backend != value.backend):
        raise ValueError("mixed Backend Operator inputs are not allowed")
    b = value._backend
    output = b.conv2d_forward(value._storage, weight._storage, None if bias is None else bias._storage, stride, padding)

    parents = (value, weight) if bias is None else (value, weight, bias)
    def rule(grad):
        grad_value, grad_weight, grad_bias = b.conv2d_backward(value._storage, weight._storage, grad, stride, padding)
        return (grad_value, grad_weight) if bias is None else (grad_value, grad_weight, grad_bias)
    return _result(output, parents, rule)
