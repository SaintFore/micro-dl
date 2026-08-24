"""Backend-neutral dynamic Autograd contracts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from micro_dl import Tensor
from micro_dl.autograd import GradFn


def _backends():
    names = ["numpy"]
    try: Tensor([0.0], backend="cupy")
    except RuntimeError: pass
    else: names.append("cupy")
    return names


class ScaleGradFn(GradFn):
    def __init__(self, parent: Tensor, factor: float) -> None:
        super().__init__(parent)
        self.factor = factor

    def backward(self, grad_output: Any):
        return (grad_output * self.factor,)


class AddGradFn(GradFn):
    def backward(self, grad_output: Any):
        return (grad_output, grad_output)


class BrokenGradFn(GradFn):
    def backward(self, grad_output: Any):
        raise RuntimeError("broken gradient")


def _result(parents, grad_fn):
    parent = parents[0]
    storage = parent._backend.array(np.zeros(parent.shape, dtype=parent.dtype), copy=True)
    return Tensor._from_storage(
        storage,
        backend=parent.backend,
        requires_grad=True,
        grad_fn=grad_fn,
    )


def _scaled(parent: Tensor, factor: float) -> Tensor:
    return _result((parent,), ScaleGradFn(parent, factor))


@pytest.mark.parametrize("backend", _backends())
def test_chain_and_explicit_gradient_propagate_to_leaf(backend):
    leaf = Tensor([0.0, 0.0], requires_grad=True, backend=backend)
    output = _scaled(_scaled(leaf, 2.0), 3.0)

    output.backward(np.array([3.0, 4.0], dtype=np.float32))

    np.testing.assert_allclose(leaf.grad_to_numpy(), [18.0, 24.0])


@pytest.mark.parametrize("backend", _backends())
def test_branch_accumulates_shared_leaf_gradient_once(backend):
    leaf = Tensor([0.0], requires_grad=True, backend=backend)
    left = _scaled(leaf, 2.0)
    right = _scaled(leaf, 3.0)
    output = _result((left, right), AddGradFn(left, right))

    output.backward()

    np.testing.assert_allclose(leaf.grad_to_numpy(), [5.0])


def test_leaf_gradients_accumulate_across_graphs_and_zero_explicitly():
    leaf = Tensor([0.0], requires_grad=True)
    _scaled(leaf, 2.0).backward()
    _scaled(leaf, 3.0).backward()
    np.testing.assert_allclose(leaf._backend.to_numpy(leaf.grad), [5.0])

    leaf.zero_grad()
    assert leaf.grad is None


def test_deep_graph_does_not_use_python_recursion():
    leaf = Tensor([0.0], requires_grad=True)
    output = leaf
    for _ in range(1500):
        output = _scaled(output, 1.0)

    output.backward()
    np.testing.assert_allclose(leaf._backend.to_numpy(leaf.grad), [1.0])


def test_graph_is_consumed_after_success_and_failure():
    leaf = Tensor([0.0], requires_grad=True)
    output = _scaled(leaf, 2.0)
    output.backward()
    with pytest.raises(RuntimeError, match="已经执行过"):
        output.backward()

    broken = _result((leaf,), BrokenGradFn(leaf))
    with pytest.raises(RuntimeError, match="broken gradient"):
        broken.backward()
    with pytest.raises(RuntimeError, match="已经执行过"):
        broken.backward()


def test_backend_transfer_is_differentiable_and_same_backend_is_identity():
    leaf = Tensor([0.0], requires_grad=True)
    assert leaf.to_backend("numpy") is leaf

    try:
        transferred = leaf.to_backend("cupy")
    except RuntimeError:
        pytest.skip("CuPy Backend unavailable")

    transferred.backward(np.array([2.0], dtype=np.float32))
    np.testing.assert_allclose(leaf._backend.to_numpy(leaf.grad), [2.0])
