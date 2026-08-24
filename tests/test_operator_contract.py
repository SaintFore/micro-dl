import numpy as np
import pytest

from micro_dl import Tensor
from micro_dl.autograd import no_grad


def _backends():
    names = ["numpy"]
    try: Tensor([0.0], backend="cupy")
    except RuntimeError: pass
    else: names.append("cupy")
    return names


@pytest.mark.parametrize("backend", _backends())
def test_broadcast_arithmetic_and_gradient(backend):
    left = Tensor([[1.0], [2.0]], requires_grad=True, backend=backend)
    right = Tensor([[3.0, 4.0]], requires_grad=True, backend=backend)
    output = ((left * right) + left).sum()

    output.backward()

    np.testing.assert_allclose(output.to_numpy(), 27.0)
    np.testing.assert_allclose(left.grad_to_numpy(), [[9.0], [9.0]])
    np.testing.assert_allclose(right.grad_to_numpy(), [[3.0, 3.0]])


def test_matrix_shape_reduction_and_activation_chain():
    x = Tensor([[1.0, -2.0]], requires_grad=True)
    weight = Tensor([[2.0], [3.0]], requires_grad=True)
    output = (x @ weight).relu().reshape(1).mean()

    output.backward()

    np.testing.assert_allclose(output.to_numpy(), [0.0])
    np.testing.assert_allclose(x._backend.to_numpy(x.grad), [[0.0, 0.0]])


def test_no_grad_is_scoped_and_does_not_build_graph():
    x = Tensor([2.0], requires_grad=True)
    with no_grad():
        result = x * 3.0
    assert result.requires_grad is False
    assert result._grad_fn is None
    assert (x * 3.0).requires_grad is True


def test_mixed_backend_operator_is_rejected():
    try:
        gpu = Tensor([1.0], backend="cupy")
    except RuntimeError:
        pytest.skip("CuPy Backend unavailable")
    with pytest.raises(ValueError, match="mixed Backend"):
        Tensor([1.0], backend="numpy") + gpu
