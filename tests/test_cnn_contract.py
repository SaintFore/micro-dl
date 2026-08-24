import numpy as np
import pytest

from micro_dl import Tensor, manual_seed
from micro_dl.nn import BatchNorm2d, Conv2d, Dropout, ResidualBlock


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_conv2d_forward_and_backward_known_example(backend):
    if backend == "cupy":
        try: Tensor([0.0], backend="cupy")
        except RuntimeError: pytest.skip("CuPy Backend unavailable")
    layer = Conv2d(1, 1, 2, bias=False, backend=backend)
    layer.weight.copy_([[[[1.0, 0.0], [0.0, -1.0]]]])
    value = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]], requires_grad=True, backend=backend)
    output = layer(value)
    np.testing.assert_allclose(output.to_numpy(), [[[[-3.0]]]])
    output.sum().backward()
    np.testing.assert_allclose(value.grad_to_numpy(), [[[[1.0, 0.0], [0.0, -1.0]]]])
    np.testing.assert_allclose(layer.weight.grad_to_numpy(), [[[[1.0, 2.0], [3.0, 4.0]]]])


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_batchnorm_dropout_and_residual_modes_and_shapes(backend):
    if backend == "cupy":
        try: Tensor([0.0], backend="cupy")
        except RuntimeError: pytest.skip("CuPy Backend unavailable")
    manual_seed(2, backend=backend)
    value = Tensor(np.arange(2 * 2 * 3 * 3, dtype=np.float32).reshape(2, 2, 3, 3), requires_grad=True, backend=backend)
    norm = BatchNorm2d(2, backend=backend)
    normalized = norm(value)
    assert normalized.shape == value.shape
    assert not np.allclose(norm.running_mean.to_numpy(), 0)
    norm.eval()
    before = norm.running_mean.to_numpy()
    assert norm(value).shape == value.shape
    np.testing.assert_allclose(norm.running_mean.to_numpy(), before)

    dropout = Dropout(0.5)
    assert dropout(value).shape == value.shape
    dropout.eval()
    assert dropout(value) is value

    block = ResidualBlock(2, backend=backend)
    result = block(value)
    assert result.shape == value.shape
    result.sum().backward()
    assert value.grad is not None


def test_conv_stride_padding_bias_shapes_and_gradients():
    layer = Conv2d(2, 3, 3, stride=2, padding=1)
    value = Tensor(np.ones((2, 2, 5, 5), dtype=np.float32), requires_grad=True)
    output = layer(value)
    assert output.shape == (2, 3, 3, 3)
    output.sum().backward()
    assert layer.bias.grad_to_numpy().shape == (3,)
    assert layer.weight.grad_to_numpy().shape == layer.weight.shape


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_residual_block_downsamples_and_changes_channels(backend):
    if backend == "cupy":
        try: Tensor([0.0], backend="cupy")
        except RuntimeError: pytest.skip("CuPy Backend unavailable")
    value = Tensor(np.ones((2, 2, 8, 8), dtype=np.float32), requires_grad=True, backend=backend)

    result = ResidualBlock(2, 4, stride=2, backend=backend)(value)

    assert result.shape == (2, 4, 4, 4)
    result.sum().backward()
    assert value.grad_to_numpy().shape == value.shape
