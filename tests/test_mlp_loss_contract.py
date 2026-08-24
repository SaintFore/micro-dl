import numpy as np
import pytest

from micro_dl import CrossEntropyLoss, Linear, ReLU, Sequential, Tensor, manual_seed


def _backends():
    names = ["numpy"]
    try:
        Tensor([0.0], backend="cupy")
    except RuntimeError:
        pass
    else:
        names.append("cupy")
    return names


@pytest.mark.parametrize("backend", _backends())
def test_cross_entropy_matches_independent_reference_and_gradient(backend):
    logits = Tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]], requires_grad=True, backend=backend)
    loss = CrossEntropyLoss(label_smoothing=0.1)(logits, [0, 2])

    shifted = np.array([[0.0, -1.0, -2.0], [-2.0, -1.0, 0.0]])
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    targets = np.eye(3)[[0, 2]] * 0.9 + 0.1 / 3
    expected = -(targets * np.log(probabilities)).sum(axis=1).mean()
    np.testing.assert_allclose(loss.to_numpy(), expected, rtol=1e-6)

    loss.backward()
    np.testing.assert_allclose(logits.grad_to_numpy(), (probabilities - targets) / 2, rtol=1e-6)


def test_extreme_logits_have_finite_cross_entropy():
    loss = CrossEntropyLoss()(Tensor([[1000.0, -1000.0]]), [1])
    assert np.isfinite(loss.item())
    assert loss.item() == pytest.approx(2000.0)


@pytest.mark.parametrize("backend", _backends())
def test_tiny_mlp_forward_backward_is_deterministic(backend):
    manual_seed(42, backend=backend)
    first = Sequential(Linear(2, 4, backend=backend), ReLU(), Linear(4, 3, backend=backend))
    manual_seed(42, backend=backend)
    second = Sequential(Linear(2, 4, backend=backend), ReLU(), Linear(4, 3, backend=backend))
    inputs = Tensor([[1.0, -1.0], [0.5, 2.0]], backend=backend)

    first_loss = CrossEntropyLoss()(first(inputs), [0, 2])
    second_loss = CrossEntropyLoss()(second(inputs), [0, 2])
    np.testing.assert_allclose(first_loss.to_numpy(), second_loss.to_numpy())
    first_loss.backward()

    assert np.isfinite(first_loss.item())
    for parameter in first.parameters():
        assert parameter.grad is not None
        assert np.isfinite(parameter.grad_to_numpy()).all()


def test_linear_weight_gradient_matches_finite_difference():
    layer = Linear(2, 2, bias=False)
    layer.weight.copy_([[0.2, -0.3], [0.4, 0.1]])
    inputs = Tensor([[0.5, -1.5]])
    output = (layer(inputs) ** 2).sum()
    output.backward()
    analytic = layer.weight.grad_to_numpy().copy()

    original = layer.weight.to_numpy()
    numeric = np.zeros_like(original)
    epsilon = 1e-3
    for index in np.ndindex(original.shape):
        plus = original.copy(); plus[index] += epsilon
        minus = original.copy(); minus[index] -= epsilon
        layer.weight.copy_(plus); high = ((layer(inputs) ** 2).sum()).item()
        layer.weight.copy_(minus); low = ((layer(inputs) ** 2).sum()).item()
        numeric[index] = (high - low) / (2 * epsilon)
    layer.weight.copy_(original)

    np.testing.assert_allclose(analytic, numeric, rtol=2e-3, atol=2e-3)
