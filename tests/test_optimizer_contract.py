import numpy as np
import pytest

from micro_dl import Adam, AdamW, CosineAnnealingLR, Parameter, SGD, StepLR, Tensor, clip_grad_norm_


def _set_gradient(parameter, values):
    (parameter * Tensor(values)).sum().backward()


def test_sgd_update_zero_grad_momentum_and_weight_decay():
    parameter = Parameter([1.0, -2.0])
    optimizer = SGD([parameter, parameter], lr=0.1, momentum=0.5, weight_decay=0.1)
    _set_gradient(parameter, [2.0, 3.0])

    optimizer.step()
    np.testing.assert_allclose(parameter.to_numpy(), [0.79, -2.28], rtol=1e-6)
    assert len(optimizer.parameters) == 1
    optimizer.zero_grad()
    assert parameter.grad is None


@pytest.mark.parametrize("backend", ["numpy", "cupy"])
def test_adam_first_update_matches_reference(backend):
    if backend == "cupy":
        try: Tensor([0.0], backend="cupy")
        except RuntimeError: pytest.skip("CuPy Backend unavailable")
    parameter = Parameter([1.0, 2.0], backend=backend)
    optimizer = Adam([parameter], lr=0.01)
    (parameter * Tensor([2.0, -4.0], backend=backend)).sum().backward()
    optimizer.step()
    np.testing.assert_allclose(parameter.to_numpy(), [0.99, 2.01], atol=1e-6)


def test_adamw_decay_is_decoupled():
    parameter = Parameter([2.0])
    optimizer = AdamW([parameter], lr=0.1, weight_decay=0.2)
    _set_gradient(parameter, [0.0])
    optimizer.step()
    np.testing.assert_allclose(parameter.to_numpy(), [1.96], atol=1e-6)


def test_gradient_clipping_and_scheduler_timing():
    parameter = Parameter([0.0, 0.0])
    _set_gradient(parameter, [3.0, 4.0])
    assert clip_grad_norm_([parameter], 2.5) == 5.0
    np.testing.assert_allclose(parameter.grad_to_numpy(), [1.5, 2.0])

    optimizer = SGD([parameter], lr=0.1)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    scheduler.step(); assert optimizer.lr == 0.1
    scheduler.step(); assert optimizer.lr == 0.05

    cosine = CosineAnnealingLR(optimizer, total_epochs=2)
    cosine.step()
    assert optimizer.lr == 0.025


def test_optimizer_state_round_trip_continues_sequence():
    first = Parameter([1.0])
    optimizer = Adam([first], lr=0.01)
    _set_gradient(first, [2.0]); optimizer.step(); optimizer.zero_grad()
    saved_parameter = first.to_numpy()
    saved_optimizer = optimizer.state_dict()

    second = Parameter(saved_parameter)
    resumed = Adam([second], lr=1.0)
    resumed.load_state_dict(saved_optimizer)
    _set_gradient(first, [3.0]); optimizer.step()
    _set_gradient(second, [3.0]); resumed.step()
    np.testing.assert_allclose(first.to_numpy(), second.to_numpy())


def test_checkpoint_restores_update_hyperparameters():
    parameter = Parameter([1.0])
    original = AdamW([parameter], lr=0.02, betas=(0.8, 0.9), eps=1e-6, weight_decay=0.3)
    state = original.state_dict()
    restored = AdamW([parameter], lr=1.0, weight_decay=0.0)
    restored.load_state_dict(state)
    assert restored.lr == 0.02
    assert (restored.beta1, restored.beta2) == (0.8, 0.9)
    assert restored.eps == 1e-6
    assert restored.decoupled_weight_decay == 0.3
