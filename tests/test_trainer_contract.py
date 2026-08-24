import numpy as np
import pytest

from micro_dl import CrossEntropyLoss, Linear, Module, SGD, Sequential, Tensor, Trainer


def _data():
    return [
        (Tensor([[1.0, 0.0], [0.0, 1.0]]), [0, 1]),
        (Tensor([[1.0, 1.0], [-1.0, -1.0]]), [0, 1]),
    ]


def test_trainer_updates_parameters_returns_results_and_restores_mode():
    model = Sequential(Linear(2, 2))
    before = [parameter.to_numpy() for parameter in model.parameters()]
    model.eval()
    trainer = Trainer(model, CrossEntropyLoss(), SGD(model.parameters(), lr=0.1))

    result = trainer.fit(_data(), epochs=3, validation_data=_data())

    assert model.training is False
    assert len(result.epochs) == 3
    assert result.best_epoch in {0, 1, 2}
    assert all(np.isfinite(epoch.train_loss) for epoch in result.epochs)
    assert all(np.isfinite(epoch.validation_loss) for epoch in result.epochs)
    assert all(0.0 <= epoch.validation_accuracy <= 1.0 for epoch in result.epochs)
    assert any(
        not np.allclose(old, parameter.to_numpy())
        for old, parameter in zip(before, model.parameters())
    )


def test_evaluate_restores_training_mode_and_does_not_create_gradients():
    model = Sequential(Linear(2, 2)).train()
    trainer = Trainer(model, CrossEntropyLoss(), SGD(model.parameters(), lr=0.1))

    result = trainer.evaluate(_data())

    assert np.isfinite(result.loss)
    assert 0.0 <= result.accuracy <= 1.0
    assert model.training is True
    assert all(parameter.grad is None for parameter in model.parameters())


def test_evaluate_reports_top1_accuracy_from_logits_and_integer_targets():
    class Identity(Module):
        def forward(self, inputs):
            return inputs

    model = Identity()
    trainer = Trainer(model, CrossEntropyLoss(), SGD(model.parameters(), lr=0.1))
    data = [
        (Tensor([[4.0, 1.0], [1.0, 4.0], [3.0, 2.0], [1.0, 5.0]]), [0, 1, 1, 1]),
    ]

    result = trainer.evaluate(data)

    assert result.accuracy == 0.75


def test_scheduler_and_clipping_run_once_at_their_lifecycle_points(monkeypatch):
    model = Sequential(Linear(2, 2))
    optimizer = SGD(model.parameters(), lr=0.1)
    calls = []

    class Scheduler:
        def step(self): calls.append("schedule")

    def clip(parameters, limit):
        assert list(parameters)
        calls.append(("clip", limit))

    monkeypatch.setattr("micro_dl.optim.clip_grad_norm_", clip)
    Trainer(
        model,
        CrossEntropyLoss(),
        optimizer,
        scheduler=Scheduler(),
        max_grad_norm=1.5,
    ).fit(_data(), epochs=2)

    assert calls.count(("clip", 1.5)) == 4
    assert calls.count("schedule") == 2


def test_multi_epoch_training_rejects_one_shot_iterator():
    model = Sequential(Linear(2, 2))
    trainer = Trainer(model, CrossEntropyLoss(), SGD(model.parameters(), lr=0.1))
    with pytest.raises(TypeError, match="re-iterable"):
        trainer.fit(iter(_data()), epochs=2)


def test_nonfinal_best_state_is_restored_exactly(monkeypatch):
    model = Sequential(Linear(2, 2))
    trainer = Trainer(model, CrossEntropyLoss(), SGD(model.parameters(), lr=0.1))
    snapshots = []
    original_state_dict = model.state_dict
    original_load = model.load_state_dict

    def recording_state():
        state = original_state_dict()
        snapshots.append({name: value.copy() for name, value in state.items()})
        return state

    restored = {}

    def recording_load(state):
        restored.update({name: value.copy() for name, value in state.items()})
        return original_load(state)

    from micro_dl import EvaluationResult

    scores = iter([EvaluationResult(0.5, 0.75), EvaluationResult(0.8, 0.5)])
    monkeypatch.setattr(model, "state_dict", recording_state)
    monkeypatch.setattr(model, "load_state_dict", recording_load)
    monkeypatch.setattr(trainer, "evaluate", lambda data: next(scores))

    result = trainer.fit(_data(), epochs=2, validation_data=_data())

    assert result.best_epoch == 0
    for name, value in snapshots[0].items():
        np.testing.assert_allclose(restored[name], value)


def test_accuracy_selection_marks_higher_accuracy_even_when_loss_is_worse(monkeypatch):
    from micro_dl import EvaluationResult

    model = Sequential(Linear(2, 2))
    trainer = Trainer(
        model,
        CrossEntropyLoss(),
        SGD(model.parameters(), lr=0.1),
        selection_metric="validation_accuracy",
    )
    evaluations = iter([EvaluationResult(0.4, 0.70), EvaluationResult(0.5, 0.80)])
    monkeypatch.setattr(trainer, "evaluate", lambda data: next(evaluations))
    selected = []

    result = trainer.fit(
        _data(),
        epochs=2,
        validation_data=_data(),
        on_epoch=lambda epoch, value, is_best: selected.append(is_best),
    )

    assert selected == [True, True]
    assert result.best_epoch == 1


def test_resume_restores_historical_best_when_new_epochs_do_not_improve(monkeypatch):
    from micro_dl import EvaluationResult

    model = Sequential(Linear(2, 2))
    trainer = Trainer(model, CrossEntropyLoss(), SGD(model.parameters(), lr=0.1), selection_metric="validation_accuracy")
    historical = {name: np.full_like(value, 7.0) for name, value in model.state_dict().items()}
    monkeypatch.setattr(trainer, "evaluate", lambda data: EvaluationResult(0.5, 0.70))

    trainer.fit(
        _data(), epochs=1, validation_data=_data(),
        initial_best_score=0.80, initial_best_state=historical,
    )

    for name, value in model.state_dict().items():
        np.testing.assert_array_equal(value, historical[name])
