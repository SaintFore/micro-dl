"""The single owner of training and validation lifecycle orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
import math
from time import perf_counter
from typing import Any, Callable, Iterable, Protocol

from micro_dl.autograd import no_grad


@dataclass(frozen=True)
class EpochResult:
    train_loss: float
    validation_loss: float | None = None
    validation_accuracy: float | None = None


@dataclass(frozen=True)
class EvaluationResult:
    loss: float
    accuracy: float


@dataclass(frozen=True)
class TrainResult:
    epochs: tuple[EpochResult, ...]
    best_epoch: int | None


@dataclass(frozen=True)
class SelectionPolicy:
    metric: str
    maximize: bool
    fallback_metric: str = "train_loss"
    fallback_maximize: bool = False

    @property
    def worst_score(self): return float("-inf") if self.maximize else float("inf")
    def score(self, result: EpochResult):
        value = getattr(result, self.metric)
        return (getattr(result, self.fallback_metric), self.fallback_maximize) if value is None else (value, self.maximize)
    def better(self, score, best, maximize): return score > best if maximize else score < best


_SELECTION_POLICIES = {
    "validation_loss": SelectionPolicy("validation_loss", False),
    "validation_accuracy": SelectionPolicy("validation_accuracy", True),
}


class BatchIterable(Protocol):
    def __iter__(self) -> Iterable[tuple[Any, Any]]: ...


class Trainer:
    def __init__(self, model, loss, optimizer, *, scheduler=None, max_grad_norm=None, selection_metric="validation_loss"):
        if selection_metric not in _SELECTION_POLICIES:
            raise ValueError("selection_metric must be 'validation_loss' or 'validation_accuracy'")
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.selection_policy = _SELECTION_POLICIES[selection_metric]

    def fit(
        self,
        train_data: BatchIterable,
        *,
        epochs: int,
        validation_data: BatchIterable | None = None,
        on_step: Callable[[int, float], None] | None = None,
        on_epoch: Callable[[int, EpochResult, bool], None] | None = None,
        initial_best_score: float | None = None,
        initial_best_state=None,
    ) -> TrainResult:
        if iter(train_data) is train_data and epochs > 1:
            raise TypeError("multi-epoch training requires re-iterable train_data")
        history = []
        has_initial_score = initial_best_score is not None
        best_score = self.worst_selection_score if initial_best_score is None else initial_best_score
        best_state = initial_best_state
        best_epoch = None
        original_mode = self.model.training
        try:
            for epoch in range(epochs):
                train_loss = self._train_epoch(train_data, on_step=on_step)
                evaluation = self.evaluate(validation_data) if validation_data is not None else None
                validation_loss = evaluation.loss if evaluation is not None else None
                history.append(EpochResult(train_loss, validation_loss, evaluation.accuracy if evaluation else None))
                score, maximize = self.selection_policy.score(history[-1])
                if (best_state is None and not has_initial_score) or self.selection_policy.better(score, best_score, maximize):
                    best_score = score
                    best_state = self.model.state_dict()
                    best_epoch = epoch
                if self.scheduler is not None:
                    self.scheduler.step()
                if on_epoch is not None:
                    on_epoch(epoch, history[-1], best_epoch == epoch)
            if best_state is not None:
                self.model.load_state_dict(best_state)
        finally:
            self.model.train(original_mode)
        return TrainResult(tuple(history), best_epoch)

    @property
    def selection_metric(self): return self.selection_policy.metric

    @property
    def worst_selection_score(self): return self.selection_policy.worst_score

    def selection_score(self, result: EpochResult) -> float:
        return self.selection_policy.score(result)[0]

    def _train_epoch(self, train_data: BatchIterable, *, on_step=None) -> float:
        self.model.train()
        losses = []
        for step, (inputs, targets) in enumerate(train_data, start=1):
            value = self._train_step(inputs, targets, step=step)
            losses.append(value)
            if on_step is not None:
                on_step(step, value)
        if not losses:
            raise ValueError("training epoch cannot be empty")
        return sum(losses) / len(losses)

    def _train_step(self, inputs, targets, *, step: int, timer=None) -> float:
        from micro_dl.optim import clip_grad_norm_

        with timer.measure("optimizer") if timer else nullcontext():
            self.optimizer.zero_grad()
        with timer.measure("forward") if timer else nullcontext():
            objective = self.loss(self.model(inputs), targets)
            value = objective.item()
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite training loss at step {step}: {value}")
        with timer.measure("backward") if timer else nullcontext():
            objective.backward()
            if self.max_grad_norm is not None:
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        with timer.measure("optimizer") if timer else nullcontext():
            self.optimizer.step()
        return value

    def evaluate(self, data: BatchIterable) -> EvaluationResult:
        original_mode = self.model.training
        self.model.eval()
        losses = []
        correct = 0
        total = 0
        try:
            with no_grad():
                for inputs, targets in data:
                    logits = self.model(inputs)
                    losses.append(self.loss(logits, targets).item())
                    labels = tuple(int(target) for target in targets)
                    predictions = logits.argmax(axis=1).to_numpy()
                    correct += sum(int(prediction) == label for prediction, label in zip(predictions, labels))
                    total += len(labels)
        finally:
            self.model.train(original_mode)
        if not losses:
            raise ValueError("evaluation data cannot be empty")
        return EvaluationResult(sum(losses) / len(losses), correct / total)

    def profile(self, train_data: BatchIterable, *, steps: int) -> dict[str, Any]:
        """Run bounded training steps and report synchronized phase timings."""
        if steps <= 0:
            raise ValueError("profile steps must be positive")
        backend = getattr(train_data, "backend", None)
        if backend is None:
            raise TypeError("profiling requires train_data with a Backend identity")
        iterator = iter(train_data)
        timer = _PhaseTimer(backend)
        completed = 0
        original_mode = self.model.training
        timer.start()
        try:
            self.model.train()
            for step in range(1, steps + 1):
                with timer.measure("data"):
                    try:
                        inputs, targets = next(iterator)
                    except StopIteration:
                        break
                self._train_step(inputs, targets, step=step, timer=timer)
                completed += 1
        finally:
            total = timer.stop()
            self.model.train(original_mode)
        if completed == 0:
            raise ValueError("profiling data cannot be empty")
        return {
            "steps": completed,
            "total_ms": total * 1000.0,
            "phases": {
                name: {
                    "total_ms": elapsed * 1000.0,
                    "ms_per_step": elapsed * 1000.0 / completed,
                    "percent": elapsed / total * 100.0 if total else 0.0,
                }
                for name, elapsed in timer.phases.items()
            },
        }


class _PhaseTimer:
    def __init__(self, backend: str):
        self.backend = backend
        self.phases = {name: 0.0 for name in ("data", "forward", "backward", "optimizer")}
        self._started = 0.0

    def _now(self):
        from micro_dl.backends import synchronize

        synchronize(backend=self.backend)
        return perf_counter()

    def start(self):
        self._started = self._now()

    def stop(self):
        return self._now() - self._started

    @contextmanager
    def measure(self, phase: str):
        started = self._now()
        try:
            yield
        finally:
            self.phases[phase] += self._now() - started
