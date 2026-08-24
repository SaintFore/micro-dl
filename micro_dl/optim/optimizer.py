"""Backend-neutral optimizer update rules."""

from __future__ import annotations

import math
from collections.abc import Iterable

from micro_dl.autograd import no_grad
from micro_dl.nn import Parameter
from micro_dl.tensor import Tensor


class Optimizer:
    def __init__(self, parameters: Iterable[Parameter], lr: float) -> None:
        self.parameters = list(dict.fromkeys(parameters))
        if lr <= 0:
            raise ValueError("learning rate must be positive")
        self.lr = float(lr)

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()

    def state_dict(self):
        return {"lr": self.lr}

    def load_state_dict(self, state) -> None:
        self.lr = float(state["lr"])


class SGD(Optimizer):
    def __init__(self, parameters, lr: float = 0.01, momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity = {}

    def state_dict(self):
        state = super().state_dict()
        state.update(momentum=self.momentum, weight_decay=self.weight_decay)
        state["velocity"] = [
            None if index not in self._velocity else self._velocity[index].to_numpy()
            for index in range(len(self.parameters))
        ]
        return state

    def load_state_dict(self, state) -> None:
        super().load_state_dict(state)
        self.momentum = float(state["momentum"])
        self.weight_decay = float(state["weight_decay"])
        _validate_state_count(state["velocity"], self.parameters)
        self._velocity = {
            index: Tensor(value, backend=parameter.backend)
            for index, (parameter, value) in enumerate(zip(self.parameters, state["velocity"]))
            if value is not None
        }

    def step(self) -> None:
        with no_grad():
            for index, parameter in enumerate(self.parameters):
                gradient = parameter.grad_tensor()
                if gradient is None:
                    continue
                if self.weight_decay:
                    gradient = gradient + self.weight_decay * parameter
                if self.momentum:
                    previous = self._velocity.get(index)
                    velocity = gradient if previous is None else self.momentum * previous + gradient
                    self._velocity[index] = velocity
                    gradient = velocity
                parameter.copy_(parameter - self.lr * gradient)


class Adam(Optimizer):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self._moments = {}

    def state_dict(self):
        state = super().state_dict()
        state.update(
            betas=(self.beta1, self.beta2),
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
        state["moments"] = [
            None if index not in self._moments else {
                "step": self._moments[index][0],
                "first": self._moments[index][1].to_numpy(),
                "second": self._moments[index][2].to_numpy(),
            }
            for index in range(len(self.parameters))
        ]
        return state

    def load_state_dict(self, state) -> None:
        super().load_state_dict(state)
        self.beta1, self.beta2 = state["betas"]
        self.eps = float(state["eps"])
        self.weight_decay = float(state["weight_decay"])
        _validate_state_count(state["moments"], self.parameters)
        self._moments = {}
        for index, (parameter, item) in enumerate(zip(self.parameters, state["moments"])):
            if item is not None:
                self._moments[index] = (
                    item["step"],
                    Tensor(item["first"], backend=parameter.backend),
                    Tensor(item["second"], backend=parameter.backend),
                )

    def step(self) -> None:
        with no_grad():
            for index, parameter in enumerate(self.parameters):
                gradient = parameter.grad_tensor()
                if gradient is None:
                    continue
                if self.weight_decay:
                    gradient = gradient + self.weight_decay * parameter
                step, first, second = self._moments.get(index, (0, gradient * 0.0, gradient * 0.0))
                step += 1
                first = self.beta1 * first + (1 - self.beta1) * gradient
                second = self.beta2 * second + (1 - self.beta2) * (gradient ** 2)
                self._moments[index] = (step, first, second)
                corrected_first = first / (1 - self.beta1**step)
                corrected_second = second / (1 - self.beta2**step)
                parameter.copy_(parameter - self.lr * corrected_first / (corrected_second ** 0.5 + self.eps))


class AdamW(Adam):
    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        super().__init__(parameters, lr, betas, eps, weight_decay=0.0)
        self.decoupled_weight_decay = weight_decay

    def step(self) -> None:
        with no_grad():
            for parameter in self.parameters:
                if parameter.grad is not None and self.decoupled_weight_decay:
                    parameter.copy_(parameter * (1 - self.lr * self.decoupled_weight_decay))
        super().step()

    def state_dict(self):
        state = super().state_dict()
        state["decoupled_weight_decay"] = self.decoupled_weight_decay
        return state

    def load_state_dict(self, state) -> None:
        super().load_state_dict(state)
        self.decoupled_weight_decay = float(state["decoupled_weight_decay"])


def clip_grad_norm_(parameters: Iterable[Parameter], max_norm: float) -> float:
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    parameters = list(dict.fromkeys(parameters))
    total_squared = 0.0
    for parameter in parameters:
        total_squared += parameter.grad_squared_sum()
    total = math.sqrt(total_squared)
    if total > max_norm:
        factor = max_norm / (total + 1e-12)
        for parameter in parameters:
            parameter.scale_grad_(factor)
    return total


def _validate_state_count(items, parameters) -> None:
    if len(items) != len(parameters):
        raise ValueError("optimizer state does not match Parameter count")
