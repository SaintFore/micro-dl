"""Explicit learning-rate schedules stepped by Trainer."""

import math


class StepLR:
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.epochs = 0

    def step(self) -> None:
        self.epochs += 1
        if self.epochs % self.step_size == 0:
            self.optimizer.lr *= self.gamma

    def state_dict(self): return {"epochs": self.epochs}
    def load_state_dict(self, state): self.epochs = int(state["epochs"])


class CosineAnnealingLR:
    def __init__(self, optimizer, total_epochs: int, min_lr: float = 0.0):
        if total_epochs <= 0:
            raise ValueError("total_epochs must be positive")
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.lr
        self.epochs = 0

    def step(self) -> None:
        self.epochs += 1
        progress = min(self.epochs, self.total_epochs) / self.total_epochs
        self.optimizer.lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

    def state_dict(self): return {"epochs": self.epochs, "base_lr": self.base_lr}
    def load_state_dict(self, state):
        self.epochs = int(state["epochs"])
        self.base_lr = float(state["base_lr"])
