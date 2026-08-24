"""Reusable neural-network Modules composed from Tensor Operators."""

from __future__ import annotations

from micro_dl.nn.module import Buffer, Module, Parameter, Sequential
from micro_dl.tensor import Tensor


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, *, bias: bool = True, backend: str | None = None):
        super().__init__()
        scale = (2.0 / in_features) ** 0.5
        self.weight = Parameter(
            Tensor.random_normal((in_features, out_features), scale, backend=backend)
        )
        self.bias = Parameter([0.0] * out_features, backend=backend) if bias else None

    def forward(self, value):
        output = value @ self.weight
        return output + self.bias if self.bias is not None else output


class ReLU(Module):
    def forward(self, value):
        return value.relu()


class Flatten(Module):
    def forward(self, value):
        return value.reshape(value.shape[0], -1)


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, *, stride=1, padding=0, bias=True, backend=None):
        super().__init__()
        kernel = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        scale = (2.0 / (in_channels * kernel[0] * kernel[1])) ** 0.5
        self.weight = Parameter(
            Tensor.random_normal((out_channels, in_channels, *kernel), scale, backend=backend)
        )
        self.bias = Parameter([0.0] * out_channels, backend=backend) if bias else None
        self.stride, self.padding = stride, padding

    def forward(self, value):
        from micro_dl.ops import conv2d
        return conv2d(value, self.weight, self.bias, stride=self.stride, padding=self.padding)


class BatchNorm2d(Module):
    def __init__(self, features, *, momentum=0.1, eps=1e-5, backend=None):
        super().__init__()
        self.weight = Parameter([1.0] * features, backend=backend)
        self.bias = Parameter([0.0] * features, backend=backend)
        self.register_buffer("running_mean", Buffer([0.0] * features, backend=backend))
        self.register_buffer("running_var", Buffer([1.0] * features, backend=backend))
        self.momentum, self.eps = momentum, eps

    def forward(self, value):
        if self.training:
            mean = value.mean(axis=(0, 2, 3), keepdims=True)
            centered = value - mean
            variance = (centered ** 2).mean(axis=(0, 2, 3), keepdims=True)
            self.running_mean.copy_((1 - self.momentum) * self.running_mean + self.momentum * mean.reshape(-1))
            self.running_var.copy_((1 - self.momentum) * self.running_var + self.momentum * variance.reshape(-1))
        else:
            mean = self.running_mean.reshape(1, -1, 1, 1)
            variance = self.running_var.reshape(1, -1, 1, 1)
            centered = value - mean
        normalized = centered / ((variance + self.eps) ** 0.5)
        return normalized * self.weight.reshape(1, -1, 1, 1) + self.bias.reshape(1, -1, 1, 1)


class Dropout(Module):
    def __init__(self, probability=0.5):
        super().__init__()
        if not 0 <= probability < 1: raise ValueError("probability must be in [0, 1)")
        self.probability = probability
    def forward(self, value):
        if not self.training or self.probability == 0: return value
        from micro_dl.ops import dropout
        return dropout(value, self.probability)


class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels=None, *, stride=1, backend=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv1 = Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, backend=backend)
        self.bn1 = BatchNorm2d(out_channels, backend=backend)
        self.conv2 = Conv2d(out_channels, out_channels, 3, padding=1, backend=backend)
        self.bn2 = BatchNorm2d(out_channels, backend=backend)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Sequential(
                Conv2d(in_channels, out_channels, 1, stride=stride, backend=backend),
                BatchNorm2d(out_channels, backend=backend),
            )
    def forward(self, value):
        residual = value if self.shortcut is None else self.shortcut(value)
        return (self.bn2(self.conv2(self.bn1(self.conv1(value)).relu())) + residual).relu()
