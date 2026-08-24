"""Explicit Module registries and persistent model state."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterator

from micro_dl.tensor import Tensor


class Parameter(Tensor):
    """A trainable Tensor registered as persistent Module state."""

    def __init__(self, data: Any, *, backend: str | None = None, dtype: Any = None):
        super().__init__(data, requires_grad=True, backend=backend, dtype=dtype)

    @property
    def requires_grad(self) -> bool:
        return True

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        if not value:
            raise ValueError("Parameter must require gradients")


class Buffer(Tensor):
    """A non-trainable Tensor registered as persistent Module state."""

    def __init__(self, data: Any, *, backend: str | None = None, dtype: Any = None):
        super().__init__(data, requires_grad=False, backend=backend, dtype=dtype)

    @property
    def requires_grad(self) -> bool:
        return False

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        if value:
            raise ValueError("Buffer cannot require gradients")


@dataclass(frozen=True)
class IncompatibleKeys:
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()
    mismatched_keys: tuple[str, ...] = ()


@dataclass
class Cache:
    """Ephemeral Module data excluded from registries and persistent state."""

    value: Any = None

    def clear(self) -> None:
        self.value = None


class Module:
    """Base object for explicitly registered Parameters, Buffers, and child Modules."""

    def __init__(self) -> None:
        object.__setattr__(self, "_parameters", OrderedDict())
        object.__setattr__(self, "_buffers", OrderedDict())
        object.__setattr__(self, "_modules", OrderedDict())
        object.__setattr__(self, "training", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or "_parameters" not in self.__dict__:
            object.__setattr__(self, name, value)
            return
        self._parameters.pop(name, None)
        self._buffers.pop(name, None)
        self._modules.pop(name, None)
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Buffer):
            self._buffers[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def register_buffer(self, name: str, tensor: Tensor) -> Buffer:
        if not name or "." in name:
            raise ValueError("Buffer name must be non-empty and cannot contain '.'")
        value = tensor if isinstance(tensor, Buffer) else Buffer(tensor)
        setattr(self, name, value)
        return value

    def parameters(self) -> Iterator[Parameter]:
        for _, parameter in self.named_parameters():
            yield parameter

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        seen: set[int] = set()
        yield from self._named_parameters(prefix, seen)

    def _named_parameters(self, prefix: str, seen: set[int]):
        for name, parameter in self._parameters.items():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                yield _join(prefix, name), parameter
        for name, module in self._modules.items():
            yield from module._named_parameters(_join(prefix, name), seen)

    def named_buffers(self, prefix: str = "") -> Iterator[tuple[str, Buffer]]:
        for name, buffer in self._buffers.items():
            yield _join(prefix, name), buffer
        for name, module in self._modules.items():
            yield from module.named_buffers(_join(prefix, name))

    def children(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def train(self, mode: bool = True) -> Module:
        self.training = bool(mode)
        for child in self.children():
            child.train(mode)
        return self

    def eval(self) -> Module:
        return self.train(False)

    def state_dict(self) -> OrderedDict[str, Any]:
        state = OrderedDict()
        for name, tensor in (*tuple(self.named_parameters()), *tuple(self.named_buffers())):
            state[name] = tensor.to_numpy()
        return state

    def load_state_dict(self, state: dict[str, Any], *, strict: bool = True) -> IncompatibleKeys:
        targets = dict(self.named_parameters()) | dict(self.named_buffers())
        missing = tuple(name for name in targets if name not in state)
        unexpected = tuple(name for name in state if name not in targets)
        mismatched = tuple(
            name
            for name in targets.keys() & state.keys()
            if tuple(getattr(state[name], "shape", ())) != targets[name].shape
        )
        result = IncompatibleKeys(missing, unexpected, mismatched)
        if strict and (missing or unexpected or mismatched):
            raise ValueError(
                f"state mismatch: missing={missing}, unexpected={unexpected}, shape={mismatched}"
            )
        for name, target in targets.items():
            if name in state and name not in mismatched:
                target.copy_(state[name])
        return result

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class Sequential(Module):
    """Apply registered child Modules in insertion order."""

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        for index, module in enumerate(modules):
            setattr(self, str(index), module)

    def forward(self, value):
        for module in self._modules.values():
            value = module(value)
        return value


def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name
