"""Backend registry and scoped default selection."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from micro_dl.backends.base import Backend
from micro_dl.backends.numpy_backend import NumpyBackend

_DEFAULT_BACKEND: ContextVar[str] = ContextVar("micro_dl_backend", default="numpy")
_NUMPY = NumpyBackend()
_CUPY = None


def resolve_backend(name: str) -> Backend:
    if name == "numpy":
        return _NUMPY
    if name == "cupy":
        global _CUPY
        from micro_dl.backends.cupy_backend import CupyBackend
        if _CUPY is None:
            _CUPY = CupyBackend()
        return _CUPY
    raise ValueError(f"unknown Backend: {name!r}")


def get_default_backend() -> str:
    return _DEFAULT_BACKEND.get()


def manual_seed(seed: int, *, backend: str | None = None) -> None:
    resolve_backend(backend or get_default_backend()).manual_seed(seed)


def rng_state(*, backend: str | None = None):
    return resolve_backend(backend or get_default_backend()).rng_state()


def load_rng_state(state, *, backend: str | None = None) -> None:
    resolve_backend(backend or get_default_backend()).load_rng_state(state)


def synchronize(*, backend: str | None = None) -> None:
    resolve_backend(backend or get_default_backend()).synchronize()


@contextmanager
def use_backend(name: str) -> Iterator[None]:
    resolve_backend(name)
    token = _DEFAULT_BACKEND.set(name)
    try:
        yield
    finally:
        _DEFAULT_BACKEND.reset(token)
