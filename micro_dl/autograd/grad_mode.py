"""Context-local control of dynamic graph recording."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_GRAD_ENABLED: ContextVar[bool] = ContextVar("micro_dl_grad_enabled", default=True)


def is_grad_enabled() -> bool:
    return _GRAD_ENABLED.get()


@contextmanager
def no_grad() -> Iterator[None]:
    token = _GRAD_ENABLED.set(False)
    try:
        yield
    finally:
        _GRAD_ENABLED.reset(token)
