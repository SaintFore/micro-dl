"""算子与自动微分引擎之间的最小协议。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from micro_dl.tensor import Tensor


class GradFn(ABC):
    """描述一个结果 Tensor 与其父 Tensor 之间的梯度关系。

    具体算子的 GradFn 负责保存 backward 所需的前向值，并为每个父
    Tensor 返回一个梯度。GradFn 本身不负责累积梯度。
    """

    def __init__(self, *parents: Tensor):
        self.parents = tuple(parents)

    @abstractmethod
    def backward(self, grad_output: Any) -> Sequence[Any | None]:
        """返回与 ``parents`` 顺序一一对应的梯度。"""
        raise NotImplementedError
