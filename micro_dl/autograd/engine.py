"""Computation Graph traversal, gradient accumulation, and lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from micro_dl.tensor import Tensor


def backward(output: Tensor, grad: Any | None = None) -> None:
    """Propagate one gradient through ``output``'s dynamic Computation Graph."""
    if output._graph_consumed:
        raise RuntimeError("该计算图已经执行过 backward，不能重复使用")

    if grad is None:
        if output.size != 1:
            raise ValueError("非标量 output 必须显式提供 grad")
        initial_grad = output._backend.ones_like(output._storage)
    else:
        initial_grad = output._backend.array(grad, copy=True)
        if initial_grad.shape != output.shape:
            raise ValueError(
                f"grad 形状{initial_grad.shape}与output{output.shape}不匹配"
            )

    order = _topological_order(output)
    graph_nodes = [node for node in order if node._grad_fn is not None]
    gradients: dict[int, Any] = {id(output): initial_grad}

    try:
        for node in reversed(order):
            grad_fn = node._grad_fn
            node_grad = gradients.get(id(node))
            if node_grad is None:
                continue
            if grad_fn is None:
                if node.requires_grad:
                    node.grad = (
                        node_grad
                        if node.grad is None
                        else node._backend.add(node.grad, node_grad)
                    )
                continue

            parent_grads = grad_fn.backward(node_grad)
            if parent_grads is None:
                raise TypeError("GradFn.backward 必须返回与 parents 对齐的梯度序列")
            parent_grads = tuple(parent_grads)
            if len(parent_grads) != len(grad_fn.parents):
                raise RuntimeError("GradFn.backward 返回的梯度数量与 parents 数量不一致")

            for parent, parent_grad in zip(grad_fn.parents, parent_grads):
                if parent_grad is None or not parent.requires_grad:
                    continue
                normalized = parent._backend.array(parent_grad, copy=True)
                if normalized.shape != parent.shape:
                    raise ValueError(
                        f"父节点梯度形状 {normalized.shape} 与 Tensor 形状 {parent.shape} 不匹配"
                    )
                existing = gradients.get(id(parent))
                gradients[id(parent)] = (
                    normalized
                    if existing is None
                    else parent._backend.add(existing, normalized)
                )
    finally:
        for node in graph_nodes:
            node._graph_consumed = True
            node._grad_fn = None


def _topological_order(output: Tensor) -> list[Tensor]:
    visited: set[int] = set()
    order: list[Tensor] = []
    stack = [(output, False)]
    while stack:
        node, parents_processed = stack.pop()
        identity = id(node)
        if parents_processed:
            order.append(node)
            continue
        if identity in visited:
            continue
        visited.add(identity)
        stack.append((node, True))
        if node._grad_fn is not None:
            for parent in reversed(node._grad_fn.parents):
                if id(parent) not in visited:
                    stack.append((parent, False))
    return order
