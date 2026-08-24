"""Classification objectives composed from Tensor Operators."""

from micro_dl.nn.module import Module


class CrossEntropyLoss(Module):
    def __init__(self, reduction: str = "mean", label_smoothing: float = 0.0) -> None:
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be 'none', 'mean', or 'sum'")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        if logits.ndim != 2:
            raise ValueError("CrossEntropyLoss expects (batch, classes) logits")
        from micro_dl.ops import cross_entropy

        return cross_entropy(
            logits,
            targets,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )
