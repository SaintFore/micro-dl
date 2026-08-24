"""Headless-safe reporting from structured training results."""

from pathlib import Path


def plot_training(result, output_path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(result.epochs) + 1)
    figure = Figure(figsize=(6, 4))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.plot(epochs, [item.train_loss for item in result.epochs], label="train")
    if any(item.validation_loss is not None for item in result.epochs):
        axis.plot(epochs, [item.validation_loss for item in result.epochs], label="validation")
    axis.set_xlabel("epoch")
    axis.set_ylabel("loss")
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, metadata={"Software": "micro-dl"})
    figure.clear()
    return destination
