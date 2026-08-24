"""Operational CLI for reproducible local and V100 execution."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
from typing import Annotated

import typer

from micro_dl import Tensor
from micro_dl.backends import load_rng_state as restore_backend_rng, manual_seed, rng_state as backend_rng_state
from micro_dl.data import MNIST
from micro_dl.experiments import assemble_cifar100_experiment, build_mnist_experiment
from micro_dl.run import RunConfig, load_checkpoint, prepare_dataset, save_checkpoint

app = typer.Typer(no_args_is_help=True, help="Reproducible micro-dl training operations.")


@app.command()
def doctor(backend: Annotated[str, typer.Option()] = "cupy"):
    """Verify Python, Backend, GPU and Git."""
    report = {"python": platform.python_version(), "backend": backend, "git": _git("rev-parse", "HEAD")}
    try:
        tensor = Tensor([1.0], backend=backend)
        report["tensor"] = {"shape": tensor.shape, "value": tensor.item()}
        if backend == "cupy":
            import cupy as cp
            properties = cp.cuda.runtime.getDeviceProperties(0)
            name = properties["name"]
            report["gpu"] = {"name": name.decode() if isinstance(name, bytes) else name, "memory_bytes": cp.cuda.runtime.memGetInfo()[1]}
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        typer.echo(json.dumps(report, indent=2)); raise typer.Exit(1)
    typer.echo(json.dumps(report, indent=2))


@app.command()
def inspect(run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)]):
    """Summarize a Run's status, latest metric and checkpoints."""
    result = {"run_dir": str(run_dir)}
    status_path, metrics_path = run_dir / "run.json", run_dir / "metrics.jsonl"
    if status_path.exists(): result["status"] = json.loads(status_path.read_text())
    if metrics_path.exists():
        lines = metrics_path.read_text().splitlines(); result["latest_metric"] = json.loads(lines[-1]) if lines else None
    checkpoints = run_dir / "checkpoints"
    result["checkpoints"] = sorted(path.name for path in checkpoints.glob("*.zip")) if checkpoints.exists() else []
    typer.echo(json.dumps(result, indent=2))


@app.command("checkpoint-inspect")
def checkpoint_inspect(path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)]):
    typer.echo(json.dumps(load_checkpoint(path)["manifest"], indent=2))


@app.command()
def smoke(experiment: Annotated[str, typer.Option()] = "mnist", backend: Annotated[str, typer.Option()] = "cupy"):
    """Run the selected one-batch contract smoke."""
    selected = "tests/test_data_mnist_smoke.py" if experiment == "mnist" else "tests/test_cifar100_smoke.py"
    raise typer.Exit(subprocess.run(["python", "-m", "pytest", "-q", selected, "-k", backend], check=False).returncode)


@app.command()
def profile(
    experiment: Annotated[str, typer.Option()],
    backend: Annotated[str, typer.Option()] = "cupy",
    steps: Annotated[int, typer.Option(min=1)] = 100,
    batch_size: Annotated[int, typer.Option(min=1)] = 32,
    data_dir: Annotated[Path, typer.Option()] = Path("~/data/micro-dl"),
    output: Annotated[Path | None, typer.Option(dir_okay=False)] = None,
):
    """Profile synchronized data, forward, backward and optimizer phases."""
    config = RunConfig(experiment=experiment, backend=backend, batch_size=batch_size, data_dir=str(data_dir))
    manual_seed(config.seed, backend=backend)
    try:
        trainer, loader, _ = _build(config)
    except FileNotFoundError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    report = {"experiment": experiment, "backend": backend, **trainer.profile(loader, steps=steps)}
    encoded = json.dumps(report, indent=2) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded)
    typer.echo(encoded, nl=False)


@app.command("prepare-data")
def prepare_data(dataset: Annotated[str, typer.Option()], data_dir: Annotated[Path, typer.Option()] = Path("~/data/micro-dl")):
    """Download and atomically verify a canonical dataset cache."""
    typer.echo(str(prepare_dataset(dataset, data_dir)))


@app.command()
def train(
    experiment: Annotated[str | None, typer.Option()] = None,
    config: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    backend: Annotated[str | None, typer.Option()] = None,
    epochs: Annotated[int | None, typer.Option()] = None,
    batch_size: Annotated[int | None, typer.Option()] = None,
    learning_rate: Annotated[float | None, typer.Option()] = None,
    data_dir: Annotated[Path | None, typer.Option()] = None,
    run_root: Annotated[Path | None, typer.Option()] = None,
    resume: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    fork_from: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
):
    """Train a canonical experiment; flags override TOML and defaults."""
    if resume and fork_from:
        raise typer.BadParameter("--resume and --fork-from are mutually exclusive")
    if resume and any(value is not None for value in (experiment, config, backend, batch_size, learning_rate, data_dir, run_root)):
        raise typer.BadParameter("--resume only permits --epochs; use --fork-from for configuration changes")
    source = resume or fork_from
    restored = load_checkpoint(source) if source else None
    if restored:
        resolved = RunConfig.from_toml_text(restored["config_toml"])
    elif config:
        resolved = RunConfig.from_toml(config)
    elif experiment:
        resolved = RunConfig(experiment=experiment)
    else:
        raise typer.BadParameter("provide --experiment, --config, --resume, or --fork-from")
    overrides = {"experiment": experiment, "backend": backend, "epochs": epochs, "batch_size": batch_size,
                 "learning_rate": learning_rate, "data_dir": str(data_dir) if data_dir else None,
                 "run_root": str(run_root) if run_root else None}
    resolved = replace(resolved, **{key: value for key, value in overrides.items() if value is not None})
    _require_clean_git()
    commit = _git("rev-parse", "HEAD")
    short = _git("rev-parse", "--short", "HEAD")
    manual_seed(resolved.seed, backend=resolved.backend)
    trainer, train_loader, validation_loader = _build(resolved)
    trainer.optimizer.lr = resolved.learning_rate
    checkpoint_metric = restored["manifest"].get("selection_metric") if restored else None
    if resume and checkpoint_metric != trainer.selection_metric:
        raise ValueError(
            "this checkpoint predates or differs from the Experiment selection policy; use --fork-from"
        )
    historical_best = None
    if resume:
        best_path = resume.resolve().parent / "best.zip"
        if not best_path.exists():
            raise ValueError("resume requires the matching sibling best.zip")
        historical_best = load_checkpoint(best_path)
        resume_score = float(restored["manifest"].get("best_score", trainer.worst_selection_score))
        best_manifest = historical_best["manifest"]
        if (
            best_manifest.get("selection_metric") != checkpoint_metric
            or float(best_manifest.get("best_score", trainer.worst_selection_score)) != resume_score
        ):
            raise ValueError("resume checkpoint and sibling best.zip do not describe the same best history")
    if resume:
        run_dir = resume.resolve().parent.parent
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path(resolved.run_root).expanduser() / resolved.experiment / f"{stamp}-{short}"
        run_dir.mkdir(parents=True, exist_ok=False)
    checkpoints = run_dir / "checkpoints"; checkpoints.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.toml").write_text(resolved.to_toml())
    _write_json(run_dir / "run.json", {"status": "running", "commit": commit, "pid": os.getpid()})
    start_epoch = int(restored["manifest"].get("epoch", 0)) if restored else 0
    if restored:
        trainer.model.load_state_dict(restored["model"])
        trainer.optimizer.load_state_dict(restored["optimizer"])
        if trainer.scheduler and restored["scheduler"]: trainer.scheduler.load_state_dict(restored["scheduler"])
        if restored["rng"]:
            if "train_data" in restored["rng"]:
                train_loader.load_state_dict(restored["rng"]["train_data"])
            else:
                train_loader._epoch = start_epoch
            restore_backend_rng(restored["rng"]["backend"], backend=resolved.backend)
    metrics_path = run_dir / "metrics.jsonl"
    global_step = int(restored["manifest"].get("global_step", 0)) if restored else 0
    log_path = run_dir / "train.log"
    def on_step(step, loss):
        nonlocal global_step
        global_step += 1
        _append_metric(metrics_path, {"type": "step", "step": global_step, "loss": loss})
        if global_step % resolved.log_every == 0: _log(log_path, f"step={global_step} loss={loss:.6f}")
    continues_policy = bool(resume and checkpoint_metric == trainer.selection_metric)
    best_score = float(restored["manifest"].get("best_score", trainer.worst_selection_score)) if continues_policy else trainer.worst_selection_score
    initial_best_state = None
    if continues_policy:
        initial_best_state = historical_best["model"]
    def on_epoch(index, result, is_best):
        nonlocal best_score
        epoch = start_epoch + index + 1
        metric = {
            "type": "epoch",
            "epoch": epoch,
            "train_loss": result.train_loss,
            "validation_loss": result.validation_loss,
            "validation_accuracy": result.validation_accuracy,
        }
        _append_metric(metrics_path, metric); _log(log_path, json.dumps(metric))
        score = trainer.selection_score(result)
        if is_best: best_score = score
        metadata = {"commit": commit, "epoch": epoch, "global_step": global_step, "experiment": resolved.experiment, "backend": resolved.backend, "selection_metric": trainer.selection_metric, "best_score": best_score}
        kwargs = dict(config_toml=resolved.to_toml(), model_state=trainer.model.state_dict(), optimizer_state=trainer.optimizer.state_dict(),
                      scheduler_state=trainer.scheduler.state_dict() if trainer.scheduler else {},
                      rng_state={"backend": backend_rng_state(backend=resolved.backend), "train_data": train_loader.state_dict()}, metadata=metadata)
        save_checkpoint(checkpoints / "last.zip", **kwargs)
        if is_best: save_checkpoint(checkpoints / "best.zip", **kwargs)
    try:
        with _gpu_monitor(run_dir / "gpu.csv", resolved.gpu_sample_seconds, resolved.backend == "cupy"):
            trainer.fit(train_loader, epochs=resolved.epochs, validation_data=validation_loader, on_step=on_step,
                        on_epoch=on_epoch,
                        initial_best_score=best_score if continues_policy else None,
                        initial_best_state=initial_best_state)
    except Exception as exc:
        _write_json(run_dir / "run.json", {"status": "failed", "commit": commit, "error": f"{type(exc).__name__}: {exc}"})
        raise
    _write_json(run_dir / "run.json", {"status": "completed", "commit": commit, "epochs": start_epoch + resolved.epochs})
    typer.echo(str(run_dir))


def _build(config: RunConfig):
    root = Path(config.data_dir).expanduser() / config.experiment
    if config.experiment == "mnist":
        train_set, validation_set = MNIST.from_cache(root), MNIST.from_cache(root, train=False)
        trainer, loader = build_mnist_experiment(train_set, backend=config.backend, batch_size=config.batch_size)
        _, validation = build_mnist_experiment(validation_set, backend=config.backend, batch_size=config.batch_size)
    elif config.experiment == "cifar100":
        return assemble_cifar100_experiment(
            root, backend=config.backend, batch_size=config.batch_size,
            epochs=config.epochs, seed=config.seed,
        )
    else:
        raise ValueError("experiment must be 'mnist' or 'cifar100'")
    validation.shuffle = False
    return trainer, loader, validation


def _git(*arguments): return subprocess.run(["git", *arguments], check=True, text=True, capture_output=True).stdout.strip()
def _require_clean_git():
    if _git("status", "--porcelain"):
        raise RuntimeError("training requires a clean Git worktree")
def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp"); temporary.write_text(json.dumps(value, indent=2) + "\n"); os.replace(temporary, path)
def _append_metric(path, value):
    with path.open("a") as handle: handle.write(json.dumps(value, separators=(",", ":")) + "\n")
def _log(path, message):
    typer.echo(message)
    with path.open("a") as handle: handle.write(message + "\n")


@contextmanager
def _gpu_monitor(path: Path, interval: float, enabled: bool):
    if not enabled:
        yield; return
    stop = threading.Event()
    if not path.exists():
        path.write_text("timestamp,index,utilization_gpu,memory_used_mib,memory_total_mib,temperature_c\n")
    def sample():
        while not stop.is_set():
            command = ["nvidia-smi", "--query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            if result.returncode == 0:
                with path.open("a") as handle: handle.write(result.stdout)
            stop.wait(interval)
    worker = threading.Thread(target=sample, daemon=True); worker.start()
    try:
        yield
    finally:
        stop.set(); worker.join(timeout=max(interval, 1.0) + 1.0)
