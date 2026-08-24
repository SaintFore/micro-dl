"""Resolved training configuration persisted with every Run."""

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class RunConfig:
    experiment: str
    backend: str = "cupy"
    epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    data_dir: str = "~/data/micro-dl"
    run_root: str = "~/runs/micro-dl"
    log_every: int = 10
    gpu_sample_seconds: float = 10.0
    seed: int = 0

    @classmethod
    def from_toml(cls, path):
        with Path(path).open("rb") as handle:
            return cls(**tomllib.load(handle))

    @classmethod
    def from_toml_text(cls, value: str):
        return cls(**tomllib.loads(value))

    def to_toml(self) -> str:
        lines = []
        for key, value in asdict(self).items():
            encoded = f'"{value}"' if isinstance(value, str) else str(value).lower() if isinstance(value, bool) else str(value)
            lines.append(f"{key} = {encoded}")
        return "\n".join(lines) + "\n"
