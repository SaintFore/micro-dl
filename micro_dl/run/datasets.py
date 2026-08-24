"""Explicit, checksummed dataset preparation; training never downloads data."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.request import urlopen


MNIST = {
    "train-images-idx3-ubyte.gz": ("https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz", "sha256", "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609"),
    "train-labels-idx1-ubyte.gz": ("https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz", "sha256", "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c"),
    "t10k-images-idx3-ubyte.gz": ("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz", "sha256", "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6"),
    "t10k-labels-idx1-ubyte.gz": ("https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz", "sha256", "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6"),
}
CIFAR100 = (
    "https://huggingface.co/datasets/nakroy/cifar100-python/resolve/main/cifar-100-python.tar.gz",
    "sha256",
    "85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7",
)


def prepare_dataset(name: str, root: Path) -> Path:
    destination = root.expanduser() / name
    destination.mkdir(parents=True, exist_ok=True)
    if name == "mnist":
        for filename, source in MNIST.items():
            _download(*source, destination / filename)
    elif name == "cifar100":
        archive = destination / "cifar-100-python.tar.gz"
        _download(*CIFAR100, archive)
        _extract_cifar(archive, destination)
    else:
        raise ValueError("dataset must be 'mnist' or 'cifar100'")
    return destination


def _download(url: str, algorithm: str, expected: str, destination: Path) -> None:
    if destination.exists() and _digest(destination, algorithm) == expected:
        return
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        with urlopen(url, timeout=60) as response:
            shutil.copyfileobj(response, temporary)
    try:
        actual = _digest(temporary_path, algorithm)
        if actual != expected:
            raise ValueError(f"checksum mismatch for {destination.name}: {actual}")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_cifar(archive: Path, destination: Path) -> None:
    expected = {"train", "test", "meta"}
    if all((destination / name).exists() for name in expected):
        return
    with tempfile.TemporaryDirectory(dir=destination) as temporary:
        root = Path(temporary)
        with tarfile.open(archive, "r:gz") as bundle:
            for member in bundle.getmembers():
                if member.name not in {f"cifar-100-python/{name}" for name in expected}:
                    continue
                member.name = Path(member.name).name
                bundle.extract(member, root, filter="data")
        if not all((root / name).exists() for name in expected):
            raise ValueError("CIFAR-100 archive is incomplete")
        for name in expected:
            os.replace(root / name, destination / name)
