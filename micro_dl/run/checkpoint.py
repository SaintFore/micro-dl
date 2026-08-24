"""Versioned, checksummed, non-pickle checkpoint archives."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

SCHEMA_VERSION = 1


def save_checkpoint(path, *, config_toml, model_state, optimizer_state, scheduler_state=None, rng_state=None, metadata=None):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model_arrays = {name: np.asarray(value) for name, value in model_state.items()}
    optimizer_json, optimizer_arrays = _split_arrays(optimizer_state)
    rng_json, rng_arrays = _split_arrays(rng_state or {})
    members = {
        "config.toml": config_toml.encode(),
        "model.npz": _npz(model_arrays),
        "optimizer.json": _json(optimizer_json),
        "optimizer.npz": _npz(optimizer_arrays),
        "scheduler.json": _json(scheduler_state or {}),
        "rng.json": _json(rng_json),
        "rng.npz": _npz(rng_arrays),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        **(metadata or {}),
        "checksums": {name: hashlib.sha256(value).hexdigest() for name, value in members.items()},
    }
    members["manifest.json"] = _json(manifest)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(temporary_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, value in members.items():
                archive.writestr(name, value)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(path):
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        required = {"manifest.json", "config.toml", "model.npz", "optimizer.json", "optimizer.npz", "scheduler.json", "rng.json", "rng.npz"}
        if names != required:
            raise ValueError("checkpoint members do not match schema")
        raw = {name: archive.read(name) for name in names}
    manifest = json.loads(raw["manifest.json"])
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    for name, expected in manifest["checksums"].items():
        if hashlib.sha256(raw[name]).hexdigest() != expected:
            raise ValueError(f"checkpoint checksum failed: {name}")
    model = _load_npz(raw["model.npz"])
    optimizer = _merge_arrays(json.loads(raw["optimizer.json"]), _load_npz(raw["optimizer.npz"]))
    scheduler = json.loads(raw["scheduler.json"])
    rng = _merge_arrays(json.loads(raw["rng.json"]), _load_npz(raw["rng.npz"]))
    return {
        "manifest": manifest,
        "config_toml": raw["config.toml"].decode(),
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "rng": rng,
    }


def _json(value): return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _npz(arrays):
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def _load_npz(raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _split_arrays(value):
    arrays = {}
    counter = 0
    def visit(item):
        nonlocal counter
        if isinstance(item, np.ndarray):
            key = f"array_{counter}"; counter += 1; arrays[key] = item
            return {"__array__": key}
        if isinstance(item, dict): return {str(key): visit(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)): return {"__sequence__": [visit(child) for child in item], "tuple": isinstance(item, tuple)}
        if isinstance(item, np.generic): return item.item()
        return item
    return visit(value), arrays


def _merge_arrays(value, arrays):
    if isinstance(value, dict) and set(value) == {"__array__"}: return arrays[value["__array__"]]
    if isinstance(value, dict) and "__sequence__" in value:
        items = [_merge_arrays(item, arrays) for item in value["__sequence__"]]
        return tuple(items) if value.get("tuple") else items
    if isinstance(value, dict): return {key: _merge_arrays(child, arrays) for key, child in value.items()}
    return value
