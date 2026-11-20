#  Copyright (c) 2026 Forum Software, Inc. (YHat Labs). Licensed under the Apache License, Version 2.0.
"""Model files: a local folder, or a Hugging Face repository downloaded once into the Hugging Face cache.

A released model is a folder with `config.json` (its task, architecture, network settings, licence and inference
defaults) and its weights: `model.safetensors`, or shards listed in `model.safetensors.index.json` for large models.
`config.json` may also carry `files` (any other files the model needs, e.g. a tabular model's preprocessor), `sha256`
(file name -> digest), checked when the model loads, and `requires_yhatmodels`, the oldest yhatmodels that can run it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from yhatmodels.catalog import lookup
from yhatmodels.errors import ModelLoadError
from yhatmodels.licensing import ensure_accepted

CONFIG, WEIGHTS, INDEX = "config.json", "model.safetensors", "model.safetensors.index.json"
DOWNLOAD = [CONFIG, "model*.safetensors", INDEX, "LICENSE*", "NOTICE*"]


def resolve(model: str | Path, revision: str | None = None, token: str | None = None) -> Path:
    """A local folder as it is; otherwise a released model's short name or a Hugging Face repo id: the copy already in
    the Hugging Face cache (no network: downloaded before means accepted before), else downloaded after its licence has
    been accepted when the licence asks for that (yhatmodels.licensing). Returns the folder."""
    path = Path(model).expanduser()
    if path.is_dir():
        if not (path / CONFIG).exists() or not ((path / WEIGHTS).exists() or (path / INDEX).exists()):
            missing = [f for f in (CONFIG, WEIGHTS) if not (path / f).exists()]
            raise ModelLoadError(f"{path} is missing {', '.join(missing)}")
        _check_files(path)
        return path
    from huggingface_hub import snapshot_download

    entry = lookup(str(model))
    repo = entry["repo"] if entry else str(model)
    if (cached := _cached(repo, revision)) is not None:
        return cached
    if entry:
        ensure_accepted(repo, entry["license"])  # known without a network call
    head = Path(snapshot_download(repo, revision=revision, token=token, allow_patterns=[CONFIG]))
    config = read_config(head)
    if not entry:  # another repository: its config.json names the licence
        ensure_accepted(repo, config.get("license"))
    folder = Path(snapshot_download(repo, revision=revision, token=token, allow_patterns=[*DOWNLOAD, *config.get("files", [])]))
    _check_files(folder)
    return folder


def _cached(repo: str, revision: str | None) -> Path | None:
    """The repository's complete copy in the local Hugging Face cache, or None (nothing or part of it there)."""
    from huggingface_hub import snapshot_download

    try:  # only the files we download: newer huggingface_hub calls a snapshot without the card or .gitattributes incomplete
        folder = Path(snapshot_download(repo, revision=revision, local_files_only=True, allow_patterns=DOWNLOAD))
    except OSError:  # LocalEntryNotFoundError, IncompleteSnapshotError: not (fully) downloaded yet
        return None
    if not (folder / CONFIG).exists():
        return None
    shards = set(json.loads((folder / INDEX).read_text())["weight_map"].values()) if (folder / INDEX).exists() else {WEIGHTS}
    return folder if all((folder / f).exists() for f in {*shards, *read_config(folder).get("files", [])}) else None


def _check_files(folder: Path) -> None:
    if missing := [f for f in read_config(folder).get("files", []) if not (folder / f).exists()]:
        raise ModelLoadError(f"{folder} is missing {', '.join(missing)}, which its config.json lists")


def read_config(folder: Path) -> dict:
    return json.loads((folder / CONFIG).read_text())


def weight_files(folder: Path) -> list[Path]:
    if (folder / WEIGHTS).exists():
        return [folder / WEIGHTS]
    index = json.loads((folder / INDEX).read_text())
    return [folder / name for name in sorted(set(index["weight_map"].values()))]


def load_weights(folder: Path) -> dict:
    from safetensors.torch import load_file

    state: dict = {}
    for f in weight_files(folder):
        state.update(load_file(f))
    return state


def verify(folder: Path, config: dict) -> None:
    """Files listed in config.json's `sha256` must match their digests (a truncated or swapped file fails here)."""
    for name, digest in (config.get("sha256") or {}).items():
        h = hashlib.sha256()
        with open(folder / name, "rb") as f:
            for block in iter(lambda: f.read(1 << 22), b""):
                h.update(block)
        if h.hexdigest() != digest:
            raise ModelLoadError(f"{folder / name} does not match its sha256 in config.json; download it again")


def check_version(config: dict, version: str) -> None:
    need = config.get("requires_yhatmodels")
    if need and _parts(version) < _parts(need):
        raise ModelLoadError(f"this model needs yhatmodels >= {need} (installed: {version}); upgrade with pip install -U yhatmodels")


def _parts(v: str) -> tuple[int, ...]:
    out = []
    for p in v.split("+")[0].split("."):
        digits = "".join(ch for ch in p if ch.isdigit())
        out.append(int(digits or 0))
    return tuple(out)
