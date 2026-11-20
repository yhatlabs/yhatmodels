#  Copyright (c) 2026 Forum Software, Inc. (YHat Labs). Licensed under the Apache License, Version 2.0.
"""Loading a released model for a task: its files, then its architecture from the registry, then its weights."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from yhatmodels import __version__, registry
from yhatmodels.hub import check_version, load_weights, read_config, resolve, verify


@dataclass
class Loaded:
    net: torch.nn.Module  # weights loaded strictly, on CPU
    config: dict  # config.json
    folder: Path  # where its files are (a tabular model's preprocessor, ...)


def load(model: str | Path, task: str, revision: str | None = None, token: str | None = None) -> Loaded:
    """`model` (a released name, a Hugging Face repo id or a folder) for `task`: its network, config and folder."""
    folder = resolve(model, revision, token)
    config = read_config(folder)
    check_version(config, __version__)
    arch = registry.get(config.get("architecture"), task)
    verify(folder, config)
    net = arch.build(config.get("model", {}))
    net.load_state_dict(load_weights(folder))
    return Loaded(net, config, folder)
