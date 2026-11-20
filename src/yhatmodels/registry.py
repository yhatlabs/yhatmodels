#  Copyright (c) 2026 Forum Software, Inc. (YHat Labs). Licensed under the Apache License, Version 2.0.
"""Architectures by name. A released model's config.json names its architecture; each task module (yhatmodels.ts, and
later others) registers the architectures it implements when it is imported, with the task they serve, so a model is
only ever loaded by the class made for it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from yhatmodels.errors import ModelLoadError

# The tasks a released model can serve. One network may serve several: anomaly detection reads a forecaster's own
# forecasts, so a forecasting architecture can list "anomaly" too.
TASKS = ("forecasting", "tabular", "anomaly")


@dataclass(frozen=True)
class Architecture:
    name: str  # as config.json names it, e.g. "chakrats_v1"
    tasks: tuple[str, ...]  # from TASKS
    build: Callable[[dict], torch.nn.Module]  # config.json's "model" settings -> the network, before its weights


ARCHITECTURES: dict[str, Architecture] = {}


def register(arch: Architecture) -> None:
    if unknown := set(arch.tasks) - set(TASKS):
        raise ValueError(f"unknown tasks {sorted(unknown)}; tasks are {TASKS}")
    ARCHITECTURES[arch.name] = arch


def get(name: str | None, task: str) -> Architecture:
    arch = ARCHITECTURES.get(name or "")
    if arch is None:
        known = sorted(a.name for a in ARCHITECTURES.values() if task in a.tasks)
        raise ModelLoadError(f"unknown architecture {name!r}; this yhatmodels knows {known}. A newer yhatmodels may have it.")
    if task not in arch.tasks:
        raise ModelLoadError(f"{name!r} is a {' / '.join(arch.tasks)} model, not a {task} one")
    return arch
