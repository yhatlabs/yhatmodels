#  Copyright (c) 2026 Forum Software, Inc. (YHat Labs). Licensed under the Apache License, Version 2.0.
"""Models released by YHat Labs: the short name, the Hugging Face repository, the task and the weights licence.

Knowing the licence here lets yhatmodels ask for acceptance before any download (for these models, once per person with
a yhatlabs.com account: yhatmodels.account); a model that is not listed is read from its config.json first.
"""

from __future__ import annotations

MODELS: dict[str, dict[str, str]] = {
    "ChakraTS-OS-10M": {
        "repo": "yhatlabs/ChakraTS-OS-10M",
        "task": "forecasting",
        "license": "chakra-nc-1.0",
        "description": "10.6M-parameter probabilistic time-series forecaster",
    },
}


def lookup(model: str) -> dict[str, str] | None:
    """A catalogued model by its short name or its repository id, with its `name`."""
    name = model if model in MODELS else next((n for n, m in MODELS.items() if m["repo"] == model), None)
    return {"name": name, **MODELS[name]} if name else None


def list_models(task: str | None = None) -> list[dict[str, str]]:
    """The released models (optionally of one task), each with its name, repository, task, licence and description."""
    return [{"name": name, **m} for name, m in MODELS.items() if task is None or m["task"] == task]
