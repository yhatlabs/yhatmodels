#  Copyright (c) 2026 Forum Software, Inc. (YHat Labs). Licensed under the Apache License, Version 2.0.
"""yhatmodels (YHatModels): open models from YHat Labs, run locally. The hosted models are in the `yhatlabs` client.

Each task lives in its own subpackage (yhatmodels.ts: time-series forecasting) and is imported when first used, so
`import yhatmodels` stays light as tasks are added.
"""

__version__ = "0.1.0"

from yhatmodels.catalog import list_models
from yhatmodels.licensing import accept_license

_LAZY = {"ChakraTS": "yhatmodels.ts", "plot_forecast": "yhatmodels.ts"}  # name -> the task module that has it
__all__ = ["ChakraTS", "__version__", "accept_license", "list_models", "plot_forecast"]


def __getattr__(name: str):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'yhatmodels' has no attribute {name!r}")
