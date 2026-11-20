#  Copyright (c) 2026 Forum Software, Inc. (YHat Labs). Licensed under the Apache License, Version 2.0.
"""Forecasting architectures, registered under the names config.json uses."""

from yhatmodels.registry import Architecture, register
from yhatmodels.ts.architectures.chakrats_v1 import ChakraTSNet, Config

register(Architecture("chakrats_v1", ("forecasting",), lambda settings: ChakraTSNet(Config.from_dict(settings))))
