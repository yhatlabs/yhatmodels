#  Copyright (c) 2026 Forum Software, Inc. (YHat Labs). Licensed under the Apache License, Version 2.0.


class YHatModelsError(Exception):
    """Base class for yhatmodels errors."""


class LicenseNotAcceptedError(YHatModelsError):
    """The model's license has not been accepted yet (see yhatmodels.accept_license)."""


class ModelLoadError(YHatModelsError):
    """A model folder or repository is missing a file, or its config names an architecture yhatmodels does not have."""
