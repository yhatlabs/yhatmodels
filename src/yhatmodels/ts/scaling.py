#  Copyright (c) 2026 Forum Software, Inc. (YHat Labs). Licensed under the Apache License, Version 2.0.
"""Causal per-patch normalisation.

Patch n of a causal row is scaled with the mean / std of every observed value up to the end of patch n, so no statistic
ever contains the future. Known-future covariate rows are observed over the whole window by definition and use
whole-row statistics. Indicator rows ({0, 1}) are left linear: asinh would only distort a flag.
"""

from __future__ import annotations

import torch

MIN_SCALE = 1e-5
MIN_COUNT = 2


def _float64_on_cpu(fn):
    """Apple GPUs (mps) have no float64, which these moments need over long histories: run them on CPU."""

    def wrapped(values, *args, **kwargs):
        if values.device.type != "mps":
            return fn(values, *args, **kwargs)
        out = fn(values.cpu(), *(a.cpu() if isinstance(a, torch.Tensor) else a for a in args), **kwargs)
        return tuple(o.to(values.device) for o in out) if isinstance(out, tuple) else out.to(values.device)

    return wrapped


@_float64_on_cpu
def prefix_stats(values: torch.Tensor, observed: torch.Tensor, patch: int):
    """(R, T) -> loc, scale, ok of shape (R, N): moments of the observed values in patches 0..n."""
    r, t = values.shape
    x = torch.where(observed, values, 0.0).double().view(r, t // patch, patch)
    w = observed.double().view(r, t // patch, patch)
    n = w.sum(-1).cumsum(-1)
    s1 = (x * w).sum(-1).cumsum(-1)
    s2 = (x * x * w).sum(-1).cumsum(-1)
    mean = s1 / n.clamp_min(1)
    var = (s2 / n.clamp_min(1) - mean**2).clamp_min(0)
    ok = n >= MIN_COUNT
    loc = torch.where(ok, mean, 0.0)
    scale = torch.where(ok, var.sqrt().clamp_min(MIN_SCALE), 1.0)
    return loc.float(), scale.float(), ok


@_float64_on_cpu
def whole_stats(values: torch.Tensor, observed: torch.Tensor, n_patches: int):
    w = observed.double()
    n = w.sum(-1, keepdim=True).clamp_min(1)
    x = torch.where(observed, values, 0.0).double()
    mean = (x * w).sum(-1, keepdim=True) / n
    var = (((x - mean) * w) ** 2).sum(-1, keepdim=True) / n
    loc = mean.float().expand(-1, n_patches)
    scale = var.sqrt().clamp_min(MIN_SCALE).float().expand(-1, n_patches)
    return loc, scale


def is_binary(values: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    v = torch.where(observed, values, 0.0)
    return ((v == 0) | (v == 1)).all(-1) & observed.any(-1)


def squash(x: torch.Tensor) -> torch.Tensor:
    return torch.asinh(x)


def unsquash(z: torch.Tensor) -> torch.Tensor:
    return torch.sinh(z.clamp(-30, 30))
