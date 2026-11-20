#  Copyright (c) 2026 Forum Software, Inc. (YHat Labs). Licensed under the Apache License, Version 2.0.
"""Covariate rows: numeric values pass through; categorical ones (text, objects) become numbers.

A categorical covariate of a single-target series is target-encoded: each category becomes the smoothed mean of the
target over the context where it occurred, so the model reads a magnitude that means something. With several targets
there is no single target to encode against, and the category codes are passed as they are.
"""

from __future__ import annotations

import numpy as np


def _missing(x) -> bool:
    return x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)) or type(x).__name__ in ("NAType", "NaTType")


def is_categorical(values: np.ndarray) -> bool:
    """Text (or any non-number) values; numbers with gaps (None, NaN, pandas NA) in an object array are numeric."""
    a = np.asarray(values)
    if a.dtype.kind in "US":
        return True
    if a.dtype.kind != "O":
        return False  # numbers and bools
    return any(not (_missing(x) or isinstance(x, (int, float, bool, np.number, np.bool_))) for x in a.ravel())


def numeric(values) -> np.ndarray:
    a = np.asarray(values)
    if a.dtype.kind == "O":
        return np.array([np.nan if _missing(x) else float(x) for x in a.ravel()], np.float32).reshape(a.shape)
    return a.astype(np.float32)


def codes(values: np.ndarray) -> np.ndarray:
    """Category codes 0..k-1 in sorted order of the distinct values, NaN for missing (None / NaN)."""
    v = np.asarray(values, dtype=object)
    missing = np.array([x is None or (isinstance(x, float) and np.isnan(x)) for x in v])
    cats = sorted({str(x) for x in v[~missing]})
    index = {c: i for i, c in enumerate(cats)}
    return np.array([np.nan if m else index[str(x)] for x, m in zip(v, missing, strict=True)], np.float32)


def target_encode(target: np.ndarray, codes: np.ndarray, smooth: float = 1.0) -> np.ndarray:
    """Category code -> (smooth * series mean + category sum) / (smooth + category count) over the context; NaN codes are
    a category of their own; codes never seen in the context -> the series mean. `codes` may run past the context
    (known-future rows): only its first len(target) steps set the means."""
    t = len(target)
    obs = np.isfinite(target)
    y = np.where(obs, target, 0.0).astype(np.float64)
    mean = y.sum() / obs.sum() if obs.any() else 0.0
    key = np.where(np.isnan(codes), -1.0, codes)  # codes are >= 0: -1 = the NaN category
    cats, inv = np.unique(key[:t], return_inverse=True)
    sums = np.bincount(inv, weights=y, minlength=len(cats))
    counts = np.bincount(inv, weights=obs.astype(np.float64), minlength=len(cats))
    lookup = (smooth * mean + sums) / (smooth + counts)
    idx = np.clip(np.searchsorted(cats, key), 0, len(cats) - 1)
    return np.where(cats[idx] == key, lookup[idx], mean).astype(np.float32)


def rows(covariates: dict | np.ndarray | None, target: np.ndarray, encode: bool = True) -> np.ndarray | None:
    """A series' covariates (dict name -> 1-D array, or an array (k, T)) -> (k, T) float32 rows."""
    if covariates is None:
        return None
    items = list(covariates.values()) if isinstance(covariates, dict) else list(np.atleast_2d(covariates))
    if not items:
        return None
    single = np.atleast_2d(target).shape[0] == 1
    out = []
    for v in items:
        if is_categorical(v):
            c = codes(v)
            out.append(target_encode(np.atleast_2d(target)[0], c) if encode and single else c)
        else:
            out.append(numeric(v))
    return np.stack(out)
