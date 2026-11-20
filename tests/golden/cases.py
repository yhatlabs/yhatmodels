#  Copyright (c) 2026 Forum Software, Inc. (YHat Labs). Licensed under the Apache License, Version 2.0.
"""Fixed inputs for the golden test: built from a seeded generator, so only the expected outputs are stored."""

import numpy as np

LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def cases():
    r = np.random.default_rng(20261119)
    t = np.arange(1500)
    uni = [(30 + 8 * np.sin(2 * np.pi * t[:n] / 24) + r.normal(0, 1.5, n)).astype(np.float32) for n in (60, 300, 1500)]
    uni.append(r.poisson(3, 400).astype(np.float32))  # counts: clipped at 0 and rounded
    gap = uni[2].copy()
    gap[100:160] = np.nan
    uni.append(gap)
    mv = np.stack([np.sin(t[:700] / 9), np.cos(t[:700] / 13) * 3 + 10]).astype(np.float32)
    T, H = 500, 24
    y = (50 + 10 * np.sin(t[:T] / 11) + r.normal(0, 2, T)).astype(np.float32)
    cov = {"past": [{"temp": r.normal(20, 5, T).astype(np.float32)}],
           "known": [{"price": r.normal(3, 0.3, T + H).astype(np.float32), "promo": np.array(["no", "yes", "no", "holiday"] * 131, dtype=object)[: T + H]}]}  # fmt: skip
    out = []
    for preset in ("fast", "pro"):
        out.append((f"univariate_{preset}", {"data": uni, "h": 24, "preset": preset}))
        out.append((f"grouped_{preset}", {"data": uni[:3], "h": 24, "preset": preset, "cross_series": 16}))
        out.append((f"multivariate_{preset}", {"data": [mv], "h": 36, "preset": preset}))
        out.append((f"covariates_{preset}", {"data": [y], "h": H, "preset": preset, **cov}))
    out.append(("long_horizon", {"data": uni[2], "h": 600}))
    return out


def run(model):
    """case name -> forecasts stacked into one array per case."""
    res = {}
    for name, kw in cases():
        q = model.forecast(quantiles=LEVELS, **kw)
        res[name] = np.concatenate([np.asarray(x).ravel() for x in q]) if isinstance(q, list) else np.asarray(q).ravel()
    return res
