#  Copyright (c) 2026 Forum Software, Inc. (YHat Labs). Licensed under the Apache License, Version 2.0.
"""Rolling-origin backtests: cut each series at several points of its own history, forecast from each cut, score against
what happened, and compare with simple baselines. Windows, metrics, baselines and the result follow the YHat Labs SDK's
Client.evaluate, so a backtest of this open model and of the hosted ChakraTS read the same way. Nothing after a cutoff
reaches that window's forecast, except known covariates (known in advance by definition).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any

import numpy as np

from yhatmodels.ts.covariates import rows as covariate_rows
from yhatmodels.ts.predictor import Series

DEFAULT_METRICS = ("mase", "sql", "wql", "mae", "rmse", "coverage@0.8")
ERROR_METRICS = ("mase", "sql", "wql", "mae", "rmse")
BASELINES = ("seasonal_naive", "naive")
DECILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)  # the levels every model and baseline is scored at
# Seasonal period by base frequency, as GluonTS / fev define it (MASE scaling); the seasonal-naive baseline repeats the cycle
# people see in such data instead: a week for daily data, a year for weekly data.
SEASONS = {"S": 3600, "s": 3600, "T": 1440, "min": 1440, "H": 24, "h": 24, "D": 1, "B": 5, "W": 1, "M": 12, "MS": 12, "ME": 12,
           "Q": 4, "QS": 4, "QE": 4, "A": 1, "Y": 1, "YS": 1, "YE": 1, "AS": 1}  # fmt: skip
BASELINE_SEASONS = {**SEASONS, "D": 7, "W": 52}


@dataclass
class EvaluationResult:
    """Backtest output. `scores`: one row per (model, series, [target], window) with each metric. `summary`: per model,
    the mean of each metric, plus `skill_<metric>` = 1 - geometric mean of the model's error relative to the baseline
    (higher is better; > 0 beats it). `forecasts`: every window's quantiles next to the actuals."""

    scores: Any
    summary: Any
    forecasts: Any
    baseline: str | None
    windows: int
    h: int
    warnings: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return f"EvaluationResult(models={list(self.summary.index)}, windows={self.windows}, h={self.h})"

    @property
    def calibration(self):
        """How honest the uncertainty is, per model, pooled over every scored point. kind "below": share of actuals
        below each quantile (a calibrated q10 has 10% below it); kind "interval": share inside each central interval.
        gap = observed - nominal: near 0 is calibrated; a negative interval gap means intervals too narrow."""
        return calibration_table(self.forecasts)

    def plot(self, ids=None, max_series: int = 4, model: str | None = None, ax=None):
        """Actuals across the scored windows, each window's forecast and its cutoff (needs `yhatmodels[plot]`)."""
        from yhatmodels.ts.plotting import plot_backtest

        return plot_backtest(self.forecasts, ids, max_series, model, ax)

    def plot_calibration(self, ax=None):
        """Share of actuals below each predicted quantile against its nominal level, one line per model."""
        from yhatmodels.ts.plotting import plot_calibration

        return plot_calibration(self.calibration, ax)


def season_for(freq: str | None, table: dict | None = None) -> int:
    if not freq:
        return 1
    import pandas as pd

    off = pd.tseries.frequencies.to_offset(freq)
    base = (table or SEASONS).get(off.rule_code.split("-")[0], 1)  # rule_code: "h", "min", "W-SUN", "MS", ...
    return max(1, base // off.n) if base > 1 else 1


def _ffill(x: np.ndarray) -> np.ndarray:
    out, last = x.copy(), np.nan
    for i, v in enumerate(out):
        if np.isnan(v):
            out[i] = last
        else:
            last = v
    return out


def baseline(history: np.ndarray, h: int, m: int, levels) -> np.ndarray:
    """(h, len(levels)) quantiles from a (seasonal) naive forecast with normal errors that widen with each season ahead."""
    y = _ffill(np.asarray(history, dtype=float))
    finite = y[np.isfinite(y)]
    if not len(finite):
        return np.full((h, len(levels)), np.nan)
    m = m if len(y) > m else 1
    last = y[-m:]
    point = np.array([last[t % m] for t in range(h)])
    point = np.where(np.isfinite(point), point, finite[-1])
    res = y[m:] - y[:-m]
    res = res[np.isfinite(res)]
    sigma = float(np.std(res)) if len(res) > 1 else 0.0
    sd = sigma * np.sqrt(np.ceil(np.arange(1, h + 1) / m))
    z = np.array([NormalDist().inv_cdf(q) for q in levels])
    return point[:, None] + sd[:, None] * z[None, :]


def scale_of(history: np.ndarray, m: int) -> float:
    """Mean absolute m-step change of the history: the MASE / SQL scale (NaN when the history repeats exactly)."""
    y = np.asarray(history, dtype=float)
    if len(y) <= m:
        m = 1
    d = np.abs(y[m:] - y[:-m])
    d = d[np.isfinite(d)]
    if not len(d):
        return float("nan")
    level = np.nanmean(np.abs(y)) if np.isfinite(y).any() else 0.0
    s = float(d.mean())
    return s if s > 1e-9 * max(1.0, level) else float("nan")


def score(y: np.ndarray, q: np.ndarray, levels, scale: float, metrics, strict: bool = True) -> dict[str, float]:
    """y (h,) with NaN = not observed, q (h, n) quantiles at `levels`. fev-style definitions."""
    ok = np.isfinite(y)
    y, q = y[ok], q[ok]
    if not len(y):
        return {m: float("nan") for m in metrics}
    lv = np.asarray(levels, dtype=float)
    err = y - q[:, int(np.argmin(np.abs(lv - 0.5)))]
    pin = np.maximum(lv * (y[:, None] - q), (lv - 1) * (y[:, None] - q))
    scaled = bool(scale) and math.isfinite(scale)
    out: dict[str, float] = {}
    for m in metrics:
        if m == "mae":
            out[m] = float(np.mean(np.abs(err)))
        elif m == "rmse":
            out[m] = float(np.sqrt(np.mean(err**2)))
        elif m == "mase":
            out[m] = float(np.mean(np.abs(err)) / scale) if scaled else float("nan")
        elif m == "wql":
            denom = np.sum(np.abs(y))
            out[m] = float(np.mean(2 * pin.sum(axis=0)) / denom) if denom > 0 else float("nan")
        elif m == "sql":
            out[m] = float(np.mean(2 * pin) / scale) if scaled else float("nan")
        elif m.startswith("coverage@"):
            width = float(m.split("@", 1)[1])
            width = width / 100 if width > 1 else width  # coverage@80 = coverage@0.8
            lo, hi = (1 - width) / 2, 1 - (1 - width) / 2
            il, ih = int(np.argmin(np.abs(lv - lo))), int(np.argmin(np.abs(lv - hi)))
            if abs(lv[il] - lo) > 1e-9 or abs(lv[ih] - hi) > 1e-9:
                if strict:
                    raise ValueError(f"{m} needs the {lo:g} and {hi:g} quantiles")
                out[m] = float("nan")
                continue
            out[m] = float(np.mean((y >= q[:, il]) & (y <= q[:, ih])))
        else:
            raise ValueError(f"unknown metric {m!r}; choose from {DEFAULT_METRICS} (coverage@<width> for any width)")
    return out


def summarize(scores, metrics, base: str | None):
    keys = [c for c in ("id", "target", "window") if c in scores.columns]
    summary = scores.groupby("model", sort=False)[list(metrics)].mean()
    if base is not None and base in summary.index:
        ref = scores[scores["model"] == base].set_index(keys)
        for m in metrics:
            if m not in ERROR_METRICS:
                continue
            floor = 1e-12 * max(1.0, float(scores[m].abs().max()))  # a baseline error of float noise makes the ratio meaningless
            col = []
            for model in summary.index:
                cur = scores[scores["model"] == model].set_index(keys)[m]
                den = ref[m].reindex(cur.index)
                r = (cur / den.where(den > floor)).replace([np.inf, -np.inf], np.nan).dropna()
                r = r[r > 0]
                col.append(float(1 - np.exp(np.mean(np.log(r)))) if len(r) else float("nan"))
            summary[f"skill_{m}"] = col
    return summary


@dataclass
class _Series:
    uid: Any
    y: np.ndarray  # (V, T)
    times: Any  # DatetimeIndex (T,) or None (arrays: integer steps)
    step: str | None
    past: dict | None  # name -> (T,)
    known: dict | None  # name -> (T,): over the history; a window reads them up to its cutoff + h


def _from_frame(data, past, known, id_col, time_col, target_col, freq):
    from yhatmodels.ts.forecaster import parse_frame

    fr = parse_frame(data, past, known, None, id_col, time_col, target_col, freq)
    out = [_Series(s.item, s.hist[fr.targets].to_numpy(np.float32).T, s.hist.index, s.step,
                   {c: s.hist[c].to_numpy() for c in fr.past} or None, {c: s.hist[c].to_numpy() for c in fr.known} or None)
           for s in fr.series]  # fmt: skip
    return out, fr.id_col or "id", fr.time_col, fr.targets


def _from_arrays(data, past, known, freq):
    from yhatmodels.ts.forecaster import _as_targets, _one

    targets, shape = _as_targets(data)
    if shape == "single":
        past, known = _one(past), _one(known)
    n = len(targets)
    past, known = past or [None] * n, known or [None] * n
    if len(past) != n or len(known) != n:
        raise ValueError(f"past / known need one entry per series ({n} series)")
    out = []
    for i, (y, pc, kc) in enumerate(zip(targets, past, known, strict=True)):
        for kind, cov in (("past", pc), ("known", kc)):
            for name, v in (cov or {}).items():
                if len(v) != y.shape[1]:
                    raise ValueError(
                        f"series {i}: {kind} covariate {name!r} has {len(v)} values; a backtest needs the history's {y.shape[1]}"
                    )
        out.append(_Series(i, y, None, freq, pc, kc))
    rows = max(y.shape[0] for y in targets)
    return out, "id", "step", [f"y{v}" for v in range(rows)] if rows > 1 else ["y"]


def _compare_table(f, id_col: str, time_col: str):
    """Your own forecasts: quantile columns (median, q10, q90, ...) or one point column (forecast, yhat, prediction)."""
    from yhatmodels.ts.forecaster import qname

    levels = [lv / 100 for lv in range(1, 100)]
    qcols = {qname(lv): lv for lv in levels if qname(lv) in f.columns}
    if not qcols:
        point = next((c for c in ("forecast", "yhat", "y_hat", "prediction", "value") if c in f.columns), None)
        if point is None:
            raise ValueError(
                f"compare forecasts need quantile columns (median, q10, ...) or a point column 'forecast'; columns: {list(f.columns)}"
            )
        qcols, f = {"median": 0.5}, f.assign(median=f[point])
    names = sorted(qcols, key=lambda c: qcols[c])
    import pandas as pd

    t = pd.to_datetime(f[time_col])
    table = {(u, ts): row for u, ts, row in zip(f[id_col], t, f[names].to_numpy(dtype=float), strict=True)}
    return table, [qcols[c] for c in names]


def run(model, data, h: int, *, windows: int, step: int | None, baselines, metrics, compare, season, preset, group: int,
        past, known, id_col, time_col, target_col, freq) -> EvaluationResult:  # fmt: skip
    import pandas as pd

    from yhatmodels.ts.forecaster import _check_horizon, qname

    h = _check_horizon(h)
    if windows < 1:
        raise ValueError("windows must be at least 1")
    step = step or h
    if unknown := [b for b in baselines if b not in BASELINES]:
        raise ValueError(f"unknown baselines {unknown}; available: {BASELINES}")
    framed = hasattr(data, "columns")
    if framed:
        series, id_name, time_name, target_names = _from_frame(data, past, known, id_col, time_col, target_col, freq)
    else:
        if compare:
            raise ValueError("compare= needs dated data (a DataFrame), to match your forecasts by id and time")
        series, id_name, time_name, target_names = _from_arrays(data, past, known, freq)
    name = model.info.get("name") or "ChakraTS"
    pred = model.predictor(preset)
    rows, frames, warns = [], [], []
    skipped = short = 0

    def add(who, s, w, c, per_target, lv, strict=True):
        m = season or season_for(s.step)
        for v, q in enumerate(per_target):
            y = s.y[v, c : c + h].astype(float)
            q = np.sort(np.asarray(q, dtype=float)[: len(y)], axis=1)
            sc = score(y, q, lv, scale_of(s.y[v, :c], m), metrics, strict)
            cutoff = s.times[c - 1] if s.times is not None else c - 1
            row = {"model": who, "id": s.uid, "window": w, "cutoff": cutoff, **sc}
            if s.y.shape[0] > 1:
                row["target"] = target_names[v]
            rows.append(row)
            f = pd.DataFrame(q, columns=[qname(x) for x in lv])
            f.insert(0, "actual", y)
            f.insert(0, time_name, s.times[c : c + len(y)] if s.times is not None else np.arange(c, c + len(y)))
            if s.y.shape[0] > 1:
                f.insert(0, "target", target_names[v])
            for col, val in (("cutoff", cutoff), ("window", w), (id_name, s.uid), ("model", who)):
                f.insert(0, col, val)
            frames.append(f)

    compares = {k: _compare_table(f if id_name in f.columns or len(series) != 1 else f.assign(**{id_name: series[0].uid}), id_name, time_name)
                for k, f in (compare or {}).items()}  # fmt: skip
    for w in range(windows):
        todo = []
        for s in series:
            m = season or season_for(s.step)
            T = s.y.shape[1]
            c = T - h - (windows - 1 - w) * step
            if c < (max(2 * m, 8) if m > 1 else 8) or not np.isfinite(s.y[:, :c]).any() or not np.isfinite(s.y[:, c : c + h]).any():
                skipped += 1
                continue
            pc = {k: v[:c] for k, v in s.past.items()} if s.past else None
            kc = {k: v[: c + h] for k, v in s.known.items()} if s.known else None
            tgt = s.y[:, :c]
            todo.append((s, c, Series(tgt, covariate_rows(pc, tgt), covariate_rows(kc, tgt))))
        if not todo:
            continue
        # one window's series together (grouped series must share a cutoff: another window reaches past this one's)
        for (s, c, _), q in zip(todo, model._forecast_series([x for _, _, x in todo], h, DECILES, pred, group), strict=True):
            add(name, s, w, c, q, DECILES)
        for b in baselines:
            for s, c, _ in todo:
                bm = 1 if b == "naive" else (season or season_for(s.step, BASELINE_SEASONS))
                if bm > 1 and c < 2 * bm:  # under two seasons of history: repeating one season is noise
                    bm, short = 1, short + 1
                add(b, s, w, c, [baseline(s.y[v, :c], h, bm, DECILES) for v in range(s.y.shape[0])], DECILES)
        for k, (table, clv) in compares.items():
            if len(target_names) > 1:
                raise ValueError("compare= supports single-target series")
            for s, c, _ in todo:
                got = [table.get((s.uid, t)) for t in s.times[c : c + h]]
                if all(g is not None for g in got):
                    add(k, s, w, c, [np.vstack(got)], clv, strict=False)
    if not rows:
        raise ValueError(f"no series is long enough for {windows} windows of h={h}; use fewer windows, a smaller h or step")
    if skipped:
        warns.append(f"skipped {skipped} (series, window) pairs with too little history or no observed values")
    if short:
        warns.append(f"seasonal_naive repeated the last value for {short} series windows: their history is under two seasons")
    scores = pd.DataFrame(rows)
    base = baselines[0] if baselines else None
    return EvaluationResult(scores, summarize(scores, metrics, base), pd.concat(frames, ignore_index=True), base, windows, h, warns)


def calibration_table(forecasts):
    import pandas as pd

    from yhatmodels.ts.forecaster import qname

    cols = ["model", "kind", "nominal", "observed", "gap", "n"]
    if forecasts is None or not len(forecasts):
        return pd.DataFrame(columns=cols)
    present = {qname(lv / 100): lv / 100 for lv in range(1, 100) if qname(lv / 100) in forecasts.columns}
    rows = []
    for model, f in forecasts.groupby("model", sort=False):
        y = f["actual"].to_numpy(dtype=float)
        have = {c: lv for c, lv in present.items() if f[c].notna().any()}
        ok = np.isfinite(y)
        for c, lv in sorted(have.items(), key=lambda kv: kv[1]):
            q = f[c].to_numpy(dtype=float)
            m = ok & np.isfinite(q)
            if m.any():
                obs = float(np.mean(y[m] < q[m]))
                rows.append({"model": model, "kind": "below", "nominal": lv, "observed": obs, "gap": obs - lv, "n": int(m.sum())})
        by_level = {lv: c for c, lv in have.items()}
        for lo in sorted(lv for lv in by_level if lv < 0.5):
            hi = round(1 - lo, 6)
            if hi in by_level:
                a, b = f[by_level[lo]].to_numpy(dtype=float), f[by_level[hi]].to_numpy(dtype=float)
                m = ok & np.isfinite(a) & np.isfinite(b)
                if m.any():
                    obs, nom = float(np.mean((y[m] >= a[m]) & (y[m] <= b[m]))), round(hi - lo, 6)
                    rows.append({"model": model, "kind": "interval", "nominal": nom, "observed": obs, "gap": obs - nom, "n": int(m.sum())})
    return pd.DataFrame(rows, columns=cols)
