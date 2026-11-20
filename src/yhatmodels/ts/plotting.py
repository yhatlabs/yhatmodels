#  Copyright (c) 2026 Forum Software, Inc. (YHat Labs). Licensed under the Apache License, Version 2.0.
"""Plots (`pip install "yhatmodels[plot]"`), the same as the YHat Labs SDK's:

plot_forecast(history, q)            one series' history and its forecast's median and intervals (arrays)
plot_forecast(fc, history=df)        a forecast DataFrame, one panel per series (up to max_series), after its history
EvaluationResult.plot()              a backtest: actuals, each window's forecast and its cutoff
EvaluationResult.plot_calibration()  share of actuals below each predicted quantile, against the nominal level

Every central interval present is shaded (widest lightest), the median is a line. Nothing global changes (no style, no
rcParams); the Axes (arrays) or Figure (DataFrames) is returned for you to adjust or save.
"""

from __future__ import annotations

import numpy as np

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)  # as ChakraTS.forecast


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError('plots need matplotlib: pip install "yhatmodels[plot]"') from e
    return plt


def plot_forecast(
    data,
    forecast=None,
    quantiles=DEFAULT_QUANTILES,
    *,
    history=None,
    ids=None,
    max_series: int = 4,
    target: int = 0,
    timestamps=None,
    future_timestamps=None,
    show_last: int | None = 200,
    ax=None,
    title: str | None = None,
):
    """Draw a forecast's median and every central interval its quantiles span, after the recent history.

    Arrays: plot_forecast(history, forecast): history (T,) or (V, T), forecast ChakraTS.forecast()'s arrays for it, (h, Q)
        or (V, h, Q), at the levels `quantiles` (as passed to forecast()); target: the row of a multivariate series;
        timestamps / future_timestamps: x values (default: step numbers); show_last: history steps shown (None: all).
        Returns the Axes.
    DataFrame: plot_forecast(fc, history=df): fc a forecast DataFrame from ChakraTS.forecast() (id and time columns, then
        median, q10, ...); history: the data it was forecast from (same id and time columns), optional; ids: the series
        to draw (default: the first max_series). Returns the Figure.
    """
    if hasattr(data, "columns"):
        return _plot_frame(data, history, ids, max_series, ax)
    if forecast is None:
        raise TypeError("plot_forecast(history, forecast) for arrays, or plot_forecast(fc_dataframe, history=df)")
    plt = _plt()
    y = np.atleast_2d(np.asarray(data, dtype=float))[target]
    q = np.asarray(forecast, dtype=float)
    q = q[target] if q.ndim == 3 else q
    levels = sorted(float(v) for v in quantiles)  # forecast() returns the levels in increasing order
    if q.ndim != 2 or q.shape[1] != len(levels):
        raise ValueError(f"forecast must be (h, {len(levels)}) for these quantiles; got shape {np.shape(forecast)}")
    n, h = len(y), q.shape[0]
    x_hist = np.arange(n) if timestamps is None else np.asarray(timestamps)
    x_fut = np.arange(n, n + h) if future_timestamps is None else np.asarray(future_timestamps)
    if show_last is not None:
        x_hist, y = x_hist[-show_last:], y[-show_last:]
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x_hist, y, color="0.2", linewidth=1.1, label="history")

    def index(level: float) -> int | None:
        return next((k for k, v in enumerate(levels) if abs(v - level) < 1e-9), None)

    pairs = sorted({(lo, 1 - lo) for lo in levels if lo < 0.5 and index(1 - lo) is not None})  # widest first
    for k, (lo, hi) in enumerate(pairs):
        alpha = 0.12 + 0.28 * (k + 1) / len(pairs)
        ax.fill_between(
            x_fut, q[:, index(lo)], q[:, index(hi)], color="C0", alpha=alpha, linewidth=0, label=f"{round((hi - lo) * 100)}% interval"
        )
    if (mid := index(0.5)) is not None:
        ax.plot(x_fut, q[:, mid], color="C0", linewidth=1.6, label="median")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    if title:
        ax.set_title(title)
    return ax


# --- DataFrames: forecasts, backtests, calibration (as the YHat Labs SDK draws them) ------------------------------------------
def _level(c) -> float | None:
    """'median' -> 0.5, 'q10' -> 0.1, 'q2.5' -> 0.025; other columns -> None."""
    c = str(c)
    if c == "median":
        return 0.5
    if c.startswith("q"):
        try:
            return float(c[1:]) / 100
        except ValueError:
            return None
    return None


def _bands(cols) -> list[tuple[str, str, float]]:
    """Symmetric quantile column pairs present, widest first: [("q10", "q90", 80.0), ("q25", "q75", 50.0)]."""
    lv = {c: v for c in cols if (v := _level(c)) is not None}
    out = []
    for c, v in lv.items():
        if v < 0.5 and (hi := next((d for d, w in lv.items() if abs(w - (1 - v)) < 1e-9), None)):
            out.append((c, hi, round((1 - 2 * v) * 100, 6)))
    return sorted(out, key=lambda t: -t[2])


def _axes(n: int, ax, height: float = 2.6) -> list:
    if ax is not None:
        return [ax] * n
    _, axes = _plt().subplots(n, 1, figsize=(10, height * n), squeeze=False, layout="constrained")
    return [a[0] for a in axes]


def _series_bands(a, f, x, color=None, label: str = "median") -> None:
    for i, (lo, hi, w) in enumerate(_bands(f.columns)):
        a.fill_between(x, f[lo], f[hi], alpha=0.15 + 0.12 * i, color=color, lw=0, label=f"{w:g}% interval")
    if "median" in f.columns:
        a.plot(x, f["median"], lw=1.5, color=color, label=label)


def _roles(df, extra=()) -> tuple[str | None, str]:
    """The id and time columns of a forecast or backtest frame: the columns before the quantiles, minus `extra`."""
    meta = [c for c in df.columns if _level(c) is None and c not in ("target", *extra)]
    return (meta[0], meta[1]) if len(meta) >= 2 else (None, meta[0])


def _history(a, history, id_col, time_col, uid, n: int) -> None:
    if history is None or time_col not in history.columns:
        return
    h = history if id_col is None or id_col not in history.columns else history[history[id_col] == uid]
    vcol = next((c for c in h.columns if c not in (id_col, time_col) and h[c].dtype.kind in "fiu"), None)
    if vcol is None:
        return
    import pandas as pd

    h = h.assign(**{time_col: pd.to_datetime(h[time_col])}).sort_values(time_col).tail(n)
    a.plot(h[time_col], h[vcol], color="0.3", lw=1, label="history")


def _finish(axes):
    """Compact date labels on time axes; returns the figure."""
    import matplotlib.dates as mdates

    for a in dict.fromkeys(axes):
        if isinstance(a.xaxis.get_major_formatter(), (mdates.AutoDateFormatter, mdates.ConciseDateFormatter)):
            loc = mdates.AutoDateLocator()
            a.xaxis.set_major_locator(loc)
            a.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    return axes[0].figure


def _plot_frame(df, history, ids, max_series: int, ax):
    import pandas as pd

    _plt()
    id_col, time_col = _roles(df)
    ids = [None] if id_col is None else (list(ids) if ids is not None else list(pd.unique(df[id_col]))[:max_series])
    axes = _axes(len(ids), ax)
    for a, uid in zip(axes, ids, strict=True):
        f = df if id_col is None else df[df[id_col] == uid]
        if "target" in f.columns:
            f = f[f["target"] == f["target"].iloc[0]]
        _history(a, history, id_col, time_col, uid, max(3 * len(f), 50))
        _series_bands(a, f, f[time_col])
        if uid is not None:
            a.set_title(str(uid), fontsize=10)
        a.legend(loc="upper left", fontsize=8)
    return _finish(axes)


def plot_backtest(forecasts, ids=None, max_series: int = 4, model: str | None = None, ax=None):
    """Actuals across the scored windows, each window's forecast (median and intervals) and a dotted line at every cutoff."""
    import pandas as pd

    _plt()
    if forecasts is None or not len(forecasts):
        raise ValueError("this backtest kept no forecasts to plot")
    model = model or forecasts["model"].iloc[0]
    f = forecasts[forecasts["model"] == model]
    id_col, time_col = _roles(f, ("model", "window", "cutoff", "actual"))
    ids = [None] if id_col is None else (list(ids) if ids is not None else list(pd.unique(f[id_col]))[:max_series])
    axes = _axes(len(ids), ax)
    for a, uid in zip(axes, ids, strict=True):
        g = f if id_col is None else f[f[id_col] == uid]
        if "target" in g.columns:
            g = g[g["target"] == g["target"].iloc[0]]
        actual = g.drop_duplicates(time_col).sort_values(time_col)
        a.plot(actual[time_col], actual["actual"], color="0.2", lw=1, marker=".", ms=3, label="actual")
        for i, (_, w) in enumerate(g.groupby("window", sort=True)):
            _series_bands(a, w, w[time_col], color="C0", label="median" if i == 0 else "_median")
            a.axvline(w["cutoff"].iloc[0], color="0.7", lw=0.8, ls=":")
        handles, labels = a.get_legend_handles_labels()
        seen = dict(zip(labels, handles, strict=True))
        a.legend(seen.values(), seen.keys(), loc="upper left", fontsize=8)
        a.set_title(f"{uid} ({model})" if uid is not None else str(model), fontsize=10)
    return _finish(axes)


def plot_calibration(table, ax=None):
    """Reliability plot: nominal level against the observed share of actuals below each quantile, one line per model."""
    if ax is None:
        _, ax = _plt().subplots(figsize=(4.5, 4.5), layout="constrained")
    ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--", label="calibrated")
    for model, c in table[table["kind"] == "below"].groupby("model", sort=False):
        ax.plot(c["nominal"], c["observed"], marker="o", ms=3, label=str(model))
    ax.set_xlabel("predicted quantile level")
    ax.set_ylabel("share of actuals below it")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    return ax.figure
