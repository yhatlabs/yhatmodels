#  Copyright (c) 2026 Forum Software, Inc. (YHat Labs). Licensed under the Apache License, Version 2.0.
"""The README's figures, in light and dark versions (docs/assets/):

forecast_*.png    a week of hourly electricity demand on the Texas Gulf Coast (ERCOT's COAST weather zone, public ERCOT
                  data), forecast zero-shot from two weeks of history, next to what happened.
covariates_*.png  two days of solar output forecast from history alone and with sunlight and cloud cover as known
                  covariates (the "Renewable Energy and Weather Conditions" dataset, CC0, measured weather standing in
                  for a weather forecast), next to what happened.
tables_*.png      four ERCOT regions forecast three days ahead in one call, with 80% intervals.
backtest_*.png    res.plot() and res.plot_calibration() for a 4-window backtest of the same regions, recoloured.

Both datasets as packaged in autogluon/fev_datasets ("ercot_1H", "solar_with_weather_1H").

    pip install "yhatmodels[plot]" datasets
    python docs/make_figures.py
"""

from pathlib import Path

import datasets
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from yhatmodels import ChakraTS

OUT = Path(__file__).parent / "assets"
THEMES = {
    "light": {"bg": "#fbfaf7", "ink": "#2b2621", "muted": "#8a8178", "grid": "#e6e1d8", "fc": "#8a5a2b"},
    "dark": {"bg": "#16140f", "ink": "#ece6dc", "muted": "#9c948a", "grid": "#2c2822", "fc": "#e0a86a"},
}
model = ChakraTS.from_pretrained("ChakraTS-OS-10M")


def series(config: str, item: str | None = None) -> dict:
    ds = datasets.load_dataset("autogluon/fev_datasets", config, split="train")
    return next(r for r in ds if item is None or r["id"] == item)


def draw(ax, c, history, actual, fc, ylabel, title=None, legend=True):
    ax.set_facecolor(c["bg"])
    ax.plot(history.index, history.values, color=c["ink"], lw=1.1, label="history")
    t = pd.DatetimeIndex(fc["ds"])
    if "q5" in fc:
        ax.fill_between(t, fc["q5"], fc["q95"], color=c["fc"], alpha=0.16, lw=0, label="90% interval")
    ax.fill_between(t, fc["q10"], fc["q90"], color=c["fc"], alpha=0.28, lw=0, label="80% interval")
    ax.plot(t, fc["median"], color=c["fc"], lw=1.8, label="forecast")
    ax.plot(actual.index, actual.values, color=c["ink"], lw=1, ls=(0, (2, 2)), label="what happened")
    ax.axvline(actual.index[0], color=c["muted"], lw=0.8, ls=":")
    ax.set_ylabel(ylabel, color=c["muted"], fontsize=10)
    if title:
        ax.set_title(title, color=c["ink"], fontsize=11, loc="left")
    ax.tick_params(colors=c["muted"], labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["grid"])
    ax.grid(axis="y", color=c["grid"], lw=0.8)
    loc = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    if legend:
        ax.legend(loc="upper left", ncol=5, fontsize=9, frameon=False, labelcolor=c["ink"])


# 1. A week of electricity demand, from history alone
r = series("ercot_1H", "COAST")
s = pd.Series(np.asarray(r["target"], dtype=float), index=pd.DatetimeIndex(r["timestamp"])) / 1000  # MW -> GW
cut, h = pd.Timestamp("2021-07-12 00:00"), 7 * 24
history, actual = s[s.index < cut], s[s.index >= cut].iloc[:h]
fc = model.forecast(history, h=h, quantiles=[0.05, 0.1, 0.5, 0.9, 0.95])  # a dated Series in, a DataFrame out
for name, c in THEMES.items():
    fig, ax = plt.subplots(figsize=(11, 3.9), dpi=160)
    fig.patch.set_facecolor(c["bg"])
    draw(ax, c, history.iloc[-14 * 24 :], actual, fc, "electricity demand (GW)")
    fig.tight_layout()
    fig.savefig(OUT / f"forecast_{name}.png", facecolor=c["bg"])
    plt.close(fig)

# 2. Two days of solar output, from history alone and with sunlight and cloud cover
r = series("solar_with_weather_1H")
t = pd.DatetimeIndex(r["timestamp"])
s = pd.Series(np.asarray(r["target"], dtype=float), index=t) / 1000  # Wh per hour -> kWh
weather = pd.DataFrame({k: np.asarray(r[k], dtype=float) for k in ("global_horizontal_irradiance", "clouds_all")}, index=t)
cut, h, ctx = pd.Timestamp("2022-08-19 00:00"), 48, 2048
history, actual = s[s.index < cut].iloc[-ctx:], s[s.index >= cut].iloc[:h]
frame = pd.DataFrame({"ds": t, "y": s.to_numpy(), "sunlight": weather["global_horizontal_irradiance"].to_numpy(),
                      "clouds": weather["clouds_all"].to_numpy()}).set_index("ds").loc[history.index[0] : actual.index[-1]].reset_index()  # fmt: skip
frame.loc[frame["ds"] >= cut, "y"] = np.nan  # the next two days: weather known, output not yet
plain = model.forecast(history, h=h, quantiles=[0.1, 0.5, 0.9])
informed = model.forecast(frame, h=h, quantiles=[0.1, 0.5, 0.9], known=["sunlight", "clouds"])
for name, c in THEMES.items():
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), dpi=160, sharey=True)
    fig.patch.set_facecolor(c["bg"])
    draw(axes[0], c, history.iloc[-72:], actual, plain, "solar output (kWh)", "From history alone", legend=False)
    draw(axes[1], c, history.iloc[-72:], actual, informed, "", "With sunlight and cloud cover", legend=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9, frameon=False, labelcolor=c["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / f"covariates_{name}.png", facecolor=c["bg"])
    plt.close(fig)

# 3. Several series in one table: four ERCOT regions, three days ahead, one call
ZONES = {"COAST": "Gulf Coast", "NORTH_C": "North Central", "SOUTHERN": "South", "FAR_WEST": "Far West"}
ercot = datasets.load_dataset("autogluon/fev_datasets", "ercot_1H", split="train")
cut, h = pd.Timestamp("2021-07-12 00:00"), 72
long = pd.concat([pd.DataFrame({"unique_id": z, "ds": pd.DatetimeIndex(r["timestamp"]), "y": np.asarray(r["target"], dtype=float) / 1000})
                  for r in ercot if (z := r["id"]) in ZONES], ignore_index=True)  # fmt: skip
past, future = long[long.ds < cut], long[(long.ds >= cut) & (long.ds < cut + pd.Timedelta(hours=h))]
fc = model.forecast(past.groupby("unique_id").tail(4096), h=h, level=[80])  # one call for every region
for name, c in THEMES.items():
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.4), dpi=160)
    fig.patch.set_facecolor(c["bg"])
    for ax, (z, label) in zip(axes.flat, ZONES.items(), strict=True):
        hz = past[past.unique_id == z].set_index("ds")["y"].iloc[-5 * 24 :]
        az = future[future.unique_id == z].set_index("ds")["y"]
        draw(ax, c, hz, az, fc[fc.unique_id == z], "GW" if z in ("COAST", "SOUTHERN") else "", label, legend=False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9, frameon=False, labelcolor=c["ink"])
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / f"tables_{name}.png", facecolor=c["bg"])
    plt.close(fig)

# 4. A backtest and its calibration, as res.plot() and res.plot_calibration() draw them
res = model.evaluate(past.groupby("unique_id").tail(4096), h=24, windows=4)
for name, c in THEMES.items():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), dpi=160, gridspec_kw={"width_ratios": [2.3, 1]})
    res.plot(ids=["COAST"], ax=a1)
    res.plot_calibration(ax=a2)
    # the library's plots, recoloured for the page's theme
    for coll in a1.collections:
        coll.set_facecolor(c["fc"])
    for line in a1.lines:
        line.set_color(c["ink"] if line.get_label() == "actual" else c["fc"] if "median" in line.get_label() else c["muted"])
    for line in a2.lines:
        line.set_color({"calibrated": c["muted"], "seasonal_naive": c["muted"]}.get(line.get_label(), c["fc"]))
        if line.get_label() == "seasonal_naive":
            line.set_linestyle(":")
    a1.set_title("")
    a1.set_title("res.plot(): four past days, forecast and compared", color=c["ink"], fontsize=10, loc="left")
    a2.set_title("res.plot_calibration()", color=c["ink"], fontsize=10, loc="left")
    fig.patch.set_facecolor(c["bg"])
    for ax in (a1, a2):
        ax.set_facecolor(c["bg"])
        ax.tick_params(colors=c["muted"], labelsize=8.5)
        ax.xaxis.label.set_color(c["muted"])
        ax.yaxis.label.set_color(c["muted"])
        for sp in ax.spines.values():
            sp.set_color(c["grid"])
        handles, labels = ax.get_legend_handles_labels()
        seen = dict(zip(labels, handles, strict=True))
        ax.legend(seen.values(), seen.keys(), loc="upper left", fontsize=8, frameon=False, labelcolor=c["ink"])
    fig.tight_layout()
    fig.savefig(OUT / f"backtest_{name}.png", facecolor=c["bg"])
    plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))
