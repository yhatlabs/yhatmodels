#  Copyright (c) 2026 Forum Software, Inc. (YHat Labs). Licensed under the Apache License, Version 2.0.
"""The README's results chart: open forecasting models' fev-bench skill score against their size, as of November 2026
(docs/benchmarks.csv: skill scores computed as the fev-bench leaderboard does, from the result files in the fev
repository; parameter counts from the models' Hugging Face pages). Writes docs/assets/results_light.png and
results_dark.png.

    pip install "yhatmodels[plot]"
    python docs/make_chart.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
OURS, SMALL = "ChakraTS-OS-10M", 30e6
rows = [(r["model"], float(r["params"]), float(r["fev_skill"])) for r in csv.DictReader(open(HERE / "benchmarks.csv"))]
LABELS = {OURS: (10, -5), "TimeH2O-v1-tiny": (-34, -15), "Timestral-5M": (-70, 3), "Timestral-28M": (8, -4), "Chronos-2": (8, -4),
          "t0-gamma": (6, 5), "TimesFM-3": (8, -4), "Simons-297M": (8, -9), "Toto-2.0-2.5B": (-48, 8)}  # fmt: skip
THEMES = {
    "light": {"bg": "#fbfaf7", "ink": "#2b2621", "muted": "#8a8178", "grid": "#ece7de", "ours": "#8a5a2b", "small": "#d6a26b", "large": "#c4bdb3"},
    "dark": {"bg": "#16140f", "ink": "#ece6dc", "muted": "#9c948a", "grid": "#26231d", "ours": "#e0a86a", "small": "#9c6e44", "large": "#57514a"},
}  # fmt: skip

for name, c in THEMES.items():
    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=160)
    fig.patch.set_facecolor(c["bg"])
    ax.set_facecolor(c["bg"])
    for group in ("large", "small", "ours"):  # ours on top
        for m, p, v in rows:
            g = "ours" if m == OURS else ("small" if p <= SMALL else "large")
            if g != group:
                continue
            ax.scatter(p, v, s={"ours": 130, "small": 34, "large": 26}[g], color=c[g], edgecolor=c["bg"], linewidth=0.8, zorder=3)
            if m in LABELS:
                ours = g == "ours"
                ax.annotate(m, (p, v), xytext=LABELS[m], textcoords="offset points", color=c["ink"] if ours else c["muted"],
                            fontsize=10.5 if ours else 8.5, fontweight="bold" if ours else "normal", zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(3e6, 4e9)
    ax.set_ylim(32, 53)
    ax.set_xlabel("model size (parameters)", color=c["muted"], fontsize=9.5)
    ax.set_ylabel("fev-bench skill score (%)", color=c["muted"], fontsize=9.5)
    ax.tick_params(colors=c["muted"], labelsize=9)
    ax.grid(True, color=c["grid"], lw=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(c["grid"])
    handles = [Line2D([], [], marker="o", ls="", markersize=s, color=c[g], markeredgecolor=c["bg"], label=lab)
               for g, s, lab in (("ours", 9, OURS), ("small", 6, "open models up to 30M parameters"), ("large", 6, "larger open models"))]  # fmt: skip
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9, labelcolor=c["ink"])
    fig.tight_layout()
    fig.savefig(HERE / "assets" / f"results_{name}.png", facecolor=c["bg"])
    plt.close(fig)
print("wrote results_light.png, results_dark.png")
