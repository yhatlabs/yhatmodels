#  Copyright (c) 2026 Forum Software, Inc. (YHat Labs). Licensed under the Apache License, Version 2.0.
"""ChakraTS: probabilistic forecasts from a pretrained ChakraTS checkpoint.

from yhatmodels import ChakraTS

model = ChakraTS.from_pretrained("ChakraTS-OS-10M")
q = model.forecast(history, h=24)  # (24, 3): quantiles 0.1, 0.5, 0.9
fc = model.forecast(df, h=24)      # a long DataFrame in, a long DataFrame out (median, q10, q90)

The arguments follow the YHat Labs SDK (yhatlabs.Client.forecast), so moving between this open model and the hosted
ChakraTS is a one-line change.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from yhatmodels.loading import load
from yhatmodels.ts.architectures.chakrats_v1 import ChakraTSNet
from yhatmodels.ts.covariates import rows as covariate_rows
from yhatmodels.ts.predictor import Predictor, Series

DEFAULT_QUANTILES = (0.1, 0.5, 0.9)  # names and defaults follow the YHat Labs SDK (yhatlabs.Client.forecast)
CROSS_SERIES = 16  # cross_series=True: series read together in groups of this many, as in training
ID_NAMES = ("unique_id", "id", "item_id", "series_id")  # DataFrame columns found without being named, as the SDK does
TIME_NAMES = ("ds", "timestamp", "time", "date", "datetime")
TARGET_NAMES = ("y", "target", "value")
# fast: the model's own patch size; pro: every patch size it reads, averaged, for horizons up to 400 steps (longer ones are
# forecast better at its own size). Both decode 48 patches per pass (one pass covers 768 steps at patch 16).
DEFAULT_PRESETS = {
    "fast": {"multi_scale": False, "step_patches": 48},
    "pro": {"multi_scale": True, "multi_scale_max_h": 400, "step_patches": 48},
}
# arguments of the SDK's forecast() that only the hosted ChakraTS has: refused by name
HOSTED_ONLY = {"static", "calendar", "mean", "samples", "seed", "season", "on_error", "history", "mask_columns", "store", "tune", "model"}
SOLO_ROWS = 2048  # rows per forward pass when every series is a single row (no cross-row attention)
GROUP_ROWS = 768  # rows per forward pass when series have several rows (an R x R mask per patch)
BUDGET_STEPS = 8192  # the row budgets hold for histories up to this long; a longer one fills a pass with proportionally fewer rows


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps is not None and mps.is_available() else "cpu"


class ChakraTS:
    def __init__(self, net: ChakraTSNet, config: dict, device: str | None = None) -> None:
        self.net, self.config, self.device = net, config, device or _default_device()
        inference = config.get("inference", {})
        self.max_context = int(inference.get("max_context", 8192))
        self.presets = {**DEFAULT_PRESETS, **inference.get("presets", {})}  # a model's config.json may add some
        self.default_preset = inference.get("default_preset", "fast")
        self.folder: Path | None = None  # set by from_pretrained
        self.report: dict | None = None  # set by tune(): how the tune did on your holdout

    @classmethod
    def from_pretrained(cls, model: str | Path = "ChakraTS-OS-10M", device: str | None = None, revision: str | None = None,
                        token: str | None = None) -> ChakraTS:  # fmt: skip
        """A released model's short name (see yhatmodels.list_models()), a Hugging Face repo id, or a local folder
        (config.json + weights). Weights under a non-open licence need it accepted before the first download: once per
        person with a yhatlabs.com account for YHat Labs' models (the browser opens; YHAT_MODELS_TOKEN in CI), else
        with yhatmodels.accept_license."""
        loaded = load(model, "forecasting", revision, token)
        out = cls(loaded.net, loaded.config, device)  # type: ignore[arg-type]
        out.folder = loaded.folder
        return out

    @property
    def info(self) -> dict:
        """What the model is and reads, for a model list: name, licence, parameters, longest history, patch sizes,
        native quantile levels, presets, plus config.json's own `info` (inputs it supports, training data, ...)."""
        cfg = self.net.cfg
        return {
            "name": self.config.get("model_name"),
            "task": "forecasting",
            "architecture": self.config.get("architecture"),
            "license": self.config.get("license"),
            "parameters": sum(p.numel() for p in self.net.parameters()),
            "max_context": self.max_context,
            "patch_sizes": sorted({cfg.patch, *cfg.patches}),
            "native_quantile_levels": list(cfg.levels),
            "presets": sorted(self.presets),
            "default_preset": self.default_preset,
            **self.config.get("info", {}),
        }

    def predictor(self, preset: str | None = None, **overrides: Any) -> Predictor:
        """The Predictor of a preset ("fast": the model's own patch size; "pro": every patch size it reads, averaged;
        others from config.json), with any Predictor setting overridden (flip, clip, snap, qcap, step_patches, fp32,
        max_context). For research and benchmark replication; forecast() takes presets only."""
        preset = preset or self.default_preset
        if preset not in self.presets:
            raise ValueError(f"preset must be one of {sorted(self.presets)}")
        settings = {"max_context": self.max_context, **self.presets[preset], **overrides}
        return Predictor(self.net, device=self.device, **settings)

    def forecast(
        self,
        data,
        h: int,
        *,
        quantiles: Sequence[float] | str | None = None,
        level: Sequence[float] | None = None,
        preset: str | None = None,
        cross_series: bool | int = False,
        past=None,
        known=None,
        future=None,
        id_col: str | None = None,
        time_col: str | None = None,
        target_col: str | list[str] | None = None,
        freq: str | None = None,
        **unsupported: Any,
    ):
        """Quantile forecasts of the next `h` steps of every series in `data`.

        data: a long-format pandas DataFrame (one row per series and timestamp), which gives a DataFrame back; or arrays,
            which give arrays back: one series (T,), N series (N, T), N series of V targets forecast jointly (N, V, T),
            or a list of series of different lengths, each (T,) or (V, T). numpy, torch, a pandas Series or a plain list
            of numbers all work; a pandas Series with a DatetimeIndex is one dated series, given back as a DataFrame.
            NaN marks missing values. In a DataFrame, missing timestamps are filled in as missing values, and rows after
            a series' last observed target (covariates filled, target empty) are its future rows: the known covariates'
            values over the next h steps, when `future` isn't given.
        quantiles: levels in (0, 1) (default 0.1, 0.5, 0.9), or "all" for every level the model produces.
        level: central intervals by width in percent, e.g. [80, 95], instead of quantiles (the median comes with them).
        preset: "fast" reads a series at the model's own patch size; "pro" also at every other patch size it was trained
            on and averages the forecasts (slower; helps most with covariates and short horizons). A model's config.json
            sets its default (ChakraTS-OS-10M: "pro") and may add presets.
        cross_series: True (or a group size) forecasts single-target series without covariates 16 (or that many) at a
            time, so the model's cross-series layers read related series together.
        past / known: covariates known up to the forecast origin / known over the next h steps as well.
            DataFrame data: column names, with the known columns' future values in `future` (id, time and those
            columns, one row per future step). Array data: per series, a dict name -> values ((T,) for past, (T + h,)
            for known); a list of dicts for several series. Text values are categories.
        id_col / time_col / target_col: DataFrame columns. Defaults: unique_id, id, item_id or series_id if present (else
            one series); ds, timestamp, time, date or datetime; y, target or value (else the one numeric column).
        freq: the series' frequency (a pandas alias); inferred from each series' timestamps when left out.
        The names follow the YHat Labs SDK's forecast(); its hosted-only arguments (static, calendar, mean, samples, tune,
        ...) are refused with a pointer to the API. Research settings: predictor().

        Returns, for arrays: (h, Q) for one series, (N, h, Q) for (N, T), (N, V, h, Q) for (N, V, T), and for a list
        one forecast per series, shaped as that series: (h, Q) for (T,), (V, h, Q) for (V, T); Q runs over the levels in increasing order. For a DataFrame: the id and time columns, `target` when
        there are several targets, then one column per level (median, q10, q90, ...).
        """
        if unsupported:
            hosted = sorted(set(unsupported) & HOSTED_ONLY)
            if "tune" in hosted:
                raise TypeError(
                    "tune= names a tune of the hosted ChakraTS; to tune this model, call model.tune(...) and forecast with what it returns"
                )
            if hosted:
                raise TypeError(f"{', '.join(hosted)}: features of the hosted ChakraTS (YHat Labs API), not of this open model")
            raise TypeError(f"forecast() got unexpected arguments {sorted(unsupported)}")
        h, levels = _check_horizon(h), _levels(quantiles, level, self.net.cfg.levels)
        group = _group_size(cross_series)
        if _dated_series(data):  # one dated series: keep its dates, as the SDK does
            import pandas as pd

            data = pd.DataFrame({"ds": data.index, "y": np.asarray(data, dtype=np.float32)})
            id_col, time_col, target_col = None, "ds", "y"
        if hasattr(data, "columns"):
            return self._forecast_frame(data, h, levels, preset, group, past, known, future, id_col, time_col, target_col, freq)
        if future is not None or any(x is not None for x in (id_col, time_col, target_col, freq)):
            raise TypeError(
                "future, id_col, time_col, target_col and freq are for dated data (a DataFrame, or a pandas Series with a"
                " DatetimeIndex); arrays take known covariates with their future values in known="
            )
        return self._forecast_arrays(data, h, levels, preset, group, past, known)

    def evaluate(
        self,
        data,
        h: int,
        *,
        windows: int = 3,
        step: int | None = None,
        baselines: Sequence[str] = ("seasonal_naive",),
        metrics: Sequence[str] = ("mase", "sql", "wql", "mae", "rmse", "coverage@0.8"),
        compare: dict | None = None,
        season: int | None = None,
        preset: str | None = None,
        cross_series: bool | int = False,
        past=None,
        known=None,
        id_col: str | None = None,
        time_col: str | None = None,
        target_col: str | list[str] | None = None,
        freq: str | None = None,
    ):
        """Rolling-origin backtest on your own series: `windows` cutoffs `step` apart (default h) at the end of each series,
        each forecast `h` ahead from the history before it and scored against what happened, next to simple baselines.
        The same windows, metrics and result as the YHat Labs SDK's Client.evaluate.

        data: as for forecast(): a long DataFrame, or arrays (integer steps; pass freq= for seasonal scaling). Known
            covariates are read from the data itself over the history (arrays: (T,) per covariate).
        baselines: seasonal_naive (repeats the last season: a week for daily data, a day for hourly) and naive (the last
            value). The first one is the reference for the summary's skill_<metric> columns.
        metrics: mase, sql, wql, mae, rmse and coverage@<width> (e.g. coverage@0.8: share of actuals inside the 10-90%
            interval). Every model is scored on the deciles 0.1 ... 0.9.
        compare: your own forecasts to score on the same windows, {name: DataFrame with the id and time columns and
            quantile columns (median, q10, ...) or a point column 'forecast'}.
        season: the seasonal period in steps, to fix it instead of inferring it from the frequency.

        Returns an EvaluationResult: .summary (per model: mean metrics and skill vs the baseline), .scores (per series and
        window), .forecasts (each window's quantiles next to the actuals) and .calibration."""
        from yhatmodels.ts import evaluate as ev

        return ev.run(self, data, h, windows=windows, step=step, baselines=tuple(baselines), metrics=tuple(metrics), compare=compare,
                      season=season, preset=preset, group=_group_size(cross_series), past=past, known=known, id_col=id_col,
                      time_col=time_col, target_col=target_col, freq=freq)  # fmt: skip

    def tune(
        self,
        data,
        h: int,
        *,
        steps: int = 300,
        preset: str | None = None,
        cross_series: bool | int = False,
        past=None,
        known=None,
        id_col: str | None = None,
        time_col: str | None = None,
        target_col: str | list[str] | None = None,
        freq: str | None = None,
        season: int | None = None,
        seed: int = 0,
        min_gain: float = 0.02,
        pick_steps: bool = True,
        verbose: bool = False,
    ) -> ChakraTS:
        """Fine-tune the model on your own series (paydays, promotions, your own seasonality): low-rank adapters on
        every linear layer, pretrained weights frozen, the last `h` steps of each
        series held out to check the tune against the pretrained model through forecast() itself. Returns a new model
        (this one is unchanged) with the adapters merged into its weights; `.report` says how the tune did. When the
        tune does not beat the pretrained model by `min_gain` on the holdout, the pretrained model comes back (with a
        warning). A tuned model is a Derivative under the model's licence; save() writes the licence and NOTICE with it.

        data / past / known / id_col / time_col / target_col / freq: as for evaluate(): the history, with known covariates
            over it. h: the horizon you will forecast (the holdout length). steps: training steps (batch of 8 windows).
        preset / cross_series: the forecast settings the holdout is checked with (use the ones you will forecast with).
        season: the seasonal period for the holdout's scaled loss (default: from the frequency).
        pick_steps: retrain on the whole history for the step count that scored best (else for all `steps`)."""
        from yhatmodels.ts import tune as tn

        return tn.run(self, data, h, steps=steps, preset=preset, group=_group_size(cross_series), past=past, known=known,
                      id_col=id_col, time_col=time_col, target_col=target_col, freq=freq, season=season, seed=seed,
                      min_gain=min_gain, pick_steps=pick_steps, verbose=verbose)  # fmt: skip

    def save(self, folder: str | Path) -> Path:
        """Write the model as a released-model folder (config.json with the weights' sha256, model.safetensors, and the
        licence files it came with), loadable with ChakraTS.from_pretrained(folder). A tuned model's NOTICE states that
        it was modified from its base model, as the licence asks of a Derivative."""
        import hashlib
        import json
        import shutil
        from datetime import date

        from safetensors.torch import save_file

        out = Path(folder).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        weights = out / "model.safetensors"
        save_file({k: v.detach().cpu().contiguous() for k, v in self.net.state_dict().items()}, weights)
        config = {**self.config, "sha256": {weights.name: hashlib.sha256(weights.read_bytes()).hexdigest()}}
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        if self.folder is not None and self.folder.resolve() != out.resolve():
            for f in [*self.folder.glob("LICENSE*"), *self.folder.glob("NOTICE*")]:
                shutil.copyfile(f, out / f.name)
        if config.get("base_model") and (out / "NOTICE").exists():
            note = f"\nModified: {config['model_name']} was fine-tuned from {config['base_model']} with yhatmodels on {date.today()}.\n"
            text = (out / "NOTICE").read_text()
            if "was fine-tuned from" not in text:
                (out / "NOTICE").write_text(text + note)
        return out

    def _forecast_arrays(self, data, h: int, levels: tuple, preset: str | None, group: int, past, known):
        targets, shape = _as_targets(data)
        n = len(targets)
        if shape == "single":  # one series: its covariates may come as a dict (or array) instead of a one-entry list
            past, known = _one(past), _one(known)
        past, known = past or [None] * n, known or [None] * n
        if len(past) != n or len(known) != n:
            raise ValueError(f"past / known need one entry per series ({n} series)")
        series, blank = [], []
        for i, (tgt, pc, kc) in enumerate(zip(targets, past, known, strict=True)):
            empty = ~np.isfinite(tgt).any(axis=1)
            if empty.any():  # forecast anyway (the model has nothing to read: about 0), as benchmarks need every row
                blank.append(f"series {i}" if tgt.shape[0] == 1 else f"series {i} (target row {int(np.argmax(empty))})")
            prow, krow = covariate_rows(pc, tgt), covariate_rows(kc, tgt)
            if prow is not None and prow.shape[1] != tgt.shape[1]:
                raise ValueError(f"series {i}: past covariates have {prow.shape[1]} values; they need the history's {tgt.shape[1]}")
            if krow is not None and krow.shape[1] != tgt.shape[1] + h:
                raise ValueError(f"series {i}: known covariates have {krow.shape[1]} values; they need history + h = {tgt.shape[1] + h}")
            series.append(Series(tgt, prow, krow))
        if blank:
            import warnings

            warnings.warn(
                f"{len(blank)} series with no observed values (all NaN), e.g. {blank[0]}: their forecasts carry no information",
                stacklevel=3,
            )
        out = self._forecast_series(series, h, levels, self.predictor(preset), group)
        if shape == "single":
            return out[0][0]
        if shape == "rows":
            return np.stack([q[0] for q in out])
        if shape == "cube":
            return np.stack(out)
        return [q[0] if _ndim(x) == 1 else q for x, q in zip(data, out, strict=True)]  # each as it came: (h, Q) or (V, h, Q)

    def _forecast_series(self, series: list[Series], horizon: int, levels: tuple, pred: Predictor, group_size: int) -> list[np.ndarray]:
        order = sorted(range(len(series)), key=lambda i: series[i].target.shape[-1])  # a pass pads to its own longest
        solo = [
            i for i in order if group_size > 1 and series[i].target.shape[0] == 1 and series[i].past is None and series[i].future is None
        ]
        grouped = set(solo)
        rest = [i for i in order if i not in grouped]
        out: list[np.ndarray | None] = [None] * len(series)
        if solo:
            groups, sizes = _grouped([series[i] for i in solo], group_size)
            fg = [f for ch in _chunks(groups, pred.max_context) for f in _fit(pred, ch, horizon, levels)]
            for i, q in zip(solo, [g[k : k + 1] for g, m in zip(fg, sizes, strict=True) for k in range(m)], strict=True):
                out[i] = q
        if rest:
            fr = [f for ch in _chunks([series[i] for i in rest], pred.max_context) for f in _fit(pred, ch, horizon, levels)]
            for i, q in zip(rest, fr, strict=True):
                out[i] = q
        return out  # type: ignore[return-value]

    def _forecast_frame(self, df, h, levels, preset, group, past, known, future, id_col, time_col, target_col, freq):
        import pandas as pd

        fr = parse_frame(df, past, known, future, id_col, time_col, target_col, freq)
        id_col, time_col, targets, past, known = fr.id_col, fr.time_col, fr.targets, fr.past, fr.known
        ids, ctx, pcov, kcov, stamps = [], [], [], [], []
        for s in fr.series:
            item, name, hist, tail, step = s.item, s.name, s.hist, s.tail, s.step
            if len(tail) and future is not None:
                raise ValueError(
                    f"{name} has {len(tail)} rows after its last observed target and `future` is given too; give the future rows once"
                )
            grid = hist.index
            ahead = pd.date_range(grid[-1], periods=h + 1, freq=step)[1:]
            ids.append(item)
            stamps.append(ahead)
            ctx.append(hist[targets].to_numpy(np.float32).T)
            pcov.append({c: hist[c].to_numpy() for c in past} or None)
            if known:
                if future is not None:
                    if (rows := fr.future.get(item)) is None:
                        raise ValueError(f"future has no rows for {name}")
                elif len(rows := tail) == 0:
                    raise ValueError(
                        f"known covariates need their values over the next {h} steps: give `future`, or keep those rows in the data"
                        f" after the last observed target ({name} has none)"
                    )
                rows = rows.assign(**{time_col: pd.to_datetime(rows[time_col])}).set_index(time_col).reindex(ahead)
                if (gaps := rows[known].isna().all(axis=1)).any():
                    raise ValueError(
                        f"{name}: no known-covariate values for {int(gaps.sum())} of the next {h} steps (first {gaps.index[gaps][0]})"
                    )
                kcov.append({c: np.concatenate([hist[c].to_numpy(), rows[c].to_numpy()]) for c in known})
            else:
                kcov.append(None)
        qs = self._forecast_arrays(ctx, h, levels, preset, group, pcov if past else None, kcov if known else None)
        frames = []
        for item, ts, q in zip(ids, stamps, qs, strict=True):
            for j, name in enumerate(targets):
                block = {id_col: item} if id_col else {}
                block[time_col] = ts
                if len(targets) > 1:
                    block["target"] = name
                block.update({qname(lv): q[j, :, k] for k, lv in enumerate(levels)})
                frames.append(pd.DataFrame(block))
        return pd.concat(frames, ignore_index=True)


@dataclass
class FrameSeries:
    item: Any  # the id (None: the frame is one series)
    name: str  # for messages: "series 'a'" / "the series"
    hist: Any  # DataFrame on the series' time grid, up to its last observed target (missing timestamps as NaN rows)
    tail: Any  # rows after the last observed target: the future rows of known covariates, when the data holds them
    step: str  # pandas frequency alias


@dataclass
class Frame:
    id_col: str | None
    time_col: str
    targets: list[str]
    past: list[str]
    known: list[str]
    series: list[FrameSeries]
    future: dict  # id -> its rows of `future`


def parse_frame(df, past, known, future, id_col, time_col, target_col, freq) -> Frame:
    """A long-format DataFrame -> its columns' roles and each series on its own time grid (shared by forecast, evaluate
    and tune)."""
    import pandas as pd

    cols = list(df.columns)
    past, known = _names(past), _names(known)
    id_col = id_col or _pick(cols, ID_NAMES)  # none: the whole frame is one series
    time_col = time_col or _pick(cols, TIME_NAMES)
    if time_col is None:
        raise ValueError("data needs a time column (ds, timestamp, time, date or datetime), or name it with time_col=")
    if target_col is None:
        target_col = _pick(cols, TARGET_NAMES)
    if target_col is None:
        numeric = [c for c in cols if c not in {id_col, time_col, *past, *known} and pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric) != 1:
            raise ValueError(f"name the target with target_col= (numeric columns: {numeric})")
        target_col = numeric[0]
    targets = [target_col] if isinstance(target_col, str) else list(target_col)
    if missing := [c for c in (id_col, time_col, *targets, *past, *known) if c and c not in cols]:
        raise ValueError(f"data has no column {', '.join(map(repr, missing))}")
    if future is not None and (missing := [c for c in (id_col, time_col, *known) if c and c not in future.columns]):
        raise ValueError(f"future has no column {', '.join(map(repr, missing))}")

    def by_series(frame):
        frame = frame.assign(**{time_col: pd.to_datetime(frame[time_col])}).sort_values(time_col, kind="stable")
        return frame.groupby(id_col, sort=False) if id_col else [(None, frame)]

    out = []
    for item, g in by_series(df):
        name = "the series" if item is None else f"series {item!r}"
        if (dup := g[time_col].duplicated()).any():
            hint = "" if id_col else "; if the data holds several series, name their id column with id_col="
            raise ValueError(f"{name} has duplicate timestamps, e.g. {g[time_col][dup].iloc[0]}{hint}")
        seen = g[targets].notna().any(axis=1).to_numpy()
        if not seen.any():
            raise ValueError(f"{name} has no observed values: nothing to forecast from")
        last = int(np.flatnonzero(seen)[-1])
        hist, tail = g.iloc[: last + 1], g.iloc[last + 1 :]  # the tail: rows after the last observed target = future rows
        step = freq or _infer_freq(g[time_col])
        if step is None:
            raise ValueError(f"cannot infer the frequency of {name}; pass freq= (a pandas alias such as 'h', 'D' or 'MS')")
        grid = pd.date_range(hist[time_col].iloc[0], hist[time_col].iloc[-1], freq=step)
        if not pd.DatetimeIndex(hist[time_col]).isin(grid).all():
            raise ValueError(f"the timestamps of {name} are not on a {step!r} grid; pass the right freq=")
        out.append(FrameSeries(item, name, hist.set_index(time_col).reindex(grid), tail, step))
    given = dict(tuple(by_series(future))) if future is not None else {}
    return Frame(id_col, time_col, targets, past, known, out, given)


def _grouped(series: list[Series], n: int) -> tuple[list[Series], list[int]]:
    """Single-target series (in the order given) -> groups of up to `n`, left-padded with NaN to the group's longest
    history; returns the groups and their sizes."""
    groups, sizes = [], []
    for s in range(0, len(series), n):
        rows = [np.atleast_2d(x.target)[0] for x in series[s : s + n]]
        width = max(len(r) for r in rows)
        groups.append(Series(np.stack([np.concatenate([np.full(width - len(r), np.nan, np.float32), r]) for r in rows])))
        sizes.append(len(rows))
    return groups, sizes


def _chunks(series: list[Series], max_context: int = BUDGET_STEPS):
    """Series (in length order) -> forward passes of at most a row budget, and of at most budget x BUDGET_STEPS cells
    (rows x the pass's longest history, capped at max_context), so long histories never fill a pass past what the
    budget was sized for."""
    sizes = [sum(0 if a is None else np.atleast_2d(a).shape[0] for a in (s.target, s.past, s.future)) for s in series]
    budget = SOLO_ROWS if max(sizes, default=1) == 1 else GROUP_ROWS
    cur, n_rows, width = [], 0, 0
    for s, n in zip(series, sizes, strict=True):
        w = max(width, min(np.atleast_2d(s.target).shape[-1], max_context))
        if cur and (n_rows + n > budget or (n_rows + n) * w > budget * BUDGET_STEPS):
            yield cur
            cur, n_rows, w = [], 0, min(np.atleast_2d(s.target).shape[-1], max_context)
        cur.append(s)
        n_rows, width = n_rows + n, w
    if cur:
        yield cur


# torch.OutOfMemoryError arrived in torch 2.5; the CUDA one is older; the Apple GPU raises a RuntimeError naming it
_OOM = tuple(e for e in (getattr(torch, "OutOfMemoryError", None), getattr(torch.cuda, "OutOfMemoryError", None)) if e is not None)


def _out_of_memory(e: BaseException) -> bool:
    return isinstance(e, _OOM) or (isinstance(e, RuntimeError) and "out of memory" in str(e).lower())


def _release(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    elif str(device).startswith("mps") and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def _fit(pred: Predictor, chunk: list[Series], horizon: int, levels: tuple) -> list[np.ndarray]:
    """Forecast a chunk; on GPU out-of-memory, retry it in halves."""
    try:
        return pred.predict(chunk, horizon, levels)
    except Exception as e:
        if not _out_of_memory(e) or len(chunk) == 1:
            raise
    # retried outside the except block: inside it the traceback keeps the failed pass's activations alive
    _release(pred.device)
    h = len(chunk) // 2
    return _fit(pred, chunk[:h], horizon, levels) + _fit(pred, chunk[h:], horizon, levels)


def _as_targets(context) -> tuple[list[np.ndarray], str]:
    """The series of `context` as (V, T) float32 arrays, and the input's shape: "single" (one series), "rows" ((N, T)),
    "cube" ((N, V, T)) or "list"."""
    if hasattr(context, "columns"):
        raise TypeError("a DataFrame is passed to forecast() as data itself, not inside a list")
    if hasattr(context, "detach"):  # a torch tensor, wherever it lives
        context = context.detach().cpu().numpy()
    if isinstance(context, (list, tuple)):
        if not context:
            raise ValueError("no series to forecast")
        if all(np.ndim(x) == 0 for x in context):  # a plain list of numbers is one series
            return [_row(context)], "single"
        return [_row(x) for x in context], "list"
    a = np.asarray(context, dtype=np.float32)
    if a.ndim == 1:
        return [_row(a)], "single"
    if a.ndim == 2:
        return [_row(r) for r in a], "rows"
    if a.ndim == 3:
        return [_row(v) for v in a], "cube"
    raise ValueError(f"context must be (T,), (N, T), (N, V, T) or a list of series; got shape {a.shape}")


def _ndim(x) -> int:
    return x.dim() if hasattr(x, "detach") else np.ndim(x)


def _one(covariates):
    return [covariates] if isinstance(covariates, dict) or hasattr(covariates, "shape") else covariates


def _row(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    a = np.atleast_2d(np.asarray(x, dtype=np.float32))
    if a.ndim != 2 or a.shape[1] == 0:
        raise ValueError(f"each series must be (T,) or (V, T) with T >= 1; got shape {np.shape(x)}")
    return a


def _check_horizon(horizon) -> int:
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
    return int(horizon)


def _levels(quantiles, level, native) -> tuple[float, ...]:
    """The requested levels, increasing: quantiles in (0, 1), "all" (the model's native levels), or central intervals
    (level, in percent) with the median."""
    if quantiles is not None and level is not None:
        raise ValueError("pass quantiles or level, not both")
    if level is not None:
        widths = [float(w) for w in np.atleast_1d(np.asarray(level, dtype=float))]
        if not widths or not all(0.0 < w < 100.0 for w in widths):
            raise ValueError(f"level takes central interval widths in percent, between 0 and 100; got {level!r}")
        qs = {0.5} | {round((100 - w) / 200, 10) for w in widths} | {round(1 - (100 - w) / 200, 10) for w in widths}
    elif quantiles is None:
        qs = set(DEFAULT_QUANTILES)
    elif isinstance(quantiles, str):
        if quantiles != "all":
            raise ValueError(f'quantiles takes levels in (0, 1) or "all", got {quantiles!r}')
        qs = {float(q) for q in native}
    else:
        qs = {float(q) for q in np.atleast_1d(np.asarray(quantiles, dtype=float))}
    out = tuple(sorted(qs))
    if not out or not all(0.0 < q < 1.0 for q in out):
        raise ValueError(f"quantiles must be numbers strictly between 0 and 1, got {quantiles!r}")
    return out


def _group_size(cross_series) -> int:
    if cross_series is None or cross_series is False:
        return 1
    if cross_series is True:
        return CROSS_SERIES
    if isinstance(cross_series, (int, np.integer)) and cross_series >= 1:
        return int(cross_series)
    raise ValueError(f"cross_series is True, False or a group size, got {cross_series!r}")


def _dated_series(data) -> bool:
    return hasattr(data, "index") and not hasattr(data, "columns") and type(data.index).__name__ == "DatetimeIndex"


def _infer_freq(times) -> str | None:
    """The frequency of a series' timestamps: pandas' own inference, else (when some timestamps are missing) the most
    common step, if it is a fixed length of time."""
    import warnings

    import pandas as pd

    ts = pd.DatetimeIndex(times)
    if len(ts) >= 3:
        with warnings.catch_warnings():  # pandas is moving infer_freq from a string to an offset; take either
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.simplefilter("ignore", FutureWarning)
            f = pd.infer_freq(ts)
        if f is not None:
            return f if isinstance(f, str) else f.freqstr
    steps = pd.Series(ts).diff().dropna()
    if steps.empty:
        return None
    try:
        return pd.tseries.frequencies.to_offset(steps.mode().iloc[0]).freqstr
    except ValueError:
        return None


def _names(cols) -> list[str]:
    return [] if cols is None else [cols] if isinstance(cols, str) else list(cols)


def _pick(cols, names) -> str | None:
    """The first of `names` among the columns, ignoring case."""
    low = {str(c).lower(): c for c in cols}
    return next((low[n] for n in names if n in low), None)


def qname(q: float) -> str:
    """Column name of a level, as the YHat Labs SDK names it: 0.5 -> 'median', 0.1 -> 'q10', 0.025 -> 'q2.5'."""
    return "median" if abs(q - 0.5) < 1e-12 else f"q{round(q * 100, 6):g}"
