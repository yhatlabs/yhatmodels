#  Copyright (c) 2026 Forum Software, Inc. (YHat Labs). Licensed under the Apache License, Version 2.0.
"""Fine-tuning ChakraTS on your own series: low-rank adapters (LoRA) on every
linear layer with the pretrained weights frozen, trained on windows cut from your history with the model's own training
loss (next-patch quantile loss), and checked on your last h steps against the pretrained model through the forecast path
itself. The tune is kept only when it beats the pretrained model there by `min_gain`; it is then trained again on the
whole history for the step count that won, so the latest data is in it, and merged into a copy of the weights (a tuned
model forecasts as fast as the pretrained one).

Training reads the targets and the numeric known covariates; past covariates and text known covariates still reach every
forecast, they are just not part of the loss.
"""

from __future__ import annotations

import copy
import math
import time
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn.utils import parametrize

from yhatmodels.ts.architectures.chakrats_v1 import FUTURE, TARGET
from yhatmodels.ts.covariates import rows as covariate_rows
from yhatmodels.ts.evaluate import DECILES, season_for
from yhatmodels.ts.predictor import Predictor, Series
from yhatmodels.ts.scaling import squash

RANK, ALPHA, LR, BATCH, CTX = (
    8,
    16.0,
    1e-4,
    8,
    1024,
)  # lr: a 2026-11-19 field test on 11 held-out sets kept no tune that hurt at 1e-4 (1e-3 overfit)
EVAL_EVERY, SCORE_SERIES = 25, 64  # holdout every 25 steps, on at most 64 series (a few seconds at any data size)


class _LoRA(nn.Module):
    """weight -> weight + (alpha / r) B A, as a parametrization: every read of the layer's weight sees the adapter, including
    the head's resampled use of it at other patch sizes."""

    def __init__(self, lin: nn.Linear, r: int) -> None:
        super().__init__()
        dev = lin.weight.device
        self.A = nn.Parameter(torch.randn(r, lin.in_features, device=dev) / math.sqrt(lin.in_features))
        self.B = nn.Parameter(torch.zeros(lin.out_features, r, device=dev))
        self.scale = ALPHA / r

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w + (self.B @ self.A).to(w.dtype) * self.scale


def _wrap(net: nn.Module, r: int) -> list[tuple[nn.Linear, _LoRA]]:
    for p in net.parameters():
        p.requires_grad_(False)
    out = []
    for mod in list(net.modules()):
        if isinstance(mod, nn.Linear):
            lo = _LoRA(mod, r)
            parametrize.register_parametrization(mod, "weight", lo)
            out.append((mod, lo))
    return out


def _snapshot(wrapped) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(lo.A.detach().clone(), lo.B.detach().clone()) for _, lo in wrapped]


def _restore(wrapped, snap) -> None:
    with torch.no_grad():
        for (_, lo), (a, b) in zip(wrapped, snap, strict=True):
            lo.A.copy_(a)
            lo.B.copy_(b)


def _merge(wrapped) -> None:
    for lin, _ in wrapped:
        parametrize.remove_parametrizations(lin, "weight", leave_parametrized=True)
    for lin, _ in wrapped:
        lin.weight.requires_grad_(False)


class _Windows:
    """Training windows (context L, future F) ending at random points <= `end` of each target row, with that row's numeric
    known covariates alongside."""

    def __init__(self, rows, F: int, p: int, rng: np.random.Generator) -> None:
        self.min_ctx = 2 * p
        self.rows = [(y, k, e) for y, k, e in rows if e >= F + self.min_ctx]
        if not self.rows:
            raise ValueError("not enough history to tune")
        self.F, self.p, self.rng = F, p, rng
        w = np.asarray([e - F - self.min_ctx + 1 for _, _, e in self.rows], float)
        self.prob = w / w.sum()

    def batch(self, n: int):
        pick = self.rng.choice(len(self.rows), n, p=self.prob)
        F, p = self.F, self.p
        ends = [int(self.rng.integers(F + self.min_ctx, self.rows[i][2] + 1)) for i in pick]
        L = min(CTX, min(e - F for e in ends)) // p * p
        # the forecast path averages in the mirrored (negated) series, so half the windows are negated: an adapter that
        # only saw the series as they are would pull the mirrored half the wrong way
        sign = np.where(self.rng.random(n) < 0.5, -1.0, 1.0).astype(np.float32)
        values, group, role = [], [], []
        for b, (i, e) in enumerate(zip(pick, ends, strict=True)):
            y, k, _ = self.rows[i]
            values.append(y[e - F - L : e] * sign[b])
            group.append(b)
            role.append(TARGET)
            for r in [] if k is None else k:
                values.append(r[e - F - L : e])
                group.append(b)
                role.append(FUTURE)
        return np.stack(values).astype(np.float32), np.asarray(group), np.asarray(role)


def _loss(net, values, group, role, device) -> torch.Tensor:
    """The decoder's training loss: token n's quantiles against patch n + 1, in the frame (loc, scale) token n was issued
    in, on observed target steps."""
    x = torch.tensor(values, device=device)
    g = torch.tensor(group, device=device, dtype=torch.long)
    ro = torch.tensor(role, device=device, dtype=torch.long)
    p = net.cfg.patch
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
        pred, loc, sc, ok = net(x, g, ro)
    r, t = x.shape
    obs = torch.isfinite(x).view(r, t // p, p)
    y = torch.nan_to_num(x).view(r, t // p, p)
    zt = squash((y[:, 1:] - loc[:, :-1, None]) / sc[:, :-1, None])
    mask = obs[:, 1:] & ok[:, :-1, None] & (ro == TARGET)[:, None, None]
    q = torch.tensor(net.cfg.levels, device=device)
    u = zt[..., None] - pred[:, :-1].float()
    loss = torch.maximum(q * u, (q - 1) * u).mean(-1)
    m = mask.float()
    return (loss * m).sum() / m.sum().clamp_min(1e-6)


def _pinball(Q: np.ndarray, y: np.ndarray) -> np.ndarray:
    lv = np.asarray(DECILES)
    d = y[..., None] - Q
    return np.maximum(lv * d, (lv - 1) * d).mean(-1)


def _holdout(preds: list[np.ndarray], truth: list[np.ndarray], ctx: list[np.ndarray], seasons: list[int]) -> tuple[float, float]:
    """Mean scaled pinball (pinball / the context's seasonal-naive error) and WAPE of the median, over every target row."""
    losses, num, den = [], 0.0, 0.0
    for Q, y, c, m in zip(preds, truth, ctx, seasons, strict=True):
        for v in range(y.shape[0]):
            cv = c[v]
            d = np.abs(cv[m:] - cv[:-m]) if len(cv) > m else np.abs(np.diff(cv))
            s = np.nanmean(d) if np.isfinite(d).any() else np.nan
            ok = np.isfinite(y[v])
            if not ok.any() or not (s > 0):
                continue
            losses.append(float(np.mean(_pinball(Q[v][ok], y[v][ok])) / s))
            num += float(np.abs(y[v][ok] - Q[v][ok, 4]).sum())
            den += float(np.abs(y[v][ok]).sum())
    if not losses:
        raise ValueError("no series has an observed holdout (its last h steps) to check the tune against")
    return float(np.mean(losses)), num / max(den, 1e-9)


def _numeric_rows(cov: dict | None, upto: int) -> np.ndarray | None:
    if not cov:
        return None
    rows = []
    for v in cov.values():
        a = np.asarray(v)
        if a.dtype.kind in "biuf":
            rows.append(a[:upto].astype(np.float32))
    return np.stack(rows) if rows else None


def run(model, data, h: int, *, steps: int, preset, group: int, past, known, id_col, time_col, target_col, freq,
        season, seed: int, min_gain: float, pick_steps: bool, verbose: bool):  # fmt: skip
    from yhatmodels.ts.evaluate import _from_arrays, _from_frame
    from yhatmodels.ts.forecaster import ChakraTS, _check_horizon

    h = _check_horizon(h)
    if steps < 1:
        raise ValueError("steps must be at least 1")
    t0 = time.perf_counter()
    if hasattr(data, "columns"):
        series = _from_frame(data, past, known, id_col, time_col, target_col, freq)[0]
    else:
        series = _from_arrays(data, past, known, freq)[0]
    base, device = model.net, model.device
    p = base.cfg.patch
    F = -(-h // p) * p
    rng = np.random.default_rng(seed)
    rows_cut, rows_full = [], []
    for s in series:
        T = s.y.shape[1]
        k_cut, k_full = _numeric_rows(s.known, T - h), _numeric_rows(s.known, T)
        for v in range(s.y.shape[0]):
            rows_cut.append((s.y[v, : T - h], k_cut, T - h))
            rows_full.append((s.y[v], k_full, T))
    try:
        w_cut, w_full = _Windows(rows_cut, F, p, rng), _Windows(rows_full, F, p, rng)
    except ValueError:
        raise ValueError(f"not enough history to tune: a series needs at least {h + F + 2 * p} values for h={h}") from None
    scored = sorted(rng.choice(len(series), SCORE_SERIES, replace=False)) if len(series) > SCORE_SERIES else range(len(series))
    hold, truth, ctx, seasons = [], [], [], []
    for i in scored:
        s = series[i]
        T = s.y.shape[1]
        if T - h < 1:
            continue
        tgt = s.y[:, : T - h]
        pc = {k: v[: T - h] for k, v in s.past.items()} if s.past else None
        hold.append(Series(tgt, covariate_rows(pc, tgt), covariate_rows(s.known, tgt)))  # known: T - h history + h ahead
        truth.append(s.y[:, T - h :])
        ctx.append(tgt)
        seasons.append(season or season_for(s.step))

    settings = {"max_context": model.max_context, **model.presets[preset or model.default_preset]}

    def score(net) -> tuple[float, float]:
        pred = Predictor(net, device=device, **settings)
        out = ChakraTS._forecast_series(model, hold, h, DECILES, pred, group)
        net.train()
        return _holdout(out, truth, ctx, seasons)

    zero = score(base)
    base.eval()

    def train(windows: _Windows, n_steps: int, every: int):
        torch.manual_seed(seed)
        net = copy.deepcopy(base).to(device)
        wrapped = _wrap(net, RANK)
        opt = torch.optim.AdamW([q for _, lo in wrapped for q in (lo.A, lo.B)], lr=LR, weight_decay=0.0)
        best = (math.inf, 0, None, None)
        net.train()
        for step in range(1, n_steps + 1):
            loss = _loss(net, *windows.batch(BATCH), device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if every and (step % every == 0 or step == n_steps):
                got = score(net)
                if verbose:
                    print(f"step {step}: holdout loss {got[0]:.4f} (pretrained {zero[0]:.4f})", flush=True)
                if got[0] < best[0]:
                    best = (got[0], step, got, _snapshot(wrapped))
        return net, wrapped, best

    _, _, (best_loss, best_step, best_score, _) = train(w_cut, steps, EVAL_EVERY)
    used = best_loss < zero[0] * (1 - min_gain)
    report = {
        "used": used,
        "holdout": {
            "h": h,
            "series": len(hold),
            "pretrained": {"loss": round(zero[0], 4), "wape": round(zero[1], 4)},
            "tuned": {"loss": round(best_score[0], 4), "wape": round(best_score[1], 4)} if best_score else None,
        },
        "steps": best_step if used else 0,
        "steps_max": steps,
        "rank": RANK,
        "min_gain": min_gain,
    }
    if not used:
        report["seconds"] = round(time.perf_counter() - t0, 1)
        warnings.warn(
            f"the tune did not beat the pretrained model by {min_gain:.0%} on your last {h} steps (holdout loss {best_loss:.4f}"
            f" vs {zero[0]:.4f}); returning the pretrained model",
            stacklevel=3,
        )
        out = ChakraTS(base, model.config, device)
        out.folder, out.report = model.folder, report
        return out
    # again on the whole history, for the step count that won (or all of them), so the latest data is in the adapters
    net, wrapped, _ = train(w_full, best_step if pick_steps else steps, 0)
    _merge(wrapped)
    net.eval()
    report["seconds"] = round(time.perf_counter() - t0, 1)
    name = model.config.get("model_name") or "ChakraTS"
    config = copy.deepcopy(model.config)
    config.pop("sha256", None)  # the weights changed; save() stamps the new ones
    config["model_name"] = f"{name}-tuned"
    config["base_model"] = config.get("base_model", name)
    config.setdefault("info", {})["tune"] = report
    out = ChakraTS(net, config, device)
    out.folder, out.report = model.folder, report
    return out
