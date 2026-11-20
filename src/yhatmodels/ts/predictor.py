#  Copyright (c) 2026 Forum Software, Inc. (YHat Labs). Licensed under the Apache License, Version 2.0.
"""Inference: contexts (+ covariates) -> raw-scale quantiles at any requested levels.

Horizon patches are appended to the context as unobserved inputs and read off one pass; horizons longer than
`step_patches` patches roll forward, feeding the median back as context.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from yhatmodels.ts.architectures.chakrats_v1 import FUTURE, PAST, TARGET, ChakraTSNet
from yhatmodels.ts.scaling import unsquash


@dataclass
class Series:
    target: np.ndarray  # (n_target, T)
    past: np.ndarray | None = None  # (n_past, T): known up to the forecast origin
    future: np.ndarray | None = None  # (n_future, T + H): known over the context and the horizon


class Predictor:
    """`flip`: average with the mirrored forecast of the negated target ("all": every row; True: rows whose history
    minimum is not exactly 0). `clip`: no quantile below 0 when the history never goes below 0. `snap`: round quantiles
    when the history is integer-valued. `multi_scale`: average the forecasts of every patch size the model reads whose
    token window holds the series' history, for horizons up to `multi_scale_max_h` (longer ones read the model's own
    patch size, which forecasts them better). `qcap`: clamp the squashed quantiles before unsquashing. All of these use
    the history only."""

    def __init__(
        self,
        model: ChakraTSNet,
        device: str = "cpu",
        max_context: int = 32768,
        flip: bool | str = "all",
        clip: bool = True,
        snap: bool = True,
        multi_scale: bool = False,
        multi_scale_max_h: int | None = None,
        qcap: float | None = None,
        step_patches: int = 32,
        fp32: bool = False,
    ) -> None:
        self.model, self.device, self.max_context = model.eval().to(device), device, max_context
        self.flip, self.clip, self.snap = flip, clip, snap
        self.multi_scale, self.qcap, self.step_patches = multi_scale, qcap, step_patches
        self.multi_scale_max_h = multi_scale_max_h
        self.fp32 = fp32  # CUDA only: fp32 maths instead of the bf16 autocast the model was trained in
        self.patch: int | None = None  # the size being decoded (None = the model's own)

    @property
    def cfg(self):
        return self.model.cfg

    @property
    def p(self) -> int:
        return self.patch or self.cfg.patch

    def _max_ctx(self, size: int) -> int:
        """Context steps at `size`: a finer size keeps the token count, a coarser one the window."""
        return min(self.max_context, self.max_context * size // self.cfg.patch)

    @torch.no_grad()
    def predict(self, series: list[Series], horizon: int, levels=(0.1, 0.5, 0.9)) -> list[np.ndarray]:
        """-> per series, (n_target, horizon, len(levels)) raw-scale quantiles."""
        native = np.array(self.cfg.levels)
        out = self._native(series, horizon)
        hists = [np.atleast_2d(s.target) for s in series]
        if self.flip:
            mirrored = self._native([Series(-np.atleast_2d(s.target), s.past, s.future) for s in series], horizon)
            for i, (q, m, h) in enumerate(zip(out, mirrored, hists, strict=True)):
                use = np.ones(h.shape[0], bool) if self.flip == "all" else _row_min(h) != 0
                avg = 0.5 * (q - m[..., ::-1])  # level k of x <-> level 1 - k of -x (symmetric grid)
                out[i] = np.where(use[:, None, None], avg, q)
        # clip and snap after interpolating to the requested levels, so levels off the model's grid (0.001, 0.975) keep
        # the promise too: a level below the grid extrapolates and could go negative, one between integers would not be one
        out = [_interp(q, native, np.asarray(levels)) for q in out]
        for i, h in enumerate(hists):
            if self.clip:
                nonneg = _row_min(h) >= 0
                out[i] = np.where(nonneg[:, None, None], np.maximum(out[i], 0), out[i])
            if self.snap:
                obs = np.where(np.isfinite(h), h, 0)
                integer = (obs == np.round(obs)).all(axis=1)
                out[i] = np.where(integer[:, None, None], np.round(out[i]), out[i])
        return out

    def _native(self, series: list[Series], horizon: int) -> list[np.ndarray]:
        if not self.multi_scale or not self.cfg.patches or (self.multi_scale_max_h and horizon > self.multi_scale_max_h):
            return self._decode(series, horizon)
        per: list[list[np.ndarray]] = [[] for _ in series]
        for size in self.model.sizes:
            idx = [i for i, s in enumerate(series) if size >= self.cfg.patch or _len(s) <= self._max_ctx(size)]
            if not idx:
                continue
            self.patch = size
            try:
                qs = self._decode([series[i] for i in idx], horizon)
            finally:
                self.patch = None
            for i, q in zip(idx, qs, strict=True):
                per[i].append(q)
        return [np.mean(np.stack(qs), axis=0) for qs in per]

    def _decode(self, series: list[Series], horizon: int) -> list[np.ndarray]:
        step = self.step_patches * self.cfg.patch  # passes cover the same steps at every patch size
        done: list[list[np.ndarray]] = [[] for _ in series]
        hist = [np.atleast_2d(s.target).astype(np.float32) for s in series]
        mid = list(self.cfg.levels).index(0.5)
        made = 0
        while made < horizon:
            h = min(step, horizon - made)
            chunk = [
                Series(hist[i], _pad_right(s.past, hist[i].shape[1]), None if s.future is None else s.future[:, : hist[i].shape[1] + h])
                for i, s in enumerate(series)
            ]
            for i, qi in enumerate(self._one_pass(chunk, h)):
                done[i].append(qi)
                hist[i] = np.concatenate([hist[i], qi[:, :, mid]], 1)
            made += h
        return [np.concatenate(d, 1) for d in done]

    def _one_pass(self, series: list[Series], h: int) -> list[np.ndarray]:
        p = self.p
        hp = -(-h // p)
        budget = max(p, (self._max_ctx(p) // p - hp) * p)  # context + horizon stay inside the trained window
        ctx_len = min(budget, max(s.target.shape[-1] for s in series))
        n_ctx = -(-ctx_len // p)
        width = (n_ctx + hp) * p
        rows, group, role, owner = [], [], [], []
        for gi, s in enumerate(series):
            t = np.atleast_2d(s.target)[:, -n_ctx * p :]
            for r in t:
                rows.append(_place(r, width, n_ctx * p, None))
                group.append(gi), role.append(TARGET), owner.append(gi)
            if s.past is not None:
                for r in np.atleast_2d(s.past)[:, -t.shape[1] :]:
                    rows.append(_place(r[-n_ctx * p :], width, n_ctx * p, None))
                    group.append(gi), role.append(PAST), owner.append(-1)
            if s.future is not None:
                fut = np.atleast_2d(s.future)
                for r in fut:
                    rows.append(_place(r[: fut.shape[1] - h][-n_ctx * p :], width, n_ctx * p, r[fut.shape[1] - h :]))
                    group.append(gi), role.append(FUTURE), owner.append(-1)
        v = torch.tensor(np.stack(rows), device=self.device)
        g = torch.tensor(group, device=self.device)
        r = torch.tensor(role, device=self.device)
        cuda = str(self.device).startswith("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cuda and not self.fp32):
            pred, loc, sc, _ = self.model(v, g, r, patch=self.patch)
        cut = n_ctx - 1  # the last context token forecasts the horizon, in its own frame
        q = pred[:, cut : cut + hp].reshape(len(rows), hp * p, -1)[:, :h]
        if self.qcap is not None:
            q = q.clamp(-self.qcap, self.qcap)
        lo, s_ = loc[:, cut][:, None, None], sc[:, cut][:, None, None]
        raw = (lo + s_ * unsquash(q)).float().cpu().numpy()
        own = np.array(owner)
        return [raw[own == gi] for gi in range(len(series))]


def _len(s: Series) -> int:
    return np.atleast_2d(s.target).shape[1]


def _pad_right(x: np.ndarray | None, length: int) -> np.ndarray | None:
    """Past-only covariates end at the forecast origin: NaN after it when the median is fed back."""
    if x is None:
        return None
    x = np.atleast_2d(x)
    pad = length - x.shape[1]
    return np.concatenate([x, np.full((x.shape[0], pad), np.nan, np.float32)], 1) if pad > 0 else x


def _place(ctx: np.ndarray, width: int, ctx_width: int, fut: np.ndarray | None) -> np.ndarray:
    """Right-align the context in the first `ctx_width` steps (NaN left padding); fill the horizon with known-future
    values or NaN."""
    row = np.full(width, np.nan, np.float32)
    ctx = ctx[-ctx_width:]
    row[ctx_width - len(ctx) : ctx_width] = ctx
    if fut is not None:
        row[ctx_width : ctx_width + len(fut)] = fut
    return row


def _interp(q: np.ndarray, native: np.ndarray, want: np.ndarray) -> np.ndarray:
    """Linear inside the native grid; beyond it, extend the outermost spacing."""
    out = np.empty((*q.shape[:-1], len(want)), np.float32)
    for k, lv in enumerate(want):
        j = int(np.clip(np.searchsorted(native, lv), 1, len(native) - 1))
        lo, hi = native[j - 1], native[j]
        w = (lv - lo) / (hi - lo)
        out[..., k] = q[..., j - 1] + w * (q[..., j] - q[..., j - 1])
    return out


def _row_min(h: np.ndarray) -> np.ndarray:
    """Each row's smallest observed value (NaN for a row with none, without numpy's all-NaN warning)."""
    ok = np.isfinite(h).any(axis=1)
    out = np.full(h.shape[0], np.nan)
    out[ok] = np.nanmin(h[ok], axis=1)
    return out
