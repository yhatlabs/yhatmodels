#  Copyright (c) 2026 Forum Software, Inc. (YHat Labs). Licensed under the Apache License, Version 2.0.
"""ChakraTS v1: a causal patch decoder over a (rows x patches) grid.

Each row of the grid is one variable of one series: a target, a past-only covariate or a known-future covariate.
Token n reads the patches up to n of its own row (time attention) and the same patch of the other rows of its series
(variate attention), and predicts patch n + 1 as quantiles. Horizon patches are appended as unobserved inputs and the
whole horizon is read off one pass. Variate attention never lets a covariate read a target, and never lets a
known-future row read a causal row, which would relay a target's future back to it through the covariate.

Inputs are normalised per patch with the statistics of everything observed so far (`scaling.py`), embedded as
[values, missing, within-patch time], and read with RMSNorm, QK-norm, partial rotary positions with an xPos-style
distance decay, sigmoid-gated attention and SwiGLU, with U-Net style skips between the first and second half of the
blocks. The same weights also read coarser or finer patches (`patches`), resampled on the fly.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
from torch import nn

from yhatmodels.ts.scaling import is_binary, prefix_stats, squash, whole_stats

TARGET, PAST, FUTURE = 0, 1, 2


@dataclass(frozen=True)
class Config:
    patch: int = 16
    d: int = 192
    heads: int = 3
    ff: int = 512
    schedule: tuple[str, ...] = ("t", "tv")  # one entry per block: "t" time attention, "tv" time + variate attention
    levels: tuple[float, ...] = (0.01, *[round(0.05 * i, 2) for i in range(1, 20)], 0.99)  # native quantile levels
    width_factor: float = 3.0
    unet: bool = False  # first-half block outputs added (learned gain) to the second half
    res_scale: bool = False  # residual branches scaled by (3 * blocks) ** -0.5
    time_channel: bool = False  # a third embedding channel with the within-patch time index
    xpos: bool = False  # rotary scaling so attention decays with patch distance
    xpos_span: int = 512  # patches over which the xPos decay runs
    gate_pre: bool = False  # gate the attention output before (True) or after the output projection
    patches: tuple[int, ...] = ()  # extra patch sizes the weights also read
    out_bound: float = 0.0  # > 0: squashed quantiles pass through b * tanh(q / b)

    @classmethod
    def from_dict(cls, d: dict, strict: bool = True) -> Config:
        """config.json's "model" settings. Settings this architecture does not implement are refused; with strict (the
        default, as for every released model) so is any key it does not know, which would otherwise be dropped and
        run a newer checkpoint wrongly. strict=False ignores unknown keys (training-only settings)."""
        if d.get("arm", "decoder") != "decoder" or d.get("dual_window", 0) or d.get("frame_window", 0) or d.get("mtp", 1) != 1:
            raise ValueError("this checkpoint needs an architecture variant yhatmodels does not implement")
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - names - {"arm", "dual_window", "frame_window", "mtp"})
        if strict and unknown:
            raise ValueError(f"settings {unknown} are unknown to this yhatmodels; a newer version may support this model")
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names}
        return cls(**kw)


def resample_matrix(n_from: int, n_to: int) -> torch.Tensor:
    """(n_to, n_from) area resampling: entry [i, j] = share of target step i covered by source step j, rows summing to 1."""
    i = torch.arange(n_to, dtype=torch.float64)
    j = torch.arange(n_from, dtype=torch.float64)
    lo = torch.maximum(i[:, None] / n_to, j[None, :] / n_from)
    hi = torch.minimum((i[:, None] + 1) / n_to, (j[None, :] + 1) / n_from)
    return ((hi - lo).clamp_min(0) * n_to).float()


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x)


class Residual(nn.Module):
    def __init__(self, i: int, h: int, o: int) -> None:
        super().__init__()
        self.a, self.b, self.skip = nn.Linear(i, h), nn.Linear(h, o), nn.Linear(i, o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(F.silu(self.a(x))) + self.skip(x)


def rope_tables(pos: torch.Tensor, dh: int, xpos: bool, span: int = 512) -> tuple:
    """cos / sin for queries and keys, shared by every time layer of a forward. pos (B, L) -> ((cq, sq), (ck, sk)), each
    (B, 1, L, rot / 2). With xpos, zeta ** (+-pos / span) is folded in, so a score decays with query-key distance."""
    rot = dh // 2
    idx = torch.arange(0, rot, 2, device=pos.device).float()
    ang = pos[:, None, :, None].float() * 10_000.0 ** (-idx / rot)
    cos, sin = ang.cos(), ang.sin()
    if not xpos:
        return (cos, sin), (cos, sin)
    scale = ((idx / rot + 0.4) / 1.4) ** (pos[:, None, :, None].float() / span)
    return (cos * scale, sin * scale), (cos / scale, sin / scale)


def rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the first half of each head's dims. x: (B, H, L, Dh)."""
    rot = x.shape[-1] // 2
    cos, sin = cos.to(x), sin.to(x)
    x1, x2, rest = x[..., : rot // 2], x[..., rot // 2 : rot], x[..., rot:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos, rest], -1)


class Attention(nn.Module):
    def __init__(self, d: int, heads: int, use_rope: bool, gate_pre: bool = False) -> None:
        super().__init__()
        self.h, self.dh, self.use_rope, self.gate_pre = heads, d // heads, use_rope, gate_pre
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, d, bias=False)
        self.qn, self.kn = RMSNorm(self.dh), RMSNorm(self.dh)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos: tuple | None) -> torch.Tensor:
        b, length, d = x.shape
        q, k, v = self.qkv(x).view(b, length, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = self.qn(q), self.kn(k)
        if self.use_rope:
            q, k = rope_apply(q, *pos[0]), rope_apply(k, *pos[1])
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self._gated(o.transpose(1, 2).reshape(b, length, d), x)

    def self_only(self, x: torch.Tensor) -> torch.Tensor:
        """Exact result when every query's only allowed key is itself: attention returns its own value."""
        _, _, v = self.qkv(x).chunk(3, -1)
        return self._gated(v, x)

    def _gated(self, o: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate(x))
        return self.out(o * g) if self.gate_pre else self.out(o) * g


class SwiGLU(nn.Module):
    def __init__(self, d: int, ff: int) -> None:
        super().__init__()
        self.wi, self.wo = nn.Linear(d, 2 * ff, bias=False), nn.Linear(ff, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, g = self.wi(x).chunk(2, -1)
        return self.wo(a * F.silu(g))


class Block(nn.Module):
    def __init__(self, cfg: Config, variate: bool) -> None:
        super().__init__()
        self.tau = (3 * len(cfg.schedule)) ** -0.5 if cfg.res_scale else 1.0
        self.tn, self.time = RMSNorm(cfg.d), Attention(cfg.d, cfg.heads, use_rope=True, gate_pre=cfg.gate_pre)
        self.variate = Attention(cfg.d, cfg.heads, use_rope=False, gate_pre=cfg.gate_pre) if variate else None
        self.vn = RMSNorm(cfg.d) if variate else None
        self.fn, self.ff = RMSNorm(cfg.d), SwiGLU(cfg.d, cfg.ff)

    def forward(self, x, time_mask, pos, var_mask):
        t = self.tau
        x = x + t * self.time(self.tn(x), time_mask, pos)
        if self.variate is not None and isinstance(var_mask, str):  # "solo": every row alone in its series
            x = x + t * self.variate.self_only(self.vn(x))
        elif self.variate is not None:
            y = self.vn(x).transpose(0, 1)  # (L, R, d): attend across rows at each patch
            x = x + t * self.variate(y, var_mask, None).transpose(0, 1)
        return x + t * self.ff(self.fn(x))


class ChakraTSNet(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        p, d, q = cfg.patch, cfg.d, len(cfg.levels)
        self.embed = Residual((3 if cfg.time_channel else 2) * p, 4 * d, d)
        self.role = nn.Embedding(3, d)
        self.query = nn.Parameter(torch.zeros(d))  # unused by this decoder; kept so released checkpoints load strictly
        self.blocks = nn.ModuleList(Block(cfg, "v" in kind) for kind in cfg.schedule)
        self.n_skip = len(cfg.schedule) // 2 if cfg.unet else 0
        if cfg.unet:
            self.skip_gain = nn.Parameter(torch.ones(self.n_skip))
        self.norm = RMSNorm(d)
        self.head = Residual(d, 4 * d, p * q)
        self.sizes = tuple(sorted({p, *cfg.patches}))
        if cfg.patches:
            self.pemb = nn.Parameter(torch.zeros(len(self.sizes), d))
            for s in cfg.patches:
                self.register_buffer(f"rs_in_{s}", resample_matrix(s, p), persistent=False)
                self.register_buffer(f"rs_out_{s}", resample_matrix(p, s), persistent=False)
        gaps = [0.0] * q
        m = cfg.levels.index(0.5)
        for i in range(q):
            if i != m:
                j = i - 1 if i > m else i + 1
                gaps[i] = abs(cfg.levels[i] - cfg.levels[j])
        self.register_buffer("gaps", torch.tensor(gaps), persistent=False)

    # ---- inputs ---------------------------------------------------------------------------------------------------
    def scale(self, values, observed, role, patch):
        """Normalised inputs plus the per-patch (loc, scale, ok) forecasts are issued in. role (R, N)."""
        n = values.shape[1] // patch
        loc, sc, ok = prefix_stats(values, observed, patch)
        fut = role == FUTURE
        if fut.any():  # known-future rows are known over the whole window: whole-row statistics
            wl, ws = whole_stats(values, observed, n)
            loc = torch.where(fut, wl, loc)
            sc = torch.where(fut, ws, sc)
        # only known-future rows may stay linear as {0, 1} flags: their whole row is legitimately known
        flag = fut & is_binary(values, observed)[:, None]
        z = torch.where(observed, (values - loc.repeat_interleave(patch, 1)) / sc.repeat_interleave(patch, 1), 0.0)
        flag_t = flag.expand(-1, n).repeat_interleave(patch, 1)
        return torch.where(flag_t, torch.where(observed, values, 0.0), squash(z)), loc, sc, ok

    def tokens(self, z, observed, role, patch):
        r, t = z.shape
        parts = [z.view(r, t // patch, patch), (~observed).float().view(r, t // patch, patch)]
        if self.cfg.time_channel:
            parts.append((torch.arange(patch, device=z.device, dtype=z.dtype) / patch).expand(r, t // patch, patch))
        feats = torch.cat(parts, -1)
        x = self.embed(feats) if patch == self.cfg.patch else self._embed_resized(feats, patch)
        if self.cfg.patches:
            x = x + self.pemb[self.sizes.index(patch)]
        return x + self.role(role)

    def _embed_resized(self, feats, p):
        """The patch embedding reading size-p patches through the (P, p) area resampling of its input weights."""
        e, base, rs = self.embed, self.cfg.patch, getattr(self, f"rs_in_{p}")
        blocks = feats.shape[-1] // p

        def w(m):
            return torch.cat([m[:, k * base : (k + 1) * base] @ rs for k in range(blocks)], 1)

        return F.linear(F.silu(F.linear(feats, w(e.a.weight), e.a.bias)), e.b.weight, e.b.bias) + F.linear(
            feats, w(e.skip.weight), e.skip.bias
        )

    @staticmethod
    def positions(observed, group, patch):
        """Patch positions counted from each series' first observed patch (rows of one series share a clock)."""
        r, t = observed.shape
        obs_patch = observed.view(r, t // patch, patch).any(-1)
        n = obs_patch.shape[1]
        # in float (exact: at most 32,768 patches): Apple GPUs before torch 2.8 have no int64 scatter_reduce
        first = torch.where(obs_patch.any(-1), obs_patch.float().argmax(-1), 0).float()
        gfirst = torch.full((int(group.max()) + 1,), float(n), device=first.device)
        gfirst = gfirst.scatter_reduce(0, group, first, reduce="amin", include_self=False)
        pos = torch.arange(n, device=observed.device)[None, :] - gfirst[group][:, None].long()
        started = obs_patch.long().cumsum(-1) > 0
        return pos.clamp_min(0), obs_patch | started

    # ---- masks ----------------------------------------------------------------------------------------------------
    @staticmethod
    def firewall(group, role):
        """(R, R) bool, True = row i may read row j."""
        same = group[:, None] == group[None, :]
        q, k = role[:, None], role[None, :]
        cov_reads_target = (q != TARGET) & (k == TARGET)
        future_reads_causal = (q == FUTURE) & (k != FUTURE)
        eye = torch.eye(same.shape[-1], dtype=torch.bool, device=same.device)
        return (same & ~cov_reads_target & ~future_reads_causal) | eye

    @staticmethod
    def time_mask(valid, role):
        """(R, N, N): a token reads itself and the valid earlier tokens of its row; a known-future token its whole row."""
        n = valid.shape[1]
        causal = torch.ones(n, n, dtype=torch.bool, device=valid.device).tril()
        keys = valid[:, None, :]
        fut = role == FUTURE
        m = torch.where(fut[:, :, None], keys.expand(-1, n, n), causal & keys)
        return m | torch.eye(n, dtype=torch.bool, device=valid.device)

    # ---- forward --------------------------------------------------------------------------------------------------
    def forward(self, values: torch.Tensor, group: torch.Tensor, role: torch.Tensor, patch: int | None = None):
        """values (R, T) with NaN = unobserved, T a multiple of `patch`; group, role (R,). Returns the squashed quantiles
        (R, N, P, Q), token n forecasting patch n + 1, and the per-patch loc / scale / ok of the frame they are in."""
        p = patch or self.cfg.patch
        observed = torch.isfinite(values)
        values = torch.nan_to_num(values)
        role_p = role[:, None].expand(-1, values.shape[1] // p)
        z, loc, sc, ok = self.scale(values, observed, role_p, p)
        x = self.tokens(z, observed, role_p, p)
        pos, valid = self.positions(observed, group, p)
        r = x.shape[0]
        tm = self.time_mask(valid, role_p)[:, None]
        if bool((torch.bincount(group)[group] == 1).all()):
            vm = "solo"
        else:
            eye = torch.eye(r, dtype=torch.bool, device=x.device)
            vm = ((self.firewall(group, role)[None] & valid.T[:, None, :]) | eye)[:, None].contiguous()
        x = self._blocks(x, tm, pos, vm)
        return self._head(x, p), loc, sc, ok

    def _blocks(self, x, tm, pos, vm):
        pos = rope_tables(pos, self.cfg.d // self.cfg.heads, self.cfg.xpos, self.cfg.xpos_span)
        skips, first_pop = [], len(self.blocks) - self.n_skip
        for i, blk in enumerate(self.blocks):
            if i >= first_pop:
                x = x + self.skip_gain[i - first_pop] * skips.pop()
            x = blk(x, tm, pos, vm)
            if i < self.n_skip:
                skips.append(x)
        return x

    def _head(self, x, p):
        r, n, _ = x.shape
        q = len(self.cfg.levels)
        z = self.norm(x)
        out = self.head(z) if p == self.cfg.patch else self._head_resized(z, p)
        qs = monotone(out.view(r, n, p, q).float(), self.gaps, self.cfg.levels.index(0.5), self.cfg.width_factor)
        b = self.cfg.out_bound
        return b * torch.tanh(qs / b) if b > 0 else qs

    def _head_resized(self, z, p):
        """The head emitting size-p patches: each output step of each level is the area-resampled mix of the head's rows."""
        h, base, q, rs = self.head, self.cfg.patch, len(self.cfg.levels), getattr(self, f"rs_out_{p}")

        def w(m):
            return torch.einsum("ji,iqd->jqd", rs, m.view(base, q, -1)).reshape(p * q, -1)

        def b(v):
            return (rs @ v.view(base, q)).reshape(-1)

        return F.linear(F.silu(h.a(z)), w(h.b.weight), b(h.b.bias)) + F.linear(z, w(h.skip.weight), b(h.skip.bias))


def monotone(raw: torch.Tensor, gaps: torch.Tensor, mid: int, width: float) -> torch.Tensor:
    """Free median; every other level = median -/+ cumulative positive steps sized by the probability gap to its inner
    neighbour, so quantiles never cross."""
    inc = width * gaps * F.softplus(raw)
    med = raw[..., mid : mid + 1]
    up = inc[..., mid + 1 :].cumsum(-1)
    down = inc[..., :mid].flip(-1).cumsum(-1).flip(-1)
    return torch.cat([med - down, med, med + up], -1)
