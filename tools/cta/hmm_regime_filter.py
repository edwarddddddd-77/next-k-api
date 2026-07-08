"""3-state HMM regime filter - Python port of TheRealDrip2Rip Pine indicator.

States: 0=Trend Up, 1=Range, 2=Trend Down.
Returns per-bar posterior probabilities and confirmed official regime.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class HMMConfig:
    obs_len: int = 4
    stat_len: int = 150
    mu_k: float = 1.0
    stick: float = 0.97
    confirm_bars: int = 3
    dom_thresh: float = 0.50


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _gpdf(x: float, mean: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    z = (x - mean) / sigma
    return float(np.exp(-0.5 * z * z) / sigma)


def compute_hmm_regime(df: pd.DataFrame, cfg: HMMConfig = HMMConfig()) -> pd.DataFrame:
    """Run forward-filter HMM on OHLCV bars. Adds columns p_up/p_range/p_down/official."""
    out = df.copy().reset_index(drop=True)
    atrv = _atr(out, 14)
    ret = np.where(atrv > 0, (out["close"] - out["close"].shift(1)) / atrv, 0.0)
    obs = pd.Series(ret, index=out.index).ewm(span=cfg.obs_len, adjust=False).mean()
    sigma = obs.rolling(cfg.stat_len, min_periods=20).std()

    n = len(out)
    p_up = np.full(n, np.nan)
    p_rng = np.full(n, np.nan)
    p_dn = np.full(n, np.nan)
    official = np.full(n, -1, dtype=int)
    raw_state = np.full(n, -1, dtype=int)

    alpha = np.array([1 / 3, 1 / 3, 1 / 3], dtype=float)
    off = 1.0 - cfg.stick
    cand_prev = -1
    streak = 0
    off_regime = -1

    for i in range(n):
        sig = sigma.iloc[i]
        ob = obs.iloc[i]
        if pd.isna(ob) or pd.isna(sig) or sig <= 0:
            continue

        a0, a1, a2 = alpha
        pr0 = a0 * cfg.stick + a1 * (off * 0.5) + a2 * (off * 0.25)
        pr1 = a0 * (off * 0.75) + a1 * cfg.stick + a2 * (off * 0.75)
        pr2 = a0 * (off * 0.25) + a1 * (off * 0.5) + a2 * cfg.stick

        m_u = cfg.mu_k * sig
        e0 = _gpdf(ob, m_u, sig)
        e1 = _gpdf(ob, 0.0, sig * 0.8)
        e2 = _gpdf(ob, -m_u, sig)
        u0, u1, u2 = pr0 * e0, pr1 * e1, pr2 * e2
        tot = u0 + u1 + u2
        if tot <= 0:
            continue
        alpha = np.array([u0 / tot, u1 / tot, u2 / tot])
        p_up[i], p_rng[i], p_dn[i] = alpha

        raw = 0 if alpha[0] >= alpha[1] and alpha[0] >= alpha[2] else (1 if alpha[1] >= alpha[2] else 2)
        raw_state[i] = raw
        pmax = alpha[raw]
        cand = raw if pmax >= cfg.dom_thresh else -1

        if cand == cand_prev and cand != -1:
            streak += 1
        elif cand == -1:
            streak = 0
        else:
            streak = 1
        cand_prev = cand
        if cand != -1 and cand != off_regime and streak >= cfg.confirm_bars:
            off_regime = cand
        official[i] = off_regime

    out["p_up"] = p_up
    out["p_range"] = p_rng
    out["p_down"] = p_dn
    out["hmm_raw"] = raw_state
    out["hmm_official"] = official
    return out


def regime_at_time(regime_df: pd.DataFrame, t_ms: int) -> int:
    """Last known official regime at or before t_ms. -1 = unknown."""
    sub = regime_df[regime_df["open_time"] <= t_ms]
    if sub.empty:
        return -1
    v = int(sub.iloc[-1]["hmm_official"])
    return v
