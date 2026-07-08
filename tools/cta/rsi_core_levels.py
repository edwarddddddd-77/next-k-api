"""RSI Core Levels Heatmap [BigBeluga] -?simplified Python port.

Maps RSI/signal crossovers to active support/resistance price levels on chart.
https://www.tradingview.com/script/g4p5t6bB-RSI-Core-Levels-Heatmap-BigBeluga/
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class RSILevel:
    direction: int  # 1 support, -1 resistance
    bot: float
    top: float
    born_ms: int
    rsi_val: float
    active: bool = True


@dataclass
class RSICoreConfig:
    rsi_len: int = 14
    signal_len: int = 9
    lookback: int = 10
    max_active: int = 30
    overlap_pct: float = 0.003  # block new level if within 0.3% of existing
    zone_pct: float = 0.001  # level thickness for overlap box


@dataclass
class RSICoreState:
    levels: list[RSILevel] = field(default_factory=list)
    rsi: float = 50.0
    signal: float = 50.0


def _rsi(cl: pd.Series, length: int) -> pd.Series:
    delta = cl.diff()
    up = delta.clip(lower=0)
    dn = (-delta).clip(lower=0)
    avg_up = up.ewm(alpha=1 / length, adjust=False).mean()
    avg_dn = dn.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_up / avg_dn.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _overlaps(levels: list[RSILevel], bot: float, top: float) -> bool:
    for item in levels:
        if not item.active:
            continue
        if (top <= item.top and top >= item.bot) or (bot >= item.bot and bot <= item.top):
            return True
        if (top >= item.top and bot <= item.bot):
            return True
    return False


def _prune_fifo(levels: list[RSILevel], max_active: int) -> None:
    active = [lv for lv in levels if lv.active]
    if len(active) <= max_active:
        return
    drop = len(active) - max_active
    for lv in levels:
        if drop <= 0:
            break
        if lv.active:
            lv.active = False
            drop -= 1


def build_rsi_core_levels_df(df: pd.DataFrame, cfg: RSICoreConfig = RSICoreConfig()) -> pd.DataFrame:
    """Per-bar active RSI core levels. Returns df with open_time + rsi + n_support + n_resist."""
    cl = df["close"].astype(float)
    hi = df["high"].astype(float)
    lo = df["low"].astype(float)
    rsi = _rsi(cl, cfg.rsi_len)
    signal = rsi.ewm(span=cfg.signal_len, adjust=False).mean()

    levels: list[RSILevel] = []
    n = len(df)
    rsi_out = np.full(n, np.nan)
    sig_out = np.full(n, np.nan)
    sup_px = np.full(n, np.nan)  # nearest active support mid
    res_px = np.full(n, np.nan)  # nearest active resistance mid

    for i in range(n):
        t = int(df.iloc[i]["open_time"])
        c = float(cl.iloc[i])
        rv = float(rsi.iloc[i]) if not np.isnan(rsi.iloc[i]) else np.nan
        sv = float(signal.iloc[i]) if not np.isnan(signal.iloc[i]) else np.nan
        rsi_out[i], sig_out[i] = rv, sv

        for lv in levels:
            if not lv.active:
                continue
            if lv.direction == 1 and c < lv.bot:
                lv.active = False
            elif lv.direction == -1 and c > lv.top:
                lv.active = False

        if i >= 1 and not np.isnan(rv) and not np.isnan(sv):
            prev_r, prev_s = float(rsi.iloc[i - 1]), float(signal.iloc[i - 1])
            bull_cross = prev_r <= prev_s and rv > sv
            bear_cross = prev_r >= prev_s and rv < sv
            lb = max(0, i - cfg.lookback + 1)
            if bull_cross:
                px = float(lo.iloc[lb : i + 1].min())
                band = px * cfg.zone_pct
                bot, top = px - band, px + band
                if not _overlaps(levels, bot, top):
                    levels.append(RSILevel(1, bot, top, t, rv))
            if bear_cross:
                px = float(hi.iloc[lb : i + 1].max())
                band = px * cfg.zone_pct
                bot, top = px - band, px + band
                if not _overlaps(levels, bot, top):
                    levels.append(RSILevel(-1, bot, top, t, rv))

        _prune_fifo(levels, cfg.max_active)
        supports = [lv for lv in levels if lv.active and lv.direction == 1]
        resists = [lv for lv in levels if lv.active and lv.direction == -1]
        if supports:
            sup_px[i] = min((lv.top + lv.bot) / 2 for lv in supports)
        if resists:
            res_px[i] = max((lv.top + lv.bot) / 2 for lv in resists)

    out = df[["open_time"]].copy()
    out["rsi"] = rsi_out
    out["rsi_signal"] = sig_out
    out["near_support"] = sup_px
    out["near_resist"] = res_px
    return out


def near_level_row(row: pd.Series, *, side: str, entry_px: float, proximity_pct: float) -> bool:
    if side == "long":
        lv = row.get("near_support")
    else:
        lv = row.get("near_resist")
    if lv is None or (isinstance(lv, float) and np.isnan(lv)):
        return False
    lv = float(lv)
    return abs(entry_px - lv) / entry_px <= proximity_pct


def row_at_time(levels_df: pd.DataFrame, t_ms: int) -> pd.Series:
    idx = int(levels_df["open_time"].searchsorted(t_ms, side="right") - 1)
    if idx < 0:
        return pd.Series(dtype=float)
    return levels_df.iloc[idx]

