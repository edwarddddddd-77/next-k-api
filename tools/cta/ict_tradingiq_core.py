"""Trading-IQ ICT Library primitives (Python port).

Aligned to the open-source Pine library `ICTlibrary` by Trading-IQ:
https://www.tradingview.com/script/UaWyGzvU-Trading-IQ-ICT-Library/

Covers FVG, order/breaker blocks, displacements, liquidity sweeps, HTF bias,
and helper structures used by ICT Master Suite strategy models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# Trading-IQ documented defaults
OTE_LEVEL_DEFAULT = 0.79
OTE_TP_EXTENSION = 0.5
M2022_ENTRY_PCT = 0.50
SILVER_BULLET_WINDOWS_NY = [
    ("03:00", "04:00"),
    ("10:00", "11:00"),
    ("14:00", "15:00"),
]
LIQUIDITY_RAID_SESSION_NY = ("13:30", "16:00")


@dataclass
class FVG:
    direction: int  # 1 up, -1 down
    top: float
    bot: float
    born_ms: int
    born_idx: int
    used: bool = False

    @property
    def mid(self) -> float:
        return (self.top + self.bot) / 2.0

    def invalidated(self, close: float) -> bool:
        if self.direction == 1:
            return close < self.bot
        return close > self.top


@dataclass
class OrderBlock:
    direction: int  # 1 bull, -1 bear
    top: float
    bot: float
    born_ms: int
    status: str = "OB"  # OB or BB

    @property
    def is_breaker(self) -> bool:
        return self.status == "BB"


@dataclass
class SweepEvent:
    direction: int  # 1 = swept lows (bullish setup), -1 = swept highs
    price: float
    born_ms: int
    born_idx: int


def ms_to_ny(ms: int) -> pd.Timestamp:
    return pd.Timestamp(ms, unit="ms", tz="UTC").tz_convert("America/New_York")


def in_ny_window(ms: int, start_hm: str, end_hm: str) -> bool:
    ts = ms_to_ny(ms)
    lo = pd.Timestamp(f"{ts.strftime('%Y-%m-%d')} {start_hm}", tz="America/New_York")
    hi = pd.Timestamp(f"{ts.strftime('%Y-%m-%d')} {end_hm}", tz="America/New_York")
    return lo <= ts < hi


def log_zscore(close: pd.Series, *, window: int = 20) -> pd.Series:
    lr = np.log(close / close.shift(1))
    mu = lr.rolling(window, min_periods=5).mean()
    sd = lr.rolling(window, min_periods=5).std().replace(0, np.nan)
    return (lr - mu) / sd


def iqzz_direction(
    high: pd.Series,
    low: pd.Series,
    *,
    atr: pd.Series,
    mult: float = 1.0,
) -> pd.Series:
    """Simplified IQZZ: +1 up, -1 down, 0 undecided."""
    n = len(high)
    out = np.zeros(n, dtype=int)
    if n < 3:
        return pd.Series(out, index=high.index)
    piv_hi = float(high.iloc[0])
    piv_lo = float(low.iloc[0])
    direction = 0
    for i in range(1, n):
        thr = float(atr.iloc[i]) * mult if not np.isnan(atr.iloc[i]) else 0.0
        h = float(high.iloc[i])
        l = float(low.iloc[i])
        if direction >= 0 and h - piv_lo >= thr:
            direction = 1
            piv_hi = h
        elif direction <= 0 and piv_hi - l >= thr:
            direction = -1
            piv_lo = l
        if direction == 1 and h > piv_hi:
            piv_hi = h
        if direction == -1 and l < piv_lo:
            piv_lo = l
        out[i] = direction
    return pd.Series(out, index=high.index)


def detect_fvgs(df: pd.DataFrame) -> List[FVG]:
    """Standard 3-candle FVG on chart TF (Trading-IQ FVG UDT)."""
    out: List[FVG] = []
    for i in range(2, len(df)):
        lo_i = float(df.iloc[i]["low"])
        hi_i = float(df.iloc[i]["high"])
        hi_2 = float(df.iloc[i - 2]["high"])
        lo_2 = float(df.iloc[i - 2]["low"])
        t = int(df.iloc[i]["open_time"])
        if lo_i > hi_2:
            out.append(FVG(1, lo_i, hi_2, t, i))
        elif hi_i < lo_2:
            out.append(FVG(-1, lo_2, hi_i, t, i))
    return out


def prune_fvgs(fvgs: List[FVG], close: float) -> List[FVG]:
    return [g for g in fvgs if not g.invalidated(close)]


def detect_displacement(df: pd.DataFrame, *, z_thr: float = 1.2) -> pd.Series:
    """Displacement via log-return z-score (Trading-IQ displacement())."""
    z = log_zscore(df["close"].astype(float))
    body = (df["close"] - df["open"]).abs()
    avg_body = body.rolling(20, min_periods=5).mean()
    up = (df["close"] > df["open"]) & (z > z_thr) & (body > avg_body)
    dn = (df["close"] < df["open"]) & (z < -z_thr) & (body > avg_body)
    return np.where(up, 1, np.where(dn, -1, 0))


def build_order_blocks(df: pd.DataFrame, *, max_blocks: int = 40) -> Tuple[List[OrderBlock], List[OrderBlock]]:
    """OB.sign: -1 bullish OB, +1 bearish OB; violation -> BB."""
    disp = detect_displacement(df)
    bull_obs: List[OrderBlock] = []
    bear_obs: List[OrderBlock] = []
    bull_bb: List[OrderBlock] = []
    bear_bb: List[OrderBlock] = []

    for i in range(1, len(df)):
        t = int(df.iloc[i]["open_time"])
        cl = float(df.iloc[i]["close"])
        if disp[i] == 1:
            j = i - 1
            while j >= 0 and df.iloc[j]["close"] >= df.iloc[j]["open"]:
                j -= 1
            if j >= 0:
                bull_obs.append(
                    OrderBlock(1, float(df.iloc[j]["high"]), float(df.iloc[j]["low"]), t, "OB")
                )
                bull_obs = bull_obs[-max_blocks:]
        elif disp[i] == -1:
            j = i - 1
            while j >= 0 and df.iloc[j]["close"] <= df.iloc[j]["open"]:
                j -= 1
            if j >= 0:
                bear_obs.append(
                    OrderBlock(-1, float(df.iloc[j]["high"]), float(df.iloc[j]["low"]), t, "OB")
                )
                bear_obs = bear_obs[-max_blocks:]

        for ob in bull_obs:
            if ob.status == "OB" and cl < ob.bot:
                ob.status = "BB"
                bear_bb.append(OrderBlock(-1, ob.top, ob.bot, t, "BB"))
                bear_bb = bear_bb[-max_blocks:]
        for ob in bear_obs:
            if ob.status == "OB" and cl > ob.top:
                ob.status = "BB"
                bull_bb.append(OrderBlock(1, ob.top, ob.bot, t, "BB"))
                bull_bb = bull_bb[-max_blocks:]

    return bull_bb, bear_bb


def detect_sweeps(df: pd.DataFrame, *, lookback: int = 20) -> Tuple[pd.Series, pd.Series]:
    swing_hi = df["high"].shift(1).rolling(lookback, min_periods=3).max()
    swing_lo = df["low"].shift(1).rolling(lookback, min_periods=3).min()
    sweep_low = (df["low"] < swing_lo) & (df["close"] > swing_lo)
    sweep_high = (df["high"] > swing_hi) & (df["close"] < swing_hi)
    return sweep_low.fillna(False), sweep_high.fillna(False)


def add_swings(df: pd.DataFrame, *, left: int = 2, right: int = 2) -> pd.DataFrame:
    out = df.copy()
    n = len(out)
    swing_hi = np.zeros(n, dtype=bool)
    swing_lo = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        hi = float(out.iloc[i]["high"])
        lo = float(out.iloc[i]["low"])
        if all(hi > float(out.iloc[j]["high"]) for j in range(i - left, i + right + 1) if j != i):
            swing_hi[i] = True
        if all(lo < float(out.iloc[j]["low"]) for j in range(i - left, i + right + 1) if j != i):
            swing_lo[i] = True
    out["swing_hi"] = swing_hi
    out["swing_lo"] = swing_lo
    out["last_swing_hi"] = np.where(swing_hi, out["high"], np.nan)
    out["last_swing_lo"] = np.where(swing_lo, out["low"], np.nan)
    out["last_swing_hi"] = out["last_swing_hi"].ffill()
    out["last_swing_lo"] = out["last_swing_lo"].ffill()
    return out


def htf_bias_hl(h: float, l: float, h2: float, l2: float) -> int:
    """Silver Bullet sessionBias from userTF H/L/H2/L2."""
    if l > l2 and h >= h2:
        return 1
    if h < h2 and l <= l2:
        return -1
    return 0


def merge_htf_columns(df_ltf: pd.DataFrame, df_htf: pd.DataFrame) -> pd.DataFrame:
    h = df_htf.copy()
    h["H"] = h["high"]
    h["L"] = h["low"]
    h["H2"] = h["high"].shift(2)
    h["L2"] = h["low"].shift(2)
    h["bias"] = [
        htf_bias_hl(float(r.H), float(r.L), float(r.H2) if not np.isnan(r.H2) else float(r.H),
                  float(r.L2) if not np.isnan(r.L2) else float(r.L))
        for r in h.itertuples()
    ]
    m = h[["open_time", "bias", "H", "L", "H2", "L2"]].rename(columns={"open_time": "htf_open"})
    return pd.merge_asof(
        df_ltf.sort_values("open_time"),
        m.sort_values("htf_open"),
        left_on="open_time",
        right_on="htf_open",
        direction="backward",
    )


def add_daily_levels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["open_time"], unit="ms", utc=True)
    day = ts.dt.strftime("%Y-%m-%d")
    daily = out.assign(_day=day).groupby("_day").agg(dh=("high", "max"), dl=("low", "min"))
    daily["prev_dh"] = daily["dh"].shift(1)
    daily["prev_dl"] = daily["dl"].shift(1)
    m = out.assign(_day=day).merge(daily[["prev_dh", "prev_dl"]], left_on="_day", right_index=True, how="left")
    out["prev_day_high"] = m["prev_dh"].values
    out["prev_day_low"] = m["prev_dl"].values
    return out


def nearest_fvg_to_level(fvgs: List[FVG], level: float, direction: int) -> Optional[FVG]:
    """m2022.mEntryDistance -?FVG closest to target % of range."""
    cands = [g for g in fvgs if g.direction == direction and not g.used]
    if not cands:
        return None
    return min(cands, key=lambda g: abs(g.mid - level))


def unicorn_long_entry(block_top: float, fvg_top: float) -> float:
    return max(block_top, fvg_top)


def unicorn_short_entry(block_bot: float, fvg_bot: float) -> float:
    return min(block_bot, fvg_bot)


def ote_long_prices(swing_lo: float, swing_hi: float, level: float = OTE_LEVEL_DEFAULT) -> Tuple[float, float, float]:
    entry = swing_hi - level * (swing_hi - swing_lo)
    tp = swing_hi + OTE_TP_EXTENSION * (swing_hi - swing_lo)
    return entry, swing_lo, tp


def ote_short_prices(swing_lo: float, swing_hi: float, level: float = OTE_LEVEL_DEFAULT) -> Tuple[float, float, float]:
    entry = swing_lo + level * (swing_hi - swing_lo)
    tp = swing_lo - OTE_TP_EXTENSION * (swing_hi - swing_lo)
    return entry, swing_hi, tp
