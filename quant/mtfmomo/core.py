"""MtfMomo2xA 信号核心（对齐 Jesse MtfMomo2xA）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from quant.common.jesse_indicators import atr_last as jesse_atr_last

# (open_time_ms, open, high, low, close)
HourOhlc = tuple[int, float, float, float, float]

@dataclass(frozen=True)
class MomoLevels:
    atr: float
    donchian_upper: float
    donchian_lower: float
    ema_exit: float
    trend_agree: int


def ema_last(closes: Sequence[float], period: int) -> float | None:
    if len(closes) < period:
        return None
    arr = np.asarray(closes, dtype=float)
    alpha = 2.0 / (period + 1.0)
    val = float(arr[0])
    for px in arr[1:]:
        val = alpha * float(px) + (1.0 - alpha) * val
    return val


def atr_last(high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int = 14) -> float | None:
    return jesse_atr_last(high, low, close, period)


def resample_utc(
    bars: Sequence[HourOhlc],
    period_hours: int,
) -> tuple[list[float], list[float], list[float]]:
    """按 UTC 整点桶聚合（4h / 24h），对齐交易所 K 线边界。"""
    if period_hours <= 1 or not bars:
        closes = [float(b[4]) for b in bars]
        highs = [float(b[2]) for b in bars]
        lows = [float(b[3]) for b in bars]
        return closes, highs, lows
    period_ms = int(period_hours) * 3_600_000
    buckets: dict[int, list[float]] = {}
    for ts, o, h, l, c in bars:
        key = int(ts) // period_ms
        row = buckets.get(key)
        if row is None:
            buckets[key] = [float(o), float(h), float(l), float(c)]
        else:
            row[1] = max(row[1], float(h))
            row[2] = min(row[2], float(l))
            row[3] = float(c)
    ordered = [buckets[k] for k in sorted(buckets.keys())]
    highs = [r[1] for r in ordered]
    lows = [r[2] for r in ordered]
    closes = [r[3] for r in ordered]
    return closes, highs, lows


def resample_closes_ohlc(
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    factor: int,
) -> tuple[list[float], list[float], list[float]]:
    """Legacy 固定窗口聚合（测试兼容）。"""
    if factor <= 1 or not closes:
        return list(closes), list(highs), list(lows)
    out_c: list[float] = []
    out_h: list[float] = []
    out_l: list[float] = []
    n = len(closes)
    start = n % factor
    i = start
    while i < n:
        chunk_c = closes[i : i + factor]
        chunk_h = highs[i : i + factor]
        chunk_l = lows[i : i + factor]
        if chunk_c:
            out_c.append(float(chunk_c[-1]))
            out_h.append(float(max(chunk_h)))
            out_l.append(float(min(chunk_l)))
        i += factor
    return out_c, out_h, out_l

def trend_dir(closes: Sequence[float], period: int) -> int:
    if len(closes) < period + 4:
        return 0
    en = ema_last(closes, period)
    ep = ema_last(closes[:-2], period)
    if en is None or ep is None:
        return 0
    c = float(closes[-1])
    if c > en and en > ep:
        return 1
    if c < en and en < ep:
        return -1
    return 0


def donchian_prev(high: Sequence[float], low: Sequence[float], period: int) -> tuple[float | None, float | None]:
    if len(high) < period + 1:
        return None, None
    window_h = high[-(period + 1) : -1]
    window_l = low[-(period + 1) : -1]
    if not window_h:
        return None, None
    return float(max(window_h)), float(min(window_l))


def compute_levels(
    hour_bars: Sequence[HourOhlc],
    *,
    entry_lb: int = 26,
    ema_exit: int = 35,
    ema_4h: int = 21,
    ema_1d: int = 16,
    anchor_4h_closes: Sequence[float] | None = None,
    anchor_1d_closes: Sequence[float] | None = None,
) -> MomoLevels | None:
    if len(hour_bars) < 60:
        return None
    hour_closes = [float(b[4]) for b in hour_bars]
    hour_highs = [float(b[2]) for b in hour_bars]
    hour_lows = [float(b[3]) for b in hour_bars]
    atr = atr_last(hour_highs, hour_lows, hour_closes, 14)
    if atr is None or atr <= 0:
        return None
    upper, lower = donchian_prev(hour_highs, hour_lows, entry_lb)
    if upper is None or lower is None:
        return None
    ema_x = ema_last(hour_closes, ema_exit)
    if ema_x is None:
        return None
    if anchor_4h_closes is not None and anchor_1d_closes is not None:
        c4 = list(anchor_4h_closes)
        c1d = list(anchor_1d_closes)
    else:
        c4, _, _ = resample_utc(hour_bars, 4)
        c1d, _, _ = resample_utc(hour_bars, 24)
    d4 = trend_dir(c4, ema_4h)
    d1 = trend_dir(c1d, ema_1d)
    agree = d4 if (d4 != 0 and d4 == d1) else 0
    return MomoLevels(
        atr=float(atr),
        donchian_upper=float(upper),
        donchian_lower=float(lower),
        ema_exit=float(ema_x),
        trend_agree=int(agree),
    )


def compute_levels_from_series(
    hour_closes: Sequence[float],
    hour_highs: Sequence[float],
    hour_lows: Sequence[float],
    *,
    entry_lb: int = 26,
    ema_exit: int = 35,
    ema_4h: int = 21,
    ema_1d: int = 16,
) -> MomoLevels | None:
    """仅序列输入时使用固定窗口聚合（测试）。"""
    if len(hour_closes) < 60:
        return None
    bars: list[HourOhlc] = [
        (i * 3_600_000, 0.0, float(h), float(l), float(c))
        for i, (c, h, l) in enumerate(zip(hour_closes, hour_highs, hour_lows))
    ]
    return compute_levels(
        bars,
        entry_lb=entry_lb,
        ema_exit=ema_exit,
        ema_4h=ema_4h,
        ema_1d=ema_1d,
    )

def entry_signal(close: float, levels: MomoLevels) -> int:
    if levels.trend_agree == 1 and close > levels.donchian_upper:
        return 1
    if levels.trend_agree == -1 and close < levels.donchian_lower:
        return -1
    return 0


def stop_tp_prices(entry: float, side: int, levels: MomoLevels, *, stop_atr: float, tp_atr: float) -> tuple[float, float]:
    if side > 0:
        return entry - stop_atr * levels.atr, entry + tp_atr * levels.atr
    return entry + stop_atr * levels.atr, entry - tp_atr * levels.atr


def should_ema_exit(close: float, side: int, ema_exit: float) -> bool:
    if side > 0 and close < ema_exit:
        return True
    if side < 0 and close > ema_exit:
        return True
    return False


def bar_hits_stop_tp(
    *,
    side: int,
    high: float,
    low: float,
    stop: float,
    tp: float,
) -> str | None:
    if side > 0:
        if low <= stop:
            return "stop"
        if high >= tp:
            return "tp"
    else:
        if high >= stop:
            return "stop"
        if low <= tp:
            return "tp"
    return None
