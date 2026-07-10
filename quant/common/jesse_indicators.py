"""Jesse ta.* 对齐指标（默认 period：kama/atr/adx/chop=14，bb=20）。"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def wilder_rma(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(values, np.nan, dtype=float)
    if len(values) < period:
        return out
    out[period - 1] = float(np.nanmean(values[:period]))
    for i in range(period, len(values)):
        out[i] = (out[i - 1] * (period - 1) + values[i]) / period
    return out


def atr_last(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int = 14,
) -> float | None:
    """Jesse ta.atr — Wilder RMA of true range."""
    if len(close) < period + 1:
        return None
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    rma = wilder_rma(tr, period)
    val = rma[-1]
    return float(val) if np.isfinite(val) else None


def kama_last(
    closes: Sequence[float],
    *,
    period: int = 14,
    fast_length: int = 2,
    slow_length: int = 30,
) -> float | None:
    """Jesse ta.kama — Kaufman Adaptive Moving Average."""
    arr = np.asarray(closes, dtype=float)
    if len(arr) < period + 1:
        return None
    fast_sc = 2.0 / (fast_length + 1.0)
    slow_sc = 2.0 / (slow_length + 1.0)
    kama = float(arr[0])
    for i in range(1, len(arr)):
        if i < period:
            kama = float(arr[i])
            continue
        change = abs(float(arr[i]) - float(arr[i - period]))
        volatility = float(np.sum(np.abs(np.diff(arr[i - period : i + 1]))))
        er = change / volatility if volatility > 0 else 0.0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama = kama + sc * (float(arr[i]) - kama)
    return float(kama)


def adx_last(high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int = 14) -> float | None:
    """Jesse ta.adx — Wilder ADX."""
    if len(close) <= 2 * period:
        return None
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    up = h[1:] - h[:-1]
    down = l[:-1] - l[1:]
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    atr_s = wilder_rma(tr, period)
    plus_di = 100.0 * wilder_rma(plus_dm, period) / np.maximum(atr_s, 1e-12)
    minus_di = 100.0 * wilder_rma(minus_dm, period) / np.maximum(atr_s, 1e-12)
    dx = 100.0 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-12)
    dx = np.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)
    adx_s = wilder_rma(dx, period)
    for i in range(len(adx_s) - 1, -1, -1):
        val = float(adx_s[i])
        if np.isfinite(val):
            return val
    return None


def chop_last(
    high: Sequence[float],
    low: Sequence[float],
    close: Sequence[float],
    period: int = 14,
    *,
    scalar: float = 100.0,
    drift: int = 1,
) -> float | None:
    """Jesse ta.chop — Choppiness Index."""
    if len(close) < period + drift:
        return None
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    end = len(c) - drift
    start = end - period
    if start < 1:
        return None
    tr = np.maximum(
        h[1:] - l[1:],
        np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])),
    )
    sum_tr = float(np.sum(tr[start:end]))
    hh = float(np.max(h[start:end]))
    ll = float(np.min(l[start:end]))
    if hh <= ll or sum_tr <= 0:
        return float(scalar)
    return float(scalar * math.log10(sum_tr / (hh - ll)) / math.log10(period))


def bollinger_bands_width_ratio(
    closes: Sequence[float],
    period: int = 20,
    mult: float = 2.0,
) -> float | None:
    """Jesse ta.bollinger_bands_width — 返回比率（策略侧 ×100 与 7 比较）。"""
    if len(closes) < period:
        return None
    arr = np.asarray(closes[-period:], dtype=float)
    mid = float(np.mean(arr))
    if mid <= 0:
        return None
    std = float(np.std(arr, ddof=0))
    upper = mid + mult * std
    lower = mid - mult * std
    return (upper - lower) / mid


def bollinger_bands_width_pct(closes: Sequence[float], period: int = 20, mult: float = 2.0) -> float | None:
    ratio = bollinger_bands_width_ratio(closes, period, mult)
    return None if ratio is None else float(ratio * 100.0)
