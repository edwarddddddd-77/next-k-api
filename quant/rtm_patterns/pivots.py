"""Pivot detection and price-structure helpers for RTM patterns."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from quant.rtm_patterns.config import RTMConfig
from quant.rtm_patterns.types import Pivot, SRFlipLevel


def find_pivots(
    high: np.ndarray,
    low: np.ndarray,
    *,
    left: int,
    right: int,
) -> list[Pivot]:
    """Local swing highs and lows (non-repainting, confirmed after `right` bars)."""
    n = len(high)
    pivots: list[Pivot] = []
    if n < left + right + 1:
        return pivots

    for i in range(left, n - right):
        h_win = high[i - left : i + right + 1]
        l_win = low[i - left : i + right + 1]
        if high[i] == np.max(h_win) and np.sum(h_win == high[i]) == 1:
            pivots.append(Pivot(i, float(high[i]), "high"))
        if low[i] == np.min(l_win) and np.sum(l_win == low[i]) == 1:
            pivots.append(Pivot(i, float(low[i]), "low"))
    return pivots


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    if n < period:
        return out
    for i in range(period - 1, n):
        out[i] = float(np.mean(tr[i - period + 1 : i + 1]))
    return out


def is_equal(a: float, b: float, ref: float, tolerance_pct: float) -> bool:
    if ref <= 0:
        return False
    return abs(a - b) / ref * 100.0 <= tolerance_pct


def pivots_before(pivots: Sequence[Pivot], bar_index: int) -> list[Pivot]:
    return [p for p in pivots if p.index <= bar_index]


def alternating_sequence(pivots: Sequence[Pivot], count: int) -> list[Pivot] | None:
    """Last `count` pivots with alternating high/low."""
    if len(pivots) < count:
        return None
    seq = list(pivots[-count:])
    for i in range(1, len(seq)):
        if seq[i].kind == seq[i - 1].kind:
            return None
    return seq


def bearish_qm_structure(seq4: Sequence[Pivot]) -> tuple[float, float, float, float, tuple[int, ...]] | None:
    """H1 -> L1 -> H2(HH) -> L2(LL). Returns (qml, head, ll, l1, indices)."""
    if len(seq4) != 4:
        return None
    h1, l1, h2, l2 = seq4
    if h1.kind != "high" or l1.kind != "low" or h2.kind != "high" or l2.kind != "low":
        return None
    if h2.price <= h1.price or l2.price >= l1.price:
        return None
    return h1.price, h2.price, l2.price, l1.price, tuple(p.index for p in seq4)


def bullish_qm_structure(seq4: Sequence[Pivot]) -> tuple[float, float, float, float, tuple[int, ...]] | None:
    """L1 -> H1 -> L2(LL) -> H2(HH). Returns (qml, head, hh, h1, indices)."""
    if len(seq4) != 4:
        return None
    l1, h1, l2, h2 = seq4
    if l1.kind != "low" or h1.kind != "high" or l2.kind != "low" or h2.kind != "high":
        return None
    if l2.price >= l1.price or h2.price <= h1.price:
        return None
    return l1.price, l2.price, h2.price, h1.price, tuple(p.index for p in seq4)


def equal_level_clusters(
    pivots: Sequence[Pivot],
    *,
    kind: str,
    tolerance_pct: float,
    min_touches: int,
    max_touches: int,
) -> list[tuple[float, list[Pivot]]]:
    """Group pivots of same kind within tolerance."""
    filtered = [p for p in pivots if p.kind == kind]
    clusters: list[tuple[float, list[Pivot]]] = []
    used: set[int] = set()
    for i, p in enumerate(filtered):
        if i in used:
            continue
        group = [p]
        used.add(i)
        for j in range(i + 1, len(filtered)):
            if j in used:
                continue
            ref = (p.price + filtered[j].price) / 2.0
            if is_equal(p.price, filtered[j].price, ref, tolerance_pct):
                group.append(filtered[j])
                used.add(j)
        if min_touches <= len(group) <= max_touches:
            level = float(np.mean([g.price for g in group]))
            clusters.append((level, group))
    return clusters


def is_bearish_engulfing(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    idx: int,
    *,
    body_ratio: float,
) -> bool:
    if idx < 1:
        return False
    prev_body = abs(close[idx - 1] - open_[idx - 1])
    body = abs(close[idx] - open_[idx])
    if body <= 0 or prev_body <= 0:
        return False
    bull_prev = close[idx - 1] > open_[idx - 1]
    bear_now = close[idx] < open_[idx]
    if not (bull_prev and bear_now):
        return False
    return body >= prev_body * body_ratio and close[idx] < open_[idx - 1] and open_[idx] > close[idx - 1]


def is_bullish_engulfing(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    idx: int,
    *,
    body_ratio: float,
) -> bool:
    if idx < 1:
        return False
    prev_body = abs(close[idx - 1] - open_[idx - 1])
    body = abs(close[idx] - open_[idx])
    if body <= 0 or prev_body <= 0:
        return False
    bear_prev = close[idx - 1] < open_[idx - 1]
    bull_now = close[idx] > open_[idx]
    if not (bear_prev and bull_now):
        return False
    return body >= prev_body * body_ratio and close[idx] > open_[idx - 1] and open_[idx] < close[idx - 1]


def trend_direction(close: np.ndarray, idx: int, lookback: int = 20) -> int:
    """1 up, -1 down, 0 flat."""
    start = max(0, idx - lookback)
    if idx <= start:
        return 0
    delta = close[idx] - close[start]
    if delta > 0:
        return 1
    if delta < 0:
        return -1
    return 0


def count_zigzags(pivots: Sequence[Pivot], start_idx: int, end_idx: int) -> int:
    sub = [p for p in pivots if start_idx <= p.index <= end_idx]
    if len(sub) < 2:
        return 0
    flips = 0
    for i in range(1, len(sub)):
        if sub[i].kind != sub[i - 1].kind:
            flips += 1
    return flips


def is_bearish_rejection(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    idx: int,
    level: float,
    *,
    body_ratio: float,
    wick_ratio: float,
) -> bool:
    """Sweep/touch above level, close below, with wick or engulf."""
    if is_bearish_engulfing(open_, high, low, close, idx, body_ratio=body_ratio):
        return high[idx] >= level and close[idx] < level
    body = abs(close[idx] - open_[idx])
    upper_wick = high[idx] - max(open_[idx], close[idx])
    if body <= 0:
        return False
    return high[idx] > level and close[idx] < level and upper_wick >= body * wick_ratio


def is_bullish_rejection(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    idx: int,
    level: float,
    *,
    body_ratio: float,
    wick_ratio: float,
) -> bool:
    if is_bullish_engulfing(open_, high, low, close, idx, body_ratio=body_ratio):
        return low[idx] <= level and close[idx] > level
    body = abs(close[idx] - open_[idx])
    lower_wick = min(open_[idx], close[idx]) - low[idx]
    if body <= 0:
        return False
    return low[idx] < level and close[idx] > level and lower_wick >= body * wick_ratio


def compute_sr_flips(
    close: np.ndarray,
    pivots: Sequence[Pivot],
    atr_vals: np.ndarray,
    *,
    break_atr_mult: float,
) -> list[SRFlipLevel]:
    """Track levels that broke and flipped S<->R (cheat-sheet SR flip / IQM)."""
    flips: list[SRFlipLevel] = []
    n = len(close)

    for p in pivots:
        atr_val = atr_vals[min(p.index, n - 1)]
        if atr_val is None or np.isnan(atr_val):
            continue
        threshold = atr_val * break_atr_mult

        if p.kind == "low":
            for i in range(p.index + 1, n):
                if close[i] < p.price - threshold:
                    flips.append(SRFlipLevel(p.price, "support", p.index, i))
                    break
        else:
            for i in range(p.index + 1, n):
                if close[i] > p.price + threshold:
                    flips.append(SRFlipLevel(p.price, "resistance", p.index, i))
                    break
    return flips


def count_level_role_flips(
    pivots: Sequence[Pivot],
    level: float,
    tolerance_pct: float,
) -> tuple[int, int]:
    """Count high and low touches at a level (Double SSR)."""
    highs = lows = 0
    for p in pivots:
        if is_equal(p.price, level, level, tolerance_pct):
            if p.kind == "high":
                highs += 1
            else:
                lows += 1
    return highs, lows


def range_contraction(high: np.ndarray, low: np.ndarray, idx: int, lookback: int) -> bool:
    """Later-half range smaller than earlier-half (compression staircase)."""
    if idx < lookback:
        return False
    half = lookback // 2
    if half < 2:
        return False
    early = float(np.max(high[idx - lookback : idx - half]) - np.min(low[idx - lookback : idx - half]))
    late = float(np.max(high[idx - half : idx + 1]) - np.min(low[idx - half : idx + 1]))
    return late < early * 0.75
