"""Donchian breakout signal core (aligned with breakoutscanner)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np

BreakoutDirection = Literal["bullish", "bearish"]
BreakoutMode = Literal["standard", "strict"]


@dataclass(frozen=True)
class DonchianSignal:
    side: int
    entry: float
    stop: float
    tp1: float
    tp2: float
    tp3: float
    level: float
    volume_ratio: float
    atr: float
    strength: int
    direction: BreakoutDirection
    mode: BreakoutMode


def _atr_series(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    if len(close) < period + 1:
        return np.array([])
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    if len(tr) < period:
        return np.array([])
    out = np.zeros(len(tr), dtype=float)
    out[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, len(tr)):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def _strong_close(close: float, high: float, low: float, direction: BreakoutDirection, pct: float) -> bool:
    rng = high - low
    if rng <= 0:
        return True
    if direction == "bullish":
        return (close - low) / rng >= pct
    return (high - close) / rng >= pct


def breakout_levels(
    entry: float,
    side: int,
    *,
    level: float,
    bar_low: float,
    bar_high: float,
    atr: float,
    sl_level_buffer: float,
    sl_atr_mult: float,
    tp1_rr: float,
    tp2_rr: float,
    tp3_rr: float,
) -> tuple[float, float, float, float, float]:
    if side > 0:
        stop = min(level * (1.0 - sl_level_buffer), bar_low, entry - sl_atr_mult * atr)
        risk = max(entry - stop, entry * 0.005)
        return stop, entry + risk * tp1_rr, entry + risk * tp2_rr, entry + risk * tp3_rr, risk
    stop = max(level * (1.0 + sl_level_buffer), bar_high, entry + sl_atr_mult * atr)
    risk = max(stop - entry, entry * 0.005)
    return stop, entry - risk * tp1_rr, entry - risk * tp2_rr, entry - risk * tp3_rr, risk


def detect_donchian_signal(
    bars: Sequence[tuple],
    *,
    lookback: int = 20,
    vol_lookback: int = 20,
    vol_mult: float = 1.3,
    strong_close_pct: float = 0.60,
    mode: BreakoutMode = "strict",
    strict_vol_mult: float = 1.6,
    strict_atr_mult: float = 1.3,
    atr_period: int = 14,
    direction_filter: Optional[BreakoutDirection] = None,
    tp1_rr: float = 2.0,
    tp2_rr: float = 3.5,
    tp3_rr: float = 3.5,
    sl_level_buffer: float = 0.015,
    sl_atr_mult: float = 1.5,
) -> Optional[DonchianSignal]:
    """
    bars: (ts, open, high, low, close, volume)
    Evaluate latest completed bar only.
    """
    min_len = max(lookback + vol_lookback + 5, atr_period + lookback + 5)
    if len(bars) < min_len:
        return None

    vm = strict_vol_mult if mode == "strict" else vol_mult
    atr_m = strict_atr_mult if mode == "strict" else 0.0

    opens = np.array([float(b[1]) for b in bars], dtype=float)
    highs = np.array([float(b[2]) for b in bars], dtype=float)
    lows = np.array([float(b[3]) for b in bars], dtype=float)
    closes = np.array([float(b[4]) for b in bars], dtype=float)
    volumes = np.array([float(b[5]) for b in bars], dtype=float)

    prior_h = highs[-(lookback + 1) : -1]
    prior_l = lows[-(lookback + 1) : -1]
    resistance = float(np.max(prior_h))
    support = float(np.min(prior_l))

    avg_vol = float(np.mean(volumes[-(vol_lookback + 1) : -1]))
    vol = float(volumes[-1])
    if avg_vol > 0 and vol < vm * avg_vol:
        return None

    close = float(closes[-1])
    open_ = float(opens[-1])
    high = float(highs[-1])
    low = float(lows[-1])

    direction: Optional[BreakoutDirection] = None
    level = resistance
    if close > resistance and (direction_filter in (None, "bullish")):
        direction = "bullish"
    elif close < support and (direction_filter in (None, "bearish")):
        direction = "bearish"
        level = support
    else:
        return None

    if not _strong_close(close, high, low, direction, strong_close_pct):
        return None

    atr_arr = _atr_series(highs, lows, closes, atr_period)
    atr = float(atr_arr[-1]) if atr_arr.size else max(high - low, close * 0.01)

    if mode == "strict":
        tr = max(high - low, abs(high - closes[-2]), abs(low - closes[-2]))
        if atr <= 0 or tr <= atr_m * atr:
            return None

    vol_ratio = vol / avg_vol if avg_vol > 0 else 1.0
    side = 1 if direction == "bullish" else -1
    strength = 0
    if vol_ratio >= vm:
        strength += 1
    if vol_ratio >= vm * 1.5:
        strength += 1
    if abs(close - open_) >= 0.5 * atr:
        strength += 1
    if mode == "strict":
        strength += 1

    stop, tp1, tp2, tp3, _ = breakout_levels(
        close,
        side,
        level=level,
        bar_low=low,
        bar_high=high,
        atr=atr,
        sl_level_buffer=sl_level_buffer,
        sl_atr_mult=sl_atr_mult,
        tp1_rr=tp1_rr,
        tp2_rr=tp2_rr,
        tp3_rr=tp3_rr,
    )
    return DonchianSignal(
        side=side,
        entry=close,
        stop=stop,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        level=level,
        volume_ratio=float(vol_ratio),
        atr=atr,
        strength=min(4, strength),
        direction=direction,
        mode=mode,
    )


def resolve_exit_target_price(
    *,
    tp1: float,
    tp2: float,
    tp3: float,
    exit_target: str = "tp1",
) -> tuple[float, str]:
    key = str(exit_target or "tp1").strip().lower()
    if key in ("tp2", "2"):
        return float(tp2), "tp2"
    if key in ("tp3", "3"):
        return float(tp3), "tp3"
    return float(tp1), "tp1"


def bar_exit_reason(
    *,
    side: int,
    high: float,
    low: float,
    stop: float,
    tp1: float,
    tp2: float,
    tp3: float,
    prev_high: float,
    prev_low: float,
    exit_target: str = "tp1",
) -> str | None:
    target_px, target_tag = resolve_exit_target_price(
        tp1=tp1, tp2=tp2, tp3=tp3, exit_target=exit_target
    )
    if side > 0:
        sl_hit = low <= stop and prev_low > stop
        tp_hit = high >= target_px and prev_high < target_px
    else:
        sl_hit = high >= stop and prev_high < stop
        tp_hit = low <= target_px and prev_low < target_px
    if sl_hit and tp_hit:
        return "sl"
    if sl_hit:
        return "sl"
    if tp_hit:
        return target_tag
    return None
