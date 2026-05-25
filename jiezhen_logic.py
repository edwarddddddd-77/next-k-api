"""接针量化：EMA 方向 + ATR/振幅挂单距离（纸面触价开仓）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple


@dataclass(frozen=True)
class SpikePlan:
    mark: float
    ema: float | None
    atr: float
    average_amplitude_pct: float
    price_atr_ratio: float
    selected_distance_pct: float
    target_long: float
    target_short: float
    is_bullish: bool
    is_bearish: bool
    long_fill: bool
    short_fill: bool


def klines_chronological(rows: Sequence[Sequence[Any]]) -> List[List[Any]]:
    """币安 /fapi/v1/klines 为时间升序（旧→新），保持顺序即可。"""
    return [list(r) for r in rows]


def ema_last(closes: Sequence[float], period: int) -> float:
    if not closes:
        return 0.0
    if period <= 0:
        return float(closes[-1])
    k = 2.0 / (period + 1)
    ema = float(closes[0])
    for p in closes[1:]:
        ema = float(p) * k + ema * (1.0 - k)
    return ema


def calculate_atr(klines_chrono: Sequence[Sequence[Any]], period: int) -> float:
    trs: List[float] = []
    for i in range(1, len(klines_chrono)):
        high = float(klines_chrono[i][2])
        low = float(klines_chrono[i][3])
        prev_close = float(klines_chrono[i - 1][4])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    if not trs:
        return 0.0
    use = trs[-period:] if len(trs) >= period else trs
    return sum(use) / len(use)


def calculate_average_amplitude(
    klines_chrono: Sequence[Sequence[Any]], period: int
) -> float:
    n = len(klines_chrono)
    if n == 0:
        return 0.0
    start = max(0, n - period)
    amplitudes: List[float] = []
    for i in range(start, n):
        high = float(klines_chrono[i][2])
        low = float(klines_chrono[i][3])
        close = float(klines_chrono[i][4])
        if close <= 0:
            continue
        amplitudes.append(((high - low) / close) * 100.0)
    if not amplitudes:
        return 0.0
    return sum(amplitudes) / len(amplitudes)


def selected_distance_pct(
    *,
    mark: float,
    atr: float,
    average_amplitude_pct: float,
    value_multiplier: float,
    min_distance_pct: float,
    distance_mode: str,
) -> tuple[float, float]:
    price_atr_ratio = (mark / atr) / 100.0 if atr > 0 and mark > 0 else 0.0
    mode = (distance_mode or "min").strip().lower()
    if mode in ("avg", "average", "mean", "zhen2", "zhen_2"):
        selected = (average_amplitude_pct + price_atr_ratio) / 2.0 * value_multiplier
    else:
        selected = min(average_amplitude_pct, price_atr_ratio) * value_multiplier
        selected = max(selected, min_distance_pct)
    return selected, price_atr_ratio


def trend_flags(
    close_last: float, *, ema_period: int, ema_value: float | None
) -> Tuple[bool, bool]:
    if ema_period <= 0:
        return True, True
    if ema_value is None:
        return False, False
    return close_last > ema_value, close_last < ema_value


def _touch_long(klines_chrono: Sequence[Sequence[Any]], target_long: float, lookback: int) -> bool:
    """最近 lookback 根 K 的最低价触及/跌破接针买价（相对当前 mark 的挂单价）。"""
    if not klines_chrono or target_long <= 0:
        return False
    n = max(1, min(lookback, len(klines_chrono)))
    window = klines_chrono[-n:]
    min_low = min(float(b[3]) for b in window)
    return min_low <= target_long


def _touch_short(klines_chrono: Sequence[Sequence[Any]], target_short: float, lookback: int) -> bool:
    n = max(1, min(lookback, len(klines_chrono)))
    window = klines_chrono[-n:]
    max_high = max(float(b[2]) for b in window)
    return max_high >= target_short


def build_spike_plan(
    *,
    mark: float,
    klines_raw: Sequence[Sequence[Any]],
    ema_period: int,
    atr_period: int,
    amplitude_period: int,
    value_multiplier: float,
    min_distance_pct: float,
    distance_mode: str,
    touch_lookback_bars: int = 3,
) -> SpikePlan | None:
    if mark <= 0 or not klines_raw:
        return None
    kl = klines_chronological(klines_raw)
    closes = [float(k[4]) for k in kl]
    if not closes:
        return None
    ema_val = None if ema_period <= 0 else ema_last(closes, ema_period)
    atr = calculate_atr(kl, atr_period)
    avg_amp = calculate_average_amplitude(kl, amplitude_period)
    selected, price_atr_ratio = selected_distance_pct(
        mark=mark,
        atr=atr,
        average_amplitude_pct=avg_amp,
        value_multiplier=value_multiplier,
        min_distance_pct=min_distance_pct,
        distance_mode=distance_mode,
    )
    target_long = mark * (1.0 - selected / 100.0)
    target_short = mark * (1.0 + selected / 100.0)
    bullish, bearish = trend_flags(closes[-1], ema_period=ema_period, ema_value=ema_val)
    lookback = max(1, touch_lookback_bars)
    return SpikePlan(
        mark=mark,
        ema=ema_val,
        atr=atr,
        average_amplitude_pct=avg_amp,
        price_atr_ratio=price_atr_ratio,
        selected_distance_pct=selected,
        target_long=target_long,
        target_short=target_short,
        is_bullish=bullish,
        is_bearish=bearish,
        long_fill=bool(bullish and _touch_long(kl, target_long, lookback)),
        short_fill=bool(bearish and _touch_short(kl, target_short, lookback)),
    )
