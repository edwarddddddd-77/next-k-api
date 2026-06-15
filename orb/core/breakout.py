"""ORB 突破确认（5m 收盘）。"""

from __future__ import annotations

from typing import List


def breakout_long(closes: List[float], *, or_high: float, confirm_bars: int, no_soften: bool) -> bool:
    k = max(1, int(confirm_bars))
    if len(closes) < k + 1:
        return False
    prev = float(closes[-(k + 1)])
    tail = [float(x) for x in closes[-k:]]
    if prev > or_high or not all(x > or_high for x in tail):
        return False
    if k >= 2 and no_soften and tail[-1] < tail[-2]:
        return False
    return True


def breakout_short(closes: List[float], *, or_low: float, confirm_bars: int, no_soften: bool) -> bool:
    k = max(1, int(confirm_bars))
    if len(closes) < k + 1:
        return False
    prev = float(closes[-(k + 1)])
    tail = [float(x) for x in closes[-k:]]
    if prev < or_low or not all(x < or_low for x in tail):
        return False
    if k >= 2 and no_soften and tail[-1] > tail[-2]:
        return False
    return True


def entry_price_for_side(*, side: str, or_high: float, or_low: float, tick_size: float, tick_offset: int) -> float:
    tick = max(0, int(tick_offset)) * float(tick_size)
    if str(side).upper() == "LONG":
        return round(or_high + tick, 8)
    return round(or_low - tick, 8)
