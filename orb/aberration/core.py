"""Aberration 信号核心（FMZ #126612 社区版逻辑，纯 Python 可测）。"""

from __future__ import annotations

import math
from typing import Literal, Optional, Tuple

AberrationAction = Optional[Literal["long", "short", "close_long", "close_short"]]


def aberration_bands(
    closes: list[float],
    *,
    n_period: int = 35,
    k_up: float = 2.0,
    k_down: float = 2.0,
) -> Optional[Tuple[float, float, float]]:
    """返回 (upper, middle, lower)；closes 为已完成 K 线收盘价（不含当前 bar）。"""
    n = int(n_period)
    if n <= 0 or len(closes) < n:
        return None
    window = closes[-n:]
    avg = sum(window) / n
    var = sum((x - avg) ** 2 for x in window) / n
    std = math.sqrt(max(0.0, var))
    upper = avg + float(k_up) * std
    lower = avg - float(k_down) * std
    return upper, avg, lower


def aberration_action(
    pos: float,
    close: float,
    upper: float,
    middle: float,
    lower: float,
) -> AberrationAction:
    """pos>0 多仓，pos<0 空仓，pos==0 空仓。"""
    px = float(close)
    if pos == 0:
        if px > upper:
            return "long"
        if px < lower:
            return "short"
        return None
    if pos < 0 and px > middle:
        return "close_short"
    if pos > 0 and px < middle:
        return "close_long"
    return None
