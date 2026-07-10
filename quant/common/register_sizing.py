"""vnpy 注册阶段仓位估算（用近期 ATR，避免价格 proxy）。"""

from __future__ import annotations

import time
from typing import Sequence

from quant.common.jesse_indicators import atr_last
from quant.market import fetch_klines_forward, klines_to_df


def recent_atr(
    symbol: str,
    interval: str,
    *,
    exchange_id: str,
    days: int = 7,
    atr_period: int = 14,
) -> float | None:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - max(3, int(days)) * 86_400_000
    try:
        rows = fetch_klines_forward(symbol, interval, start_ms, end_ms, exchange_id=exchange_id)
    except Exception:
        return None
    df = klines_to_df(rows)
    if df.empty or len(df) < atr_period + 2:
        return None
    highs: Sequence[float] = [float(x) for x in df["high"].tolist()]
    lows: Sequence[float] = [float(x) for x in df["low"].tolist()]
    closes: Sequence[float] = [float(x) for x in df["close"].tolist()]
    atr = atr_last(highs, lows, closes, atr_period)
    if atr is None or atr <= 0:
        return None
    return float(atr)
