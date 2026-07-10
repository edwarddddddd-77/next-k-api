"""多周期 K 线拉取（对齐 Jesse get_candles）。"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

from quant.market import fetch_klines_forward, klines_to_df

_CACHE: Dict[Tuple[str, str, int, str], Tuple[float, List[float]]] = {}
_REFRESH_SEC = {
    "1h": 5 * 60,
    "4h": 15 * 60,
    "1d": 60 * 60,
}


def _cache_ttl(interval: str) -> int:
    return _REFRESH_SEC.get(interval, 15 * 60)


def fetch_tf_closes(
    symbol: str,
    interval: str,
    *,
    days: int,
    exchange_id: str,
) -> List[float]:
    key = (symbol, interval, max(7, int(days)), exchange_id)
    now = time.time()
    cached = _CACHE.get(key)
    if cached is not None and now < cached[0]:
        return list(cached[1])

    end_ms = int(now * 1000)
    start_ms = end_ms - max(7, int(days)) * 86_400_000
    try:
        rows = fetch_klines_forward(symbol, interval, start_ms, end_ms, exchange_id=exchange_id)
    except Exception:
        if cached is not None:
            return list(cached[1])
        return []
    df = klines_to_df(rows)
    if df.empty:
        if cached is not None:
            return list(cached[1])
        return []
    closes = [float(x) for x in df["close"].tolist()]
    _CACHE[key] = (now + _cache_ttl(interval), closes)
    return closes


def clear_tf_klines_cache() -> None:
    _CACHE.clear()
