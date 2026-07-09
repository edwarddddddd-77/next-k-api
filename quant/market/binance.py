"""Binance U 本位行情（委托 binance_fapi）。"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from binance_fapi import (
    check_fapi_connectivity,
    fetch_klines_forward as _fetch_klines_forward,
    fetch_mark_price as _fetch_mark_price,
)
from quant.market.klines import klines_to_df

PROVIDER_BINANCE = "binance"


def fetch_mark_price(symbol: str) -> Optional[float]:
    return _fetch_mark_price(symbol)


def fetch_klines_forward(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: Optional[int] = None,
) -> List[List[Any]]:
    return _fetch_klines_forward(symbol, interval, start_ms, end_ms)


def check_connectivity(*, timeout_sec: float | None = None) -> tuple[bool, str]:
    return check_fapi_connectivity(timeout_sec=timeout_sec)


__all__ = [
    "PROVIDER_BINANCE",
    "check_connectivity",
    "fetch_klines_forward",
    "fetch_mark_price",
    "klines_to_df",
]
