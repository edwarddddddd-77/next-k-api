"""行情源注册表 — 解析见 quant.common.exchange_env。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from quant.common.exchange_env import resolve_market_data_exchange_id
from quant.market.context import get_runtime_market_data_exchange
from quant.market.klines import klines_to_df

PROVIDER_BINANCE = "binance"
PROVIDER_BITGET = "bitget"
PROVIDER_BITGET_SPOT = "bitget_spot"
SUPPORTED_MARKET_DATA_PROVIDERS = ("binance", "bitget", "bitget_spot")
DEFAULT_MARKET_DATA_PROVIDER = "binance"


@dataclass(frozen=True)
class MarketDataAdapter:
    id: str
    label: str
    fetch_mark_price: Callable[[str], Optional[float]]
    fetch_klines_forward: Callable[[str, str, int, Optional[int]], List[List[Any]]]
    klines_to_df: Callable[[List[List[Any]]], Any]
    check_connectivity: Callable[..., tuple[bool, str]]


def _binance_adapter() -> MarketDataAdapter:
    from quant.market import binance as binance_market

    return MarketDataAdapter(
        id=PROVIDER_BINANCE,
        label="Binance USDT-M",
        fetch_mark_price=binance_market.fetch_mark_price,
        fetch_klines_forward=binance_market.fetch_klines_forward,
        klines_to_df=klines_to_df,
        check_connectivity=binance_market.check_connectivity,
    )


def _bitget_adapter() -> MarketDataAdapter:
    from quant.market import bitget as bitget_market

    return MarketDataAdapter(
        id=PROVIDER_BITGET,
        label="Bitget USDT-M",
        fetch_mark_price=bitget_market.fetch_mark_price,
        fetch_klines_forward=bitget_market.fetch_klines_forward,
        klines_to_df=klines_to_df,
        check_connectivity=bitget_market.check_connectivity,
    )


def _bitget_spot_adapter() -> MarketDataAdapter:
    from quant.market import bitget_spot as bitget_spot_market

    return MarketDataAdapter(
        id=PROVIDER_BITGET_SPOT,
        label="Bitget Spot",
        fetch_mark_price=bitget_spot_market.fetch_mark_price,
        fetch_klines_forward=bitget_spot_market.fetch_klines_forward,
        klines_to_df=klines_to_df,
        check_connectivity=bitget_spot_market.check_connectivity,
    )


_ADAPTERS: Dict[str, MarketDataAdapter] = {}


def get_market_adapter(exchange_id: str | None = None) -> MarketDataAdapter:
    if exchange_id is None:
        exchange_id = get_runtime_market_data_exchange()
    ex = resolve_market_data_exchange_id(exchange_id)
    if ex not in _ADAPTERS:
        if ex == PROVIDER_BITGET_SPOT:
            _ADAPTERS[ex] = _bitget_spot_adapter()
        elif ex == PROVIDER_BITGET:
            _ADAPTERS[ex] = _bitget_adapter()
        else:
            _ADAPTERS[ex] = _binance_adapter()
    return _ADAPTERS[ex]


def fetch_mark_price(symbol: str, exchange_id: str | None = None) -> Optional[float]:
    return get_market_adapter(exchange_id).fetch_mark_price(symbol)


def fetch_klines_forward(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int | None = None,
    *,
    exchange_id: str | None = None,
) -> List[List[Any]]:
    return get_market_adapter(exchange_id).fetch_klines_forward(symbol, interval, start_ms, end_ms)


def check_market_connectivity(
    exchange_id: str | None = None,
    *,
    timeout_sec: float | None = None,
) -> tuple[bool, str]:
    return get_market_adapter(exchange_id).check_connectivity(timeout_sec=timeout_sec)
