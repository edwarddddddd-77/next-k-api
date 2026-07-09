"""量化行情层 — MARKET_DATA_EXCHANGE 切换 binance / bybit。"""

from quant.common.exchange_env import resolve_market_data_exchange_id
from quant.market.klines import klines_to_df
from quant.market.registry import (
    MarketDataAdapter,
    check_market_connectivity,
    fetch_klines_forward,
    fetch_mark_price,
    get_market_adapter,
)

__all__ = [
    "MarketDataAdapter",
    "check_market_connectivity",
    "fetch_klines_forward",
    "fetch_mark_price",
    "get_market_adapter",
    "klines_to_df",
    "resolve_market_data_exchange_id",
]
