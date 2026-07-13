"""实盘交易所注册表 — 解析见 quant.common.exchange_env。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Type

from quant.common.exchange_env import resolve_live_exchange_id
from quant.common.kline_cache import norm_symbol
from quant.engine.exchanges.context import get_runtime_live_exchange

EXCHANGE_BINANCE = "binance"
EXCHANGE_BITGET = "bitget"
EXCHANGE_BITGET_SPOT = "bitget_spot"
SUPPORTED_LIVE_EXCHANGES = ("binance", "bitget", "bitget_spot")
DEFAULT_LIVE_EXCHANGE = "binance"


@dataclass(frozen=True)
class LiveExchangeAdapter:
    id: str
    label: str
    gateway_class: Type
    gateway_name: str
    credentials_configured: Callable[[], bool]
    connect_setting: Callable[[], dict]
    vnpy_vt_symbol: Callable[[str], str]
    symbol_from_vt: Callable[[str], str]
    fetch_position_amounts: Callable[[List[str]], Dict[str, float]]
    fetch_position_snapshots: Callable[[List[str]], Dict[str, Dict[str, float]]]
    ensure_pool_leverage: Callable[[List[str], Any], None]
    credentials_missing_reason: str = "live_credentials_missing"
    contracts_not_ready_reason: str = "contracts_not_ready"


def _binance_adapter() -> LiveExchangeAdapter:
    from quant.engine.exchanges.binance import account as binance_account
    from quant.engine.exchanges.binance.gateway import (
        GATEWAY_NAME,
        VnpyBinanceLinearGateway,
        binance_connect_setting,
        binance_credentials_configured,
        symbol_from_vt,
        vnpy_vt_symbol,
    )

    return LiveExchangeAdapter(
        id=EXCHANGE_BINANCE,
        label="Binance USDT-M",
        gateway_class=VnpyBinanceLinearGateway,
        gateway_name=GATEWAY_NAME,
        credentials_configured=binance_credentials_configured,
        connect_setting=binance_connect_setting,
        vnpy_vt_symbol=vnpy_vt_symbol,
        symbol_from_vt=symbol_from_vt,
        fetch_position_amounts=binance_account.fetch_position_amounts,
        fetch_position_snapshots=binance_account.fetch_position_snapshots,
        ensure_pool_leverage=binance_account.ensure_pool_leverage,
        credentials_missing_reason="binance_credentials_missing",
        contracts_not_ready_reason="binance_contracts_not_ready",
    )


def _bitget_spot_adapter() -> LiveExchangeAdapter:
    from quant.engine.exchanges.bitget_spot import account as bitget_spot_account
    from quant.engine.exchanges.bitget_spot.gateway import (
        GATEWAY_NAME,
        VnpyBitgetSpotGateway,
        bitget_spot_connect_setting,
        bitget_spot_credentials_configured,
        symbol_from_vt,
        vnpy_vt_symbol,
    )

    return LiveExchangeAdapter(
        id=EXCHANGE_BITGET_SPOT,
        label="Bitget Spot",
        gateway_class=VnpyBitgetSpotGateway,
        gateway_name=GATEWAY_NAME,
        credentials_configured=bitget_spot_credentials_configured,
        connect_setting=bitget_spot_connect_setting,
        vnpy_vt_symbol=vnpy_vt_symbol,
        symbol_from_vt=symbol_from_vt,
        fetch_position_amounts=bitget_spot_account.fetch_position_amounts,
        fetch_position_snapshots=bitget_spot_account.fetch_position_snapshots,
        ensure_pool_leverage=bitget_spot_account.ensure_pool_leverage,
        credentials_missing_reason="bitget_spot_credentials_missing",
        contracts_not_ready_reason="bitget_spot_contracts_not_ready",
    )


def _bitget_adapter() -> LiveExchangeAdapter:
    from quant.engine.exchanges.bitget import account as bitget_account
    from quant.engine.exchanges.bitget.gateway import (
        GATEWAY_NAME,
        VnpyBitgetGateway,
        bitget_connect_setting,
        bitget_credentials_configured,
        symbol_from_vt,
        vnpy_vt_symbol,
    )

    return LiveExchangeAdapter(
        id=EXCHANGE_BITGET,
        label="Bitget USDT-M",
        gateway_class=VnpyBitgetGateway,
        gateway_name=GATEWAY_NAME,
        credentials_configured=bitget_credentials_configured,
        connect_setting=bitget_connect_setting,
        vnpy_vt_symbol=vnpy_vt_symbol,
        symbol_from_vt=symbol_from_vt,
        fetch_position_amounts=bitget_account.fetch_position_amounts,
        fetch_position_snapshots=bitget_account.fetch_position_snapshots,
        ensure_pool_leverage=bitget_account.ensure_pool_leverage,
        credentials_missing_reason="bitget_credentials_missing",
        contracts_not_ready_reason="bitget_contracts_not_ready",
    )


_ADAPTERS: Dict[str, LiveExchangeAdapter] = {}


def get_adapter(exchange_id: str | None = None) -> LiveExchangeAdapter:
    if exchange_id is None:
        exchange_id = get_runtime_live_exchange()
    ex = resolve_live_exchange_id(exchange_id)
    if ex not in _ADAPTERS:
        if ex == EXCHANGE_BITGET_SPOT:
            _ADAPTERS[ex] = _bitget_spot_adapter()
        elif ex == EXCHANGE_BITGET:
            _ADAPTERS[ex] = _bitget_adapter()
        else:
            _ADAPTERS[ex] = _binance_adapter()
    return _ADAPTERS[ex]


def get_live_adapter() -> LiveExchangeAdapter:
    return get_adapter()


def vnpy_vt_symbol(symbol: str, exchange_id: str | None = None) -> str:
    return get_adapter(exchange_id).vnpy_vt_symbol(norm_symbol(symbol))


def symbol_from_vt(vt_symbol: str, exchange_id: str | None = None) -> str:
    return get_adapter(exchange_id).symbol_from_vt(vt_symbol)
