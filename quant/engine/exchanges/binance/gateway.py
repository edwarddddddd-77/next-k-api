"""Binance USDT-M vnpy Gateway（Trading ORB 实盘）。"""

from __future__ import annotations

import logging
import os

from quant.common.kline_cache import norm_symbol
from quant.engine.bootstrap import ensure_vnpy_path
from quant.engine.exchanges.base import VnpyLiveGatewayMixin
from quant.engine.lane import get_enabled_vnpy_lanes
from quant.engine.registry import plugin_for_lane

ensure_vnpy_path()

from vnpy_binance.linear_gateway import BinanceLinearGateway  # noqa: E402

logger = logging.getLogger(__name__)

GATEWAY_NAME = BinanceLinearGateway.default_name
_VT_SUFFIX = "_SWAP_BINANCE"


def binance_credentials_configured() -> bool:
    return bool(
        (os.getenv("BINANCE_API_KEY") or "").strip()
        and (os.getenv("BINANCE_API_SECRET") or "").strip()
    )


def binance_connect_setting() -> dict:
    server = (os.getenv("BINANCE_SERVER") or "REAL").strip().upper()
    if server not in ("REAL", "TESTNET"):
        server = "REAL"
    kline = (os.getenv("BINANCE_KLINE_STREAM") or "").strip()
    if not kline:
        has_stream = any(
            (plugin := plugin_for_lane(name)) is not None and plugin.uses_kline_stream
            for name, _ in get_enabled_vnpy_lanes()
        )
        kline = "True" if has_stream else "False"
    if kline.lower() in ("1", "true", "yes", "on"):
        kline = "True"
    else:
        kline = "False"
    proxy_port = int((os.getenv("BINANCE_PROXY_PORT") or "0").strip() or 0)
    return {
        "API Key": (os.getenv("BINANCE_API_KEY") or "").strip(),
        "API Secret": (os.getenv("BINANCE_API_SECRET") or "").strip(),
        "Server": server,
        "Kline Stream": kline,
        "Proxy Host": (os.getenv("BINANCE_PROXY_HOST") or "").strip(),
        "Proxy Port": proxy_port,
    }


def vnpy_vt_symbol(symbol: str) -> str:
    """官方合约 vt_symbol：ETHUSDT_SWAP_BINANCE.GLOBAL"""
    sym = norm_symbol(symbol)
    return f"{sym}{_VT_SUFFIX}.GLOBAL"


def symbol_from_vt(vt_symbol: str) -> str:
    raw = str(vt_symbol or "").split(".", 1)[0]
    if raw.endswith(_VT_SUFFIX):
        raw = raw[: -len(_VT_SUFFIX)]
    return norm_symbol(raw)


class VnpyBinanceLinearGateway(VnpyLiveGatewayMixin, BinanceLinearGateway):
    """官方 Gateway + 共用实盘守卫与复利记账。"""

    def __init__(self, event_engine, gateway_name: str = GATEWAY_NAME) -> None:
        super().__init__(event_engine, gateway_name)


__all__ = [
    "GATEWAY_NAME",
    "VnpyBinanceLinearGateway",
    "binance_connect_setting",
    "binance_credentials_configured",
    "symbol_from_vt",
    "vnpy_vt_symbol",
]
