"""Bybit Linear vnpy Gateway（Trading ORB 实盘）。"""

from __future__ import annotations

import os

from quant.common.kline_cache import norm_symbol
from quant.engine.bootstrap import ensure_vnpy_path
from quant.engine.exchanges.base import VnpyLiveGatewayMixin

ensure_vnpy_path()

_VT_SUFFIX = "_SWAP_BYBIT"
GATEWAY_NAME = "BYBIT"


def bybit_credentials_configured() -> bool:
    return bool(
        (os.getenv("BYBIT_API_KEY") or "").strip()
        and (os.getenv("BYBIT_API_SECRET") or "").strip()
    )


def bybit_connect_setting() -> dict:
    server = (os.getenv("BYBIT_SERVER") or "REAL").strip().upper()
    if server not in ("REAL", "TESTNET", "DEMO"):
        server = "REAL"
    proxy_port = int((os.getenv("BYBIT_PROXY_PORT") or "0").strip() or 0)
    return {
        "API Key": (os.getenv("BYBIT_API_KEY") or "").strip(),
        "Secret Key": (os.getenv("BYBIT_API_SECRET") or "").strip(),
        "Server": server,
        "Proxy Host": (os.getenv("BYBIT_PROXY_HOST") or "").strip(),
        "Proxy Port": proxy_port,
    }


def vnpy_vt_symbol(symbol: str) -> str:
    """官方合约 vt_symbol：ETHUSDT_SWAP_BYBIT.GLOBAL"""
    sym = norm_symbol(symbol)
    return f"{sym}{_VT_SUFFIX}.GLOBAL"


def symbol_from_vt(vt_symbol: str) -> str:
    raw = str(vt_symbol or "").split(".", 1)[0]
    if raw.endswith(_VT_SUFFIX):
        raw = raw[: -len(_VT_SUFFIX)]
    return norm_symbol(raw)


try:
    from vnpy_bybit import BybitGateway  # noqa: E402

    GATEWAY_NAME = BybitGateway.default_name

    class VnpyBybitGateway(VnpyLiveGatewayMixin, BybitGateway):
        """官方 BybitGateway + 共用实盘守卫与复利记账。"""

        def __init__(self, event_engine, gateway_name: str = GATEWAY_NAME) -> None:
            super().__init__(event_engine, gateway_name)

except ImportError:

    class VnpyBybitGateway:  # type: ignore[no-redef]
        """占位 — 需 pip install vnpy_bybit。"""

        default_name = GATEWAY_NAME

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("Bybit 实盘需要 pip install vnpy_bybit")


__all__ = [
    "GATEWAY_NAME",
    "VnpyBybitGateway",
    "bybit_connect_setting",
    "bybit_credentials_configured",
    "symbol_from_vt",
    "vnpy_vt_symbol",
]
