"""Bitget USDT 永续 vnpy Gateway。"""

from __future__ import annotations

import os

from quant.common.kline_cache import norm_symbol
from quant.engine.bootstrap import ensure_vnpy_path
from quant.engine.exchanges.base import VnpyLiveGatewayMixin
from quant.engine.lane import get_enabled_vnpy_lanes
from quant.engine.registry import plugin_for_lane

ensure_vnpy_path()

_VT_SUFFIX = "_SWAP_BITGET"
GATEWAY_NAME = "BITGET"


def bitget_credentials_configured() -> bool:
    return bool(
        (os.getenv("BITGET_API_KEY") or "").strip()
        and (os.getenv("BITGET_API_SECRET") or "").strip()
        and (os.getenv("BITGET_PASSPHRASE") or os.getenv("BITGET_API_PASSPHRASE") or "").strip()
    )


def bitget_connect_setting() -> dict:
    server = (os.getenv("BITGET_SERVER") or "REAL").strip().upper()
    if server not in ("REAL", "DEMO"):
        server = "REAL"
    kline = (os.getenv("BITGET_KLINE_STREAM") or "").strip()
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
    proxy_port = int((os.getenv("BITGET_PROXY_PORT") or "0").strip() or 0)
    setting = {
        "API Key": (os.getenv("BITGET_API_KEY") or "").strip(),
        "Secret Key": (os.getenv("BITGET_API_SECRET") or "").strip(),
        "Passphrase": (os.getenv("BITGET_PASSPHRASE") or os.getenv("BITGET_API_PASSPHRASE") or "").strip(),
        "Server": server,
        "Proxy Host": (os.getenv("BITGET_PROXY_HOST") or "").strip(),
        "Proxy Port": proxy_port,
    }
    setting["Kline Stream"] = kline
    return setting


def vnpy_vt_symbol(symbol: str) -> str:
    sym = norm_symbol(symbol)
    return f"{sym}{_VT_SUFFIX}.GLOBAL"


def symbol_from_vt(vt_symbol: str) -> str:
    raw = str(vt_symbol or "").split(".", 1)[0]
    if raw.endswith(_VT_SUFFIX):
        raw = raw[: -len(_VT_SUFFIX)]
    return norm_symbol(raw)


def _load_bitget_gateway():
    try:
        from vnpy_bitgets.bitget_gateway import BitgetGateway

        return BitgetGateway
    except ImportError:
        pass
    try:
        from vnpy_bitget import BitgetGateway

        return BitgetGateway
    except ImportError:
        pass
    try:
        from bitgets.bitget_gateway import BitgetGateway

        return BitgetGateway
    except ImportError:
        return None


_BitgetGatewayBase = _load_bitget_gateway()

if _BitgetGatewayBase is not None:
    GATEWAY_NAME = getattr(_BitgetGatewayBase, "default_name", GATEWAY_NAME)

    class VnpyBitgetGateway(VnpyLiveGatewayMixin, _BitgetGatewayBase):
        """Bitget Gateway + 共用实盘守卫与复利记账。"""

        def __init__(self, event_engine, gateway_name: str = GATEWAY_NAME) -> None:
            super().__init__(event_engine, gateway_name)

else:

    class VnpyBitgetGateway:  # type: ignore[no-redef]
        """占位 — 需安装 Bitget vnpy 插件（如 vnpy_bitgets）。"""

        default_name = GATEWAY_NAME

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("Bitget 实盘需要安装 vnpy Bitget 插件（如 vnpy_bitgets）")


__all__ = [
    "GATEWAY_NAME",
    "VnpyBitgetGateway",
    "bitget_connect_setting",
    "bitget_credentials_configured",
    "symbol_from_vt",
    "vnpy_vt_symbol",
]
