"""Bitget 实盘适配。"""

from quant.engine.exchanges.bitget import account
from quant.engine.exchanges.bitget.gateway import (
    GATEWAY_NAME,
    VnpyBitgetGateway,
    bitget_connect_setting,
    bitget_credentials_configured,
    symbol_from_vt,
    vnpy_vt_symbol,
)

__all__ = [
    "GATEWAY_NAME",
    "VnpyBitgetGateway",
    "account",
    "bitget_connect_setting",
    "bitget_credentials_configured",
    "symbol_from_vt",
    "vnpy_vt_symbol",
]
