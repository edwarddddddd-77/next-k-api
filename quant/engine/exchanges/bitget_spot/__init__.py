"""Bitget 现货实盘适配。"""

from quant.engine.exchanges.bitget_spot import account
from quant.engine.exchanges.bitget_spot.gateway import (
    GATEWAY_NAME,
    VnpyBitgetSpotGateway,
    bitget_spot_connect_setting,
    bitget_spot_credentials_configured,
    symbol_from_vt,
    vnpy_vt_symbol,
)

__all__ = [
    "GATEWAY_NAME",
    "VnpyBitgetSpotGateway",
    "account",
    "bitget_spot_connect_setting",
    "bitget_spot_credentials_configured",
    "symbol_from_vt",
    "vnpy_vt_symbol",
]
