"""币安 U 本位杠杆封顶（Moss crypto 子集）。"""

from __future__ import annotations

import copy
from typing import Dict

from moss_quant.universe import symbol_to_base

# 保守默认；可按 exchangeInfo 扩展
_BN_MAX_LEVERAGE: Dict[str, float] = {
    "BTC": 50.0,
    "ETH": 50.0,
    "SOL": 25.0,
    "BNB": 25.0,
    "XRP": 20.0,
    "DOGE": 20.0,
    "ADA": 15.0,
    "AVAX": 15.0,
    "LINK": 15.0,
    "DOT": 15.0,
    "MATIC": 15.0,
    "LTC": 15.0,
    "BCH": 15.0,
    "ATOM": 10.0,
    "APT": 10.0,
    "ARB": 10.0,
    "OP": 10.0,
    "NEAR": 10.0,
    "FIL": 10.0,
    "HBAR": 10.0,
    "UNI": 10.0,
    "SUI": 10.0,
    "TRX": 10.0,
    "HYPE": 10.0,
}

_DEFAULT_CAP = 10.0


def max_leverage_for_symbol(symbol: str) -> float:
    base = symbol_to_base(symbol)
    return float(_BN_MAX_LEVERAGE.get(base, _DEFAULT_CAP))


def cap_params_for_symbol(params: dict, symbol: str) -> dict:
    out = copy.deepcopy(params)
    cap = max_leverage_for_symbol(symbol)
    for key in ("base_leverage", "max_leverage"):
        if key in out:
            out[key] = min(float(out[key]), cap)
    if float(out.get("max_leverage", 1)) < float(out.get("base_leverage", 1)):
        out["max_leverage"] = out["base_leverage"]
    return out
