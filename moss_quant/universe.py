"""Moss 内置 crypto 标的 ∩ 币安 U 本位永续。"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from watchlist_symbols import drop_blacklisted_symbols, filter_symbols_to_binance_usdt_perps

# Moss data_cache 中 crypto 类 base（排除股票/商品 HIP-3）
_MOSS_CRYPTO_BASES = (
    "ADA",
    "APT",
    "ARB",
    "ATOM",
    "AVAX",
    "BCH",
    "BNB",
    "BTC",
    "DOGE",
    "DOT",
    "ETH",
    "FIL",
    "HBAR",
    "HYPE",
    "LINK",
    "LTC",
    "NEAR",
    "OP",
    "SOL",
    "SUI",
    "TRX",
    "UNI",
    "XRP",
)


def moss_catalog_bases() -> List[str]:
    extra = os.getenv("MOSS_QUANT_EXTRA_BASES", "").strip()
    bases = list(_MOSS_CRYPTO_BASES)
    if extra:
        for x in extra.split(","):
            b = x.strip().upper()
            if b and b not in bases:
                bases.append(b)
    return sorted(bases)


def base_to_binance_symbol(base: str) -> str:
    b = str(base or "").strip().upper()
    if not b:
        return ""
    if b.endswith("USDT"):
        return b
    return f"{b}USDT"


def symbol_to_base(symbol: str) -> str:
    s = str(symbol or "").strip().upper()
    for q in ("USDT", "USDC", "BUSD"):
        if s.endswith(q) and len(s) > len(q):
            return s[: -len(q)]
    return s.replace("/", "").replace("-", "")


def list_universe() -> List[Dict[str, Any]]:
    """返回可在 Next-K 使用的 Moss 标的（币安永续）。"""
    raw = [base_to_binance_symbol(b) for b in moss_catalog_bases()]
    filtered = drop_blacklisted_symbols(filter_symbols_to_binance_usdt_perps(raw))
    out: List[Dict[str, Any]] = []
    for sym in filtered:
        base = symbol_to_base(sym)
        out.append(
            {
                "symbol": sym,
                "base": base,
                "display": f"{base}/USDT",
                "timeframe": "15m",
            }
        )
    return out


def is_symbol_allowed(symbol: str) -> bool:
    sym = str(symbol or "").strip().upper()
    allowed = {u["symbol"] for u in list_universe()}
    return sym in allowed


def active_symbols_taken(conn, *, exclude_profile_id: Optional[int] = None) -> set[str]:
    q = "SELECT symbol FROM moss_profiles WHERE enabled = 1"
    params: list[Any] = []
    if exclude_profile_id is not None:
        q += " AND id != ?"
        params.append(int(exclude_profile_id))
    rows = conn.execute(q, params).fetchall()
    return {str(r[0]).upper() for r in rows if r[0]}
