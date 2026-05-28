"""Moss 内置 crypto 标的 ∩ 币安 U 本位永续。"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from watchlist_symbols import drop_blacklisted_symbols, filter_symbols_to_binance_usdt_perps

# Moss crypto 标的（工厂 data_cache ∪ 币安 U 本位永续；排除股票/商品 HIP-3）
_MOSS_CRYPTO_BASES = (
    "AAVE",
    "ADA",
    "ALGO",
    "APT",
    "ARB",
    "ATOM",
    "AVAX",
    "BCH",
    "BNB",
    "BONK",
    "BTC",
    "DOGE",
    "DOT",
    "ENA",
    "ETC",
    "ETH",
    "FIL",
    "HBAR",
    "HYPE",
    "ICP",
    "IMX",
    "INJ",
    "LINK",
    "LTC",
    "NEAR",
    "OP",
    "PENDLE",
    "PEPE",
    "POL",
    "RENDER",
    "SEI",
    "SHIB",
    "SOL",
    "STRK",
    "SUI",
    "TIA",
    "TON",
    "TRX",
    "UNI",
    "WIF",
    "WLD",
    "XLM",
    "XRP",
)


# 币安合约报价单位与 base 名不一致（千枚计价）
_BINANCE_CONTRACT_PREFIX: Dict[str, str] = {
    "PEPE": "1000PEPE",
    "SHIB": "1000SHIB",
    "BONK": "1000BONK",
}


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
    contract = _BINANCE_CONTRACT_PREFIX.get(b, b)
    return f"{contract}USDT"


def symbol_to_base(symbol: str) -> str:
    s = str(symbol or "").strip().upper()
    for q in ("USDT", "USDC", "BUSD"):
        if s.endswith(q) and len(s) > len(q):
            core = s[: -len(q)]
            if core.startswith("1000") and len(core) > 4:
                return core[4:]
            return core
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


def normalize_usdt_perp_symbol(symbol: str) -> str:
    """规范为 XXXUSDT（去空格/斜杠，无后缀则补 USDT）。"""
    s = str(symbol or "").strip().upper().replace("/", "").replace("-", "")
    if not s:
        return ""
    if not s.endswith("USDT"):
        s += "USDT"
    return s


def is_symbol_allowed(symbol: str) -> bool:
    """纸面 Profile / 每日寻优宇宙：仅 Moss 内置 ∩ 币安永续。"""
    sym = normalize_usdt_perp_symbol(symbol)
    allowed = {u["symbol"] for u in list_universe()}
    return sym in allowed


def is_research_symbol_allowed(symbol: str) -> bool:
    """回测 / 寻优 / 进化：默认任意 XXXUSDT；可关闭 relax 后仅允许币安永续 TRADING。"""
    from moss_quant import config as cfg

    sym = normalize_usdt_perp_symbol(symbol)
    if not sym or len(sym) < 6:
        return False
    if not re.fullmatch(r"[A-Z0-9]+USDT", sym):
        return False
    if cfg.MOSS_QUANT_RESEARCH_RELAX_SYMBOL_CHECK:
        return True
    kept = filter_symbols_to_binance_usdt_perps([sym])
    return bool(kept)


def active_symbols_taken(conn, *, exclude_profile_id: Optional[int] = None) -> set[str]:
    q = "SELECT symbol FROM moss_profiles WHERE enabled = 1"
    params: list[Any] = []
    if exclude_profile_id is not None:
        q += " AND id != ?"
        params.append(int(exclude_profile_id))
    rows = conn.execute(q, params).fetchall()
    return {str(r[0]).upper() for r in rows if r[0]}
