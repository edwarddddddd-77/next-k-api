"""币安市值排行 → Moss 扩展寻优候选（排除稳定币与每日扫描宇宙）。"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

from watchlist_symbols import drop_blacklisted_symbols, filter_symbols_to_binance_usdt_perps

from moss_quant.universe import (
    base_to_binance_symbol,
    list_daily_core_universe,
    symbol_to_base,
)

logger = logging.getLogger(__name__)

_BINANCE_MCAP_URL = (
    "https://www.binance.com/bapi/composite/v1/public/marketing/symbol/list"
)

# 稳定币 / 法币锚定（按 base 名排除）
_STABLE_BASES: Set[str] = {
    "USDT",
    "USDC",
    "BUSD",
    "DAI",
    "TUSD",
    "FDUSD",
    "USDP",
    "USDD",
    "EUR",
    "AEUR",
    "USD1",
    "XUSD",
    "BFUSD",
}


def fetch_binance_spot_market_caps() -> Dict[str, float]:
    """base 名 → 流通市值（USD）。"""
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("requests required for binance mcap API") from e

    r = requests.get(_BINANCE_MCAP_URL, timeout=15)
    r.raise_for_status()
    payload = r.json()
    out: Dict[str, float] = {}
    for item in payload.get("data") or []:
        name = str(item.get("name") or "").strip().upper()
        try:
            mc = float(item.get("marketCap") or 0)
        except (TypeError, ValueError):
            continue
        if name and mc > 0:
            out[name] = mc
    return out


def _daily_core_universe_from_db() -> List[Dict[str, Any]]:
    """读 moss_daily_core_symbols；失败时回退内置 25。"""
    try:
        from accumulation_radar import init_db

        conn = init_db()
        try:
            return list_daily_core_universe(conn)
        finally:
            conn.close()
    except Exception:
        logger.warning(
            "[moss] daily core universe db read failed, using builtin fallback",
            exc_info=True,
        )
        return list_daily_core_universe()


def daily_scan_symbol_set() -> Set[str]:
    return {str(u["symbol"]).upper() for u in _daily_core_universe_from_db()}


def daily_scan_base_set() -> Set[str]:
    return {str(u["base"]).upper() for u in _daily_core_universe_from_db()}


def _daily_core_exclude_sets() -> tuple[Set[str], Set[str]]:
    universe = _daily_core_universe_from_db()
    bases = {str(u["base"]).upper() for u in universe if u.get("base")}
    syms = {str(u["symbol"]).upper() for u in universe if u.get("symbol")}
    return bases, syms


def build_mcap_scan_candidates(
    *,
    mcap_limit: int = 100,
    mcap_map: Dict[str, float] | None = None,
) -> List[Dict[str, Any]]:
    """
    取币安市值前 mcap_limit 名（可交易 U 本位永续），去掉稳定币与 moss_daily_core_symbols 每日寻优表标的。
    返回按市值降序的候选列表。
    """
    mcap_limit = max(1, int(mcap_limit))
    caps = mcap_map if mcap_map is not None else fetch_binance_spot_market_caps()
    daily_bases, daily_syms = _daily_core_exclude_sets()

    ranked_bases = sorted(caps.items(), key=lambda x: x[1], reverse=True)
    raw_symbols: List[str] = []
    meta_by_sym: Dict[str, Dict[str, Any]] = {}

    for base, mc in ranked_bases:
        if base in _STABLE_BASES:
            continue
        if base in daily_bases:
            continue
        sym = base_to_binance_symbol(base)
        if not sym or sym in daily_syms:
            continue
        if sym in meta_by_sym:
            continue
        raw_symbols.append(sym)
        meta_by_sym[sym] = {
            "symbol": sym,
            "base": symbol_to_base(sym),
            "market_cap_usd": float(mc),
        }

    filtered = drop_blacklisted_symbols(
        filter_symbols_to_binance_usdt_perps(raw_symbols)
    )
    out: List[Dict[str, Any]] = []
    for sym in filtered:
        if sym in meta_by_sym:
            out.append(dict(meta_by_sym[sym]))
        if len(out) >= mcap_limit:
            break

    logger.info(
        "[moss] mcap scan candidates=%s (cap_top=%s, excluded_daily=%s)",
        len(out),
        mcap_limit,
        len(daily_bases),
    )
    return out
