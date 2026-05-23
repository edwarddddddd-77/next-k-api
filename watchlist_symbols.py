"""共享：worth_watch 标的池 + 币安 U 本位永续过滤（Supertrend / ZCT 共用）。"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Set, Tuple

_PERP_SYMBOLS_CACHE: Optional[Tuple[float, Set[str]]] = None
_PERP_SYMBOLS_CACHE_TTL_SEC = max(
    60, int(os.getenv("ZCT_PERP_SYMBOLS_CACHE_TTL_SEC", "3600").strip() or "3600")
)


def binance_usdt_perp_symbol_set() -> Set[str]:
    global _PERP_SYMBOLS_CACHE
    now = time.time()
    if _PERP_SYMBOLS_CACHE is not None:
        ts, allowed = _PERP_SYMBOLS_CACHE
        if now - ts < _PERP_SYMBOLS_CACHE_TTL_SEC:
            return allowed
    try:
        from accumulation_radar import get_all_perp_symbols

        allowed = set(get_all_perp_symbols())
    except Exception:
        if _PERP_SYMBOLS_CACHE is not None:
            return _PERP_SYMBOLS_CACHE[1]
        raise
    _PERP_SYMBOLS_CACHE = (now, allowed)
    return allowed


def filter_symbols_to_binance_usdt_perps(raw: List[str]) -> List[str]:
    """
    worth_watch 里可能出现非 U 本位永续、已下架或格式不一致的 symbol。
    仅保留币安 /fapi exchangeInfo 中 status=TRADING 的 USDT 永续，其余跳过。
    """
    if not raw:
        return []
    try:
        allowed = binance_usdt_perp_symbol_set()
    except Exception as e:
        print(f"[watchlist] get_all_perp_symbols 失败，沿用原始列表: {e}")
        return [s.strip().upper() for s in raw if s and str(s).strip()]
    kept: List[str] = []
    skipped: List[str] = []
    seen: Set[str] = set()
    for s in raw:
        u = str(s).strip().upper()
        if not u or u in seen:
            continue
        seen.add(u)
        if u in allowed:
            kept.append(u)
        else:
            skipped.append(u)
    if skipped:
        preview = skipped[:15]
        tail = "..." if len(skipped) > 15 else ""
        print(
            f"[watchlist] 跳过 {len(skipped)} 个非 USDT 永续或未上市合约: "
            f"{preview}{tail}"
        )
    return kept


def hot_oi_watchlist_symbols() -> List[str]:
    """worth_watch_hot_oi 当前标的（已过滤为币安 U 本位永续）。"""
    from accumulation_radar import init_db

    conn = init_db()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT symbol FROM worth_watch_hot_oi
            ORDER BY COALESCE(rank_in_category, 999) ASC, symbol ASC
            """
        )
        raw = [str(x[0]).strip().upper() for x in cur.fetchall() if x and x[0]]
        filtered = filter_symbols_to_binance_usdt_perps(raw)
        if raw and not filtered:
            print(
                "[warn] hot_oi: worth_watch_hot_oi 有标的但均无有效币安 U 本位永续(TRADING)，"
                "请核对是否与合约代码一致（如 1000SHIB 对应 1000SHIBUSDT）"
            )
        return filtered
    except Exception as e:
        print(f"[hot_oi] worth_watch_hot_oi 读取失败: {e}")
        return []
    finally:
        conn.close()
