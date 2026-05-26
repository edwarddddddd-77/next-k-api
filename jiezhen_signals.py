"""接针标的池：严选 jz_universe（默认）或 worth_watch_hot_oi（兼容）。"""

from __future__ import annotations

import sqlite3
from typing import List, Optional, Tuple

import jiezhen_config as cfg
from watchlist_symbols import (
    drop_blacklisted_symbols,
    hot_oi_watchlist_symbols,
    symbol_blacklist,
)


def resolve_jiezhen_universe(
    conn: Optional[sqlite3.Connection] = None,
) -> Tuple[List[str], dict]:
    """
    返回 (symbols, meta)。
    curated：读 jz_universe（必要时刷新）；hot_oi：旧逻辑。
    """
    if cfg.jiezhen_universe_curated():
        return _resolve_curated(conn)
    return _resolve_hot_oi_legacy()


def _resolve_curated(
    conn: Optional[sqlite3.Connection],
) -> Tuple[List[str], dict]:
    from jiezhen_universe import (
        load_jiezhen_universe_rows,
        load_jiezhen_universe_symbols,
        refresh_jiezhen_universe_if_stale,
    )

    close = False
    if conn is None:
        from accumulation_radar import init_db

        conn = init_db()
        close = True
    try:
        refresh_meta = refresh_jiezhen_universe_if_stale(conn)
        cur = conn.cursor()
        syms = load_jiezhen_universe_symbols(cur)
        rows = load_jiezhen_universe_rows(cur)
        meta = {
            "source": "jz_universe",
            "mode": "curated_v2",
            "count": len(syms),
            "universe_max": cfg.JIEZHEN_UNIVERSE_MAX,
            "refresh": refresh_meta,
            "entries": rows[:50],
        }
        if not syms:
            meta["warning"] = (
                "empty_jz_universe_run_oi_cron_or_lower_JIEZHEN_SEL_*_thresholds"
            )
        bl = symbol_blacklist()
        if bl:
            meta["blacklist"] = list(bl)
        return syms, meta
    finally:
        if close:
            conn.close()


def _resolve_hot_oi_legacy() -> Tuple[List[str], dict]:
    raw = drop_blacklisted_symbols(hot_oi_watchlist_symbols())
    meta = {"source": "worth_watch_hot_oi", "mode": "hot_oi", "raw_count": len(raw)}
    bl = symbol_blacklist()
    if bl:
        meta["blacklist"] = list(bl)
    if not raw:
        meta["warning"] = "empty_universe_run_oi_cron_first"
        return [], meta
    cap = cfg.JIEZHEN_UNIVERSE_MAX
    syms = raw[:cap]
    meta["capped_count"] = len(syms)
    meta["universe_max"] = cap
    return syms, meta
