"""接针标的池：worth_watch_hot_oi（热度+OI）。"""

from __future__ import annotations

from typing import List, Tuple

import jiezhen_config as cfg
from watchlist_symbols import hot_oi_watchlist_symbols


def resolve_jiezhen_universe() -> Tuple[List[str], dict]:
    """
    返回 (symbols, meta)。
    需先跑 oi 小时任务写入 worth_watch_hot_oi。
    """
    raw = hot_oi_watchlist_symbols()
    meta = {"source": "worth_watch_hot_oi", "raw_count": len(raw)}
    if not raw:
        meta["warning"] = "empty_universe_run_oi_cron_first"
        return [], meta
    cap = cfg.JIEZHEN_UNIVERSE_MAX
    syms = raw[:cap]
    meta["capped_count"] = len(syms)
    meta["universe_max"] = cap
    return syms, meta
