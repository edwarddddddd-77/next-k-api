"""Supertrend 标的池：默认 🔥⚡ 热度+OI（worth_watch_hot_oi）。"""

from __future__ import annotations

import os
from typing import List

import supertrend_config as cfg


def resolve_symbols() -> List[str]:
    mode = cfg.ST_UNIVERSE_MODE
    if mode == "hot_oi":
        syms = _hot_oi_symbols()
    elif mode == "static":
        raw = (os.getenv("ST_SYMBOLS", "") or "").strip()
        syms = [s.strip().upper() for s in raw.split(",") if s.strip()]
        syms = _filter_perps(syms)
    else:
        syms = _hot_oi_symbols()
    if cfg.ST_MAX_SYMBOLS > 0:
        syms = syms[: cfg.ST_MAX_SYMBOLS]
    return syms


def _hot_oi_symbols() -> List[str]:
    from watchlist_symbols import hot_oi_watchlist_symbols

    return hot_oi_watchlist_symbols()


def _filter_perps(raw: List[str]) -> List[str]:
    from watchlist_symbols import filter_symbols_to_binance_usdt_perps

    return filter_symbols_to_binance_usdt_perps(raw)
