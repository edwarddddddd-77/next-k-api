"""Aberration 复利读 orb_vnpy_symbol_bots（与 ORB/ICT 同表）。"""

from __future__ import annotations

import sqlite3
from typing import Optional

from orb.aberration.config import AberrationVnpyConfig
from orb.core.kline_cache import norm_symbol
from orb.trading_orb.db import load_wallet


def symbol_equity_usdt(
    cfg: AberrationVnpyConfig,
    symbol: str,
    *,
    cur: Optional[sqlite3.Cursor] = None,
) -> float:
    base = float(cfg.equity_usdt or 500.0)
    if not cfg.compound or cur is None:
        return base
    return load_wallet(cur, norm_symbol(symbol), default=base)
