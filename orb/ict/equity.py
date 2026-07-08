"""ICT 复利读 orb_vnpy_symbol_bots（与 ORB 同表，默认 ICT_VNPY_EQUITY_USDT）。"""

from __future__ import annotations

import sqlite3
from typing import Optional

from orb.core.kline_cache import norm_symbol
from orb.ict.config import IctVnpyConfig
from orb.trading_orb.db import load_wallet


def symbol_equity_usdt(
    cfg: IctVnpyConfig,
    symbol: str,
    *,
    cur: Optional[sqlite3.Cursor] = None,
) -> float:
    base = float(cfg.equity_usdt or 1000.0)
    if not cfg.compound or cur is None:
        return base
    return load_wallet(cur, norm_symbol(symbol), default=base)
