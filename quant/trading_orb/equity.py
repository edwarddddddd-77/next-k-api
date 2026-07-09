"""每标机器人权益（复利读 orb_vnpy_symbol_bots）。"""

from __future__ import annotations

import sqlite3
from typing import Optional

from quant.common.kline_cache import norm_symbol
from quant.trading_orb.config import OrbVnpyConfig
from quant.trading_orb.db import load_wallet


def symbol_equity_usdt(
    cfg: OrbVnpyConfig,
    symbol: str,
    *,
    cur: Optional[sqlite3.Cursor] = None,
) -> float:
    base = float(cfg.equity_usdt or 14.0)
    if not cfg.compound or cur is None:
        return base
    return load_wallet(cur, norm_symbol(symbol), default=base)
