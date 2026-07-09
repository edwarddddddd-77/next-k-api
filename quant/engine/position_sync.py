"""启动时从交易所同步持仓到 vnpy 策略 pos。"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from quant.common.kline_cache import norm_symbol
from quant.engine.exchanges.registry import get_live_adapter

logger = logging.getLogger(__name__)


def sync_cta_positions(
    cta_engine: Any,
    symbols: List[str],
    *,
    strategy_prefix: str = "orb",
    restore_levels: bool = False,
) -> Dict[str, float]:
    """将交易所净持仓写入各 vnpy 策略 pos；无仓则 cancel_all。"""
    adapter = get_live_adapter()
    if restore_levels:
        snapshots = adapter.fetch_position_snapshots(symbols)
        amounts = {sym: float(s.get("amount") or 0.0) for sym, s in snapshots.items()}
    else:
        amounts = adapter.fetch_position_amounts(symbols)
        snapshots = {}
    out: Dict[str, float] = {}
    for raw in symbols:
        sym = norm_symbol(raw)
        name = f"{strategy_prefix}_{sym.lower()}"
        strat = cta_engine.strategies.get(name)
        if strat is None:
            continue
        amt = float(amounts.get(sym, 0.0) or 0.0)
        out[sym] = amt
        old = float(getattr(strat, "pos", 0) or 0.0)
        if abs(old - amt) > 1e-9:
            logger.warning("[vnpy] sync pos %s: strategy=%s exchange=%s", sym, old, amt)
        strat.pos = amt
        if amt == 0.0:
            try:
                strat.cancel_all()
            except Exception as exc:
                logger.warning("[vnpy] cancel_all after sync %s: %s", sym, exc)
        elif restore_levels:
            snap = snapshots.get(sym, {})
            entry = float(snap.get("entry") or 0.0)
            restore = getattr(strat, "restore_synced_position", None)
            if callable(restore) and entry > 0:
                try:
                    restore(entry_px=entry, pos=amt)
                except Exception as exc:
                    logger.warning("[vnpy] restore levels %s failed: %s", sym, exc)
        if cta_engine:
            try:
                cta_engine.sync_strategy_data(strat)
            except Exception:
                pass
    return out
