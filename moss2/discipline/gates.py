"""L1 开仓闸门：EV、连亏、composite margin。"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from moss2 import config as cfg
from moss2.discipline.metrics import trade_stats_from_rows


def recent_settled_trades(
    conn: sqlite3.Connection, profile_id: int, *, limit: int = 30
) -> List[dict]:
    from moss2.db import get_profile

    prof = get_profile(conn, profile_id) or {}
    base_notional = float(
        prof.get("virtual_equity_usdt") or cfg.MOSS2_PROFILE_CAPITAL
    )
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT pnl_usdt, virtual_notional_usdt, exit_rule, outcome_at_utc
           FROM moss2_signals
           WHERE profile_id=? AND outcome IS NOT NULL AND pnl_usdt IS NOT NULL
           ORDER BY outcome_at_utc DESC LIMIT ?""",
        (profile_id, limit),
    ).fetchall()
    out = []
    for r in rows:
        pnl = float(r["pnl_usdt"] or 0)
        notional = float(r["virtual_notional_usdt"] or 0) or base_notional
        pnl_pct = (pnl / notional) if notional > 0 else 0.0
        out.append({"pnl_usdt": pnl, "pnl_pct": pnl_pct})
    return out


def check_open_gate(
    conn: sqlite3.Connection,
    profile_id: int,
    *,
    composite: float,
    entry_threshold: float,
) -> Tuple[bool, str, Dict[str, Any]]:
    """返回 (allowed, reason, debug)。"""
    debug: Dict[str, Any] = {}
    margin = abs(composite) - float(entry_threshold)
    debug["composite"] = round(composite, 4)
    debug["entry_threshold"] = float(entry_threshold)
    debug["margin"] = round(margin, 4)

    extra = float(cfg.MOSS2_ENTRY_MARGIN or 0)
    if margin < extra:
        return False, "margin_below_threshold", debug

    if not cfg.MOSS2_DISCIPLINE_ENABLED or not cfg.MOSS2_DISCIPLINE_BLOCK_EV:
        return True, "ok", debug

    trades = recent_settled_trades(conn, profile_id)
    if len(trades) < cfg.MOSS2_DISCIPLINE_MIN_SETTLED:
        return True, "ok_insufficient_history", debug

    stats = trade_stats_from_rows(trades)
    debug["recent_ev"] = stats.get("ev_per_trade_pct")
    debug["recent_trades"] = stats.get("trade_count")
    if float(stats.get("ev_per_trade_pct") or 0) < 0:
        return False, "recent_ev_negative", debug
    if int(stats.get("max_consecutive_losses") or 0) >= cfg.MOSS2_DISCIPLINE_MAX_CONSEC_LOSS:
        return False, "max_consec_loss", debug

    return True, "ok", debug


def regime_notional_scale(regime_label: str) -> float:
    if not cfg.MOSS2_REGIME_SNOW_ENABLED:
        return 1.0
    r = (regime_label or "").upper()
    if r in cfg.MOSS2_REGIME_SNOW_REGIMES:
        return float(cfg.MOSS2_REGIME_SNOW_NOTIONAL_SCALE)
    return 1.0
