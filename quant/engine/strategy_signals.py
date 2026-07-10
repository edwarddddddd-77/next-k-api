"""vnpy 策略开单信号 — 持久化与查询（next-k-api，非 Protocol）。

仅记录策略算法产生的 **开仓** 信号（action=open），例如 ORB 突破进场。
不记录：交易所持仓同步、强平平仓、成交回报、手动下单。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from quant.common.kline_cache import norm_symbol

LANE_TRADING_ORB = "trading_orb"
LANE_MTFMOMO = "mtfmomo"
LANE_KAMA_TREND = "kama_trend"
LANE_SQUEEZE_BREAKOUT = "squeeze_breakout"
VALID_LANES = {
    LANE_TRADING_ORB,
    LANE_MTFMOMO,
    LANE_KAMA_TREND,
    LANE_SQUEEZE_BREAKOUT,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _db_conn() -> sqlite3.Connection:
    from accumulation_radar import init_db

    conn = init_db()
    conn.row_factory = sqlite3.Row
    return conn


def migrate_strategy_signals_table(c: sqlite3.Cursor) -> None:
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS strategy_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lane TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            action TEXT NOT NULL DEFAULT 'open',
            entry_price REAL,
            sl_price REAL,
            tp_price REAL,
            status TEXT NOT NULL DEFAULT 'emitted',
            skip_reason TEXT,
            detail_json TEXT,
            bar_ms INTEGER,
            created_at_utc TEXT NOT NULL
        )
        """
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS ix_strategy_signals_lane_time "
        "ON strategy_signals(lane, created_at_utc DESC)"
    )


def record_strategy_open_signal(
    *,
    lane: str,
    symbol: str,
    side: str,
    entry_price: Optional[float] = None,
    sl_price: Optional[float] = None,
    tp_price: Optional[float] = None,
    status: str = "emitted",
    skip_reason: Optional[str] = None,
    detail: Optional[Dict[str, Any]] = None,
    bar_ms: int = 0,
) -> None:
    """策略产生的开仓信号（vnpy lane 应调用此函数，而非平仓/同步路径）。"""
    record_strategy_signal(
        lane=lane,
        symbol=symbol,
        side=side,
        action="open",
        entry_price=entry_price,
        sl_price=sl_price,
        tp_price=tp_price,
        status=status,
        skip_reason=skip_reason,
        detail=detail,
        bar_ms=bar_ms,
    )


def record_strategy_signal(
    *,
    lane: str,
    symbol: str,
    side: str,
    action: str = "open",
    entry_price: Optional[float] = None,
    sl_price: Optional[float] = None,
    tp_price: Optional[float] = None,
    status: str = "emitted",
    skip_reason: Optional[str] = None,
    detail: Optional[Dict[str, Any]] = None,
    bar_ms: int = 0,
) -> None:
    lane_s = str(lane or "").strip()
    if lane_s not in VALID_LANES:
        return
    sym = norm_symbol(symbol)
    side_u = str(side or "").upper()
    if not sym or side_u not in ("LONG", "SHORT"):
        return
    conn = _db_conn()
    try:
        cur = conn.cursor()
        migrate_strategy_signals_table(cur)
        cur.execute(
            """
            INSERT INTO strategy_signals (
                lane, symbol, side, action, entry_price, sl_price, tp_price,
                status, skip_reason, detail_json, bar_ms, created_at_utc
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                lane_s,
                sym,
                side_u,
                str(action or "open").lower(),
                float(entry_price) if entry_price is not None else None,
                float(sl_price) if sl_price is not None else None,
                float(tp_price) if tp_price is not None else None,
                str(status or "emitted"),
                skip_reason,
                json.dumps(detail or {}, ensure_ascii=False),
                int(bar_ms or 0),
                _utc_now(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _row_to_signal(row: sqlite3.Row) -> Dict[str, Any]:
    detail: Dict[str, Any] = {}
    raw = row["detail_json"]
    if raw:
        try:
            detail = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            detail = {}
    return {
        "id": int(row["id"]),
        "lane": row["lane"],
        "symbol": row["symbol"],
        "side": row["side"],
        "action": row["action"] or "open",
        "entry_price": row["entry_price"],
        "sl_price": row["sl_price"],
        "tp_price": row["tp_price"],
        "status": row["status"] or "emitted",
        "skip_reason": row["skip_reason"],
        "detail": detail,
        "bar_ms": row["bar_ms"],
        "received_at": row["created_at_utc"],
    }


def list_strategy_signals(*, lane: str, limit: int = 100) -> Dict[str, Any]:
    lane_s = str(lane or "").strip()
    if lane_s not in VALID_LANES:
        return {"ok": False, "lane": lane_s, "count": 0, "signals": [], "error": "invalid_lane"}
    lim = max(1, min(int(limit or 100), 500))
    conn = _db_conn()
    try:
        cur = conn.cursor()
        migrate_strategy_signals_table(cur)
        cur.execute(
            """
            SELECT * FROM strategy_signals
            WHERE lane = ? AND (action = 'open' OR action IS NULL OR action = '')
            ORDER BY id DESC
            LIMIT ?
            """,
            (lane_s, lim),
        )
        rows = [_row_to_signal(r) for r in cur.fetchall()]
        return {"ok": True, "lane": lane_s, "count": len(rows), "signals": rows}
    finally:
        conn.close()
