"""vnpy 策略发出信号 — 持久化与查询（next-k-api，非 Protocol）。"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from orb.core.kline_cache import norm_symbol

LANE_TRADING_ORB = "trading_orb"
LANE_ICT_2022 = "ict_2022"
VALID_LANES = {LANE_TRADING_ORB, LANE_ICT_2022}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    from accumulation_radar import init_db

    conn = init_db()
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


def _legacy_trade_lane(detail_raw: Optional[str]) -> str:
    if not detail_raw:
        return LANE_TRADING_ORB
    try:
        detail = json.loads(detail_raw)
    except (TypeError, json.JSONDecodeError):
        return LANE_TRADING_ORB
    if isinstance(detail, dict) and str(detail.get("lane") or "") == LANE_ICT_2022:
        return LANE_ICT_2022
    return LANE_TRADING_ORB


def _legacy_trades_as_signals(cur: sqlite3.Cursor, *, lane: str, limit: int) -> List[Dict[str, Any]]:
    from orb.trading_orb.db import migrate_orb_vnpy_tables

    migrate_orb_vnpy_tables(cur)
    cur.execute(
        """
        SELECT id, symbol, side, entry, detail_json, bar_ms, created_at_utc, event
        FROM orb_vnpy_trades
        WHERE event = 'open'
        ORDER BY id DESC
        LIMIT ?
        """,
        (max(limit * 3, 50),),
    )
    out: List[Dict[str, Any]] = []
    for row in cur.fetchall():
        trade_lane = _legacy_trade_lane(row["detail_json"])
        if trade_lane != lane:
            continue
        out.append(
            {
                "id": -int(row["id"]),
                "lane": trade_lane,
                "symbol": row["symbol"],
                "side": row["side"],
                "action": "open",
                "entry_price": row["entry"],
                "sl_price": None,
                "tp_price": None,
                "status": "filled",
                "skip_reason": None,
                "detail": {},
                "bar_ms": row["bar_ms"],
                "received_at": row["created_at_utc"],
            }
        )
        if len(out) >= limit:
            break
    return out


def _signal_dedup_key(row: Dict[str, Any]) -> tuple:
    entry = row.get("entry_price")
    entry_key = round(float(entry), 4) if entry is not None else None
    ts = str(row.get("received_at") or "")[:16]
    return (row.get("symbol"), row.get("side"), entry_key, ts)


def list_strategy_signals(*, lane: str, limit: int = 100) -> Dict[str, Any]:
    lane_s = str(lane or "").strip()
    if lane_s not in VALID_LANES:
        return {"ok": False, "lane": lane_s, "count": 0, "signals": [], "error": "invalid_lane"}
    lim = max(1, min(int(limit or 100), 500))
    from accumulation_radar import init_db

    conn = init_db()
    try:
        cur = conn.cursor()
        migrate_strategy_signals_table(cur)
        cur.execute(
            """
            SELECT * FROM strategy_signals
            WHERE lane = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (lane_s, lim),
        )
        rows = [_row_to_signal(r) for r in cur.fetchall()]
        if len(rows) < lim:
            seen = {_signal_dedup_key(r) for r in rows}
            for legacy in _legacy_trades_as_signals(cur, lane=lane_s, limit=lim - len(rows)):
                key = _signal_dedup_key(legacy)
                if key in seen:
                    continue
                rows.append(legacy)
                seen.add(key)
        rows.sort(key=lambda r: str(r.get("received_at") or ""), reverse=True)
        return {"ok": True, "lane": lane_s, "count": len(rows[:lim]), "signals": rows[:lim]}
    finally:
        conn.close()
