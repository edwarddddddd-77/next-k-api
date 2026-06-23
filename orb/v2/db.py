"""ORB 2.0 SQLite 辅助表。"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, Optional


def migrate_orb_v2_tables(c: sqlite3.Cursor) -> None:
    from orb.v2.robots import migrate_orb_robots

    migrate_orb_robots(c)
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS orb_v2_breakout_seen (
            session_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            first_seen_at_utc TEXT NOT NULL,
            scan_open_ms INTEGER,
            p_true REAL,
            opened INTEGER NOT NULL DEFAULT 0,
            reason TEXT,
            PRIMARY KEY (session_date, symbol)
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS orb_v2_gate_day (
            session_date TEXT PRIMARY KEY,
            scored_signals INTEGER NOT NULL DEFAULT 0,
            recent_p_json TEXT NOT NULL DEFAULT '[]',
            day_aborted INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS orb_v2_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ran_at_utc TEXT NOT NULL,
            symbols_scanned INTEGER DEFAULT 0,
            opens INTEGER DEFAULT 0,
            gate_skips INTEGER DEFAULT 0,
            detail_json TEXT
        )
        """
    )


def mark_breakout_seen(
    cur: sqlite3.Cursor,
    *,
    session_date: str,
    symbol: str,
    now_utc: str,
    scan_open_ms: int,
    p_true: float,
    opened: bool,
    reason: str,
) -> None:
    sym = str(symbol).strip().upper()
    params = (
        str(session_date),
        sym,
        now_utc,
        int(scan_open_ms),
        float(p_true),
        str(reason),
    )
    if opened:
        # Gate 拒绝会先写入 opened=0；成功开仓必须能覆盖，否则 session_traded 锁不住同日再开。
        cur.execute(
            """
            INSERT INTO orb_v2_breakout_seen
                (session_date, symbol, first_seen_at_utc, scan_open_ms, p_true, opened, reason)
            VALUES (?, ?, ?, ?, ?, 1, ?)
            ON CONFLICT(session_date, symbol) DO UPDATE SET
                scan_open_ms = excluded.scan_open_ms,
                p_true = excluded.p_true,
                opened = 1,
                reason = excluded.reason
            """,
            params,
        )
        return
    cur.execute(
        """
        INSERT OR IGNORE INTO orb_v2_breakout_seen
            (session_date, symbol, first_seen_at_utc, scan_open_ms, p_true, opened, reason)
        VALUES (?, ?, ?, ?, ?, 0, ?)
        """,
        params,
    )


def rollback_breakout_opened(
    cur: sqlite3.Cursor, session_date: str, symbol: str, *, reason: str = "live_open_failed"
) -> bool:
    """Undo opened=1 when protocol live open failed after paper row was written."""
    sym = str(symbol).strip().upper()
    cur.execute(
        """
        UPDATE orb_v2_breakout_seen
        SET opened = 0, reason = ?
        WHERE session_date = ? AND symbol = ? AND opened = 1
        """,
        (str(reason), str(session_date), sym),
    )
    return cur.rowcount > 0


def breakout_seen_today(cur: sqlite3.Cursor, symbol: str, session_date: str) -> bool:
    if not session_date:
        return False
    cur.execute(
        """
        SELECT 1 FROM orb_v2_breakout_seen
        WHERE session_date = ? AND symbol = ?
        LIMIT 1
        """,
        (str(session_date), str(symbol).strip().upper()),
    )
    return cur.fetchone() is not None


def breakout_opened_today(cur: sqlite3.Cursor, symbol: str, session_date: str) -> bool:
    """当日该标的是否已成功开仓（opened=1）。"""
    if not session_date:
        return False
    cur.execute(
        """
        SELECT 1 FROM orb_v2_breakout_seen
        WHERE session_date = ? AND symbol = ? AND opened = 1
        LIMIT 1
        """,
        (str(session_date), str(symbol).strip().upper()),
    )
    return cur.fetchone() is not None


def count_v2_opens_today(cur: sqlite3.Cursor, session_date: str) -> int:
    cur.execute(
        """
        SELECT COUNT(*) FROM orb_v2_breakout_seen
        WHERE session_date = ? AND opened = 1
        """,
        (str(session_date),),
    )
    row = cur.fetchone()
    if row is None:
        return 0
    return int(row[0] if not hasattr(row, "keys") else row[0])


def load_gate_day_meta(cur: sqlite3.Cursor, session_date: str) -> Dict[str, Any]:
    cur.execute(
        """
        SELECT scored_signals, recent_p_json, day_aborted
        FROM orb_v2_gate_day WHERE session_date = ?
        """,
        (str(session_date),),
    )
    row = cur.fetchone()
    if row is None:
        return {"scored_signals": 0, "recent_p": [], "day_aborted": False}
    if hasattr(row, "keys"):
        recent_raw = row["recent_p_json"]
        scored = row["scored_signals"]
        aborted = row["day_aborted"]
    else:
        scored, recent_raw, aborted = row[0], row[1], row[2]
    try:
        recent = json.loads(recent_raw or "[]")
    except json.JSONDecodeError:
        recent = []
    return {
        "scored_signals": int(scored or 0),
        "recent_p": [float(x) for x in recent if x is not None],
        "day_aborted": bool(aborted),
    }


def save_gate_day_meta(
    cur: sqlite3.Cursor,
    session_date: str,
    *,
    scored_signals: int,
    recent_p: list,
    day_aborted: bool,
) -> None:
    cur.execute(
        """
        INSERT INTO orb_v2_gate_day(session_date, scored_signals, recent_p_json, day_aborted)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(session_date) DO UPDATE SET
            scored_signals=excluded.scored_signals,
            recent_p_json=excluded.recent_p_json,
            day_aborted=excluded.day_aborted
        """,
        (
            str(session_date),
            int(scored_signals),
            json.dumps([float(x) for x in recent_p]),
            1 if day_aborted else 0,
        ),
    )
