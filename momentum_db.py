"""动量策略 SQLite 表（accumulation.db）。"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional


def migrate_mom_tables(c: sqlite3.Cursor) -> None:
    c.execute(
        """CREATE TABLE IF NOT EXISTS mom_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recorded_at_utc TEXT NOT NULL,
        side TEXT NOT NULL,
        symbol TEXT NOT NULL,
        signal_type TEXT,
        entry_price REAL,
        virtual_notional_usdt REAL,
        event_timestamp_ms INTEGER,
        mark_price REAL,
        unrealized_pnl_usdt REAL,
        outcome TEXT,
        outcome_at_utc TEXT,
        exit_price REAL,
        pnl_usdt REAL,
        exit_rule TEXT,
        meta_json TEXT,
        updated_at_utc TEXT
    )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS mom_settlements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        settled_at_utc TEXT NOT NULL,
        signal_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        outcome TEXT NOT NULL,
        entry_price REAL,
        exit_price REAL,
        pnl_usdt REAL,
        virtual_notional_usdt REAL,
        exit_rule TEXT
    )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS mom_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ran_at_utc TEXT NOT NULL,
        long_target TEXT,
        short_target TEXT,
        opens INTEGER,
        closes INTEGER,
        skipped TEXT,
        detail_json TEXT
    )"""
    )
    c.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS ux_mom_open_side
        ON mom_signals(side) WHERE outcome IS NULL
        """
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS ix_mom_settle_time ON mom_settlements(settled_at_utc)"
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS ix_mom_signals_recorded ON mom_signals(recorded_at_utc)"
    )
    _ensure_updated_at_utc_column(c)
    _ensure_trail_columns(c)


def _ensure_trail_columns(c: sqlite3.Cursor) -> None:
    c.execute("PRAGMA table_info(mom_signals)")
    cols = {str(row[1]) for row in c.fetchall()}
    if "peak_profit_pct" not in cols:
        c.execute("ALTER TABLE mom_signals ADD COLUMN peak_profit_pct REAL")
    if "trail_tier" not in cols:
        c.execute("ALTER TABLE mom_signals ADD COLUMN trail_tier TEXT")


def _ensure_updated_at_utc_column(c: sqlite3.Cursor) -> None:
    c.execute("PRAGMA table_info(mom_signals)")
    cols = {str(row[1]) for row in c.fetchall()}
    if "updated_at_utc" not in cols:
        c.execute("ALTER TABLE mom_signals ADD COLUMN updated_at_utc TEXT")


def peak_profit_from_row(row: sqlite3.Row) -> float:
    try:
        keys = row.keys()
        if "peak_profit_pct" in keys and row["peak_profit_pct"] is not None:
            return float(row["peak_profit_pct"])
    except (TypeError, ValueError, IndexError):
        pass
    return 0.0


def trail_tier_from_row(row: sqlite3.Row) -> str:
    try:
        keys = row.keys()
        if "trail_tier" in keys and row["trail_tier"]:
            return str(row["trail_tier"])
    except (TypeError, ValueError, IndexError):
        pass
    return "none"


def fetch_open_by_side(cur: sqlite3.Cursor, side: str) -> Optional[sqlite3.Row]:
    cur.execute(
        """
        SELECT * FROM mom_signals
        WHERE side = ? AND outcome IS NULL
        ORDER BY id DESC LIMIT 1
        """,
        (side.upper(),),
    )
    return cur.fetchone()


def count_open_positions(cur: sqlite3.Cursor) -> int:
    cur.execute(
        """
        SELECT COUNT(*) FROM mom_signals
        WHERE outcome IS NULL AND side IN ('LONG', 'SHORT')
        """
    )
    return int(cur.fetchone()[0] or 0)


def archive_settlement(
    cur: sqlite3.Cursor,
    *,
    signal_id: int,
    symbol: str,
    side: str,
    outcome: str,
    entry_price: float,
    exit_price: float,
    pnl_usdt: float,
    notional: float,
    exit_rule: str,
    settled_at_utc: str,
) -> None:
    cur.execute(
        """
        INSERT INTO mom_settlements (
            settled_at_utc, signal_id, symbol, side, outcome,
            entry_price, exit_price, pnl_usdt, virtual_notional_usdt, exit_rule
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            settled_at_utc,
            signal_id,
            symbol,
            side,
            outcome,
            entry_price,
            exit_price,
            pnl_usdt,
            notional,
            exit_rule,
        ),
    )


def _settled_at_utc_to_ms(settled_at_utc: str | None) -> Optional[int]:
    if not settled_at_utc:
        return None
    raw = str(settled_at_utc).replace("Z", "+00:00")
    try:
        from datetime import datetime, timezone

        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except ValueError:
        return None


def last_close_utc_ms(
    cur: sqlite3.Cursor, *, symbol: str, side: str
) -> Optional[int]:
    ms, _ = last_close_info(cur, symbol=symbol, side=side)
    return ms


def last_close_info(
    cur: sqlite3.Cursor, *, symbol: str, side: str
) -> tuple[Optional[int], Optional[str]]:
    """该标的+方向最近一次平仓时间(ms)与 exit_rule。"""
    cur.execute(
        """
        SELECT settled_at_utc, exit_rule FROM mom_settlements
        WHERE symbol = ? AND side = ?
        ORDER BY id DESC LIMIT 1
        """,
        (symbol.upper(), side.upper()),
    )
    row = cur.fetchone()
    if not row:
        return None, None
    return _settled_at_utc_to_ms(row[0]), (
        str(row[1]) if row[1] is not None else None
    )


def clear_mom_lane_tables(conn: sqlite3.Connection) -> dict[str, int]:
    cur = conn.cursor()
    out: dict[str, int] = {}
    for table, key in (
        ("mom_settlements", "deleted_mom_settlements"),
        ("mom_signals", "deleted_mom_signals"),
        ("mom_runs", "deleted_mom_runs"),
    ):
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        if not cur.fetchone():
            out[key] = 0
            continue
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        n = int(cur.fetchone()[0] or 0)
        cur.execute(f"DELETE FROM {table}")
        out[key] = n
    conn.commit()
    return out
