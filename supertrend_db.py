"""Supertrend SQLite 表（accumulation.db）。"""

from __future__ import annotations

import sqlite3
from typing import Any, Optional


def migrate_st_tables(c: sqlite3.Cursor) -> None:
    c.execute(
        """CREATE TABLE IF NOT EXISTS st_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recorded_at_utc TEXT NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        trend INTEGER,
        signal_type TEXT,
        entry_price REAL,
        sl_price REAL,
        tp_price REAL,
        st_up REAL,
        st_dn REAL,
        st_atr REAL,
        timeframe TEXT,
        st_period INTEGER,
        st_multiplier REAL,
        entry_bar_open_ms INTEGER,
        outcome TEXT,
        outcome_at_utc TEXT,
        exit_price REAL,
        pnl_r REAL,
        pnl_usdt REAL,
        virtual_notional_usdt REAL,
        exit_rule TEXT,
        meta_json TEXT
    )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS st_settlements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        settled_at_utc TEXT NOT NULL,
        signal_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT,
        outcome TEXT NOT NULL,
        entry_price REAL,
        exit_price REAL,
        pnl_r REAL,
        pnl_usdt REAL,
        virtual_notional_usdt REAL,
        exit_rule TEXT
    )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS st_indicator_state (
        symbol TEXT PRIMARY KEY,
        last_bar_open_ms INTEGER NOT NULL,
        last_trend INTEGER NOT NULL,
        updated_at_utc TEXT NOT NULL
    )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS st_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ran_at_utc TEXT NOT NULL,
        symbols_scanned INTEGER,
        flips INTEGER,
        opens INTEGER,
        closes INTEGER,
        skipped TEXT,
        detail_json TEXT
    )"""
    )
    c.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_st_signals_symbol ON st_signals(symbol)"
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS ix_st_settle_symbol ON st_settlements(symbol)"
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS ix_st_settle_time ON st_settlements(settled_at_utc)"
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS st_symbol_cooldown (
        symbol TEXT PRIMARY KEY,
        until_bar_open_ms INTEGER NOT NULL,
        until_utc_ms INTEGER,
        reason TEXT NOT NULL,
        updated_at_utc TEXT NOT NULL,
        blocked_side TEXT
    )"""
    )
    _ensure_cooldown_blocked_side_column(c)


def _ensure_cooldown_blocked_side_column(c: sqlite3.Cursor) -> None:
    c.execute("PRAGMA table_info(st_symbol_cooldown)")
    cols = {str(row[1]) for row in c.fetchall()}
    if "blocked_side" not in cols:
        c.execute("ALTER TABLE st_symbol_cooldown ADD COLUMN blocked_side TEXT")


def fetch_open_row(cur: sqlite3.Cursor, symbol: str) -> Optional[sqlite3.Row]:
    cur.execute(
        """
        SELECT * FROM st_signals
        WHERE symbol = ? AND outcome IS NULL AND side IN ('LONG', 'SHORT')
        """,
        (symbol,),
    )
    return cur.fetchone()


def list_open_position_symbols(cur: sqlite3.Cursor) -> list[str]:
    cur.execute(
        """
        SELECT DISTINCT symbol FROM st_signals
        WHERE outcome IS NULL AND side IN ('LONG', 'SHORT')
        ORDER BY symbol ASC
        """
    )
    return [str(r[0]) for r in cur.fetchall() if r and r[0]]


def count_open_positions(cur: sqlite3.Cursor) -> int:
    cur.execute(
        """
        SELECT COUNT(*) FROM st_signals
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
    pnl_r: Optional[float],
    pnl_usdt: Optional[float],
    notional: float,
    exit_rule: str,
    settled_at_utc: str,
) -> None:
    cur.execute(
        """
        INSERT INTO st_settlements (
            settled_at_utc, signal_id, symbol, side, outcome,
            entry_price, exit_price, pnl_r, pnl_usdt, virtual_notional_usdt, exit_rule
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            settled_at_utc,
            signal_id,
            symbol,
            side,
            outcome,
            entry_price,
            exit_price,
            pnl_r,
            pnl_usdt,
            notional,
            exit_rule,
        ),
    )


def upsert_indicator_state(
    cur: sqlite3.Cursor,
    *,
    symbol: str,
    bar_open_ms: int,
    trend: int,
    updated_at_utc: str,
) -> None:
    cur.execute(
        """
        INSERT INTO st_indicator_state (symbol, last_bar_open_ms, last_trend, updated_at_utc)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            last_bar_open_ms = excluded.last_bar_open_ms,
            last_trend = excluded.last_trend,
            updated_at_utc = excluded.updated_at_utc
        """,
        (symbol, bar_open_ms, trend, updated_at_utc),
    )


def get_indicator_state(cur: sqlite3.Cursor, symbol: str) -> Optional[Any]:
    cur.execute(
        "SELECT last_bar_open_ms, last_trend FROM st_indicator_state WHERE symbol = ?",
        (symbol,),
    )
    return cur.fetchone()


def upsert_symbol_cooldown(
    cur: sqlite3.Cursor,
    *,
    symbol: str,
    until_bar_open_ms: int,
    until_utc_ms: Optional[int],
    reason: str,
    updated_at_utc: str,
    blocked_side: Optional[str] = None,
) -> None:
    cur.execute(
        """
        INSERT INTO st_symbol_cooldown (
            symbol, until_bar_open_ms, until_utc_ms, reason, updated_at_utc, blocked_side
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            until_bar_open_ms = MAX(excluded.until_bar_open_ms, st_symbol_cooldown.until_bar_open_ms),
            until_utc_ms = CASE
                WHEN excluded.until_utc_ms IS NULL THEN st_symbol_cooldown.until_utc_ms
                WHEN st_symbol_cooldown.until_utc_ms IS NULL THEN excluded.until_utc_ms
                ELSE MAX(excluded.until_utc_ms, st_symbol_cooldown.until_utc_ms)
            END,
            reason = excluded.reason,
            updated_at_utc = excluded.updated_at_utc,
            blocked_side = COALESCE(excluded.blocked_side, st_symbol_cooldown.blocked_side)
        """,
        (symbol, until_bar_open_ms, until_utc_ms, reason, updated_at_utc, blocked_side),
    )


def cooldown_blocks_entry(
    cur: sqlite3.Cursor,
    symbol: str,
    *,
    bar_open_ms: int,
    now_utc_ms: int,
    entry_side: str,
) -> Optional[str]:
    cur.execute(
        """
        SELECT until_bar_open_ms, until_utc_ms, reason, blocked_side
        FROM st_symbol_cooldown WHERE symbol = ?
        """,
        (symbol,),
    )
    row = cur.fetchone()
    if not row:
        return None
    blocked_side = row[3] if len(row) > 3 else None
    if blocked_side and str(blocked_side).upper() != str(entry_side).upper():
        return None
    until_bar = int(row[0] or 0)
    until_utc = row[1]
    if until_bar > bar_open_ms:
        return str(row[2] or "cooldown_bar")
    if until_utc is not None and int(until_utc) > now_utc_ms:
        return str(row[2] or "cooldown_time")
    return None


def purge_expired_cooldowns(
    cur: sqlite3.Cursor,
    *,
    bar_open_ms: int,
    now_utc_ms: int,
) -> None:
    cur.execute(
        """
        DELETE FROM st_symbol_cooldown
        WHERE (until_bar_open_ms IS NULL OR until_bar_open_ms <= ?)
          AND (until_utc_ms IS NULL OR until_utc_ms <= ?)
        """,
        (bar_open_ms, now_utc_ms),
    )


def clear_st_lane_tables(conn: sqlite3.Connection) -> dict[str, int]:
    """清空 Supertrend 车道全部表（维护用）。"""
    cur = conn.cursor()
    out: dict[str, int] = {}
    for table, key in (
        ("st_settlements", "deleted_st_settlements"),
        ("st_signals", "deleted_st_signals"),
        ("st_indicator_state", "deleted_st_indicator_state"),
        ("st_runs", "deleted_st_runs"),
        ("st_symbol_cooldown", "deleted_st_symbol_cooldown"),
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


def count_symbol_losses_today(cur: sqlite3.Cursor, symbol: str, day_prefix: str) -> int:
    cur.execute(
        """
        SELECT COUNT(*) FROM st_settlements
        WHERE symbol = ? AND settled_at_utc >= ?
          AND pnl_usdt IS NOT NULL AND pnl_usdt < 0
        """,
        (symbol, f"{day_prefix}T00:00:00Z"),
    )
    return int(cur.fetchone()[0] or 0)
