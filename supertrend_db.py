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
