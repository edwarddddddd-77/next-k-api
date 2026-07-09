"""Trading ORB vnpy SQLite（orb_vnpy_* 表）。"""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, Optional


def migrate_orb_vnpy_tables(c: sqlite3.Cursor) -> None:
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS orb_vnpy_symbol_bots (
            symbol TEXT PRIMARY KEY,
            wallet_usdt REAL NOT NULL,
            updated_at_utc TEXT NOT NULL
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS orb_vnpy_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_date TEXT NOT NULL,
            symbol TEXT NOT NULL,
            event TEXT NOT NULL,
            side TEXT,
            entry REAL,
            exit_px REAL,
            notional_usdt REAL,
            pnl_usdt_gross REAL,
            fee_usdt REAL,
            pnl_usdt REAL,
            outcome TEXT,
            detail_json TEXT,
            bar_ms INTEGER,
            created_at_utc TEXT NOT NULL
        )
        """
    )


def load_wallet(cur: sqlite3.Cursor, symbol: str, *, default: float) -> float:
    cur.execute("SELECT wallet_usdt FROM orb_vnpy_symbol_bots WHERE symbol = ?", (symbol,))
    row = cur.fetchone()
    if row is None:
        return float(default)
    return float(row[0] or default)


def save_wallet(cur: sqlite3.Cursor, symbol: str, wallet: float, *, now_utc: str) -> None:
    cur.execute(
        """
        INSERT INTO orb_vnpy_symbol_bots (symbol, wallet_usdt, updated_at_utc)
        VALUES (?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            wallet_usdt = excluded.wallet_usdt,
            updated_at_utc = excluded.updated_at_utc
        """,
        (symbol, round(float(wallet), 4), now_utc),
    )


def insert_trade(
    cur: sqlite3.Cursor,
    *,
    session_date: str,
    symbol: str,
    event: str,
    side: str = "",
    entry: float = 0.0,
    exit_px: float = 0.0,
    notional_usdt: float = 0.0,
    pnl_usdt_gross: float = 0.0,
    fee_usdt: float = 0.0,
    pnl_usdt: float = 0.0,
    outcome: str = "",
    detail: Optional[Dict[str, Any]] = None,
    bar_ms: int = 0,
    now_utc: str,
) -> None:
    cur.execute(
        """
        INSERT INTO orb_vnpy_trades (
            session_date, symbol, event, side, entry, exit_px, notional_usdt,
            pnl_usdt_gross, fee_usdt, pnl_usdt, outcome, detail_json, bar_ms, created_at_utc
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            session_date,
            symbol,
            event,
            side,
            entry,
            exit_px,
            notional_usdt,
            pnl_usdt_gross,
            fee_usdt,
            pnl_usdt,
            outcome,
            json.dumps(detail or {}, ensure_ascii=False),
            int(bar_ms),
            now_utc,
        ),
    )
