"""Anchor Drift SQLite（锚定价持久化）。"""

from __future__ import annotations

import sqlite3
from typing import Optional

from quant.common.kline_cache import norm_symbol


def migrate_anchor_drift_tables(c: sqlite3.Cursor) -> None:
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS anchor_drift_state (
            symbol TEXT PRIMARY KEY,
            anchor_session TEXT NOT NULL,
            anchor_price REAL NOT NULL,
            anchor_ms INTEGER NOT NULL,
            updated_at_utc TEXT NOT NULL
        )
        """
    )


def load_anchor(
    cur: sqlite3.Cursor,
    symbol: str,
) -> Optional[tuple[str, float, int]]:
    sym = norm_symbol(symbol)
    cur.execute(
        """
        SELECT anchor_session, anchor_price, anchor_ms
        FROM anchor_drift_state
        WHERE symbol = ?
        """,
        (sym,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return str(row[0]), float(row[1]), int(row[2])


def save_anchor(
    cur: sqlite3.Cursor,
    *,
    symbol: str,
    anchor_session: str,
    anchor_price: float,
    anchor_ms: int,
    now_utc: str,
) -> None:
    sym = norm_symbol(symbol)
    cur.execute(
        """
        INSERT INTO anchor_drift_state (
            symbol, anchor_session, anchor_price, anchor_ms, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(symbol) DO UPDATE SET
            anchor_session = excluded.anchor_session,
            anchor_price = excluded.anchor_price,
            anchor_ms = excluded.anchor_ms,
            updated_at_utc = excluded.updated_at_utc
        """,
        (sym, anchor_session, float(anchor_price), int(anchor_ms), now_utc),
    )
