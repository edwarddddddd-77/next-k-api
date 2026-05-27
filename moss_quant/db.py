"""Moss 量化 SQLite 表。"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def migrate_moss_tables(c: sqlite3.Cursor) -> None:
    c.execute(
        """CREATE TABLE IF NOT EXISTS moss_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        symbol TEXT NOT NULL,
        template TEXT,
        enabled INTEGER NOT NULL DEFAULT 0,
        initial_params_json TEXT NOT NULL,
        tactical_params_json TEXT NOT NULL,
        virtual_equity_usdt REAL NOT NULL DEFAULT 10000,
        evolution_enabled INTEGER NOT NULL DEFAULT 1,
        created_at_utc TEXT NOT NULL,
        updated_at_utc TEXT NOT NULL
    )"""
    )
    c.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS ux_moss_profiles_symbol_enabled
        ON moss_profiles(symbol) WHERE enabled = 1"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS moss_backtest_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        mode TEXT NOT NULL,
        symbol TEXT NOT NULL,
        range_start_utc TEXT,
        range_end_utc TEXT,
        segment_bars INTEGER,
        initial_params_json TEXT,
        result_json TEXT,
        evolution_log_json TEXT,
        schedule_json TEXT,
        summary_json TEXT,
        created_at_utc TEXT NOT NULL,
        FOREIGN KEY (profile_id) REFERENCES moss_profiles(id)
    )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS moss_kline_meta (
        symbol TEXT NOT NULL,
        interval TEXT NOT NULL,
        first_open_time_ms INTEGER,
        last_open_time_ms INTEGER,
        bar_count INTEGER,
        updated_at_utc TEXT NOT NULL,
        PRIMARY KEY (symbol, interval)
    )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS moss_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER NOT NULL,
        recorded_at_utc TEXT NOT NULL,
        side TEXT NOT NULL,
        symbol TEXT NOT NULL,
        entry_price REAL,
        virtual_notional_usdt REAL,
        mark_price REAL,
        composite REAL,
        regime TEXT,
        unrealized_pnl_usdt REAL,
        outcome TEXT,
        outcome_at_utc TEXT,
        exit_price REAL,
        pnl_usdt REAL,
        exit_rule TEXT,
        meta_json TEXT,
        updated_at_utc TEXT,
        FOREIGN KEY (profile_id) REFERENCES moss_profiles(id)
    )"""
    )
    c.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS ux_moss_open_profile
        ON moss_signals(profile_id) WHERE outcome IS NULL AND side IN ('LONG','SHORT')"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS moss_settlements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        settled_at_utc TEXT NOT NULL,
        signal_id INTEGER NOT NULL,
        profile_id INTEGER NOT NULL,
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
        """CREATE TABLE IF NOT EXISTS moss_paper_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ran_at_utc TEXT NOT NULL,
        profiles_scanned INTEGER,
        opens INTEGER,
        closes INTEGER,
        detail_json TEXT
    )"""
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS ix_moss_runs_profile ON moss_backtest_runs(profile_id)"
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS ix_moss_signals_profile ON moss_signals(profile_id)"
    )


def row_to_profile(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["enabled"] = bool(d.get("enabled"))
    d["evolution_enabled"] = bool(d.get("evolution_enabled"))
    d["initial_params"] = json.loads(d.pop("initial_params_json") or "{}")
    d["tactical_params"] = json.loads(d.pop("tactical_params_json") or "{}")
    return d


def get_profile(conn: sqlite3.Connection, profile_id: int) -> Optional[Dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM moss_profiles WHERE id = ?", (int(profile_id),)
    ).fetchone()
    return row_to_profile(row) if row else None


def count_enabled_profiles(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM moss_profiles WHERE enabled = 1"
    ).fetchone()
    return int(row[0] or 0)


def list_enabled_profiles(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM moss_profiles WHERE enabled = 1 ORDER BY id ASC"
    ).fetchall()
    return [row_to_profile(r) for r in rows]
