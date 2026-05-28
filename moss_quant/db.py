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
    c.execute(
        """CREATE TABLE IF NOT EXISTS moss_daily_optimize_batches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ran_at_utc TEXT NOT NULL,
        finished_at_utc TEXT,
        status TEXT NOT NULL,
        symbols_total INTEGER,
        symbols_ok INTEGER,
        capital REAL,
        data_source TEXT,
        kline_start TEXT,
        kline_end TEXT,
        error TEXT
    )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS moss_daily_optimize_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        template TEXT,
        tactical_params_json TEXT,
        summary_json TEXT,
        score REAL,
        profile_id INTEGER,
        FOREIGN KEY (batch_id) REFERENCES moss_daily_optimize_batches(id)
    )"""
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS ix_moss_daily_items_batch ON moss_daily_optimize_items(batch_id)"
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS moss_mcap_scan_batches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ran_at_utc TEXT NOT NULL,
        finished_at_utc TEXT,
        status TEXT NOT NULL,
        symbols_total INTEGER,
        symbols_ok INTEGER,
        capital REAL,
        data_source TEXT,
        kline_start TEXT,
        kline_end TEXT,
        display_top_n INTEGER,
        mcap_pool_limit INTEGER,
        error TEXT
    )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS moss_mcap_scan_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        batch_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        market_cap_usd REAL,
        mcap_rank INTEGER,
        template TEXT,
        tactical_params_json TEXT,
        summary_json TEXT,
        score REAL,
        FOREIGN KEY (batch_id) REFERENCES moss_mcap_scan_batches(id)
    )"""
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS ix_moss_mcap_items_batch ON moss_mcap_scan_items(batch_id)"
    )
    _ensure_profile_source_column(c)


def _ensure_profile_source_column(c: sqlite3.Cursor) -> None:
    cols = {row[1] for row in c.execute("PRAGMA table_info(moss_profiles)").fetchall()}
    if "profile_source" not in cols:
        c.execute(
            "ALTER TABLE moss_profiles ADD COLUMN profile_source TEXT NOT NULL DEFAULT 'manual'"
        )


DAILY_PROFILE_SOURCE = "daily_auto"  # 历史遗留；新逻辑不再自动创建
FROM_DAILY_PROFILE_SOURCE = "from_daily"
MANUAL_PROFILE_SOURCE = "manual"
DAILY_PROFILE_NAME_PREFIX = "daily-"


def daily_profile_name(symbol: str) -> str:
    return DAILY_PROFILE_NAME_PREFIX + str(symbol).strip().upper()


def get_daily_profile_by_symbol(
    conn: sqlite3.Connection, symbol: str
) -> Optional[Dict[str, Any]]:
    sym = str(symbol).strip().upper()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """SELECT * FROM moss_profiles
           WHERE profile_source = ? AND symbol = ? LIMIT 1""",
        (DAILY_PROFILE_SOURCE, sym),
    ).fetchone()
    return row_to_profile(row) if row else None


def row_to_profile(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["enabled"] = bool(d.get("enabled"))
    d["evolution_enabled"] = bool(d.get("evolution_enabled"))
    d["profile_source"] = str(d.get("profile_source") or "manual")
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


def profile_has_open_position(conn: sqlite3.Connection, profile_id: int) -> bool:
    row = conn.execute(
        """SELECT 1 FROM moss_signals
           WHERE profile_id = ? AND outcome IS NULL AND side IN ('LONG','SHORT')
           LIMIT 1""",
        (int(profile_id),),
    ).fetchone()
    return row is not None


def list_profiles_for_strategy_sync(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """寻优后策略同步：已启用 Profile + 仍有持仓的 Profile（含已停用但有仓）。"""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT DISTINCT p.* FROM moss_profiles p
           WHERE p.enabled = 1
              OR EXISTS (
                  SELECT 1 FROM moss_signals s
                  WHERE s.profile_id = p.id
                    AND s.outcome IS NULL
                    AND s.side IN ('LONG','SHORT')
              )
           ORDER BY p.id ASC"""
    ).fetchall()
    return [row_to_profile(r) for r in rows]


def get_profile_by_symbol(
    conn: sqlite3.Connection, symbol: str
) -> Optional[Dict[str, Any]]:
    sym = str(symbol).strip().upper()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM moss_profiles WHERE symbol = ? ORDER BY id DESC LIMIT 1",
        (sym,),
    ).fetchone()
    return row_to_profile(row) if row else None


def list_profiles_for_paper_scan(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """纸面扫描：已启用 Profile + 仍有持仓的 Profile（便于平仓）。"""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT DISTINCT p.* FROM moss_profiles p
           WHERE p.enabled = 1
              OR EXISTS (
                  SELECT 1 FROM moss_signals s
                  WHERE s.profile_id = p.id
                    AND s.outcome IS NULL
                    AND s.side IN ('LONG','SHORT')
              )
           ORDER BY p.id ASC"""
    ).fetchall()
    return [row_to_profile(r) for r in rows]


def delete_profile(conn: sqlite3.Connection, profile_id: int) -> Optional[Dict[str, int]]:
    """删除 Profile 及其关联纸面/回测记录（有持仓则失败）。"""
    pid = int(profile_id)
    prof = get_profile(conn, pid)
    if not prof:
        return None
    open_n = int(
        conn.execute(
            """SELECT COUNT(*) FROM moss_signals
               WHERE profile_id = ? AND outcome IS NULL AND side IN ('LONG','SHORT')""",
            (pid,),
        ).fetchone()[0]
        or 0
    )
    if open_n > 0:
        raise ValueError("profile_has_open_position")
    deleted = {
        "signals": int(
            conn.execute(
                "SELECT COUNT(*) FROM moss_signals WHERE profile_id = ?", (pid,)
            ).fetchone()[0]
            or 0
        ),
        "settlements": int(
            conn.execute(
                "SELECT COUNT(*) FROM moss_settlements WHERE profile_id = ?", (pid,)
            ).fetchone()[0]
            or 0
        ),
        "backtest_runs": int(
            conn.execute(
                "SELECT COUNT(*) FROM moss_backtest_runs WHERE profile_id = ?", (pid,)
            ).fetchone()[0]
            or 0
        ),
    }
    conn.execute("DELETE FROM moss_settlements WHERE profile_id = ?", (pid,))
    conn.execute("DELETE FROM moss_signals WHERE profile_id = ?", (pid,))
    conn.execute("DELETE FROM moss_backtest_runs WHERE profile_id = ?", (pid,))
    conn.execute("DELETE FROM moss_profiles WHERE id = ?", (pid,))
    deleted["profile_id"] = pid
    return deleted
