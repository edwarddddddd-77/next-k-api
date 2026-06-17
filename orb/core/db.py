"""ORB 纸面 SQLite 表。"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def migrate_orb_tables(c: sqlite3.Cursor) -> None:
    c.execute(
        """CREATE TABLE IF NOT EXISTS orb_signals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recorded_at_utc TEXT NOT NULL,
        updated_at_utc TEXT,
        symbol TEXT NOT NULL UNIQUE,
        play TEXT NOT NULL,
        side TEXT NOT NULL,
        confidence TEXT,
        entry_price REAL,
        entry_bar_open_ms INTEGER,
        sl_price REAL,
        tp_price REAL,
        r_unit REAL,
        virtual_notional_usdt REAL DEFAULT 1000,
        or_high REAL,
        or_low REAL,
        or_width_pct REAL,
        session_date TEXT,
        volume REAL,
        vol_ma REAL,
        mark_price REAL,
        unrealized_pnl_usdt REAL,
        outcome TEXT,
        outcome_at_utc TEXT,
        exit_price REAL,
        pnl_r REAL,
        pnl_usdt REAL,
        exit_rule TEXT,
        reasons_json TEXT,
        scan_params_json TEXT,
        notes TEXT
    )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS orb_settlements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        settled_at_utc TEXT NOT NULL,
        signal_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        play TEXT,
        outcome TEXT NOT NULL,
        entry_price REAL,
        exit_price REAL,
        pnl_r REAL,
        pnl_usdt REAL,
        virtual_notional_usdt REAL,
        exit_rule TEXT,
        session_date TEXT
    )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS orb_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ran_at_utc TEXT NOT NULL,
        symbols_scanned INTEGER DEFAULT 0,
        opens INTEGER DEFAULT 0,
        resolves INTEGER DEFAULT 0,
        detail_json TEXT
    )"""
    )
    for sql in (
        "CREATE INDEX IF NOT EXISTS ix_orb_recorded ON orb_signals(recorded_at_utc)",
        "CREATE INDEX IF NOT EXISTS ix_orb_session ON orb_signals(session_date)",
        "CREATE INDEX IF NOT EXISTS ix_orb_settle_time ON orb_settlements(settled_at_utc)",
    ):
        try:
            c.execute(sql)
        except sqlite3.OperationalError:
            pass
    c.execute(
        """CREATE TABLE IF NOT EXISTS orb_symbol_bots (
        symbol TEXT PRIMARY KEY,
        virtual_equity_usdt REAL NOT NULL,
        enabled INTEGER NOT NULL DEFAULT 1,
        created_at_utc TEXT NOT NULL,
        updated_at_utc TEXT NOT NULL
    )"""
    )
    try:
        from orb.v2.robots import migrate_orb_robots

        migrate_orb_robots(c)
    except ImportError:
        pass


def symbol_session_traded(cur: sqlite3.Cursor, symbol: str, session_date: str) -> bool:
    if not session_date:
        return False
    sym = str(symbol).strip().upper()
    day = str(session_date)
    cur.execute(
        """
        SELECT 1 FROM orb_settlements
        WHERE symbol = ? AND session_date = ?
        LIMIT 1
        """,
        (sym, day),
    )
    if cur.fetchone() is not None:
        return True
    cur.execute(
        """
        SELECT 1 FROM orb_signals
        WHERE symbol = ? AND session_date = ?
          AND side IN ('LONG','SHORT') AND entry_bar_open_ms IS NOT NULL
        LIMIT 1
        """,
        (sym, day),
    )
    return cur.fetchone() is not None


def ensure_symbol_bots(
    cur: sqlite3.Cursor,
    symbols: List[str],
    *,
    initial_equity_usdt: float,
) -> None:
    """为每个标的确保存在独立机器人记录（一标的一 bot）。"""
    now = _utc_now()
    init = max(0.0, float(initial_equity_usdt or 0.0))
    for sym in symbols:
        s = str(sym).strip().upper()
        if not s:
            continue
        cur.execute(
            """
            INSERT INTO orb_symbol_bots(symbol, virtual_equity_usdt, enabled, created_at_utc, updated_at_utc)
            VALUES (?, ?, 1, ?, ?)
            ON CONFLICT(symbol) DO NOTHING
            """,
            (s, init, now, now),
        )


def symbol_bot_enabled(cur: sqlite3.Cursor, symbol: str) -> bool:
    sym = str(symbol).strip().upper()
    cur.execute("SELECT enabled FROM orb_symbol_bots WHERE symbol = ?", (sym,))
    row = cur.fetchone()
    if row is None:
        return True
    return int(row[0] or 0) != 0


def symbol_bot_settled_pnl(cur: sqlite3.Cursor, symbol: str) -> float:
    sym = str(symbol).strip().upper()
    cur.execute(
        "SELECT COALESCE(SUM(pnl_usdt), 0) FROM orb_settlements WHERE symbol = ?",
        (sym,),
    )
    return float(cur.fetchone()[0] or 0)


def symbol_bot_wallet_balance(
    conn: sqlite3.Connection,
    symbol: str,
    *,
    initial_equity_usdt: float,
    sync: bool = True,
) -> float:
    """单标机器人钱包：配置初始本金 + 该标的已实现盈亏。"""
    sym = str(symbol).strip().upper()
    initial = max(0.0, float(initial_equity_usdt or 0.0))
    cur = conn.cursor()
    settled = symbol_bot_settled_pnl(cur, sym)
    balance = round(initial + settled, 4)
    if sync:
        cur.execute("SELECT 1 FROM orb_symbol_bots WHERE symbol = ?", (sym,))
        if cur.fetchone() is not None:
            cur.execute(
                "UPDATE orb_symbol_bots SET virtual_equity_usdt=?, updated_at_utc=? WHERE symbol=?",
                (balance, _utc_now(), sym),
            )
    return balance


def list_symbol_bot_summaries(
    conn: sqlite3.Connection,
    *,
    symbols: List[str],
    initial_equity_usdt: float,
) -> List[Dict[str, Any]]:
    """按标的汇总机器人状态（一标的一 bot）。"""
    prev_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        init = max(0.0, float(initial_equity_usdt or 0.0))
        sym_order = [str(s).strip().upper() for s in symbols if str(s).strip()]
        sym_set = set(sym_order)
        cur.execute(
            """
            SELECT symbol,
                   COUNT(*) AS settled_count,
                   SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) AS losses,
                   COALESCE(SUM(pnl_usdt), 0) AS total_pnl_usdt
            FROM orb_settlements
            GROUP BY symbol
            """
        )
        settle_by_sym = {str(r["symbol"]).upper(): dict(r) for r in cur.fetchall()}
        cur.execute(
            """
            SELECT symbol, side, virtual_notional_usdt
            FROM orb_signals
            WHERE outcome IS NULL AND side IN ('LONG','SHORT') AND sl_price IS NOT NULL
            """
        )
        open_by_sym: Dict[str, Dict[str, Any]] = {}
        for r in cur.fetchall():
            open_by_sym[str(r["symbol"]).upper()] = dict(r)
        cur.execute("SELECT symbol, enabled FROM orb_symbol_bots")
        bot_rows = {str(r["symbol"]).upper(): dict(r) for r in cur.fetchall()}
        all_syms = sym_order + sorted(
            (set(settle_by_sym) | set(open_by_sym) | set(bot_rows)) - sym_set
        )
        out: List[Dict[str, Any]] = []
        for sym in all_syms:
            bot = bot_rows.get(sym, {})
            st = settle_by_sym.get(sym, {})
            wins = int(st.get("wins") or 0)
            losses = int(st.get("losses") or 0)
            touch = wins + losses
            settled = int(st.get("settled_count") or 0)
            pnl = round(float(st.get("total_pnl_usdt") or 0), 4)
            wallet = round(
                symbol_bot_wallet_balance(conn, sym, initial_equity_usdt=init, sync=False),
                4,
            )
            op = open_by_sym.get(sym)
            out.append(
                {
                    "symbol": sym,
                    "enabled": bool(int(bot.get("enabled", 1) or 1)),
                    "initial_equity_usdt": round(init, 4),
                    "wallet_balance_usdt": wallet,
                    "realized_pnl_usdt": pnl,
                    "settled_count": settled,
                    "wins": wins,
                    "losses": losses,
                    "touch_win_rate": round(wins / touch, 4) if touch else None,
                    "open_side": str(op["side"]).upper() if op else None,
                    "open_notional_usdt": round(float(op["virtual_notional_usdt"] or 0), 4) if op else None,
                }
            )
        return out
    finally:
        conn.row_factory = prev_factory


def fetch_open_hold(cur: sqlite3.Cursor, symbol: str, *, default_notional: float) -> Optional[sqlite3.Row]:
    cur.execute(
        """
        SELECT id, symbol, side, play, entry_price, sl_price, tp_price,
               COALESCE(virtual_notional_usdt, ?) AS notion, session_date, robot_id
        FROM orb_signals
        WHERE symbol = ? AND outcome IS NULL
          AND sl_price IS NOT NULL AND side IN ('LONG','SHORT')
        """,
        (default_notional, str(symbol).strip().upper()),
    )
    return cur.fetchone()


def count_open_positions(cur: sqlite3.Cursor) -> int:
    cur.execute(
        """
        SELECT COUNT(*) FROM orb_signals
        WHERE outcome IS NULL AND side IN ('LONG','SHORT') AND sl_price IS NOT NULL
        """
    )
    return int(cur.fetchone()[0] or 0)


def fetch_open_for_resolve(cur: sqlite3.Cursor, *, default_notional: float) -> list[tuple[Any, ...]]:
    cur.execute(
        """
        SELECT id, symbol, side, play, entry_price, sl_price, tp_price,
               entry_bar_open_ms, COALESCE(virtual_notional_usdt, ?) AS notion, robot_id
        FROM orb_signals
        WHERE outcome IS NULL AND sl_price IS NOT NULL AND entry_bar_open_ms IS NOT NULL
          AND side IN ('LONG','SHORT')
        ORDER BY id ASC
        """,
        (default_notional,),
    )
    return list(cur.fetchall())


def archive_settlement(
    cur: sqlite3.Cursor,
    *,
    signal_id: int,
    symbol: str,
    side: str,
    play: Optional[str],
    outcome: str,
    entry_price: float,
    exit_price: float,
    pnl_r: float,
    pnl_usdt: float,
    notional: float,
    exit_rule: str,
    settled_at_utc: str,
    session_date: Optional[str] = None,
    robot_id: Optional[int] = None,
) -> None:
    cur.execute(
        """
        INSERT INTO orb_settlements (
            settled_at_utc, signal_id, symbol, side, play, outcome,
            entry_price, exit_price, pnl_r, pnl_usdt, virtual_notional_usdt,
            exit_rule, session_date, robot_id
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            settled_at_utc,
            signal_id,
            symbol,
            side,
            play,
            outcome,
            entry_price,
            exit_price,
            pnl_r,
            pnl_usdt,
            notional,
            exit_rule,
            session_date,
            int(robot_id) if robot_id is not None else None,
        ),
    )


def clear_orb_tables(conn: sqlite3.Connection) -> dict[str, int]:
    cur = conn.cursor()
    out: dict[str, int] = {}
    for table, key in (
        ("orb_settlements", "deleted_settlements"),
        ("orb_signals", "deleted_signals"),
        ("orb_v2_runs", "deleted_v2_runs"),
        ("orb_v2_breakout_seen", "deleted_v2_breakout_seen"),
        ("orb_v2_gate_day", "deleted_v2_gate_day"),
        ("orb_robots", "deleted_robots"),
        ("orb_robot_resets", "deleted_robot_resets"),
        ("orb_runs", "deleted_runs"),
        ("orb_symbol_bots", "deleted_symbol_bots"),
    ):
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
        if not cur.fetchone():
            out[key] = 0
            continue
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        n = int(cur.fetchone()[0] or 0)
        cur.execute(f"DELETE FROM {table}")
        out[key] = n
    conn.commit()
    return out
