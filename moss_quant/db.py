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
    c.execute(
        """CREATE TABLE IF NOT EXISTS moss_daily_core_symbols (
        symbol TEXT PRIMARY KEY,
        base TEXT NOT NULL,
        sort_order INTEGER NOT NULL DEFAULT 0,
        enabled INTEGER NOT NULL DEFAULT 1,
        note TEXT,
        updated_at_utc TEXT NOT NULL
    )"""
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS ix_moss_daily_core_enabled ON moss_daily_core_symbols(enabled, sort_order)"
    )
    seed_moss_daily_core_symbols(c)
    _ensure_profile_source_column(c)
    _ensure_moss_wallet_table(c)


def _ensure_moss_wallet_table(c: sqlite3.Cursor) -> None:
    from moss_quant import config as cfg

    c.execute(
        """CREATE TABLE IF NOT EXISTS moss_wallet (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        initial_capital_usdt REAL NOT NULL,
        balance_usdt REAL NOT NULL,
        updated_at_utc TEXT NOT NULL
    )"""
    )
    row = c.execute("SELECT id FROM moss_wallet WHERE id = 1").fetchone()
    initial = float(cfg.MOSS_QUANT_DEFAULT_CAPITAL)
    now = _utc_now()
    if not row:
        try:
            settled = float(
                c.execute(
                    "SELECT COALESCE(SUM(pnl_usdt), 0) FROM moss_settlements"
                ).fetchone()[0]
                or 0
            )
        except sqlite3.OperationalError:
            settled = 0.0
        balance = initial + settled
        c.execute(
            """INSERT INTO moss_wallet(id, initial_capital_usdt, balance_usdt, updated_at_utc)
               VALUES (1, ?, ?, ?)""",
            (initial, balance, now),
        )


def backfill_settlements_from_closed_signals(conn: sqlite3.Connection) -> Dict[str, Any]:
    """已平仓信号若缺结算行则补写（修复历史删 Profile 时误删 settlements）。"""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT s.id, s.outcome_at_utc, s.updated_at_utc, s.recorded_at_utc,
                  s.profile_id, s.symbol, s.side, s.outcome, s.entry_price, s.exit_price,
                  s.pnl_usdt, s.exit_rule, s.virtual_notional_usdt
           FROM moss_signals s
           WHERE s.outcome IS NOT NULL
             AND s.side IN ('LONG', 'SHORT')
             AND s.pnl_usdt IS NOT NULL
             AND NOT EXISTS (
                 SELECT 1 FROM moss_settlements t WHERE t.signal_id = s.id
             )"""
    ).fetchall()
    inserted = 0
    pnl_sum = 0.0
    for row in rows:
        settled_at = (
            str(row["outcome_at_utc"] or row["updated_at_utc"] or row["recorded_at_utc"])
            or _utc_now()
        )
        pnl = float(row["pnl_usdt"] or 0)
        conn.execute(
            """INSERT INTO moss_settlements(
                   settled_at_utc, signal_id, profile_id, symbol, side, outcome,
                   entry_price, exit_price, pnl_usdt, virtual_notional_usdt, exit_rule)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                settled_at,
                int(row["id"]),
                int(row["profile_id"]),
                str(row["symbol"] or "").upper(),
                str(row["side"] or ""),
                str(row["outcome"] or "flat"),
                row["entry_price"],
                row["exit_price"],
                pnl,
                row["virtual_notional_usdt"],
                row["exit_rule"],
            ),
        )
        inserted += 1
        pnl_sum += pnl
    return {"inserted": inserted, "pnl_usdt": round(pnl_sum, 4)}


def backfill_settlements_from_paper_runs(conn: sqlite3.Connection) -> Dict[str, Any]:
    """从纸面扫描日志中的 CLOSE 记录回补缺失结算（信号已被删时）。"""
    import json

    conn.row_factory = sqlite3.Row
    run_rows = conn.execute(
        "SELECT id, ran_at_utc, detail_json FROM moss_paper_runs ORDER BY id ASC"
    ).fetchall()
    inserted = 0
    pnl_sum = 0.0
    for run in run_rows:
        ran_at = str(run["ran_at_utc"] or _utc_now())
        try:
            details = json.loads(run["detail_json"] or "[]")
        except json.JSONDecodeError:
            continue
        if not isinstance(details, list):
            continue
        for d in details:
            if not isinstance(d, dict) or str(d.get("action") or "") != "close":
                continue
            pid = d.get("profile_id")
            pnl = d.get("pnl")
            sym = str(d.get("symbol") or "").upper()
            if pid is None or pnl is None or not sym:
                continue
            pnl_f = float(pnl)
            side = str(d.get("side") or "LONG").upper()
            exists = conn.execute(
                """SELECT 1 FROM moss_settlements
                   WHERE profile_id=? AND symbol=? AND ABS(pnl_usdt - ?) < 0.02
                   LIMIT 1""",
                (int(pid), sym, pnl_f),
            ).fetchone()
            if exists:
                continue
            conn.execute(
                """INSERT INTO moss_signals(
                       profile_id, recorded_at_utc, side, symbol,
                       entry_price, virtual_notional_usdt, mark_price,
                       outcome, outcome_at_utc, exit_price, pnl_usdt, exit_rule,
                       updated_at_utc)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    int(pid),
                    ran_at,
                    side,
                    sym,
                    None,
                    None,
                    None,
                    "win" if pnl_f > 0 else ("loss" if pnl_f < 0 else "flat"),
                    ran_at,
                    None,
                    pnl_f,
                    str(d.get("rule") or "paper_run_backfill"),
                    ran_at,
                ),
            )
            sig_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
            conn.execute(
                """INSERT INTO moss_settlements(
                       settled_at_utc, signal_id, profile_id, symbol, side, outcome,
                       entry_price, exit_price, pnl_usdt, virtual_notional_usdt, exit_rule)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    ran_at,
                    sig_id,
                    int(pid),
                    sym,
                    side,
                    "win" if pnl_f > 0 else ("loss" if pnl_f < 0 else "flat"),
                    None,
                    None,
                    pnl_f,
                    None,
                    str(d.get("rule") or "paper_run_backfill"),
                ),
            )
            inserted += 1
            pnl_sum += pnl_f
    return {"inserted": inserted, "pnl_usdt": round(pnl_sum, 4)}


def reconcile_moss_wallet(conn: sqlite3.Connection) -> Dict[str, Any]:
    """回补缺失结算并重算全局钱包。"""
    a = backfill_settlements_from_closed_signals(conn)
    b = backfill_settlements_from_paper_runs(conn)
    wallet = get_moss_wallet(conn, reconcile=False)
    return {
        "backfill_from_signals": a,
        "backfill_from_paper_runs": b,
        "wallet": wallet,
    }


def get_moss_wallet(
    conn: sqlite3.Connection, *, reconcile: bool = True
) -> Dict[str, Any]:
    """全局纸面钱包（与 Profile 解耦）；余额 = 初始 + 全部已结算盈亏。"""
    from moss_quant import config as cfg

    if reconcile:
        backfill_settlements_from_closed_signals(conn)
        backfill_settlements_from_paper_runs(conn)
    _ensure_moss_wallet_table(conn.cursor())
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT initial_capital_usdt, balance_usdt, updated_at_utc FROM moss_wallet WHERE id = 1"
    ).fetchone()
    initial = float(cfg.MOSS_QUANT_DEFAULT_CAPITAL)
    if not row:
        initial = float(cfg.MOSS_QUANT_DEFAULT_CAPITAL)
        return {
            "initial_capital_usdt": initial,
            "balance_usdt": initial,
            "realized_pnl_usdt": 0.0,
            "updated_at_utc": _utc_now(),
        }
    initial = float(row["initial_capital_usdt"] or cfg.MOSS_QUANT_DEFAULT_CAPITAL)
    settled = float(
        conn.execute("SELECT COALESCE(SUM(pnl_usdt), 0) FROM moss_settlements").fetchone()[
            0
        ]
        or 0
    )
    balance = initial + settled
    stored = float(row["balance_usdt"] or balance)
    if abs(stored - balance) > 0.01:
        conn.execute(
            "UPDATE moss_wallet SET balance_usdt=?, updated_at_utc=? WHERE id=1",
            (balance, _utc_now()),
        )
    return {
        "initial_capital_usdt": round(initial, 4),
        "balance_usdt": round(balance, 4),
        "realized_pnl_usdt": round(settled, 4),
        "updated_at_utc": str(row["updated_at_utc"] or _utc_now()),
    }


def wallet_equity_for_sizing(conn: sqlite3.Connection) -> float:
    """开仓名义仓位按全局钱包余额计算。"""
    return float(get_moss_wallet(conn)["balance_usdt"])


def sync_moss_wallet_from_settlements(conn: sqlite3.Connection) -> Dict[str, Any]:
    """按全部结算记录重算钱包余额（平仓后调用；删除 Profile 不影响已实现）。"""
    _ensure_moss_wallet_table(conn.cursor())
    wallet = get_moss_wallet(conn)
    now = _utc_now()
    conn.execute(
        "UPDATE moss_wallet SET balance_usdt=?, updated_at_utc=? WHERE id=1",
        (wallet["balance_usdt"], now),
    )
    return wallet


def list_settlement_stats_by_profile(
    conn: sqlite3.Connection,
) -> List[Dict[str, Any]]:
    """按 Profile 汇总已结算盈亏（删 Profile 后历史结算仍保留 profile_id）。"""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT profile_id, symbol,
                  COUNT(*) AS settled_count,
                  COALESCE(SUM(pnl_usdt), 0) AS total_pnl_usdt
           FROM moss_settlements
           GROUP BY profile_id, symbol
           ORDER BY profile_id ASC"""
    ).fetchall()
    return [
        {
            "profile_id": int(r["profile_id"]),
            "symbol": str(r["symbol"] or "").upper(),
            "settled_count": int(r["settled_count"] or 0),
            "total_pnl_usdt": round(float(r["total_pnl_usdt"] or 0), 4),
        }
        for r in rows
    ]


def list_open_unrealized_by_profile(
    conn: sqlite3.Connection,
) -> List[Dict[str, Any]]:
    """各 Profile 当前持仓浮盈（未平仓）。"""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT profile_id, symbol,
                  COUNT(*) AS open_count,
                  COALESCE(SUM(unrealized_pnl_usdt), 0) AS unrealized_pnl_usdt
           FROM moss_signals
           WHERE outcome IS NULL AND side IN ('LONG', 'SHORT')
           GROUP BY profile_id, symbol
           ORDER BY profile_id ASC"""
    ).fetchall()
    return [
        {
            "profile_id": int(r["profile_id"]),
            "symbol": str(r["symbol"] or "").upper(),
            "open_count": int(r["open_count"] or 0),
            "unrealized_pnl_usdt": round(float(r["unrealized_pnl_usdt"] or 0), 4),
        }
        for r in rows
    ]


def list_settlement_stats_by_symbol(
    conn: sqlite3.Connection,
) -> List[Dict[str, Any]]:
    """按标的汇总已结算盈亏。"""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT symbol,
                  COUNT(*) AS settled_count,
                  COALESCE(SUM(pnl_usdt), 0) AS total_pnl_usdt
           FROM moss_settlements
           GROUP BY symbol
           ORDER BY symbol ASC"""
    ).fetchall()
    return [
        {
            "symbol": str(r["symbol"] or "").upper(),
            "settled_count": int(r["settled_count"] or 0),
            "total_pnl_usdt": round(float(r["total_pnl_usdt"] or 0), 4),
        }
        for r in rows
    ]


def reset_moss_wallet(conn: sqlite3.Connection) -> None:
    from moss_quant import config as cfg

    _ensure_moss_wallet_table(conn.cursor())
    initial = float(cfg.MOSS_QUANT_DEFAULT_CAPITAL)
    now = _utc_now()
    conn.execute(
        """INSERT INTO moss_wallet(id, initial_capital_usdt, balance_usdt, updated_at_utc)
           VALUES (1, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             initial_capital_usdt=excluded.initial_capital_usdt,
             balance_usdt=excluded.balance_usdt,
             updated_at_utc=excluded.updated_at_utc""",
        (initial, initial, now),
    )


def _ensure_profile_source_column(c: sqlite3.Cursor) -> None:
    cols = {row[1] for row in c.execute("PRAGMA table_info(moss_profiles)").fetchall()}
    if "profile_source" not in cols:
        c.execute(
            "ALTER TABLE moss_profiles ADD COLUMN profile_source TEXT NOT NULL DEFAULT 'manual'"
        )


def seed_moss_daily_core_symbols(c: sqlite3.Cursor) -> None:
    """写入每日核心币（INSERT OR IGNORE，不覆盖已有行；缺行自动补 ICP/TON 等）。"""
    from moss_quant.universe import MOSS_DAILY_CORE_BASES, base_to_binance_symbol

    now = _utc_now()
    for i, base in enumerate(MOSS_DAILY_CORE_BASES):
        sym = base_to_binance_symbol(base)
        if not sym:
            continue
        c.execute(
            """INSERT OR IGNORE INTO moss_daily_core_symbols(
                   symbol, base, sort_order, enabled, note, updated_at_utc)
               VALUES (?,?,?,?,?,?)""",
            (
                sym,
                base,
                i + 1,
                1,
                "daily_core",
                now,
            ),
        )


def list_daily_core_bases(conn: sqlite3.Connection) -> List[str]:
    """enabled=1 的每日核心 base 列表（按 sort_order）。"""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT base FROM moss_daily_core_symbols
           WHERE enabled = 1
           ORDER BY sort_order ASC, symbol ASC"""
    ).fetchall()
    bases = [str(r["base"]).upper() for r in rows if r["base"]]
    if bases:
        return bases
    from moss_quant.universe import MOSS_DAILY_CORE_BASES

    return list(MOSS_DAILY_CORE_BASES)


def daily_core_symbol_set(conn: sqlite3.Connection) -> set[str]:
    return {
        str(r["symbol"]).upper()
        for r in list_daily_core_symbols(conn)
        if r.get("symbol")
    }


def add_symbol_to_daily_core(
    conn: sqlite3.Connection,
    symbol: str,
    *,
    note: str = "from_mcap_scan",
) -> Dict[str, Any]:
    """将标的加入 moss_daily_core_symbols（每日寻优必扫表）。"""
    from moss_quant.universe import base_to_binance_symbol, symbol_to_base

    from watchlist_symbols import filter_symbols_to_binance_usdt_perps

    sym = base_to_binance_symbol(str(symbol or "").strip().upper())
    if not sym:
        raise ValueError("invalid_symbol")
    if not filter_symbols_to_binance_usdt_perps([sym]):
        raise ValueError("symbol_not_on_binance_perp")
    base = symbol_to_base(sym)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT symbol, enabled FROM moss_daily_core_symbols WHERE symbol=?",
        (sym,),
    ).fetchone()
    now = _utc_now()
    if row:
        if int(row["enabled"] or 0):
            return {
                "ok": True,
                "symbol": sym,
                "base": base,
                "added": False,
                "already_in_daily_core": True,
            }
        conn.execute(
            """UPDATE moss_daily_core_symbols
               SET enabled=1, note=?, updated_at_utc=?
               WHERE symbol=?""",
            (note, now, sym),
        )
        conn.commit()
        return {
            "ok": True,
            "symbol": sym,
            "base": base,
            "added": True,
            "re_enabled": True,
        }
    max_order = int(
        conn.execute(
            "SELECT COALESCE(MAX(sort_order), 0) FROM moss_daily_core_symbols"
        ).fetchone()[0]
        or 0
    )
    conn.execute(
        """INSERT INTO moss_daily_core_symbols(
               symbol, base, sort_order, enabled, note, updated_at_utc)
           VALUES (?,?,?,?,?,?)""",
        (sym, base, max_order + 1, 1, note, now),
    )
    conn.commit()
    return {
        "ok": True,
        "symbol": sym,
        "base": base,
        "added": True,
        "sort_order": max_order + 1,
    }


def list_daily_core_symbols(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT symbol, base, sort_order, enabled, note, updated_at_utc
           FROM moss_daily_core_symbols
           WHERE enabled = 1
           ORDER BY sort_order ASC, symbol ASC"""
    ).fetchall()
    return [dict(r) for r in rows]


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


def mark_profile_open_signals_external_closed(
    conn: sqlite3.Connection,
    profile_id: int,
    *,
    exit_rule: str = "external_closed",
) -> int:
    now = _utc_now()
    cur = conn.execute(
        """UPDATE moss_signals
           SET outcome='external_closed',
               outcome_at_utc=?,
               exit_rule=?,
               updated_at_utc=?,
               unrealized_pnl_usdt=0
           WHERE profile_id=? AND outcome IS NULL AND side IN ('LONG','SHORT')""",
        (now, exit_rule, now, int(profile_id)),
    )
    return int(cur.rowcount or 0)


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
    # 保留 moss_settlements / moss_signals：全局已实现盈亏与历史成交不随 Profile 删除而减少
    conn.execute("DELETE FROM moss_backtest_runs WHERE profile_id = ?", (pid,))
    conn.execute("DELETE FROM moss_profiles WHERE id = ?", (pid,))
    deleted["signals_preserved"] = deleted.pop("signals")
    deleted["settlements_preserved"] = deleted.pop("settlements")
    deleted["profile_id"] = pid
    return deleted
