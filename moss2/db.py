"""Moss2 SQLite（表前缀 moss2_，与 moss_* 隔离）。"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from moss2 import config as moss2_config


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def migrate_moss2_tables(c: sqlite3.Cursor) -> None:
    c.execute(
        """CREATE TABLE IF NOT EXISTS moss2_profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        symbol TEXT NOT NULL,
        variant TEXT NOT NULL DEFAULT 'en',
        template TEXT,
        enabled INTEGER NOT NULL DEFAULT 0,
        initial_params_json TEXT NOT NULL,
        tactical_params_json TEXT NOT NULL,
        virtual_equity_usdt REAL NOT NULL DEFAULT 10000,
        created_at_utc TEXT NOT NULL,
        updated_at_utc TEXT NOT NULL
    )"""
    )
    c.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS ux_moss2_profiles_symbol_enabled
        ON moss2_profiles(symbol, variant) WHERE enabled = 1"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS moss2_backtest_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        variant TEXT NOT NULL,
        symbol TEXT NOT NULL,
        data_csv TEXT,
        initial_params_json TEXT,
        result_json TEXT NOT NULL,
        summary_json TEXT,
        created_at_utc TEXT NOT NULL,
        FOREIGN KEY (profile_id) REFERENCES moss2_profiles(id)
    )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS moss2_signals (
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
        outcome TEXT,
        outcome_at_utc TEXT,
        exit_price REAL,
        pnl_usdt REAL,
        exit_rule TEXT,
        meta_json TEXT,
        updated_at_utc TEXT,
        FOREIGN KEY (profile_id) REFERENCES moss2_profiles(id)
    )"""
    )
    c.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS ux_moss2_open_profile
        ON moss2_signals(profile_id) WHERE outcome IS NULL AND side IN ('LONG','SHORT')"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS moss2_paper_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ran_at_utc TEXT NOT NULL,
        profiles_scanned INTEGER,
        opens INTEGER,
        closes INTEGER,
        detail_json TEXT
    )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS moss2_settlements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        settled_at_utc TEXT NOT NULL,
        signal_id INTEGER NOT NULL,
        profile_id INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        outcome TEXT NOT NULL,
        entry_price REAL,
        exit_price REAL,
        pnl_usdt REAL NOT NULL,
        virtual_notional_usdt REAL,
        exit_rule TEXT,
        FOREIGN KEY (signal_id) REFERENCES moss2_signals(id)
    )"""
    )
    c.execute(
        """CREATE UNIQUE INDEX IF NOT EXISTS ux_moss2_settlements_signal
        ON moss2_settlements(signal_id)"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS moss2_wallet (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        initial_capital_usdt REAL NOT NULL,
        balance_usdt REAL NOT NULL,
        updated_at_utc TEXT NOT NULL
    )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS moss2_discipline_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        profile_id INTEGER,
        symbol TEXT NOT NULL,
        variant TEXT NOT NULL,
        template TEXT,
        params_version TEXT,
        data_csv TEXT,
        discipline_json TEXT NOT NULL,
        created_at_utc TEXT NOT NULL,
        FOREIGN KEY (profile_id) REFERENCES moss2_profiles(id)
    )"""
    )
    _migrate_moss2_profile_columns(c)


def _migrate_moss2_profile_columns(c: sqlite3.Cursor) -> None:
    cols = {row[1] for row in c.execute("PRAGMA table_info(moss2_profiles)").fetchall()}
    alters = [
        ("params_version", "TEXT DEFAULT 'v1'"),
        ("approved_params_version", "TEXT"),
        ("params_hash", "TEXT"),
        ("canary_scale", "REAL DEFAULT 1.0"),
        ("candidate_params_json", "TEXT"),
        ("evolution_status", "TEXT DEFAULT 'baseline'"),
        ("last_evolve_at_utc", "TEXT"),
    ]
    for name, typedef in alters:
        if name not in cols:
            c.execute(f"ALTER TABLE moss2_profiles ADD COLUMN {name} {typedef}")
    sig_cols = {row[1] for row in c.execute("PRAGMA table_info(moss2_signals)").fetchall()}
    if "unrealized_pnl_usdt" not in sig_cols:
        c.execute(
            "ALTER TABLE moss2_signals ADD COLUMN unrealized_pnl_usdt REAL DEFAULT 0"
        )


def row_to_profile(row: sqlite3.Row) -> Dict[str, Any]:
    keys = row.keys()
    return {
        "id": row["id"],
        "name": row["name"],
        "symbol": row["symbol"],
        "variant": row["variant"],
        "template": row["template"],
        "enabled": bool(row["enabled"]),
        "initial_params": json.loads(row["initial_params_json"]),
        "tactical_params": json.loads(row["tactical_params_json"] or "{}"),
        "virtual_equity_usdt": float(row["virtual_equity_usdt"]),
        "created_at_utc": row["created_at_utc"],
        "updated_at_utc": row["updated_at_utc"],
        "initial_params_json": row["initial_params_json"],
        "tactical_params_json": row["tactical_params_json"],
        "params_version": row["params_version"] if "params_version" in keys else "v1",
        "approved_params_version": row["approved_params_version"]
        if "approved_params_version" in keys
        else None,
        "params_hash": row["params_hash"] if "params_hash" in keys else None,
        "canary_scale": float(row["canary_scale"] or 1.0)
        if "canary_scale" in keys
        else 1.0,
        "candidate_params_json": row["candidate_params_json"]
        if "candidate_params_json" in keys
        else None,
        "evolution_status": row["evolution_status"]
        if "evolution_status" in keys
        else "baseline",
        "last_evolve_at_utc": row["last_evolve_at_utc"]
        if "last_evolve_at_utc" in keys
        else None,
    }


def list_profiles(conn: sqlite3.Connection, *, enabled_only: bool = False) -> List[dict]:
    conn.row_factory = sqlite3.Row
    q = "SELECT * FROM moss2_profiles"
    if enabled_only:
        q += " WHERE enabled = 1"
    q += " ORDER BY id"
    return [row_to_profile(r) for r in conn.execute(q).fetchall()]


def list_profiles_for_paper_scan(conn: sqlite3.Connection) -> List[dict]:
    """已启用 + 仍有持仓（便于平仓）。"""
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT DISTINCT p.* FROM moss2_profiles p
           WHERE p.enabled = 1
              OR EXISTS (
                  SELECT 1 FROM moss2_signals s
                  WHERE s.profile_id = p.id
                    AND s.outcome IS NULL
                    AND s.side IN ('LONG','SHORT')
              )
           ORDER BY p.id ASC"""
    ).fetchall()
    return [row_to_profile(r) for r in rows]


def get_profile(conn: sqlite3.Connection, profile_id: int) -> Optional[dict]:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM moss2_profiles WHERE id = ?", (profile_id,)
    ).fetchone()
    return row_to_profile(row) if row else None


def create_profile(
    conn: sqlite3.Connection,
    *,
    name: str,
    symbol: str,
    variant: str,
    template: str,
    enabled: bool,
    initial_params: dict,
    tactical_params: Optional[dict] = None,
    virtual_equity_usdt: float,
) -> int:
    now = _utc_now()
    cur = conn.execute(
        """INSERT INTO moss2_profiles(
               name, symbol, variant, template, enabled,
               initial_params_json, tactical_params_json,
               virtual_equity_usdt, params_version, approved_params_version,
               canary_scale, evolution_status,
               created_at_utc, updated_at_utc)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            name,
            symbol.upper(),
            variant,
            template,
            1 if enabled else 0,
            json.dumps(initial_params, ensure_ascii=False),
            json.dumps(tactical_params or {}, ensure_ascii=False),
            virtual_equity_usdt,
            "v1",
            "v1",
            1.0,
            "baseline",
            now,
            now,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def patch_profile(conn: sqlite3.Connection, profile_id: int, **fields) -> None:
    now = _utc_now()
    sets = ["updated_at_utc = ?"]
    args: list = [now]
    if "enabled" in fields:
        sets.append("enabled = ?")
        args.append(1 if fields["enabled"] else 0)
    if "tactical_params" in fields:
        import copy

        from moss2.params import (
            resolve_params_dict,
            tactical_field_names,
        )

        from moss2.config import profile_variant

        prof = get_profile(conn, profile_id) or {}
        variant = profile_variant(prof)
        merged = copy.deepcopy(json.loads(prof.get("initial_params_json") or "{}"))
        merged.update(json.loads(prof.get("tactical_params_json") or "{}"))
        merged.update(fields["tactical_params"] or {})
        resolved = resolve_params_dict(merged, variant=variant)  # type: ignore[arg-type]
        tact_names = tactical_field_names(variant)  # type: ignore[arg-type]
        tactical = {k: resolved[k] for k in tact_names if k in resolved}
        sets.append("tactical_params_json = ?")
        args.append(json.dumps(tactical, ensure_ascii=False))
    if "name" in fields:
        sets.append("name = ?")
        args.append(fields["name"])
    if "template" in fields:
        sets.append("template = ?")
        args.append(fields["template"])
    if "initial_params" in fields:
        sets.append("initial_params_json = ?")
        args.append(json.dumps(fields["initial_params"], ensure_ascii=False))
    for col in (
        "params_version",
        "approved_params_version",
        "params_hash",
        "canary_scale",
        "candidate_params_json",
        "evolution_status",
        "last_evolve_at_utc",
    ):
        if col in fields:
            sets.append(f"{col} = ?")
            val = fields[col]
            if col == "candidate_params_json" and isinstance(val, dict):
                val = json.dumps(val, ensure_ascii=False)
            args.append(val)
    args.append(profile_id)
    conn.execute(
        f"UPDATE moss2_profiles SET {', '.join(sets)} WHERE id = ?", args
    )
    conn.commit()


def insert_backtest_run(
    conn: sqlite3.Connection,
    *,
    profile_id: Optional[int],
    variant: str,
    symbol: str,
    data_csv: Optional[str],
    initial_params: dict,
    result: dict,
    summary: dict,
) -> int:
    cur = conn.execute(
        """INSERT INTO moss2_backtest_runs(
               profile_id, variant, symbol, data_csv,
               initial_params_json, result_json, summary_json, created_at_utc)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            profile_id,
            variant,
            symbol,
            data_csv,
            json.dumps(initial_params, ensure_ascii=False),
            json.dumps(result, ensure_ascii=False),
            json.dumps(summary, ensure_ascii=False),
            _utc_now(),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def profile_has_open_position(conn: sqlite3.Connection, profile_id: int) -> bool:
    row = conn.execute(
        """SELECT 1 FROM moss2_signals
           WHERE profile_id = ? AND outcome IS NULL AND side IN ('LONG','SHORT')
           LIMIT 1""",
        (int(profile_id),),
    ).fetchone()
    return row is not None


def delete_profile(conn: sqlite3.Connection, profile_id: int) -> bool:
    pid = int(profile_id)
    prof = get_profile(conn, pid)
    if not prof:
        return False
    if profile_has_open_position(conn, pid):
        raise ValueError("profile_has_open_position")
    conn.execute("DELETE FROM moss2_settlements WHERE profile_id=?", (pid,))
    conn.execute("DELETE FROM moss2_signals WHERE profile_id=?", (pid,))
    conn.execute("DELETE FROM moss2_backtest_runs WHERE profile_id=?", (pid,))
    conn.execute(
        "DELETE FROM moss2_discipline_snapshots WHERE profile_id=?", (pid,)
    )
    cur = conn.execute("DELETE FROM moss2_profiles WHERE id=?", (pid,))
    conn.commit()
    return cur.rowcount > 0


def list_open_signals(conn: sqlite3.Connection) -> List[dict]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT s.*, p.name AS profile_name, p.template, p.variant, p.enabled
           FROM moss2_signals s
           JOIN moss2_profiles p ON p.id = s.profile_id
           WHERE s.outcome IS NULL AND s.side IN ('LONG','SHORT')
           ORDER BY s.recorded_at_utc DESC"""
    ).fetchall()
    from moss2.exit_levels import parse_exit_levels_from_meta

    out = []
    for r in rows:
        levels = parse_exit_levels_from_meta(
            r["meta_json"] if "meta_json" in r.keys() else None
        )
        out.append(
            {
                "id": r["id"],
                "profile_id": r["profile_id"],
                "profile_name": r["profile_name"],
                "template": r["template"],
                "variant": r["variant"],
                "enabled": bool(r["enabled"]),
                "side": r["side"],
                "symbol": r["symbol"],
                "entry_price": r["entry_price"],
                "mark_price": r["mark_price"],
                "virtual_notional_usdt": r["virtual_notional_usdt"],
                "composite": r["composite"],
                "regime": r["regime"],
                "unrealized_pnl_usdt": r["unrealized_pnl_usdt"]
                if "unrealized_pnl_usdt" in r.keys()
                else None,
                "recorded_at_utc": r["recorded_at_utc"],
                "stop_loss": levels.get("stop_loss"),
                "take_profit": levels.get("take_profit"),
                "atr14": levels.get("atr14"),
            }
        )
    return out


def list_signals(
    conn: sqlite3.Connection, *, profile_id: Optional[int] = None, limit: int = 100
) -> List[dict]:
    conn.row_factory = sqlite3.Row
    if profile_id:
        rows = conn.execute(
            """SELECT * FROM moss2_signals WHERE profile_id=?
               ORDER BY (outcome IS NULL) DESC, recorded_at_utc DESC LIMIT ?""",
            (profile_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT * FROM moss2_signals
               ORDER BY (outcome IS NULL) DESC, recorded_at_utc DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def count_moss2_profiles(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM moss2_profiles").fetchone()
    return int(row[0] or 0)


def count_enabled_profiles(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) FROM moss2_profiles WHERE enabled = 1"
    ).fetchone()
    return int(row[0] or 0)


def count_open_positions(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        """SELECT COUNT(*) FROM moss2_signals
           WHERE outcome IS NULL AND side IN ('LONG','SHORT')"""
    ).fetchone()
    return int(row[0] or 0)


def aggregate_moss2_wallet_initial(conn: sqlite3.Connection) -> float:
    """全局纸面初始 = Σ(每 Profile virtual_equity_usdt)。"""
    row = conn.execute(
        "SELECT COALESCE(SUM(virtual_equity_usdt), 0) FROM moss2_profiles"
    ).fetchone()
    return round(float(row[0] or 0), 4)


def backfill_settlements_from_closed_signals(conn: sqlite3.Connection) -> Dict[str, Any]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT s.id, s.outcome_at_utc, s.updated_at_utc, s.recorded_at_utc,
                  s.profile_id, s.symbol, s.side, s.outcome, s.entry_price, s.exit_price,
                  s.pnl_usdt, s.exit_rule, s.virtual_notional_usdt
           FROM moss2_signals s
           WHERE s.outcome IS NOT NULL
             AND s.side IN ('LONG', 'SHORT')
             AND s.pnl_usdt IS NOT NULL
             AND NOT EXISTS (
                 SELECT 1 FROM moss2_settlements t WHERE t.signal_id = s.id
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
        outcome = str(row["outcome"] or "flat")
        if outcome == "closed":
            outcome = "win" if pnl > 0 else ("loss" if pnl < 0 else "flat")
        conn.execute(
            """INSERT INTO moss2_settlements(
                   settled_at_utc, signal_id, profile_id, symbol, side, outcome,
                   entry_price, exit_price, pnl_usdt, virtual_notional_usdt, exit_rule)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                settled_at,
                int(row["id"]),
                int(row["profile_id"]),
                str(row["symbol"] or "").upper(),
                str(row["side"] or ""),
                outcome,
                row["entry_price"],
                row["exit_price"],
                pnl,
                row["virtual_notional_usdt"],
                row["exit_rule"],
            ),
        )
        inserted += 1
        pnl_sum += pnl
    if inserted:
        conn.commit()
    return {"inserted": inserted, "pnl_usdt": round(pnl_sum, 4)}


def get_moss2_wallet(
    conn: sqlite3.Connection, *, reconcile: bool = True
) -> Dict[str, Any]:
    if reconcile:
        backfill_settlements_from_closed_signals(conn)
    initial = aggregate_moss2_wallet_initial(conn)
    settled = float(
        conn.execute(
            "SELECT COALESCE(SUM(pnl_usdt), 0) FROM moss2_settlements"
        ).fetchone()[0]
        or 0
    )
    balance = round(initial + settled, 4)
    now = _utc_now()
    conn.execute(
        """INSERT INTO moss2_wallet(id, initial_capital_usdt, balance_usdt, updated_at_utc)
           VALUES (1, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             initial_capital_usdt=excluded.initial_capital_usdt,
             balance_usdt=excluded.balance_usdt,
             updated_at_utc=excluded.updated_at_utc""",
        (initial, balance, now),
    )
    conn.commit()
    return {
        "initial_capital_usdt": initial,
        "balance_usdt": balance,
        "realized_pnl_usdt": round(settled, 4),
        "profile_count": count_moss2_profiles(conn),
        "updated_at_utc": now,
    }


def sync_moss2_wallet_from_settlements(conn: sqlite3.Connection) -> Dict[str, Any]:
    wallet = get_moss2_wallet(conn)
    now = _utc_now()
    conn.execute(
        "UPDATE moss2_wallet SET balance_usdt=?, updated_at_utc=? WHERE id=1",
        (wallet["balance_usdt"], now),
    )
    conn.commit()
    return wallet


def list_settlement_stats_by_profile(conn: sqlite3.Connection) -> List[dict]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT profile_id, symbol,
                  COUNT(*) AS settled_count,
                  COALESCE(SUM(pnl_usdt), 0) AS total_pnl_usdt
           FROM moss2_settlements
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


def list_settlement_stats_by_symbol(conn: sqlite3.Connection) -> List[dict]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT symbol,
                  COUNT(*) AS settled_count,
                  COALESCE(SUM(pnl_usdt), 0) AS total_pnl_usdt
           FROM moss2_settlements
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


def insert_moss2_settlement(
    conn: sqlite3.Connection,
    *,
    signal_id: int,
    profile_id: int,
    symbol: str,
    side: str,
    outcome: str,
    entry_price: float,
    exit_price: float,
    pnl_usdt: float,
    notional: float,
    exit_rule: str,
    settled_at: str,
) -> None:
    conn.execute(
        """INSERT INTO moss2_settlements(
               settled_at_utc, signal_id, profile_id, symbol, side, outcome,
               entry_price, exit_price, pnl_usdt, virtual_notional_usdt, exit_rule)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (
            settled_at,
            signal_id,
            profile_id,
            symbol.upper(),
            side.upper(),
            outcome,
            entry_price,
            exit_price,
            pnl_usdt,
            notional,
            exit_rule,
        ),
    )


def summarize_lane(
    conn: sqlite3.Connection, *, refresh_open_marks: bool = True
) -> Dict[str, Any]:
    from moss2.paper_wallet import paper_trading_leverage, refresh_live_open_signals

    conn.row_factory = sqlite3.Row
    profiles = list_profiles(conn)
    enabled_n = sum(1 for p in profiles if p.get("enabled"))
    legacy_hl = sum(
        1 for p in profiles if str(p.get("variant") or "").lower() != "en"
    )
    wallet = get_moss2_wallet(conn)
    per_profile = list_settlement_stats_by_profile(conn)
    open_map = (
        refresh_live_open_signals(conn)
        if refresh_open_marks
        else {}
    )
    if not refresh_open_marks:
        from moss2.paper_wallet import fetch_open_positions_map

        open_map = fetch_open_positions_map(conn)
    unrealized = round(
        sum(float(pos.get("upnl") or pos.get("unrealized_pnl_usdt") or 0) for pos in open_map.values()),
        4,
    )
    balance = float(wallet["balance_usdt"])
    initial = float(wallet["initial_capital_usdt"])
    open_by_profile: List[Dict[str, Any]] = []
    for pid, pos in open_map.items():
        open_by_profile.append(
            {
                "profile_id": int(pid),
                "symbol": str(pos.get("symbol") or "").upper(),
                "open_count": 1,
                "unrealized_pnl_usdt": round(
                    float(pos.get("upnl") or pos.get("unrealized_pnl_usdt") or 0),
                    4,
                ),
            }
        )
    open_by_profile.sort(key=lambda x: x["profile_id"])
    paper_mode = True
    real_on = moss2_config.real_mode_enabled()
    mode_lbl = "paper"
    return {
        "ok": True,
        "mode": mode_lbl,
        "lane": "moss2",
        "ops_variant": moss2_config.MOSS2_OPS_VARIANT,
        "protocol_venue": moss2_config.MOSS2_PROTOCOL_VENUE,
        "legacy_non_ops_profiles": legacy_hl,
        "profile_count": len(profiles),
        "enabled_profiles": enabled_n,
        "open_positions": len(open_map),
        "settled_count": int(
            sum(int(row.get("settled_count") or 0) for row in per_profile)
        ),
        "total_pnl_usdt": round(float(wallet.get("realized_pnl_usdt") or 0), 4),
        "unrealized_pnl_usdt": unrealized,
        "equity_usdt": round(balance + unrealized, 4),
        "wallet_initial_usdt": round(initial, 4),
        "wallet_balance_usdt": round(balance, 4),
        "available_balance_usdt": None,
        "profile_capital_usdt": float(moss2_config.MOSS2_PROFILE_CAPITAL),
        "leverage": paper_trading_leverage(),
        "real_mode": real_on,
        "paper_source_of_truth": moss2_config.MOSS2_PAPER_SOURCE_OF_TRUTH,
        "per_profile": per_profile,
        "per_symbol": list_settlement_stats_by_symbol(conn),
        "open_by_profile": open_by_profile,
        "total_equity_usdt": round(balance + unrealized, 2),
    }


def refresh_open_signal_marks(
    conn: sqlite3.Connection,
    *,
    profile_id: int,
    mark: float,
    composite: Optional[float] = None,
) -> None:
    """持仓行更新标记价（看板展示用）。"""
    if composite is not None:
        conn.execute(
            """UPDATE moss2_signals SET mark_price=?, composite=?, updated_at_utc=?
               WHERE profile_id=? AND outcome IS NULL AND side IN ('LONG','SHORT')""",
            (mark, composite, _utc_now(), profile_id),
        )
    else:
        conn.execute(
            """UPDATE moss2_signals SET mark_price=?, updated_at_utc=?
               WHERE profile_id=? AND outcome IS NULL AND side IN ('LONG','SHORT')""",
            (mark, _utc_now(), profile_id),
        )
    conn.commit()


def latest_paper_run(conn: sqlite3.Connection) -> Optional[dict]:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """SELECT id, ran_at_utc, profiles_scanned, opens, closes, detail_json
           FROM moss2_paper_runs ORDER BY id DESC LIMIT 1"""
    ).fetchone()
    if not row:
        return None
    details: list = []
    extra: Dict[str, Any] = {}
    if row["detail_json"]:
        try:
            raw = json.loads(row["detail_json"])
            if isinstance(raw, dict):
                details = raw.get("details") or []
                extra = {k: v for k, v in raw.items() if k != "details"}
            elif isinstance(raw, list):
                details = raw
        except json.JSONDecodeError:
            details = []
    out = {
        "run_id": int(row["id"]),
        "ran_at_utc": row["ran_at_utc"],
        "profiles_scanned": int(row["profiles_scanned"] or 0),
        "opens": int(row["opens"] or 0),
        "closes": int(row["closes"] or 0),
        "details": details,
    }
    out.update(extra)
    return out


def insert_paper_run(conn: sqlite3.Connection, stats: dict) -> None:
    conn.execute(
        """INSERT INTO moss2_paper_runs(
               ran_at_utc, profiles_scanned, opens, closes, detail_json)
           VALUES (?,?,?,?,?)""",
        (
            _utc_now(),
            int(stats.get("profiles_scanned") or 0),
            int(stats.get("opens") or 0),
            int(stats.get("closes") or 0),
            json.dumps(stats, ensure_ascii=False, default=str),
        ),
    )
    conn.commit()


def patch_profile_evolution(
    conn: sqlite3.Connection, profile_id: int, **fields
) -> None:
    patch_profile(conn, profile_id, **fields)


def insert_discipline_snapshot(
    conn: sqlite3.Connection,
    *,
    profile_id: Optional[int],
    symbol: str,
    variant: str,
    template: str,
    params_version: str,
    data_csv: Optional[str],
    discipline: dict,
) -> int:
    cur = conn.execute(
        """INSERT INTO moss2_discipline_snapshots(
               profile_id, symbol, variant, template, params_version,
               data_csv, discipline_json, created_at_utc)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            profile_id,
            symbol,
            variant,
            template,
            params_version,
            data_csv,
            json.dumps(discipline, ensure_ascii=False),
            _utc_now(),
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def list_discipline_snapshots(
    conn: sqlite3.Connection, profile_id: int, *, limit: int = 12
) -> List[dict]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT * FROM moss2_discipline_snapshots
           WHERE profile_id=? ORDER BY id DESC LIMIT ?""",
        (profile_id, limit),
    ).fetchall()
    out = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "created_at_utc": r["created_at_utc"],
                "params_version": r["params_version"],
                "discipline": json.loads(r["discipline_json"] or "{}"),
            }
        )
    return out
