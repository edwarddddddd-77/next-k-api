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


def delete_profile(conn: sqlite3.Connection, profile_id: int) -> bool:
    conn.execute("DELETE FROM moss2_signals WHERE profile_id=?", (profile_id,))
    conn.execute("DELETE FROM moss2_backtest_runs WHERE profile_id=?", (profile_id,))
    conn.execute(
        "DELETE FROM moss2_discipline_snapshots WHERE profile_id=?", (profile_id,)
    )
    cur = conn.execute("DELETE FROM moss2_profiles WHERE id=?", (profile_id,))
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
    out = []
    for r in rows:
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
                "recorded_at_utc": r["recorded_at_utc"],
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
               ORDER BY recorded_at_utc DESC LIMIT ?""",
            (profile_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT * FROM moss2_signals ORDER BY recorded_at_utc DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def summarize_lane(conn: sqlite3.Connection) -> Dict[str, Any]:
    conn.row_factory = sqlite3.Row
    profiles = list_profiles(conn)
    enabled = [p for p in profiles if p.get("enabled")]
    open_rows = list_open_signals(conn)
    settled = conn.execute(
        """SELECT COUNT(*) AS n, COALESCE(SUM(pnl_usdt),0) AS pnl
           FROM moss2_signals WHERE outcome IS NOT NULL"""
    ).fetchone()
    total_equity = sum(float(p.get("virtual_equity_usdt") or 0) for p in profiles)
    realized = round(float(settled["pnl"] or 0), 4) if settled else 0.0
    mode = "live" if moss2_config.real_mode_enabled() else "paper"
    legacy_hl = sum(
        1 for p in profiles if str(p.get("variant") or "").lower() != "en"
    )
    return {
        "ok": True,
        "mode": mode,
        "lane": "moss2",
        "ops_variant": moss2_config.MOSS2_OPS_VARIANT,
        "protocol_venue": moss2_config.MOSS2_PROTOCOL_VENUE,
        "legacy_non_ops_profiles": legacy_hl,
        "profile_count": len(profiles),
        "enabled_profiles": len(enabled),
        "open_positions": len(open_rows),
        "settled_count": int(settled["n"] or 0) if settled else 0,
        "total_pnl_usdt": realized,
        "total_equity_usdt": round(total_equity, 2),
        "profile_capital_usdt": float(moss2_config.MOSS2_PROFILE_CAPITAL),
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
