#!/usr/bin/env python3
"""
触轨池 SQLite：`zct_vwap_touch_pool`（当前入选）与 `zct_vwap_touch_pool_runs`（审计）。

每轮写入前在单事务内 **先清空入选表** 再 INSERT；无入选则表为空。runs 仅追加一条。

入选表写完后：默认从 `zct_vwap_signals` 删除 **已不在本轮入选** 且 **无未结方向持仓**
（`outcome IS NULL` 且 `side IN ('LONG','SHORT')` 且 `sl_price IS NOT NULL` 视为仍持仓，保留行）。

**触轨主 lane**（`ZCT_TOUCH_POOL_UNIVERSE=1`）下，每次 `run_scan` 入库/结算结束后亦会按 **当前入选表** 执行同一清理，避免看板残留已出池标的。
环境变量 **`ZCT_TOUCH_POOL_PRUNE_SIGNALS=0|false|off`** 可关闭该清理。
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional

_DEFAULT_POOL = "zct_vwap_touch_pool"
_DEFAULT_RUNS = "zct_vwap_touch_pool_runs"


def _pool_table() -> str:
    t = os.getenv("ZCT_TOUCH_POOL_TABLE", _DEFAULT_POOL).strip()
    return t if all(c.isalnum() or c == "_" for c in t) else _DEFAULT_POOL


def _runs_table() -> str:
    t = os.getenv("ZCT_TOUCH_POOL_RUNS_TABLE", _DEFAULT_RUNS).strip()
    return t if all(c.isalnum() or c == "_" for c in t) else _DEFAULT_RUNS


def touch_pool_physical_table_names() -> tuple[str, str]:
    """当前配置的入选表 / 审计表名（供日志等）。"""
    return _pool_table(), _runs_table()


def _signals_table_ident() -> str:
    """与 `zct_vwap_signal_scanner.ZCT_DB_SIGNALS_TABLE` 同规则，避免循环 import。"""
    raw = (os.getenv("ZCT_DB_SIGNALS_TABLE") or "zct_vwap_signals").strip()
    if raw and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", raw):
        return raw
    return "zct_vwap_signals"


def _touch_pool_prune_signals_enabled() -> bool:
    v = (os.getenv("ZCT_TOUCH_POOL_PRUNE_SIGNALS") or "1").strip().lower()
    return v not in ("0", "false", "no", "off", "disabled")


def touch_pool_list_symbols(conn: Optional[sqlite3.Connection] = None) -> List[str]:
    """当前触轨入选表中的 symbol 列表（升序）。表不存在或无行时返回 []。"""
    pt, _ = touch_pool_physical_table_names()
    own = conn is None
    if own:
        from accumulation_radar import init_db

        conn = init_db()
    assert conn is not None
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (pt,),
        )
        if not cur.fetchone():
            return []
        cur.execute(f"SELECT symbol FROM {pt} ORDER BY symbol ASC")
        return [str(r[0]).strip().upper() for r in cur.fetchall() if r and r[0]]
    finally:
        if own:
            conn.close()


def touch_pool_ensure_schema(conn: sqlite3.Connection) -> None:
    pt, rt = _pool_table(), _runs_table()
    c = conn.cursor()
    c.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {pt} (
            symbol TEXT PRIMARY KEY,
            updated_at_ms INTEGER NOT NULL,
            days REAL NOT NULL,
            signal_interval TEXT NOT NULL,
            win INTEGER NOT NULL,
            loss INTEGER NOT NULL,
            win_plus_loss INTEGER NOT NULL,
            win_rate_touch_sl_tp REAL,
            expired INTEGER NOT NULL,
            unresolved INTEGER NOT NULL,
            user_start_open_ms INTEGER,
            hist_end_open_ms INTEGER,
            trades_emitted INTEGER,
            criteria_json TEXT NOT NULL
        )
        """
    )
    c.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {rt} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at_ms INTEGER NOT NULL,
            matched_count INTEGER NOT NULL,
            scanned_count INTEGER NOT NULL,
            criteria_json TEXT NOT NULL,
            pool_json TEXT NOT NULL
        )
        """
    )
    conn.commit()


def touch_pool_prune_signals_vs_allowlist(
    conn: sqlite3.Connection, allowed_symbols: List[str]
) -> int:
    """
    删除 `zct_vwap_signals` 中 symbol 不在 allowed 列表且 **无未结方向持仓** 的行。
    allowed 为空表示入选表当前无标的：删尽所有「非持仓中」行。
    不 commit；由调用方在同一事务或随后 commit。
    """
    if not _touch_pool_prune_signals_enabled():
        return 0
    sig_tbl = _signals_table_ident()
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (sig_tbl,),
    )
    if not cur.fetchone():
        return 0
    hold = (
        "outcome IS NULL AND sl_price IS NOT NULL AND side IN ('LONG', 'SHORT')"
    )
    au = [str(s).strip().upper() for s in allowed_symbols if str(s).strip()]
    if not au:
        cur.execute(f"DELETE FROM {sig_tbl} WHERE NOT ({hold})")
    else:
        ph = ",".join("?" * len(au))
        cur.execute(
            f"DELETE FROM {sig_tbl} WHERE symbol NOT IN ({ph}) AND NOT ({hold})",
            au,
        )
    return int(cur.rowcount or 0)


def touch_pool_prune_signals_for_current_pool(conn: sqlite3.Connection) -> int:
    """按当前 `zct_vwap_touch_pool` 入选清理 signals；不 commit。"""
    return touch_pool_prune_signals_vs_allowlist(conn, touch_pool_list_symbols(conn))


def _touch_pool_remove_symbols_no_commit(
    conn: sqlite3.Connection, symbols: List[str]
) -> int:
    syms = sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
    if not syms:
        return 0
    pt = _pool_table()
    cur = conn.cursor()
    ph = ",".join("?" * len(syms))
    cur.execute(f"DELETE FROM {pt} WHERE symbol IN ({ph})", syms)
    return int(cur.rowcount or 0)


def touch_pool_remove_symbols(conn: sqlite3.Connection, symbols: List[str]) -> int:
    """从触轨池删除指定标的；不删 signals。单事务 commit。"""
    syms = sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
    if not syms:
        return 0
    cur = conn.cursor()
    cur.execute("BEGIN IMMEDIATE")
    try:
        n = _touch_pool_remove_symbols_no_commit(conn, syms)
        conn.commit()
        return n
    except Exception:
        conn.rollback()
        raise


def touch_pool_refresh_kept_symbols(
    conn: sqlite3.Connection,
    kept_rows: List[Dict[str, Any]],
    *,
    criteria: Dict[str, Any],
    backtest_meta: Dict[str, Any],
    run_ms: int,
    days: float,
    signal_interval: str,
) -> int:
    """滚动清洗后更新仍留在池内的标的统计（不 commit）。"""
    if not kept_rows:
        return 0
    pt = _pool_table()
    crit_json = json.dumps(criteria, ensure_ascii=False)
    cur = conn.cursor()
    n = 0
    for m in kept_rows:
        sym = str(m.get("symbol", "")).strip().upper()
        if not sym:
            continue
        w = int(m.get("win", 0) or 0)
        l_ = int(m.get("loss", 0) or 0)
        cur.execute(
            f"""
            UPDATE {pt}
            SET updated_at_ms = ?, days = ?, signal_interval = ?,
                win = ?, loss = ?, win_plus_loss = ?,
                win_rate_touch_sl_tp = ?,
                expired = ?, unresolved = ?,
                user_start_open_ms = ?, hist_end_open_ms = ?, trades_emitted = ?,
                criteria_json = ?
            WHERE symbol = ?
            """,
            (
                run_ms,
                float(days),
                str(signal_interval),
                w,
                l_,
                w + l_,
                m.get("win_rate_touch_sl_tp"),
                int(m.get("expired", 0) or 0),
                int(m.get("unresolved", 0) or 0),
                backtest_meta.get("user_start_open_ms"),
                backtest_meta.get("hist_end_open_ms"),
                backtest_meta.get("trades_emitted"),
                crit_json,
                sym,
            ),
        )
        if int(cur.rowcount or 0) > 0:
            n += 1
    return n


def touch_pool_append_rolling_audit(
    conn: sqlite3.Connection,
    audit: Dict[str, Any],
) -> None:
    """追加滚动清洗审计到 runs 表（不 commit）。"""
    _, rt = touch_pool_physical_table_names()
    run_ms = int(audit.get("generated_at_ms") or int(time.time() * 1000))
    crit = dict(audit.get("criteria") or {})
    crit.setdefault("scan_phase", "rolling_clean")
    removed = list(audit.get("removed") or [])
    kept = list(audit.get("kept") or [])
    cur = conn.cursor()
    cur.execute(
        f"""
        INSERT INTO {rt} (run_at_ms, matched_count, scanned_count, criteria_json, pool_json)
        VALUES (?,?,?,?,?)
        """,
        (
            run_ms,
            len(kept),
            int(audit.get("checked") or 0),
            json.dumps(crit, ensure_ascii=False),
            json.dumps(audit, ensure_ascii=False),
        ),
    )


def touch_pool_apply_rolling_clean(
    conn: sqlite3.Connection,
    *,
    to_remove: List[str],
    kept_rows: List[Dict[str, Any]],
    remaining_symbols: List[str],
    audit: Dict[str, Any],
    days: float,
    signal_interval: str,
) -> Dict[str, int]:
    """
    单事务：DELETE 出池标的 → 刷新保留行 → prune signals → 写 runs 审计。
    """
    run_ms = int(audit.get("generated_at_ms") or int(time.time() * 1000))
    meta = dict(audit.get("backtest_meta") or {})
    crit = dict(audit.get("criteria") or {})
    cur = conn.cursor()
    cur.execute("BEGIN IMMEDIATE")
    try:
        pool_deleted = _touch_pool_remove_symbols_no_commit(conn, to_remove)
        kept_updated = touch_pool_refresh_kept_symbols(
            conn,
            kept_rows,
            criteria=crit,
            backtest_meta=meta,
            run_ms=run_ms,
            days=float(days),
            signal_interval=str(signal_interval),
        )
        signals_pruned = touch_pool_prune_signals_vs_allowlist(
            conn, remaining_symbols
        )
        touch_pool_append_rolling_audit(conn, audit)
        conn.commit()
        return {
            "pool_deleted": pool_deleted,
            "kept_updated": kept_updated,
            "signals_pruned": signals_pruned,
        }
    except Exception:
        conn.rollback()
        raise


def touch_pool_write_db(conn: sqlite3.Connection, out: Dict[str, Any]) -> int:
    """
    先清空 `zct_vwap_touch_pool` 全表，再写入本轮 matched；最后追加一条 runs 审计。
    默认再清理 `zct_vwap_signals`：不在本轮入选且无未结方向单的标的行删除（见模块说明）。
    单事务：失败则回滚，入选表不处于半写入状态。
    """
    pt, rt = _pool_table(), _runs_table()
    crit = json.dumps(out.get("criteria") or {}, ensure_ascii=False)
    run_ms = int(out.get("generated_at_ms") or int(time.time() * 1000))
    matched: List[Dict[str, Any]] = list(out.get("matched") or [])
    meta = out.get("backtest_meta") or {}
    scanned = len(out.get("symbols_scanned") or [])
    days = float((out.get("criteria") or {}).get("days") or 0)
    sig_iv = str((out.get("criteria") or {}).get("signal_interval") or "1m")

    rows: List[tuple] = []
    for m in matched:
        sym = str(m.get("symbol", "")).strip().upper()
        if not sym:
            continue
        rows.append(
            (
                sym,
                run_ms,
                days,
                sig_iv,
                int(m.get("win", 0) or 0),
                int(m.get("loss", 0) or 0),
                int(m.get("win_plus_loss", 0) or 0),
                m.get("win_rate_touch_sl_tp"),
                int(m.get("expired", 0) or 0),
                int(m.get("unresolved", 0) or 0),
                meta.get("user_start_open_ms"),
                meta.get("hist_end_open_ms"),
                meta.get("trades_emitted"),
                crit,
            )
        )

    cur = conn.cursor()
    cur.execute("BEGIN IMMEDIATE")
    try:
        cur.execute(f"DELETE FROM {pt}")
        if rows:
            cur.executemany(
                f"""
                INSERT INTO {pt} (
                    symbol, updated_at_ms, days, signal_interval,
                    win, loss, win_plus_loss, win_rate_touch_sl_tp,
                    expired, unresolved,
                    user_start_open_ms, hist_end_open_ms, trades_emitted,
                    criteria_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
        cur.execute(
            f"""
            INSERT INTO {rt} (run_at_ms, matched_count, scanned_count, criteria_json, pool_json)
            VALUES (?,?,?,?,?)
            """,
            (
                run_ms,
                len(rows),
                scanned,
                crit,
                json.dumps(out, ensure_ascii=False),
            ),
        )
        if _touch_pool_prune_signals_enabled():
            touch_pool_prune_signals_vs_allowlist(conn, [t[0] for t in rows])
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return len(rows)
