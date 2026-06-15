"""ORB V2 共享 8-robot 资金池（Live 与回测共用）。"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from orb.core.config import OrbConfig


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def robot_count_from_env() -> int:
    raw = os.getenv("ORB_V2_ROBOT_COUNT", "8")
    try:
        return max(1, int(float(str(raw).strip())))
    except ValueError:
        return 8


def robot_equity_from_env() -> float:
    raw = os.getenv("ORB_V2_ROBOT_EQUITY", "10000")
    try:
        return max(0.0, float(str(raw).strip()))
    except ValueError:
        return 10_000.0


def init_robot_wallets(*, count: int = 8, equity_usdt: float = 10_000.0) -> List[float]:
    eq = float(equity_usdt)
    n = max(1, int(count))
    return [eq] * n


def robot_equity_for_signals(wallets: List[float], cfg: OrbConfig) -> float:
    positive = [float(w) for w in wallets if float(w) > 0]
    return max(positive) if positive else float(cfg.per_symbol_bot_equity())


def next_robot_index(used: set[int], wallets: List[float]) -> Optional[int]:
    """0-based index；无 reuse 时每日每 robot 最多一单。"""
    for i, w in enumerate(wallets):
        if i not in used and float(w) > 0:
            return i
    return None


def next_free_robot(busy: Dict[int, Dict[str, Any]], wallets: List[float]) -> Optional[int]:
    """0-based index；reuse 模式下 robot 不在 busy 即可用。"""
    for i, w in enumerate(wallets):
        if i not in busy and float(w) > 0:
            return i
    return None


def release_robots_through(
    busy: Dict[int, Dict[str, Any]],
    wallets: List[float],
    scan_ms: int,
) -> None:
    done = [ridx for ridx, occ in busy.items() if int(occ.get("exit_ms") or 0) <= int(scan_ms)]
    for ridx in done:
        occ = busy.pop(ridx)
        wallets[ridx] = round(float(wallets[ridx]) + float(occ.get("pnl_usdt") or 0), 4)


def migrate_orb_robots(c: sqlite3.Cursor) -> None:
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS orb_robots (
            robot_id INTEGER PRIMARY KEY,
            initial_equity_usdt REAL NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at_utc TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL
        )
        """
    )
    for ddl in (
        "ALTER TABLE orb_signals ADD COLUMN robot_id INTEGER",
        "ALTER TABLE orb_settlements ADD COLUMN robot_id INTEGER",
    ):
        try:
            c.execute(ddl)
        except sqlite3.OperationalError:
            pass


def ensure_orb_robots(
    cur: sqlite3.Cursor,
    *,
    count: int,
    initial_equity_usdt: float,
) -> None:
    migrate_orb_robots(cur)
    now = _utc_now()
    init = max(0.0, float(initial_equity_usdt or 0.0))
    n = max(1, int(count))
    for rid in range(1, n + 1):
        cur.execute(
            """
            INSERT INTO orb_robots(robot_id, initial_equity_usdt, enabled, created_at_utc, updated_at_utc)
            VALUES (?, ?, 1, ?, ?)
            ON CONFLICT(robot_id) DO NOTHING
            """,
            (rid, init, now, now),
        )


def robot_settled_pnl(cur: sqlite3.Cursor, robot_id: int) -> float:
    cur.execute(
        "SELECT COALESCE(SUM(pnl_usdt), 0) FROM orb_settlements WHERE robot_id = ?",
        (int(robot_id),),
    )
    return float(cur.fetchone()[0] or 0)


def busy_robot_ids(cur: sqlite3.Cursor) -> set[int]:
    cur.execute(
        """
        SELECT DISTINCT robot_id FROM orb_signals
        WHERE outcome IS NULL AND side IN ('LONG','SHORT') AND sl_price IS NOT NULL
          AND robot_id IS NOT NULL
        """
    )
    return {int(r[0]) for r in cur.fetchall() if r[0] is not None}


def next_free_robot_id(cur: sqlite3.Cursor, *, count: int) -> Optional[int]:
    busy = busy_robot_ids(cur)
    n = max(1, int(count))
    for rid in range(1, n + 1):
        if rid in busy:
            continue
        cur.execute("SELECT enabled FROM orb_robots WHERE robot_id = ?", (rid,))
        row = cur.fetchone()
        if row is not None and int(row[0] or 0) == 0:
            continue
        return rid
    return None


def robot_wallet_balance(
    conn: sqlite3.Connection,
    robot_id: int,
    *,
    initial_equity_usdt: float,
    sync: bool = True,
) -> float:
    rid = int(robot_id)
    initial = max(0.0, float(initial_equity_usdt or 0.0))
    cur = conn.cursor()
    settled = robot_settled_pnl(cur, rid)
    balance = round(initial + settled, 4)
    if sync:
        cur.execute("SELECT 1 FROM orb_robots WHERE robot_id = ?", (rid,))
        if cur.fetchone() is not None:
            cur.execute(
                "UPDATE orb_robots SET updated_at_utc = ? WHERE robot_id = ?",
                (_utc_now(), rid),
            )
    return balance


def list_robot_wallet_balances(
    conn: sqlite3.Connection,
    *,
    count: int,
    initial_equity_usdt: float,
) -> List[float]:
    n = max(1, int(count))
    return [
        robot_wallet_balance(conn, rid, initial_equity_usdt=initial_equity_usdt, sync=False)
        for rid in range(1, n + 1)
    ]


def sync_robot_wallet(conn: sqlite3.Connection, robot_id: int, *, initial_equity_usdt: float) -> float:
    return robot_wallet_balance(
        conn,
        int(robot_id),
        initial_equity_usdt=initial_equity_usdt,
        sync=True,
    )
