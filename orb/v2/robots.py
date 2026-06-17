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
    raw = os.getenv("ORB_V2_ROBOT_EQUITY", "1000")
    try:
        return max(0.0, float(str(raw).strip()))
    except ValueError:
        return 1000.0


def robot_reset_cap_from_env() -> float:
    raw = os.getenv("ORB_V2_ROBOT_RESET_CAP", "2500")
    try:
        return max(0.0, float(str(raw).strip()))
    except ValueError:
        return 2500.0


def robot_reset_floor_from_env() -> float:
    raw = os.getenv("ORB_V2_ROBOT_RESET_FLOOR", "1500")
    try:
        return max(0.0, float(str(raw).strip()))
    except ValueError:
        return 1500.0


def robot_reset_policy() -> Dict[str, Any]:
    cap = robot_reset_cap_from_env()
    floor = robot_reset_floor_from_env()
    return {"cap_usdt": cap, "floor_usdt": floor, "enabled": cap > floor}


def _migrate_orb_robot_resets(c: sqlite3.Cursor) -> None:
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS orb_robot_resets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reset_at_utc TEXT NOT NULL,
            robot_id INTEGER NOT NULL,
            trigger_signal_id INTEGER,
            session_date TEXT,
            balance_before REAL NOT NULL,
            balance_after REAL NOT NULL,
            withdrawn_usdt REAL NOT NULL,
            cap_usdt REAL NOT NULL,
            floor_usdt REAL NOT NULL
        )
        """
    )
    c.execute(
        "CREATE INDEX IF NOT EXISTS ix_orb_robot_resets_robot ON orb_robot_resets(robot_id, id DESC)"
    )


def apply_robot_wallet_after_pnl(
    balance: float,
    pnl: float,
) -> tuple[float, Optional[Dict[str, Any]]]:
    """结算后更新余额；若 >= cap 则提现至 floor（Live / 回测共用）。"""
    policy = robot_reset_policy()
    new_balance = round(float(balance) + float(pnl), 4)
    if not policy.get("enabled"):
        return new_balance, None
    cap = float(policy["cap_usdt"])
    floor = float(policy["floor_usdt"])
    if new_balance + 1e-9 < cap:
        return new_balance, None
    withdrawn = round(new_balance - floor, 4)
    if withdrawn <= 0:
        return new_balance, None
    evt = {
        "balance_before": round(new_balance, 4),
        "balance_after": round(floor, 4),
        "withdrawn_usdt": withdrawn,
        "cap_usdt": cap,
        "floor_usdt": floor,
    }
    return round(floor, 4), evt


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
        wallets[ridx], _ = apply_robot_wallet_after_pnl(
            wallets[ridx],
            float(occ.get("pnl_usdt") or 0),
        )


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
    _migrate_orb_robot_resets(c)
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


def next_free_robot_id(
    cur: sqlite3.Cursor,
    *,
    count: int,
    initial_equity_usdt: Optional[float] = None,
) -> Optional[int]:
    busy = busy_robot_ids(cur)
    n = max(1, int(count))
    init = robot_equity_from_env() if initial_equity_usdt is None else float(initial_equity_usdt)
    conn = cur.connection
    for rid in range(1, n + 1):
        if rid in busy:
            continue
        cur.execute("SELECT enabled FROM orb_robots WHERE robot_id = ?", (rid,))
        row = cur.fetchone()
        if row is not None and int(row[0] or 0) == 0:
            continue
        if conn is not None and robot_wallet_balance(conn, rid, initial_equity_usdt=init, sync=False) <= 0:
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


def list_robot_summaries(
    conn: sqlite3.Connection,
    *,
    count: int,
    initial_equity_usdt: float,
) -> List[Dict[str, Any]]:
    """按 robot_id 汇总 V2 资金池状态（R1..Rn）。"""
    prev_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        init = max(0.0, float(initial_equity_usdt or 0.0))
        n = max(1, int(count))
        cur.execute(
            """
            SELECT robot_id,
                   COUNT(*) AS settled_count,
                   SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) AS losses,
                   COALESCE(SUM(pnl_usdt), 0) AS trading_pnl_usdt
            FROM orb_settlements
            WHERE robot_id IS NOT NULL AND COALESCE(outcome, '') != 'robot_reset'
            GROUP BY robot_id
            """
        )
        settle_by_rid = {int(r["robot_id"]): dict(r) for r in cur.fetchall() if r["robot_id"] is not None}
        cur.execute(
            """
            SELECT robot_id, reset_at_utc, balance_before, balance_after, withdrawn_usdt
            FROM orb_robot_resets AS r
            WHERE id = (
                SELECT MAX(id) FROM orb_robot_resets WHERE robot_id = r.robot_id
            )
            """
        )
        last_reset_by_rid = {int(r["robot_id"]): dict(r) for r in cur.fetchall()}
        cur.execute(
            """
            SELECT robot_id, symbol, side, virtual_notional_usdt
            FROM orb_signals
            WHERE outcome IS NULL AND side IN ('LONG','SHORT') AND sl_price IS NOT NULL
              AND robot_id IS NOT NULL
            """
        )
        open_by_rid: Dict[int, Dict[str, Any]] = {}
        for r in cur.fetchall():
            open_by_rid[int(r["robot_id"])] = dict(r)
        cur.execute("SELECT robot_id, enabled FROM orb_robots")
        bot_rows = {int(r["robot_id"]): dict(r) for r in cur.fetchall()}
        out: List[Dict[str, Any]] = []
        for rid in range(1, n + 1):
            bot = bot_rows.get(rid, {})
            st = settle_by_rid.get(rid, {})
            wins = int(st.get("wins") or 0)
            losses = int(st.get("losses") or 0)
            touch = wins + losses
            settled = int(st.get("settled_count") or 0)
            trading_pnl = round(float(st.get("trading_pnl_usdt") or 0), 4)
            wallet = round(
                robot_wallet_balance(conn, rid, initial_equity_usdt=init, sync=False),
                4,
            )
            op = open_by_rid.get(rid)
            sym = str(op["symbol"]).upper() if op and op.get("symbol") else None
            last_reset = last_reset_by_rid.get(rid)
            row: Dict[str, Any] = {
                "robot_id": rid,
                "label": f"R{rid}",
                "symbol": sym,
                "enabled": bool(int(bot.get("enabled", 1) or 1)),
                "initial_equity_usdt": round(init, 4),
                "wallet_balance_usdt": wallet,
                "realized_pnl_usdt": trading_pnl,
                "settled_count": settled,
                "wins": wins,
                "losses": losses,
                "touch_win_rate": round(wins / touch, 4) if touch else None,
                "open_side": str(op["side"]).upper() if op else None,
                "open_notional_usdt": round(float(op["virtual_notional_usdt"] or 0), 4) if op else None,
            }
            if last_reset:
                row["last_reset"] = {
                    "reset_at_utc": str(last_reset.get("reset_at_utc") or ""),
                    "balance_before": round(float(last_reset.get("balance_before") or 0), 4),
                    "balance_after": round(float(last_reset.get("balance_after") or 0), 4),
                    "withdrawn_usdt": round(float(last_reset.get("withdrawn_usdt") or 0), 4),
                }
            out.append(row)
        return out
    finally:
        conn.row_factory = prev_factory


def sync_robot_wallet(conn: sqlite3.Connection, robot_id: int, *, initial_equity_usdt: float) -> float:
    return robot_wallet_balance(
        conn,
        int(robot_id),
        initial_equity_usdt=initial_equity_usdt,
        sync=True,
    )


def total_robot_withdrawn(cur: sqlite3.Cursor) -> float:
    cur.execute("SELECT COALESCE(SUM(withdrawn_usdt), 0) FROM orb_robot_resets")
    return round(float(cur.fetchone()[0] or 0), 4)


def list_recent_robot_resets(cur: sqlite3.Cursor, *, limit: int = 8) -> List[Dict[str, Any]]:
    lim = max(1, min(int(limit), 50))
    cur.execute(
        """
        SELECT id, reset_at_utc, robot_id, trigger_signal_id, session_date,
               balance_before, balance_after, withdrawn_usdt, cap_usdt, floor_usdt
        FROM orb_robot_resets
        ORDER BY id DESC
        LIMIT ?
        """,
        (lim,),
    )
    out: List[Dict[str, Any]] = []
    for row in cur.fetchall():
        d = dict(row)
        rid = int(d["robot_id"])
        out.append(
            {
                "id": int(d["id"]),
                "reset_at_utc": str(d.get("reset_at_utc") or ""),
                "robot_id": rid,
                "robot_label": f"R{rid}",
                "trigger_signal_id": d.get("trigger_signal_id"),
                "session_date": d.get("session_date"),
                "balance_before": round(float(d.get("balance_before") or 0), 4),
                "balance_after": round(float(d.get("balance_after") or 0), 4),
                "withdrawn_usdt": round(float(d.get("withdrawn_usdt") or 0), 4),
                "cap_usdt": round(float(d.get("cap_usdt") or 0), 4),
                "floor_usdt": round(float(d.get("floor_usdt") or 0), 4),
            }
        )
    return out


def maybe_reset_robot_wallet_after_settle(
    conn: sqlite3.Connection,
    robot_id: int,
    *,
    trigger_signal_id: Optional[int] = None,
    session_date: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """结算后若机器人余额 >= cap，则提现至 floor 并重置账本。"""
    if trigger_signal_id is None:
        return None
    policy = robot_reset_policy()
    if not policy.get("enabled"):
        return None
    cap = float(policy["cap_usdt"])
    floor = float(policy["floor_usdt"])
    init = robot_equity_from_env()
    balance = robot_wallet_balance(conn, int(robot_id), initial_equity_usdt=init, sync=False)
    _, evt = apply_robot_wallet_after_pnl(balance, 0.0)
    if evt is None:
        return None

    withdrawn = float(evt["withdrawn_usdt"])
    cur = conn.cursor()
    trigger_sym = ""
    if trigger_signal_id is not None:
        cur.execute("SELECT symbol FROM orb_signals WHERE id = ?", (int(trigger_signal_id),))
        sig_row = cur.fetchone()
        if sig_row and sig_row[0]:
            trigger_sym = str(sig_row[0]).strip().upper()

    now = _utc_now()
    from orb.core.db import archive_settlement

    archive_settlement(
        cur,
        signal_id=int(trigger_signal_id),
        symbol=trigger_sym or f"R{int(robot_id)}",
        side="RESET",
        play=None,
        outcome="robot_reset",
        entry_price=round(float(evt["balance_before"]), 4),
        exit_price=round(floor, 4),
        pnl_r=0.0,
        pnl_usdt=round(-withdrawn, 4),
        notional=0.0,
        exit_rule=f"robot_reset cap={cap:g} floor={floor:g}",
        settled_at_utc=now,
        session_date=session_date,
        robot_id=int(robot_id),
    )
    cur.execute(
        """
        INSERT INTO orb_robot_resets (
            reset_at_utc, robot_id, trigger_signal_id, session_date,
            balance_before, balance_after, withdrawn_usdt, cap_usdt, floor_usdt
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            now,
            int(robot_id),
            int(trigger_signal_id) if trigger_signal_id is not None else None,
            session_date,
            round(float(evt["balance_before"]), 4),
            round(floor, 4),
            round(withdrawn, 4),
            cap,
            floor,
        ),
    )
    sync_robot_wallet(conn, int(robot_id), initial_equity_usdt=init)
    return {
        "id": int(cur.lastrowid or 0),
        "reset_at_utc": now,
        "robot_id": int(robot_id),
        "robot_label": f"R{int(robot_id)}",
        "trigger_signal_id": trigger_signal_id,
        "session_date": session_date,
        "balance_before": round(float(evt["balance_before"]), 4),
        "balance_after": round(floor, 4),
        "withdrawn_usdt": round(withdrawn, 4),
        "cap_usdt": cap,
        "floor_usdt": floor,
    }
