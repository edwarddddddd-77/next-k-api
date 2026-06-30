"""ORB V2 共享 8-robot 资金池（Live 与回测共用）。"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from orb.core.config import OrbConfig
from orb.core.kline_cache import norm_symbol

if TYPE_CHECKING:
    from orb.ml.gate import LiveGateConfig

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def robot_count_from_env() -> int:
    raw = os.getenv("ORB_V2_ROBOT_COUNT", "8")
    try:
        return max(1, int(float(str(raw).strip())))
    except ValueError:
        return 8


def resolve_robot_pool_size(*, gate: "LiveGateConfig", symbol_count: int) -> int:
    """共享池以 live_gate 为准；绑定模式仅在 robot_count == symbol_count 时生效。"""
    env_n = robot_count_from_env()
    if robot_bound_mode(symbol_count=symbol_count, robot_count=env_n):
        return max(1, symbol_count)
    gate_n = int(getattr(gate, "robot_pool_size", 0) or gate.max_opens_per_day or 8)
    if env_n != gate_n:
        logger.info(
            "[orb_v2] shared robot pool=%s from gate (ORB_V2_ROBOT_COUNT=%s ignored)",
            gate_n,
            env_n,
        )
    return max(1, gate_n)


def robot_equity_from_env() -> float:
    raw = os.getenv("ORB_V2_ROBOT_EQUITY", "20")
    try:
        return max(0.0, float(str(raw).strip()))
    except ValueError:
        return 20.0


def _env_truthy(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def robot_bound_mode(*, symbol_count: int, robot_count: Optional[int] = None) -> bool:
    """每 robot 固定绑定 symbols 列表中同序标的一标（须 robot_count == symbol_count）。"""
    rc = robot_count if robot_count is not None else robot_count_from_env()
    if symbol_count <= 0 or rc <= 0:
        return False
    if rc != symbol_count:
        raw = os.getenv("ORB_V2_ROBOT_BOUND", "")
        if raw.strip() and _env_truthy("ORB_V2_ROBOT_BOUND"):
            logger.warning(
                "[orb_v2] ORB_V2_ROBOT_BOUND=1 ignored: robot_count=%s != symbol_count=%s "
                "(use shared %s-robot pool)",
                rc,
                symbol_count,
                rc,
            )
        return False
    raw = os.getenv("ORB_V2_ROBOT_BOUND", "")
    if raw.strip():
        return _env_truthy("ORB_V2_ROBOT_BOUND")
    return True


def robot_symbol_bindings(symbols: List[str]) -> Dict[int, str]:
    """1-based robot_id -> symbol（顺序与 symbols.txt 一致）。"""
    out: Dict[int, str] = {}
    for i, sym in enumerate(symbols):
        s = norm_symbol(sym)
        if s:
            out[i + 1] = s
    return out


def symbol_to_robot_id(symbol: str, symbols: List[str]) -> Optional[int]:
    sym = norm_symbol(symbol)
    for i, s in enumerate(symbols):
        if norm_symbol(s) == sym:
            return i + 1
    return None


def symbol_to_robot_index(symbol: str, symbols: List[str]) -> Optional[int]:
    rid = symbol_to_robot_id(symbol, symbols)
    return (rid - 1) if rid else None


def bound_robot_id_for_open(
    cur: sqlite3.Cursor,
    symbol: str,
    symbols: List[str],
    *,
    initial_equity_usdt: float,
) -> Optional[int]:
    """绑定模式下：仅当该标的专属 robot 空闲且有余额时返回 robot_id。"""
    rid = symbol_to_robot_id(symbol, symbols)
    if rid is None:
        return None
    if rid in busy_robot_ids(cur):
        return None
    cur.execute("SELECT enabled FROM orb_robots WHERE robot_id = ?", (rid,))
    row = cur.fetchone()
    if row is not None and int(row[0] or 0) == 0:
        return None
    conn = cur.connection
    if conn is not None and robot_wallet_balance(
        conn, rid, initial_equity_usdt=initial_equity_usdt, sync=False
    ) <= 0:
        return None
    return rid


def bound_robot_index_available(
    robot_busy: Dict[int, Dict[str, Any]],
    robot_wallets: List[float],
    symbol: str,
    symbols: List[str],
) -> Optional[int]:
    """绑定模式 sim：0-based robot index，专属 robot 空闲且有余额。"""
    ridx = symbol_to_robot_index(symbol, symbols)
    if ridx is None:
        return None
    if ridx in robot_busy:
        return None
    if ridx >= len(robot_wallets) or float(robot_wallets[ridx]) <= 0:
        return None
    return ridx


def apply_robot_wallet_after_pnl(balance: float, pnl: float) -> float:
    """结算后更新余额（纯复利，无 cap/floor 提现）。"""
    return round(float(balance) + float(pnl), 4)


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
        wallets[ridx] = apply_robot_wallet_after_pnl(
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
        cur.execute(
            """
            UPDATE orb_robots
            SET enabled=1, updated_at_utc=?
            WHERE robot_id = ? AND enabled = 0
            """,
            (now, rid),
        )
    cur.execute(
        "UPDATE orb_robots SET enabled=0, updated_at_utc=? WHERE robot_id > ?",
        (now, n),
    )


def robot_settled_pnl(cur: sqlite3.Cursor, robot_id: int) -> float:
    cur.execute(
        """
        SELECT COALESCE(SUM(pnl_usdt), 0) FROM orb_settlements
        WHERE robot_id = ? AND COALESCE(outcome, '') != 'robot_reset'
        """,
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
    symbols: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """按 robot_id 汇总 V2 资金池状态（R1..Rn）。"""
    prev_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    try:
        init = max(0.0, float(initial_equity_usdt or 0.0))
        n = max(1, int(count))
        bindings = (
            robot_symbol_bindings(symbols)
            if symbols and robot_bound_mode(symbol_count=len(symbols), robot_count=n)
            else {}
        )
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
            bound_sym = bindings.get(rid)
            row: Dict[str, Any] = {
                "robot_id": rid,
                "label": f"R{rid}",
                "symbol": sym or bound_sym,
                "bound_symbol": bound_sym,
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

