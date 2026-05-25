"""接针策略 — 热度+OI 纸面 API。"""

from __future__ import annotations

import logging
import sqlite3
import threading
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from utils.maintenance_auth import require_maintenance_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jiezhen", tags=["jiezhen"])


def _rows_to_dicts(rows: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [dict(r) for r in rows]


def _compute_summary(cur: sqlite3.Cursor) -> Dict[str, Any]:
    cur.execute(
        """
        SELECT COUNT(*) FROM jz_signals
        WHERE outcome IS NULL AND side IN ('LONG', 'SHORT')
        """
    )
    open_positions = int(cur.fetchone()[0] or 0)

    cur.execute("SELECT COUNT(*) FROM jz_settlements")
    settled_count = int(cur.fetchone()[0] or 0)

    cur.execute(
        """
        SELECT COALESCE(SUM(pnl_usdt), 0),
               SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN pnl_usdt < 0 THEN 1 ELSE 0 END)
        FROM jz_settlements WHERE pnl_usdt IS NOT NULL
        """
    )
    row = cur.fetchone()
    total_pnl = float(row[0] or 0) if row else 0.0
    wins = int(row[1] or 0) if row else 0
    losses = int(row[2] or 0) if row else 0
    wr_denom = wins + losses
    win_rate = (wins / wr_denom) if wr_denom > 0 else None

    cur.execute(
        """
        SELECT COALESCE(SUM(unrealized_pnl_usdt), 0) FROM jz_signals
        WHERE outcome IS NULL AND side IN ('LONG', 'SHORT')
        """
    )
    unrealized = float(cur.fetchone()[0] or 0)

    cur.execute(
        """
        SELECT side, symbol, unrealized_pnl_usdt, mark_price, entry_price, trail_tier
        FROM jz_signals
        WHERE outcome IS NULL AND side IN ('LONG', 'SHORT')
        ORDER BY side ASC, symbol ASC
        """
    )
    open_legs = [dict(r) for r in cur.fetchall()]

    last_run: str | None = None
    try:
        cur.execute(
            "SELECT ran_at_utc, universe_size FROM jz_runs ORDER BY id DESC LIMIT 1"
        )
        r = cur.fetchone()
        if r:
            last_run = str(r[0]) if r[0] else None
    except sqlite3.OperationalError:
        pass

    try:
        import jiezhen_config as jz_cfg

        notional = jz_cfg.JIEZHEN_NOTIONAL_USDT
        leverage = jz_cfg.JIEZHEN_LEVERAGE
        equity = jz_cfg.JIEZHEN_ACCOUNT_EQUITY_USDT
        scan_sec = jz_cfg.JIEZHEN_SCAN_INTERVAL_SEC
        trail_sec = jz_cfg.JIEZHEN_TRAIL_SCAN_INTERVAL_SEC
        trail_scheduler = jz_cfg.jz_trail_scheduler_enabled()
        jz_trail_enabled = jz_cfg.JIEZHEN_TRAIL_ENABLED
        universe_max = jz_cfg.JIEZHEN_UNIVERSE_MAX
    except Exception:
        notional = 1000.0
        leverage = 0.1
        equity = 10000.0
        scan_sec = 60
        trail_sec = 15
        trail_scheduler = True
        jz_trail_enabled = True
        universe_max = 20

    return {
        "ok": True,
        "lane": "jiezhen",
        "strategy": "jiezhen",
        "open_positions": open_positions,
        "settled_count": settled_count,
        "total_pnl_usdt": total_pnl,
        "unrealized_pnl_usdt": unrealized,
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "notional_usdt": notional,
        "leverage": leverage,
        "equity_usdt": equity,
        "scan_interval_seconds": scan_sec,
        "trail_scan_interval_seconds": trail_sec,
        "trail_scheduler_enabled": trail_scheduler,
        "trail_params_from_momentum_env": True,
        "jiezhen_trail_enabled": jz_trail_enabled,
        "jiezhen_trail_scheduler_enabled": trail_scheduler,
        "universe_max": universe_max,
        "last_run_utc": last_run,
        "open_legs": open_legs,
    }


@router.get("/summary")
async def get_jiezhen_summary():
    from accumulation_radar import init_db

    conn = init_db()
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        return _compute_summary(cur)
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            return {"ok": True, "lane": "jiezhen", "open_positions": 0}
        raise HTTPException(status_code=500, detail="jz_summary_error") from e
    finally:
        conn.close()


@router.get("/signals")
async def get_jiezhen_signals():
    from accumulation_radar import init_db

    conn = init_db()
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM jz_signals
            ORDER BY
              CASE WHEN outcome IS NULL AND side IN ('LONG','SHORT') THEN 0 ELSE 1 END,
              recorded_at_utc DESC
            """
        )
        return {"signals": _rows_to_dicts(cur.fetchall())}
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            return {"signals": []}
        raise HTTPException(status_code=500, detail="jz_signals_error") from e
    finally:
        conn.close()


@router.get("/settlements")
async def get_jiezhen_settlements(limit: int = 100):
    from accumulation_radar import init_db

    lim = max(1, min(500, int(limit)))
    conn = init_db()
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM jz_settlements
            ORDER BY settled_at_utc DESC LIMIT ?
            """,
            (lim,),
        )
        return {"settlements": _rows_to_dicts(cur.fetchall())}
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            return {"settlements": []}
        raise HTTPException(status_code=500, detail="jz_settlements_error") from e
    finally:
        conn.close()


@router.get("/universe")
async def get_jiezhen_universe():
    """当前热度+OI 标的池（只读）。"""
    from jiezhen_signals import resolve_jiezhen_universe

    syms, meta = resolve_jiezhen_universe()
    return {"ok": True, "symbols": syms, "meta": meta}


@router.post("/maintenance/clear-db")
async def post_jiezhen_clear_db(_: None = Depends(require_maintenance_token)):
    from accumulation_radar import init_db
    from jiezhen_db import clear_jz_lane_tables

    try:
        conn = init_db()
        try:
            deleted = clear_jz_lane_tables(conn)
        finally:
            conn.close()
        logger.warning("jz clear-db: %s", deleted)
        return {"ok": True, **deleted}
    except Exception as e:
        logger.exception("jz clear-db failed: %s", e)
        raise HTTPException(status_code=500, detail="jz_clear_db_failed") from e


@router.post("/scan")
async def post_jiezhen_scan(_: None = Depends(require_maintenance_token)):
    def _work() -> None:
        try:
            from jiezhen_scanner import run_scan

            run_scan(notify=True)
        except Exception:
            logger.exception("manual jz scan failed")

    threading.Thread(target=_work, daemon=True).start()
    return {"accepted": True, "task": "jiezhen_scan"}


@router.post("/trail-scan")
async def post_jiezhen_trail_scan(_: None = Depends(require_maintenance_token)):
    def _work() -> None:
        try:
            from jiezhen_scanner import run_trail_checks

            run_trail_checks(notify=True)
        except Exception:
            logger.exception("manual jz trail scan failed")

    threading.Thread(target=_work, daemon=True).start()
    return {"accepted": True, "task": "jiezhen_trail"}
