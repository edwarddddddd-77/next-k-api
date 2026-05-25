"""动量多一空一 — topMovers 纸面仓位 API。"""

from __future__ import annotations

import logging
import sqlite3
import threading
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from utils.maintenance_auth import require_maintenance_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/momentum", tags=["momentum"])


def _rows_to_dicts(rows: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [dict(r) for r in rows]


def _compute_summary(cur: sqlite3.Cursor) -> Dict[str, Any]:
    cur.execute(
        """
        SELECT COUNT(*) FROM mom_signals
        WHERE outcome IS NULL AND side IN ('LONG', 'SHORT')
        """
    )
    open_positions = int(cur.fetchone()[0] or 0)

    cur.execute("SELECT COUNT(*) FROM mom_settlements")
    settled_count = int(cur.fetchone()[0] or 0)

    cur.execute(
        """
        SELECT COALESCE(SUM(pnl_usdt), 0),
               SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN pnl_usdt < 0 THEN 1 ELSE 0 END)
        FROM mom_settlements WHERE pnl_usdt IS NOT NULL
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
        SELECT COALESCE(SUM(unrealized_pnl_usdt), 0) FROM mom_signals
        WHERE outcome IS NULL AND side IN ('LONG', 'SHORT')
        """
    )
    unrealized = float(cur.fetchone()[0] or 0)

    cur.execute(
        """
        SELECT side, symbol, unrealized_pnl_usdt, mark_price, entry_price
        FROM mom_signals
        WHERE outcome IS NULL AND side IN ('LONG', 'SHORT')
        ORDER BY side ASC
        """
    )
    open_legs = [dict(r) for r in cur.fetchall()]

    last_run: str | None = None
    long_target: str | None = None
    short_target: str | None = None
    try:
        cur.execute(
            "SELECT ran_at_utc, long_target, short_target FROM mom_runs ORDER BY id DESC LIMIT 1"
        )
        r = cur.fetchone()
        if r:
            last_run = str(r[0]) if r[0] else None
            long_target = r[1]
            short_target = r[2]
    except sqlite3.OperationalError:
        pass

    try:
        import momentum_config as mom_cfg

        notional = mom_cfg.MOM_NOTIONAL_USDT
        leverage = mom_cfg.MOM_LEVERAGE
        equity = mom_cfg.MOM_ACCOUNT_EQUITY_USDT
        interval = mom_cfg.MOM_SCAN_INTERVAL_MINUTES
        trail_interval_sec = mom_cfg.MOM_TRAIL_SCAN_INTERVAL_SEC
        trail_scheduler = mom_cfg.mom_trail_scheduler_enabled()
        long_event = mom_cfg.MOM_LONG_EVENT
        short_event = mom_cfg.MOM_SHORT_EVENT
    except Exception:
        notional = 1000.0
        leverage = 0.1
        equity = 10000.0
        interval = 15
        trail_interval_sec = 20
        trail_scheduler = True
        long_event = "PULLBACK"
        short_event = "RALLY"

    return {
        "ok": True,
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
        "scan_interval_minutes": interval,
        "trail_scan_interval_seconds": trail_interval_sec,
        "trail_scan_interval_minutes": max(1, (trail_interval_sec + 59) // 60),
        "trail_scheduler_enabled": trail_scheduler,
        "long_event": long_event,
        "short_event": short_event,
        "last_run_utc": last_run,
        "last_long_target": long_target,
        "last_short_target": short_target,
        "open_legs": open_legs,
    }


@router.get("/summary")
async def get_momentum_summary():
    from accumulation_radar import init_db

    conn = init_db()
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        return _compute_summary(cur)
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            return {"ok": True, "open_positions": 0, "settled_count": 0, "total_pnl_usdt": 0}
        raise HTTPException(status_code=500, detail="mom_summary_error") from e
    finally:
        conn.close()


@router.get("/signals")
async def get_momentum_signals():
    from accumulation_radar import init_db

    conn = init_db()
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM mom_signals
            ORDER BY
              CASE WHEN outcome IS NULL AND side IN ('LONG','SHORT') THEN 0 ELSE 1 END,
              recorded_at_utc DESC
            """
        )
        return {"signals": _rows_to_dicts(cur.fetchall())}
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            return {"signals": []}
        raise HTTPException(status_code=500, detail="mom_signals_error") from e
    finally:
        conn.close()


@router.get("/settlements")
async def get_momentum_settlements(limit: int = 100):
    from accumulation_radar import init_db

    lim = max(1, min(500, int(limit)))
    conn = init_db()
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM mom_settlements
            ORDER BY settled_at_utc DESC LIMIT ?
            """,
            (lim,),
        )
        return {"settlements": _rows_to_dicts(cur.fetchall())}
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            return {"settlements": []}
        raise HTTPException(status_code=500, detail="mom_settlements_error") from e
    finally:
        conn.close()


@router.get("/top-movers")
async def get_top_movers_snapshot():
    """当前 topMovers 解析结果（只读，不下单）。"""
    from momentum_signals import fetch_momentum_targets

    long_sym, short_sym, meta = fetch_momentum_targets()
    return {
        "ok": not meta.get("error"),
        "long_target": long_sym,
        "short_target": short_sym,
        "meta": meta,
    }


@router.post("/maintenance/clear-db")
async def post_momentum_clear_db(_: None = Depends(require_maintenance_token)):
    from accumulation_radar import init_db
    from momentum_db import clear_mom_lane_tables

    try:
        conn = init_db()
        try:
            deleted = clear_mom_lane_tables(conn)
        finally:
            conn.close()
        logger.warning("mom clear-db: %s", deleted)
        return {"ok": True, **deleted}
    except Exception as e:
        logger.exception("mom clear-db failed: %s", e)
        raise HTTPException(status_code=500, detail="mom_clear_db_failed") from e


@router.post("/scan")
async def post_momentum_scan(_: None = Depends(require_maintenance_token)):
    def _work() -> None:
        try:
            from momentum_scanner import run_scan

            run_scan(notify=True)
        except Exception:
            logger.exception("manual mom scan failed")

    threading.Thread(target=_work, daemon=True).start()
    return {"accepted": True, "task": "mom_scan"}


@router.post("/trail-scan")
async def post_momentum_trail_scan(_: None = Depends(require_maintenance_token)):
    def _work() -> None:
        try:
            from momentum_scanner import run_trail_checks

            run_trail_checks(notify=True)
        except Exception:
            logger.exception("manual mom trail scan failed")

    threading.Thread(target=_work, daemon=True).start()
    return {"accepted": True, "task": "mom_trail"}
