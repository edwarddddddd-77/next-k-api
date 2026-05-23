"""Supertrend 纸面信号 API。"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from utils.maintenance_auth import require_maintenance_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/supertrend", tags=["supertrend"])


def _rows_to_dicts(rows: List[sqlite3.Row]) -> List[Dict[str, Any]]:
    return [dict(r) for r in rows]


def _compute_summary(cur: sqlite3.Cursor) -> Dict[str, Any]:
    cur.execute(
        """
        SELECT COUNT(*) FROM st_signals
        WHERE outcome IS NULL AND side IN ('LONG', 'SHORT')
        """
    )
    open_positions = int(cur.fetchone()[0] or 0)

    cur.execute(
        """
        SELECT COUNT(*) FROM st_signals
        WHERE outcome IS NOT NULL
        """
    )
    settled_snapshots = int(cur.fetchone()[0] or 0)

    cur.execute("SELECT COUNT(*) FROM st_settlements")
    settled_count = int(cur.fetchone()[0] or 0)

    cur.execute(
        """
        SELECT COALESCE(SUM(pnl_usdt), 0),
               SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END),
               SUM(CASE WHEN pnl_usdt < 0 THEN 1 ELSE 0 END)
        FROM st_settlements
        WHERE pnl_usdt IS NOT NULL
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
        SELECT settled_at_utc, pnl_usdt FROM st_settlements
        WHERE pnl_usdt IS NOT NULL
        ORDER BY settled_at_utc ASC
        """
    )
    by_day: Dict[str, float] = {}
    for settled_at, pnl in cur.fetchall():
        day = str(settled_at or "")[:10]
        if not day:
            continue
        by_day[day] = by_day.get(day, 0.0) + float(pnl or 0)
    cum = 0.0
    equity_points: List[Dict[str, Any]] = []
    for day in sorted(by_day.keys()):
        cum += by_day[day]
        equity_points.append(
            {"date": day, "day_pnl_usdt": by_day[day], "cum_pnl_usdt": cum}
        )

    last_run: Optional[str] = None
    try:
        cur.execute(
            "SELECT ran_at_utc FROM st_runs ORDER BY id DESC LIMIT 1"
        )
        r = cur.fetchone()
        if r and r[0]:
            last_run = str(r[0])
    except sqlite3.OperationalError:
        pass

    try:
        import supertrend_config as st_cfg

        timeframe = st_cfg.ST_TIMEFRAME
        st_period = st_cfg.ST_ATR_PERIOD
        st_multiplier = st_cfg.ST_ATR_MULTIPLIER
    except Exception:
        timeframe = "5m"
        st_period = 10
        st_multiplier = 3.0

    return {
        "ok": True,
        "open_positions": open_positions,
        "settled_count": settled_count,
        "settled_snapshots": settled_snapshots,
        "total_pnl_usdt": total_pnl,
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "timeframe": timeframe,
        "st_period": st_period,
        "st_multiplier": st_multiplier,
        "last_run_utc": last_run,
        "equity_points": equity_points,
    }


@router.get("/summary")
async def get_st_summary():
    from accumulation_radar import init_db

    conn = init_db()
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        return _compute_summary(cur)
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            return {
                "ok": True,
                "open_positions": 0,
                "settled_count": 0,
                "total_pnl_usdt": 0,
                "win_rate": None,
                "equity_points": [],
            }
        raise HTTPException(status_code=500, detail="st_summary_error") from e
    finally:
        conn.close()


@router.get("/signals")
async def get_st_signals():
    from accumulation_radar import init_db

    conn = init_db()
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM st_signals
            ORDER BY
              CASE WHEN outcome IS NULL AND side IN ('LONG','SHORT') THEN 0 ELSE 1 END,
              recorded_at_utc DESC
            """
        )
        return {"signals": _rows_to_dicts(cur.fetchall())}
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            return {"signals": []}
        raise HTTPException(status_code=500, detail="st_signals_error") from e
    finally:
        conn.close()


@router.get("/settlements")
async def get_st_settlements(limit: int = 100):
    from accumulation_radar import init_db

    lim = max(1, min(500, int(limit)))
    conn = init_db()
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT * FROM st_settlements
            ORDER BY settled_at_utc DESC
            LIMIT ?
            """,
            (lim,),
        )
        return {"settlements": _rows_to_dicts(cur.fetchall())}
    except sqlite3.OperationalError as e:
        if "no such table" in str(e).lower():
            return {"settlements": []}
        raise HTTPException(status_code=500, detail="st_settlements_error") from e
    finally:
        conn.close()


@router.post("/scan")
async def post_st_scan(_: None = Depends(require_maintenance_token)):
    """手动触发 Supertrend 扫描（与定时任务同源）。"""
    import threading

    def _work() -> None:
        try:
            from supertrend_signal_scanner import run_scan

            run_scan(notify=True)
        except Exception:
            logger.exception("manual st scan failed")

    threading.Thread(target=_work, daemon=True).start()
    return {"accepted": True, "task": "st_scan"}
