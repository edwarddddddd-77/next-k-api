"""ORB 量价策略 API。"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from accumulation_radar import init_db
from orb.db import clear_orb_tables, ensure_symbol_bots, list_symbol_bot_summaries, migrate_orb_tables
from orb.paper import run_scan
from orb.session_today import build_session_today
from utils.maintenance_auth import require_maintenance_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/orb", tags=["orb"])


def _status(row: Dict[str, Any]) -> str:
    oc = row.get("outcome")
    if oc:
        return {
            "win": "盈利",
            "loss": "止损",
            "expired": "超时",
            "session_close": "收盘平仓",
            "early_exit": "提前离场",
            "supersede": "信号结束",
        }.get(str(oc), str(oc))
    if row.get("side") in ("LONG", "SHORT") and row.get("sl_price") is not None:
        return "持仓中"
    return "观望"


def load_summary() -> Dict[str, Any]:
    from orb.config import OrbConfig

    cfg = OrbConfig.from_env()
    conn = init_db()
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        migrate_orb_tables(cur)
        bot_equity = cfg.per_symbol_bot_equity()
        ensure_symbol_bots(cur, cfg.symbol_list(), initial_equity_usdt=bot_equity)
        conn.commit()
        cur.execute(
            "SELECT COUNT(*) FROM orb_signals WHERE outcome IS NULL AND side IN ('LONG','SHORT') AND sl_price IS NOT NULL"
        )
        open_n = int(cur.fetchone()[0] or 0)
        cur.execute("SELECT COUNT(*), COALESCE(SUM(pnl_usdt),0) FROM orb_settlements")
        r = cur.fetchone()
        settled, pnl = int(r[0] or 0), float(r[1] or 0)
        cur.execute("SELECT outcome, COUNT(*) FROM orb_settlements GROUP BY outcome")
        by_oc = {str(x[0]): int(x[1]) for x in cur.fetchall()}
        w, l = int(by_oc.get("win", 0)), int(by_oc.get("loss", 0))
        touch = w + l
        per_symbol = list_symbol_bot_summaries(
            conn, symbols=cfg.symbol_list(), initial_equity_usdt=bot_equity
        )
        return {
            "ok": True,
            "lane": "orb",
            "open_positions": open_n,
            "settled_trades": settled,
            "sum_pnl_usdt": round(pnl, 4),
            "touch_win_rate": round(w / touch, 4) if touch else None,
            "outcome_breakdown": by_oc,
            "symbol_bot_equity_usdt": round(bot_equity, 4),
            "symbol_bot_count": len(cfg.symbol_list()),
            "per_symbol": per_symbol,
            "today": build_session_today(),
        }
    finally:
        conn.close()


def load_signals(*, limit: int, offset: int, symbol: Optional[str], status: str) -> Dict[str, Any]:
    conn = init_db()
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        migrate_orb_tables(cur)
        where: List[str] = []
        params: List[Any] = []
        if symbol:
            where.append("symbol = ?")
            params.append(symbol.strip().upper())
        if status == "open":
            where.append("outcome IS NULL AND side IN ('LONG','SHORT') AND sl_price IS NOT NULL")
        elif status == "settled":
            where.append("outcome IS NOT NULL")
        sql = "SELECT * FROM orb_signals"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY recorded_at_utc DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = [dict(r) for r in cur.execute(sql, params).fetchall()]
        for d in rows:
            d["status"] = _status(d)
            if d.get("reasons_json"):
                try:
                    d["reasons"] = json.loads(d["reasons_json"])
                except json.JSONDecodeError:
                    d["reasons"] = []
        return {"ok": True, "lane": "orb", "count": len(rows), "signals": rows}
    finally:
        conn.close()


def load_latest_run() -> Dict[str, Any]:
    conn = init_db()
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        migrate_orb_tables(cur)
        cur.execute("SELECT * FROM orb_runs ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            return {"ok": True, "has_run": False}
        d = dict(row)
        if d.get("detail_json"):
            try:
                d["detail"] = json.loads(d["detail_json"])
            except json.JSONDecodeError:
                d["detail"] = None
        return {"ok": True, "has_run": True, "run": d}
    finally:
        conn.close()


@router.get("/session/today")
async def orb_session_today():
    try:
        return await run_in_threadpool(build_session_today)
    except Exception as e:
        logger.warning("orb session today failed: %s", e)
        raise HTTPException(status_code=500, detail="orb_session_today_error")


@router.get("/summary")
async def orb_summary():
    try:
        return await run_in_threadpool(load_summary)
    except Exception as e:
        logger.warning("orb summary failed: %s", e)
        raise HTTPException(status_code=500, detail="orb_summary_error")


@router.get("/signals")
async def orb_signals(
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    symbol: Optional[str] = Query(None),
    status: Optional[str] = Query(None, description="all|open|settled"),
):
    st = (status or "all").strip().lower()
    if st not in ("all", "open", "settled"):
        raise HTTPException(status_code=400, detail="invalid status")
    try:
        return await run_in_threadpool(load_signals, limit=limit, offset=offset, symbol=symbol, status=st)
    except Exception as e:
        logger.warning("orb signals failed: %s", e)
        raise HTTPException(status_code=500, detail="orb_signals_error")


@router.get("/scan/latest")
async def orb_scan_latest():
    try:
        return await run_in_threadpool(load_latest_run)
    except Exception as e:
        logger.warning("orb scan latest failed: %s", e)
        raise HTTPException(status_code=500, detail="orb_scan_latest_error")


@router.post("/maintenance/scan")
async def orb_maintenance_scan(_: None = Depends(require_maintenance_token)):
    try:
        return await run_in_threadpool(run_scan, do_resolve=True)
    except Exception as e:
        logger.exception("orb scan failed: %s", e)
        raise HTTPException(status_code=500, detail="orb_scan_failed") from e


@router.post("/maintenance/clear-db")
async def orb_clear_db(_: None = Depends(require_maintenance_token)):
    conn = init_db()
    try:
        return {"ok": True, **clear_orb_tables(conn)}
    except Exception as e:
        logger.exception("orb clear-db failed: %s", e)
        raise HTTPException(status_code=500, detail="orb_clear_db_failed") from e
