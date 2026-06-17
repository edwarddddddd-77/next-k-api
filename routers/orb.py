"""ORB 量价策略 API。"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from accumulation_radar import init_db
from orb.core.db import clear_orb_tables, migrate_orb_tables
from orb.core.live_settings import live_notify_status
from orb.v2.config import OrbV2Config
from orb.v2.db import migrate_orb_v2_tables
from orb.v2.paper import run_scan_v2
from orb.v2.robots import (
    ensure_orb_robots,
    list_recent_robot_resets,
    list_robot_summaries,
    robot_count_from_env,
    robot_equity_from_env,
    robot_reset_policy,
    total_robot_withdrawn,
)
from orb.core.session_today import build_session_today
from utils.maintenance_auth import require_maintenance_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/orb", tags=["orb"])


def _with_live_status(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload.update(live_notify_status())
    return payload


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
            "robot_reset": "提现重置",
        }.get(str(oc), str(oc))
    if row.get("side") in ("LONG", "SHORT") and row.get("sl_price") is not None:
        return "持仓中"
    return "观望"


def load_summary() -> Dict[str, Any]:
    v2 = OrbV2Config.from_env()
    cfg = v2.base
    robot_count = robot_count_from_env()
    robot_equity = robot_equity_from_env()
    conn = init_db()
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        migrate_orb_tables(cur)
        migrate_orb_v2_tables(cur)
        ensure_orb_robots(cur, count=robot_count, initial_equity_usdt=robot_equity)
        conn.commit()
        cur.execute(
            "SELECT COUNT(*) FROM orb_signals WHERE outcome IS NULL AND side IN ('LONG','SHORT') AND sl_price IS NOT NULL"
        )
        open_n = int(cur.fetchone()[0] or 0)
        cur.execute("SELECT COUNT(*) FROM orb_settlements WHERE COALESCE(outcome, '') != 'robot_reset'")
        settled = int(cur.fetchone()[0] or 0)
        cur.execute(
            "SELECT COALESCE(SUM(pnl_usdt),0) FROM orb_settlements WHERE COALESCE(outcome, '') != 'robot_reset'"
        )
        trading_pnl = float(cur.fetchone()[0] or 0)
        cur.execute("SELECT COALESCE(SUM(pnl_usdt),0) FROM orb_settlements")
        wallet_pnl = float(cur.fetchone()[0] or 0)
        withdrawn_total = total_robot_withdrawn(cur)
        cur.execute(
            "SELECT outcome, COUNT(*) FROM orb_settlements WHERE COALESCE(outcome, '') != 'robot_reset' GROUP BY outcome"
        )
        by_oc = {str(x[0]): int(x[1]) for x in cur.fetchall()}
        w, l = int(by_oc.get("win", 0)), int(by_oc.get("loss", 0))
        touch = w + l
        robots = list_robot_summaries(
            conn, count=robot_count, initial_equity_usdt=robot_equity
        )
        recent_resets = list_recent_robot_resets(cur, limit=8)
        conn.commit()
        gate = v2.load_gate()
        symbols = v2.symbol_list()
        return _with_live_status(
            {
                "ok": True,
                "lane": v2.lane,
                "strategy": "orb_v2",
                "orb_version": 2,
                "open_positions": open_n,
                "settled_trades": settled,
                "sum_pnl_usdt": round(trading_pnl, 4),
                "sum_wallet_pnl_usdt": round(wallet_pnl, 4),
                "total_withdrawn_usdt": withdrawn_total,
                "recent_robot_resets": recent_resets,
                "robot_reset_policy": robot_reset_policy(),
                "touch_win_rate": round(w / touch, 4) if touch else None,
                "outcome_breakdown": by_oc,
                "robot_count": robot_count,
                "robot_equity_usdt": round(robot_equity, 4),
                "universe_count": len(symbols),
                "symbols_file": str(v2.symbols_file),
                "robots": robots,
                "gate": {
                    "min_p_true": gate.min_p_true,
                    "max_opens_per_day": gate.max_opens_per_day,
                    "robot_reuse_after_exit": gate.robot_reuse_after_exit,
                    "day_abort_enabled": gate.day_abort_enabled,
                },
                "today": build_session_today(),
            },
        )
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
            rid = d.get("robot_id")
            d["robot_label"] = f"R{int(rid)}" if rid is not None else None
            if d.get("reasons_json"):
                try:
                    d["reasons"] = json.loads(d["reasons_json"])
                except json.JSONDecodeError:
                    d["reasons"] = []
        return {"ok": True, "lane": "orb_v2", "count": len(rows), "signals": rows}
    finally:
        conn.close()


def load_latest_run() -> Dict[str, Any]:
    conn = init_db()
    conn.row_factory = sqlite3.Row
    try:
        cur = conn.cursor()
        migrate_orb_tables(cur)
        migrate_orb_v2_tables(cur)
        cur.execute("SELECT * FROM orb_v2_runs ORDER BY id DESC LIMIT 1")
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


@router.get("/live")
async def orb_live_get():
    """读取 ORB_LIVE_ENABLED（Railway 环境变量）及 Protocol 连接状态。"""
    return live_notify_status()


@router.get("/live-bundle")
async def orb_live_bundle():
    """Live 人工包（Gate + GBM + Profiles）就绪状态，供前端提示。"""
    from orb.ml.live_bundle import live_bundle_hint

    try:
        return await run_in_threadpool(live_bundle_hint)
    except Exception as e:
        logger.warning("orb live-bundle failed: %s", e)
        raise HTTPException(status_code=500, detail="orb_live_bundle_error") from e


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
        return await run_in_threadpool(run_scan_v2, do_resolve=True)
    except Exception as e:
        logger.exception("orb scan failed: %s", e)
        raise HTTPException(status_code=500, detail="orb_scan_failed") from e


@router.post("/maintenance/clear-db")
async def orb_clear_db(_: None = Depends(require_maintenance_token)):
    conn = init_db()
    try:
        return {"ok": True, **clear_orb_tables(conn)}
    finally:
        conn.close()
