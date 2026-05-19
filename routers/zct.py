from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool

from models.api_models import ZctTouchPoolScanBody, ZctVwapManualPatchBody
from utils.maintenance_auth import require_maintenance_token

logger = logging.getLogger(__name__)

router = APIRouter(tags=["zct"])

@router.get("/api/zct-vwap/summary")
async def get_zct_vwap_summary():
    """ZCT VWAP 虚拟信号汇总：持仓笔数、已结算、累计 pnl_usdt、全局胜率、`per_symbol` 按标的胜率与笔数。"""
    try:
        from zct_vwap_api import load_zct_vwap_summary

        return load_zct_vwap_summary()
    except Exception as e:
        logger.warning("zct_vwap summary failed: %s", e)
        raise HTTPException(status_code=500, detail="zct_vwap_summary_error")


@router.get("/api/zct-vwap/equity-curve")
async def get_zct_vwap_equity_curve():
    """按结算日（UTC 日历日）累计虚拟盈亏曲线，供看板折线图。"""
    try:
        from zct_vwap_api import load_zct_equity_curve

        return load_zct_equity_curve()
    except Exception as e:
        logger.warning("zct_vwap equity curve failed: %s", e)
        raise HTTPException(status_code=500, detail="zct_vwap_equity_curve_error")


@router.get("/api/zct-vwap/signals")
async def get_zct_vwap_signals(
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    symbol: Optional[str] = Query(None, description="如 BTCUSDT"),
    status: Optional[str] = Query(
        None,
        description="all（默认）| open（持仓中）| settled（已结算）",
    ),
):
    """分页列出 ZCT VWAP 扫描入库的信号（含 SL/TP、虚拟名义与结算结果）。"""
    try:
        from zct_vwap_api import load_zct_vwap_signals

        return load_zct_vwap_signals(
            limit=limit,
            offset=offset,
            symbol=symbol,
            status=status or "all",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.warning("zct_vwap signals failed: %s", e)
        raise HTTPException(status_code=500, detail="zct_vwap_signals_error")


@router.patch("/api/zct-vwap/signals/{signal_id}")
async def patch_zct_vwap_signal(signal_id: int, body: ZctVwapManualPatchBody):
    """更新 ZCT VWAP 信号的实盘入场/平仓价与备注（不影响脚本虚拟字段）。"""
    try:
        from zct_vwap_api import patch_zct_vwap_manual

        updates = body.model_dump(exclude_unset=True)
        if not updates:
            raise HTTPException(status_code=400, detail="no_fields_to_update")
        out = patch_zct_vwap_manual(signal_id, updates)
        if not out.get("ok"):
            if out.get("error") == "not_found":
                raise HTTPException(status_code=404, detail="signal_not_found")
            raise HTTPException(status_code=500, detail="zct_vwap_patch_failed")
        return out
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.warning("zct_vwap patch failed: %s", e)
        raise HTTPException(status_code=500, detail="zct_vwap_patch_error")


async def _zct_vwap_clear_db_impl() -> dict:
    from accumulation_radar import init_db

    conn = init_db()
    try:
        cur = conn.cursor()
        n_settle = 0
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='zct_vwap_settlements'"
        )
        if cur.fetchone():
            cur.execute("SELECT COUNT(*) FROM zct_vwap_settlements")
            n_settle = int(cur.fetchone()[0] or 0)
            cur.execute("DELETE FROM zct_vwap_settlements")
        cur.execute("SELECT COUNT(*) FROM zct_vwap_signals")
        n_sig = int(cur.fetchone()[0] or 0)
        cur.execute("DELETE FROM zct_vwap_signals")
        conn.commit()
        logger.warning(
            "zct_vwap clear-db: deleted signals=%s settlements=%s",
            n_sig,
            n_settle,
        )
        return {
            "ok": True,
            "deleted_zct_vwap_signals": n_sig,
            "deleted_zct_vwap_settlements": n_settle,
        }
    finally:
        conn.close()


@router.post("/api/zct-vwap/maintenance/clear-db")
async def post_zct_vwap_clear_db(_: None = Depends(require_maintenance_token)):
    """
    清空 ZCT VWAP：`zct_vwap_signals`（每标的快照）与 `zct_vwap_settlements`（已平仓历史）。
    需 `X-Maintenance-Token`（环境变量 NEXT_K_MAINTENANCE_TOKEN）。
    """
    try:
        return await _zct_vwap_clear_db_impl()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("zct_vwap clear-db failed: %s", e)
        raise HTTPException(status_code=500, detail="zct_vwap_clear_db_failed") from e


@router.post("/api/zct-vwap/touch-pool-scan")
async def post_zct_touch_pool_scan(
    body: ZctTouchPoolScanBody = Body(...),
    _: None = Depends(require_maintenance_token),
):
    """ZCT walk-forward 触轨池；可选落库 accumulation.db（先清空入选表再写入）。"""
    iv = str(body.signal_interval or "1m").strip().lower()
    if iv not in ("1m", "5m"):
        raise HTTPException(status_code=400, detail="signal_interval must be 1m or 5m")

    src = str(body.symbols_source or "worth_watch_plus_default_22").strip().lower()
    if src not in ("request", "worth_watch_plus_default_22", "hot_oi_plus_default_22"):
        raise HTTPException(status_code=400, detail="invalid_symbols_source")

    if src == "worth_watch_plus_default_22":
        from zct_vwap_asset_pool import touch_pool_symbols_worth_watch_plus_default

        syms = touch_pool_symbols_worth_watch_plus_default()
        scan_src = "worth_watch_plus_default_22"
    elif src == "hot_oi_plus_default_22":
        from zct_vwap_asset_pool import touch_pool_symbols_hot_oi_plus_default_22

        syms = touch_pool_symbols_hot_oi_plus_default_22()
        scan_src = "hot_oi_plus_default_22"
    else:
        syms = [x.strip().upper() for x in (body.symbols or "").split(",") if x.strip()]
        scan_src = None

    if not syms:
        raise HTTPException(status_code=400, detail="empty_symbols")

    persist = bool(body.persist_db)

    def _work():
        from accumulation_radar import init_db
        from zct_vwap_asset_pool import (
            notify_touch_pool_empty_if_needed,
            run_asset_pool_scan,
        )
        from zct_vwap_touch_pool_db import touch_pool_ensure_schema, touch_pool_write_db

        out, _summary = run_asset_pool_scan(
            days=float(body.days),
            symbols=syms,
            ignore_db_cooldown=True,
            sleep_between_symbols=float(body.sleep_between_symbols),
            signal_interval=iv,
            min_touch_trades=int(body.min_touch_trades),
            strict_greater_touch=bool(body.strict_greater_touch),
            min_touch_win_rate=float(body.min_touch_win_rate),
            strict_greater_rate=bool(body.strict_greater_rate),
            min_total_trades=int(body.min_total_trades),
            max_expired_ratio=float(body.max_expired_ratio),
            min_win_loss_abs=int(body.min_win_loss_abs),
            min_touch_share=float(body.min_touch_share),
            min_profit_factor=float(body.min_profit_factor),
            max_consecutive_losses_at_end=int(body.max_consecutive_losses_at_end),
            quiet=True,
            symbols_source=scan_src,
        )
        if persist:
            conn = init_db()
            try:
                touch_pool_ensure_schema(conn)
                n = touch_pool_write_db(conn, out)
                notify_touch_pool_empty_if_needed(n, criteria=out.get("criteria") or {})
            finally:
                conn.close()
        return {"ok": True, "pool": out, "persisted_db": persist}

    try:
        return await run_in_threadpool(_work)
    except Exception as e:
        logger.exception("zct touch_pool scan failed: %s", e)
        raise HTTPException(status_code=500, detail="zct_touch_pool_scan_failed")


@router.get("/api/zct-hot-oi/summary")
async def get_zct_hot_oi_summary():
    """兼容旧路径：与 GET /api/zct-vwap/summary 相同（统一 zct_vwap_*）。"""
    try:
        from zct_vwap_api import load_zct_vwap_summary

        return load_zct_vwap_summary()
    except Exception as e:
        logger.warning("zct_hot_oi summary failed: %s", e)
        raise HTTPException(status_code=500, detail="zct_hot_oi_summary_error")


@router.get("/api/zct-hot-oi/equity-curve")
async def get_zct_hot_oi_equity_curve():
    """兼容旧路径：与 GET /api/zct-vwap/equity-curve 相同。"""
    try:
        from zct_vwap_api import load_zct_equity_curve

        return load_zct_equity_curve()
    except Exception as e:
        logger.warning("zct_hot_oi equity curve failed: %s", e)
        raise HTTPException(status_code=500, detail="zct_hot_oi_equity_curve_error")


@router.get("/api/zct-hot-oi/signals")
async def get_zct_hot_oi_signals(
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    symbol: Optional[str] = Query(None, description="如 BTCUSDT"),
    status: Optional[str] = Query(
        None,
        description="all（默认）| open（持仓中）| settled（已结算）",
    ),
):
    try:
        from zct_vwap_api import load_zct_vwap_signals

        return load_zct_vwap_signals(
            limit=limit,
            offset=offset,
            symbol=symbol,
            status=status or "all",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.warning("zct_hot_oi signals failed: %s", e)
        raise HTTPException(status_code=500, detail="zct_hot_oi_signals_error")


@router.patch("/api/zct-hot-oi/signals/{signal_id}")
async def patch_zct_hot_oi_signal(signal_id: int, body: ZctVwapManualPatchBody):
    try:
        from zct_vwap_api import patch_zct_vwap_manual

        updates = body.model_dump(exclude_unset=True)
        if not updates:
            raise HTTPException(status_code=400, detail="no_fields_to_update")
        out = patch_zct_vwap_manual(signal_id, updates)
        if not out.get("ok"):
            if out.get("error") == "not_found":
                raise HTTPException(status_code=404, detail="signal_not_found")
            raise HTTPException(status_code=500, detail="zct_hot_oi_patch_failed")
        return out
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.warning("zct_hot_oi patch failed: %s", e)
        raise HTTPException(status_code=500, detail="zct_hot_oi_patch_error")


@router.post("/api/zct-hot-oi/maintenance/clear-db")
async def post_zct_hot_oi_clear_db(_: None = Depends(require_maintenance_token)):
    """兼容旧路径：与 POST /api/zct-vwap/maintenance/clear-db 相同（清空统一 zct_vwap_*）。"""
    try:
        return await _zct_vwap_clear_db_impl()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("zct_hot_oi clear-db failed: %s", e)
        raise HTTPException(status_code=500, detail="zct_vwap_clear_db_failed") from e


@router.get("/dashboard/zct-vwap", response_class=HTMLResponse)
async def zct_vwap_dashboard_page():
    """ZCT VWAP 虚拟信号看板（静态页 + 调用上方 JSON API）。"""
    path = Path(__file__).resolve().parent.parent / "static" / "zct_vwap_dashboard.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="dashboard_zct_vwap_not_found")
    return HTMLResponse(content=path.read_text(encoding="utf-8"))

