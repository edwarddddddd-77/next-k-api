"""Hyperliquid short-term watchlist + paper copy API."""

from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from utils.rate_limit import MinIntervalGuard

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/hl-short", tags=["hl-short"])

_refresh_lock = threading.Lock()
_refresh_cooldown = MinIntervalGuard("HL_SHORT_REFRESH_COOLDOWN_SEC", 20.0)


@router.get("/watchlist")
async def get_watchlist():
    from utils.hl_short_term import load_watchlist_doc

    return await run_in_threadpool(load_watchlist_doc)


@router.get("/events")
async def get_events(limit: int = Query(50, ge=1, le=500)):
    from utils.hl_short_term import load_events

    events = await run_in_threadpool(lambda: load_events(limit=limit))
    return {"ok": True, "events": events, "count": len(events)}


@router.get("/board")
async def get_board(
    refresh: bool = Query(False, description="true 时强制拉取 Hyperliquid（受冷却限制）"),
):
    from utils.hl_short_term import build_board, load_board

    if not refresh:
        return await run_in_threadpool(lambda: build_board(refresh=False))

    allowed, wait = _refresh_cooldown.check_allow()
    if not allowed:
        snap = await run_in_threadpool(load_board)
        if snap:
            out = dict(snap)
            out["snapshot_source"] = "cache"
            out["refresh_skipped"] = True
            out["retry_after_sec"] = round(wait, 1)
            return out
        raise HTTPException(
            status_code=429,
            detail=f"refresh cooldown, retry in {wait:.0f}s",
        )

    if not _refresh_lock.acquire(blocking=False):
        snap = await run_in_threadpool(load_board)
        if snap:
            out = dict(snap)
            out["snapshot_source"] = "cache"
            out["refresh_skipped"] = True
            out["note"] = "refresh already in progress"
            return out
        raise HTTPException(status_code=409, detail="refresh in progress")

    try:
        board = await run_in_threadpool(lambda: build_board(refresh=True))
        _refresh_cooldown.mark_used()
        board = dict(board)
        board["snapshot_source"] = "live"
        return board
    except Exception as exc:
        logger.exception("hl-short board refresh failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        _refresh_lock.release()


@router.get("/paper")
async def get_paper(mark: bool = Query(False, description="true 时尝试刷新浮盈（有冷却）")):
    """Simulated ledger. Default: cached book only; mark=true is rate-limited."""
    from utils.hl_paper_copy import load_paper, refresh_marks

    if mark:
        return await run_in_threadpool(lambda: refresh_marks(force=False))
    return await run_in_threadpool(load_paper)


@router.post("/paper/reset")
async def reset_paper_ledger():
    from utils.hl_paper_copy import reset_paper

    return await run_in_threadpool(reset_paper)


@router.get("/copy/status")
async def get_copy_status():
    from utils.hl_bitget_executor import status as bitget_live_status
    from utils.hl_copy_supervisor import hl_copy_supervisor

    return {"ok": True, **hl_copy_supervisor.status, "bitget_live": bitget_live_status()}


@router.get("/live/status")
async def get_hl_bitget_live_status():
    """HL → Bitget executor flags (vnpy Bitget REST)."""
    from utils.hl_bitget_executor import status as bitget_live_status

    return {"ok": True, **bitget_live_status()}


_screen_lock = threading.Lock()
_screen_cooldown = MinIntervalGuard("HL_WR_SCREEN_COOLDOWN_SEC", 600.0)


@router.get("/screen")
async def get_wr_screen(
    refresh: bool = Query(False, description="true 时强制重跑日筛（冷却约 10 分钟）"),
):
    """短线高胜率日筛看板（默认读缓存；每日 cron 写入）。"""
    from utils.hl_wr_screen import get_board, load_board

    if not refresh:
        return await run_in_threadpool(lambda: get_board(refresh=False))

    allowed, wait = _screen_cooldown.check_allow()
    if not allowed:
        snap = await run_in_threadpool(load_board)
        if snap:
            out = dict(snap)
            out["snapshot_source"] = "cache"
            out["refresh_skipped"] = True
            out["retry_after_sec"] = round(wait, 1)
            return out
        raise HTTPException(
            status_code=429,
            detail=f"screen cooldown, retry in {wait:.0f}s",
        )

    if not _screen_lock.acquire(blocking=False):
        snap = await run_in_threadpool(load_board)
        if snap:
            out = dict(snap)
            out["snapshot_source"] = "cache"
            out["refresh_skipped"] = True
            out["note"] = "screen already in progress"
            return out
        raise HTTPException(status_code=409, detail="screen in progress")

    try:
        board = await run_in_threadpool(lambda: get_board(refresh=True))
        _screen_cooldown.mark_used()
        return board
    except Exception as exc:
        logger.exception("hl wr screen refresh failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        _screen_lock.release()
