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


_candidates_lock = threading.Lock()
_candidates_cooldown = MinIntervalGuard("HL_CANDIDATES_COOLDOWN_SEC", 900.0)


@router.get("/candidates")
async def get_desk_candidates(
    refresh: bool = Query(False, description="true 时强制重建候选池（冷却约 15 分钟）"),
):
    """跟单候选池：ready=可绑 / watch=过门槛但不够活 / bound=当前 A–J。"""
    from utils.hl_desk_candidates import get_candidates, load_candidates

    if not refresh:
        return await run_in_threadpool(lambda: get_candidates(refresh=False))

    allowed, wait = _candidates_cooldown.check_allow()
    if not allowed:
        snap = await run_in_threadpool(load_candidates)
        if snap:
            out = dict(snap)
            out["snapshot_source"] = "cache"
            out["refresh_skipped"] = True
            out["retry_after_sec"] = round(wait, 1)
            return out
        raise HTTPException(
            status_code=429,
            detail=f"candidates cooldown, retry in {wait:.0f}s",
        )

    if not _candidates_lock.acquire(blocking=False):
        snap = await run_in_threadpool(load_candidates)
        if snap:
            out = dict(snap)
            out["snapshot_source"] = "cache"
            out["refresh_skipped"] = True
            out["note"] = "candidates build already in progress"
            return out
        raise HTTPException(status_code=409, detail="candidates build in progress")

    try:
        board = await run_in_threadpool(lambda: get_candidates(refresh=True))
        _candidates_cooldown.mark_used()
        out = dict(board)
        out["snapshot_source"] = "live"
        return out
    except Exception as exc:
        logger.exception("hl desk candidates refresh failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        _candidates_lock.release()


@router.get("/f-mr")
async def get_avax_f_mr(
    coin: str = Query("AVAX", description="HL coin, default AVAX (desk F style)"),
    backtest: bool = Query(True, description="include 90d bar backtest summary"),
    mode: str = Query("trade", description="trade=balanced v2 · gate=stricter F mirror filter"),
):
    """Desk-F style 4h fade + RSI + 24h no-chase + 3d extreme 做单指标（研究用）。"""
    from utils.avax_f_mr_indicator import snapshot

    mode_s = str(mode or "trade").strip().lower()
    if mode_s not in ("trade", "gate"):
        raise HTTPException(status_code=400, detail="mode must be trade|gate")
    try:
        return await run_in_threadpool(
            lambda: snapshot(
                coin=str(coin or "AVAX").upper(),
                with_backtest=backtest,
                mode=mode_s,  # type: ignore[arg-type]
            )
        )
    except Exception as exc:
        logger.exception("f-mr indicator failed")
        raise HTTPException(status_code=502, detail=str(exc)) from exc
