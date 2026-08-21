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
_catch_up_cooldown = MinIntervalGuard("HL_CATCH_UP_COOLDOWN_SEC", 60.0)


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
    """Simulated ledger. Default: cached book only; mark=true is rate-limited.

    Live-only seats (Binance / Bitget, including Railway-enabled Bitget) are
    overlaid with exchange wallet/positions. Other seats keep the paper book.
    """
    from utils.hl_binance_executor import overlay_live_bots as overlay_binance
    from utils.hl_bitget_executor import overlay_live_bots as overlay_bitget
    from utils.hl_paper_copy import load_paper, refresh_marks, slim_paper_for_api

    def _run():
        from utils.hl_paper_copy import flush_pending_live_flatten

        book = refresh_marks(force=False) if mark else load_paper()
        flush_pending_live_flatten()
        try:
            book = overlay_binance(book)
        except Exception:
            logger.exception("binance live overlay failed")
        try:
            book = overlay_bitget(book)
        except Exception:
            logger.exception("bitget live overlay failed")
        return slim_paper_for_api(book)

    return await run_in_threadpool(_run)


@router.post("/paper/reset")
async def reset_paper_ledger():
    from utils.hl_paper_copy import reset_paper

    return await run_in_threadpool(reset_paper)


@router.post("/paper/reset/{bot_id}")
async def reset_paper_bot_ledger(bot_id: str):
    """Reset a single paper seat (positions/fills → initial balance)."""
    from fastapi import HTTPException

    from utils.hl_paper_copy import reset_paper_bot

    bid = str(bot_id or "").strip()
    if not bid:
        raise HTTPException(status_code=400, detail="bot_id required")
    try:
        return await run_in_threadpool(reset_paper_bot, bid)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/copy/status")
async def get_copy_status():
    from utils.hl_binance_executor import status as binance_live_status
    from utils.hl_bitget_executor import status as bitget_live_status
    from utils.hl_copy_supervisor import hl_copy_supervisor

    return {
        "ok": True,
        **hl_copy_supervisor.status,
        "bitget_live": bitget_live_status(),
        "binance_live": binance_live_status(),
    }


@router.get("/live/status")
async def get_hl_bitget_live_status():
    """HL → Bitget executor flags (vnpy Bitget REST)."""
    from utils.hl_bitget_executor import status as bitget_live_status

    return {"ok": True, **bitget_live_status()}


@router.post("/live/catch-up")
async def post_hl_bitget_catch_up(
    bot_id: str = Query(..., description="live seat id, e.g. bot_c"),
    coins: str = Query(
        ...,
        description="comma-separated HL coins to open (required, e.g. xyz:GOOGL)",
    ),
    refresh: bool = Query(True, description="refresh leader snapshot before sizing"),
):
    """Manual one-shot open of missed mid-book legs.

    No token. Requires explicit ``coins``. Does not resize already-held legs;
    does not write pending_fresh. Cooldown: ``HL_CATCH_UP_COOLDOWN_SEC``.
    """
    from utils.hl_bitget_executor import catch_up_orphan_coins

    allowed, wait = _catch_up_cooldown.check_allow()
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"catch_up cooldown, retry in {wait:.0f}s",
        )

    coin_list = [c.strip() for c in str(coins or "").split(",") if c.strip()]
    if not coin_list:
        raise HTTPException(status_code=400, detail="coins_required")

    out = await run_in_threadpool(
        lambda: catch_up_orphan_coins(
            str(bot_id or "").strip(),
            coin_list,
            refresh_target=bool(refresh),
        )
    )
    if not out.get("ok"):
        raise HTTPException(status_code=400, detail=out.get("error") or "catch_up_failed")
    _catch_up_cooldown.mark_used()
    return out


@router.get("/live/binance/status")
async def get_hl_binance_live_status():
    """HL → Binance executor flags (vnpy Binance USDT-M REST, same BINANCE_API_* as ORB)."""
    from utils.hl_binance_executor import status as binance_live_status

    return {"ok": True, **binance_live_status()}


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
