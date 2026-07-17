#!/usr/bin/env python3
"""跨所费率 / 价差警报 API。"""

from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, HTTPException, Query

from utils.rate_limit import MinIntervalGuard

logger = logging.getLogger(__name__)

router = APIRouter(tags=["xarb"])

_refresh_lock = threading.Lock()
_refresh_cooldown = MinIntervalGuard("XARB_REFRESH_COOLDOWN_SEC", 45.0)


@router.get("/api/xarb/board")
async def get_xarb_board(
    refresh: bool = Query(False, description="true 时强制刷新（受冷却限制）"),
):
    from xarb_radar import build_board, load_snapshot

    if not refresh:
        snap = load_snapshot()
        if snap.get("ok"):
            return snap
        try:
            board = build_board(force_refresh=True)
            try:
                from xarb_paper import auto_manage_from_board

                auto_manage_from_board(board or {})
            except Exception:
                logger.exception("xarb paper auto manage on cold build failed")
            return board
        except Exception as e:
            logger.exception("xarb board build failed")
            raise HTTPException(status_code=502, detail=f"xarb_build_failed: {e}") from e

    allowed, wait = _refresh_cooldown.check_allow()
    if not allowed:
        raise HTTPException(status_code=429, detail=f"refresh_cooldown:{wait:.0f}s")
    if not _refresh_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="refresh_in_progress")
    try:
        board = build_board(force_refresh=True)
        try:
            from xarb_paper import auto_manage_from_board

            auto_manage_from_board(board or {})
        except Exception:
            logger.exception("xarb paper auto manage on refresh failed")
        return board
    except Exception as e:
        logger.exception("xarb refresh failed")
        raise HTTPException(status_code=502, detail=f"xarb_refresh_failed: {e}") from e
    finally:
        _refresh_lock.release()


@router.get("/api/xarb/paper")
async def get_xarb_paper():
    """跨所纸面账本（含手续费表）。"""
    from xarb_paper import list_paper

    return list_paper()


@router.post("/api/xarb/paper/open")
async def post_xarb_paper_open(body: dict):
    """
    开纸面对锁。
    body: base, ex_long, ex_short, long_entry, short_entry, size_usd,
          funding_8h_long?, funding_8h_short?, note?, pair?
    """
    from xarb_paper import list_paper, open_paper

    try:
        fr_l = body.get("funding_8h_long")
        fr_s = body.get("funding_8h_short")
        row = open_paper(
            base=str(body.get("base") or ""),
            ex_long=str(body.get("ex_long") or ""),
            ex_short=str(body.get("ex_short") or ""),
            long_entry=float(body.get("long_entry")),
            short_entry=float(body.get("short_entry")),
            size_usd=float(body.get("size_usd")),
            funding_8h_long=float(fr_l) if fr_l is not None and str(fr_l) != "" else None,
            funding_8h_short=float(fr_s) if fr_s is not None and str(fr_s) != "" else None,
            note=str(body.get("note") or ""),
            pair=str(body.get("pair") or ""),
        )
        return {**row, "book": list_paper()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("xarb paper open failed")
        raise HTTPException(status_code=500, detail=f"xarb_paper_open_failed: {e}") from e


@router.post("/api/xarb/paper/close")
async def post_xarb_paper_close(body: dict):
    """平纸面。body: trade_id, long_exit, short_exit, hours_held?"""
    from xarb_paper import close_paper, list_paper

    try:
        hours = body.get("hours_held")
        row = close_paper(
            trade_id=str(body.get("trade_id") or ""),
            long_exit=float(body.get("long_exit")),
            short_exit=float(body.get("short_exit")),
            hours_held=float(hours) if hours is not None and str(hours) != "" else None,
        )
        return {**row, "book": list_paper()}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("xarb paper close failed")
        raise HTTPException(status_code=500, detail=f"xarb_paper_close_failed: {e}") from e
