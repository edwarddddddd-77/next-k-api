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
            return build_board(force_refresh=True)
        except Exception as e:
            logger.exception("xarb board build failed")
            raise HTTPException(status_code=502, detail=f"xarb_build_failed: {e}") from e

    allowed, wait = _refresh_cooldown.check_allow()
    if not allowed:
        raise HTTPException(status_code=429, detail=f"refresh_cooldown:{wait:.0f}s")
    if not _refresh_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="refresh_in_progress")
    try:
        return build_board(force_refresh=True)
    except Exception as e:
        logger.exception("xarb refresh failed")
        raise HTTPException(status_code=502, detail=f"xarb_refresh_failed: {e}") from e
    finally:
        _refresh_lock.release()
