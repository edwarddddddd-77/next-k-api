"""API: Barra sector-factor board."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from utils.rate_limit import MinIntervalGuard

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/factor-sector", tags=["factor-sector"])
_refresh_cooldown = MinIntervalGuard("FACTOR_SECTOR_REFRESH_COOLDOWN_SEC", 90.0)


@router.get("/board")
async def factor_sector_board(
    refresh: bool = Query(False, description="true 时强制重算（受冷却限制）"),
):
    from factor_sector import load_snapshot, refresh_snapshot

    try:
        if not refresh:
            snap = await run_in_threadpool(load_snapshot)
            if snap.get("ok"):
                return snap
            # cold start: build once
            return await run_in_threadpool(refresh_snapshot)

        allowed, wait = _refresh_cooldown.check_allow()
        if not allowed:
            raise HTTPException(status_code=429, detail=f"refresh_cooldown:{wait:.0f}s")
        out = await run_in_threadpool(refresh_snapshot)
        _refresh_cooldown.mark_used()
        return out
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("factor-sector board failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/health")
async def factor_sector_health():
    from factor_sector import load_snapshot

    snap = load_snapshot()
    return {
        "ok": True,
        "has_snapshot": bool(snap.get("ok")),
        "age_sec": snap.get("age_sec"),
        "latest_date": (snap.get("latest") or {}).get("date"),
    }
