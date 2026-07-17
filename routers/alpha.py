from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, HTTPException, Query

from utils.rate_limit import MinIntervalGuard

logger = logging.getLogger(__name__)

router = APIRouter(tags=["alpha"])

_refresh_lock = threading.Lock()
_holders_lock = threading.Lock()
_refresh_cooldown = MinIntervalGuard("ALPHA_BOARD_REFRESH_COOLDOWN_SEC", 60.0)
_holders_cooldown = MinIntervalGuard("ALPHA_HOLDERS_REFRESH_COOLDOWN_SEC", 90.0)


@router.get("/api/alpha/board")
async def get_alpha_board(
    refresh: bool = Query(False, description="true 时强制刷新行情（受冷却限制）"),
    limit: int = Query(40, ge=5, le=100),
):
    """
    Alpha 筹码策略看板。
    默认读快照；含最近一次 chip_watch（链上持仓）。
    """
    from alpha_radar import build_board, load_snapshot

    if not refresh:
        snap = load_snapshot()
        if snap.get("ok"):
            return snap
        try:
            return build_board(limit=limit, force_refresh=True)
        except Exception as e:
            logger.exception("alpha board build failed")
            raise HTTPException(status_code=502, detail=f"alpha_board_build_failed: {e}") from e

    allowed, wait = _refresh_cooldown.check_allow()
    if not allowed:
        raise HTTPException(status_code=429, detail=f"refresh_cooldown:{wait:.0f}s")
    if not _refresh_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="refresh_in_progress")
    try:
        data = build_board(limit=limit, force_refresh=True)
        _refresh_cooldown.mark_used()
        return data
    except Exception as e:
        logger.exception("alpha board refresh failed")
        raise HTTPException(status_code=502, detail=f"alpha_board_refresh_failed: {e}") from e
    finally:
        _refresh_lock.release()


@router.post("/api/alpha/refresh")
async def post_alpha_refresh(limit: int = Query(40, ge=5, le=100)):
    """强制刷新行情看板（冷却默认 60s）。"""
    from alpha_radar import build_board

    allowed, wait = _refresh_cooldown.check_allow()
    if not allowed:
        raise HTTPException(status_code=429, detail=f"refresh_cooldown:{wait:.0f}s")
    if not _refresh_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="refresh_in_progress")
    try:
        data = build_board(limit=limit, force_refresh=True)
        _refresh_cooldown.mark_used()
        return data
    except Exception as e:
        logger.exception("alpha board refresh failed")
        raise HTTPException(status_code=502, detail=f"alpha_board_refresh_failed: {e}") from e
    finally:
        _refresh_lock.release()


@router.get("/api/alpha/holders")
async def get_alpha_holders(
    refresh: bool = Query(False, description="true 时强制重拉多链 Top 持仓"),
    limit: int = Query(20, ge=5, le=50),
    coingecko_id: str | None = Query(None, description="只拉单个币；默认日历全部"),
):
    """多链 Top 持仓监控。默认读快照；refresh=1 实时拉取。"""
    from alpha_holders import (
        load_watch_snapshot,
        watch_calendar_tokens,
        watch_token_holders,
    )
    from alpha_radar import _load_calendar

    if not refresh and not coingecko_id:
        snap = load_watch_snapshot()
        if snap.get("ok"):
            return snap

    allowed, wait = _holders_cooldown.check_allow()
    if refresh or not load_watch_snapshot().get("ok"):
        if not allowed and refresh:
            raise HTTPException(status_code=429, detail=f"holders_cooldown:{wait:.0f}s")
        if not _holders_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="holders_refresh_in_progress")
        try:
            if coingecko_id:
                cal = _load_calendar()
                meta = next(
                    (x for x in cal if str(x.get("coingecko_id") or "") == coingecko_id),
                    {"symbol": "", "name": ""},
                )
                w = watch_token_holders(
                    coingecko_id.strip(),
                    symbol=str(meta.get("symbol") or ""),
                    name=str(meta.get("name") or ""),
                    limit=limit,
                )
                payload = {
                    "ok": True,
                    "watches": [w],
                    "generated_at_cst": w.get("generated_at_cst"),
                    "snapshot_source": "live",
                }
            else:
                payload = watch_calendar_tokens(_load_calendar(), limit=limit)
                payload["snapshot_source"] = "live"
            _holders_cooldown.mark_used()
            return payload
        except Exception as e:
            logger.exception("alpha holders refresh failed")
            raise HTTPException(status_code=502, detail=f"holders_refresh_failed: {e}") from e
        finally:
            _holders_lock.release()

    snap = load_watch_snapshot()
    if snap.get("ok"):
        return snap
    raise HTTPException(status_code=404, detail="no_holders_snapshot")


@router.post("/api/alpha/holders/refresh")
async def post_alpha_holders_refresh(limit: int = Query(20, ge=5, le=50)):
    """强制刷新日历标的多链持仓（冷却默认 90s，可能需 30–90 秒）。"""
    from alpha_holders import watch_calendar_tokens
    from alpha_radar import _load_calendar

    allowed, wait = _holders_cooldown.check_allow()
    if not allowed:
        raise HTTPException(status_code=429, detail=f"holders_cooldown:{wait:.0f}s")
    if not _holders_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="holders_refresh_in_progress")
    try:
        data = watch_calendar_tokens(_load_calendar(), limit=limit)
        data["snapshot_source"] = "live"
        _holders_cooldown.mark_used()
        return data
    except Exception as e:
        logger.exception("alpha holders refresh failed")
        raise HTTPException(status_code=502, detail=f"holders_refresh_failed: {e}") from e
    finally:
        _holders_lock.release()


@router.get("/api/alpha/calendar")
async def get_alpha_calendar():
    """仅返回上新 / 空投日历（含阶段标注；尽量用快照市场数据）。"""
    from alpha_radar import build_board, enrich_calendar, load_snapshot, _load_calendar

    snap = load_snapshot()
    if snap.get("ok") and isinstance(snap.get("calendar"), list):
        return {
            "ok": True,
            "calendar": snap["calendar"],
            "focus": snap.get("focus"),
            "generated_at_cst": snap.get("generated_at_cst"),
            "snapshot_source": snap.get("snapshot_source", "disk"),
        }
    try:
        data = build_board(force_refresh=True)
        return {
            "ok": True,
            "calendar": data.get("calendar") or [],
            "focus": data.get("focus"),
            "generated_at_cst": data.get("generated_at_cst"),
            "snapshot_source": "live",
        }
    except Exception:
        cal = enrich_calendar(_load_calendar(), {})
        return {"ok": True, "calendar": cal, "focus": None, "snapshot_source": "calendar_only"}
