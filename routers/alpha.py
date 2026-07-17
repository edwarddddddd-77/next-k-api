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


def _run_holders_refresh(*, limit: int, coingecko_id: str | None = None) -> dict:
    from alpha_holders import (
        _phase_from_item,
        load_watch_snapshot,
        save_watch_snapshot,
        watch_calendar_tokens,
        watch_token_holders,
    )
    from alpha_radar import _load_calendar, patch_board_snapshot_chip_watch

    if coingecko_id:
        cid = coingecko_id.strip()
        cal = _load_calendar()
        meta = next(
            (x for x in cal if str(x.get("coingecko_id") or "") == cid),
            {"symbol": "", "name": "", "start_at_cst": ""},
        )
        w = watch_token_holders(
            cid,
            symbol=str(meta.get("symbol") or ""),
            name=str(meta.get("name") or ""),
            limit=limit,
            phase=_phase_from_item(meta),
            calendar_item=meta if isinstance(meta, dict) else None,
        )
        # 合并进全量 watch 快照，避免单币刷新冲掉其它标的
        snap = load_watch_snapshot()
        watches = list(snap.get("watches") or []) if snap.get("ok") else []
        watches = [x for x in watches if str(x.get("coingecko_id") or "") != cid]
        watches.append(w)
        payload = {
            "ok": True,
            "watches": watches,
            "generated_at_cst": w.get("generated_at_cst"),
            "snapshot_source": "live",
            "chains_supported": snap.get("chains_supported") if snap.get("ok") else None,
        }
        try:
            save_watch_snapshot(payload)
        except Exception:
            logger.exception("save merged holders watch failed")
        # 单币刷新也写入该期历史
        try:
            from alpha_history import record_from_watch_payload

            record_from_watch_payload([w], [meta])
        except Exception:
            logger.exception("alpha history single record failed")
    else:
        payload = watch_calendar_tokens(_load_calendar(), limit=limit)
        payload["snapshot_source"] = "live"

    try:
        patch_board_snapshot_chip_watch()
    except Exception:
        logger.exception("patch board after holders refresh failed")
    return payload


@router.get("/api/alpha/board")
async def get_alpha_board(
    refresh: bool = Query(False, description="true 时强制刷新行情（受冷却限制）"),
    limit: int = Query(40, ge=5, le=100),
):
    """Alpha 筹码策略看板。默认读快照；响应会挂上最新 chip_watch。"""
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
    """
    多链 Top 持仓监控。
    - 默认读快照
    - 无快照时自动拉一次全量（冷启动）
    - refresh=1 强制重拉（受冷却限制）
    - 仅 coingecko_id：优先从快照过滤；没有该币且 refresh=1 才单拉
    """
    from alpha_holders import load_watch_snapshot

    cid = (coingecko_id or "").strip() or None
    snap = load_watch_snapshot()

    if not refresh:
        if snap.get("ok"):
            if cid:
                matched = [
                    w
                    for w in (snap.get("watches") or [])
                    if str(w.get("coingecko_id") or "") == cid
                ]
                if matched:
                    return {
                        "ok": True,
                        "watches": matched,
                        "generated_at_cst": snap.get("generated_at_cst"),
                        "snapshot_source": snap.get("snapshot_source", "disk"),
                        "filtered": True,
                    }
                # 快照里没有该币：404，避免静默返回全表
                raise HTTPException(status_code=404, detail=f"coin_not_in_snapshot:{cid}")
            return snap
        # 无快照 → 冷启动拉全量
        if not _holders_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="holders_refresh_in_progress")
        try:
            payload = _run_holders_refresh(limit=limit, coingecko_id=None)
            _holders_cooldown.mark_used()
            return payload
        except Exception as e:
            logger.exception("alpha holders cold start failed")
            raise HTTPException(status_code=502, detail=f"holders_refresh_failed: {e}") from e
        finally:
            _holders_lock.release()

    # refresh=true
    allowed, wait = _holders_cooldown.check_allow()
    if not allowed:
        raise HTTPException(status_code=429, detail=f"holders_cooldown:{wait:.0f}s")
    if not _holders_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="holders_refresh_in_progress")
    try:
        payload = _run_holders_refresh(limit=limit, coingecko_id=cid)
        _holders_cooldown.mark_used()
        return payload
    except Exception as e:
        logger.exception("alpha holders refresh failed")
        raise HTTPException(status_code=502, detail=f"holders_refresh_failed: {e}") from e
    finally:
        _holders_lock.release()


@router.post("/api/alpha/holders/refresh")
async def post_alpha_holders_refresh(limit: int = Query(20, ge=5, le=50)):
    """强制刷新日历标的多链持仓（冷却默认 90s，可能需 30–90 秒）。"""
    allowed, wait = _holders_cooldown.check_allow()
    if not allowed:
        raise HTTPException(status_code=429, detail=f"holders_cooldown:{wait:.0f}s")
    if not _holders_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="holders_refresh_in_progress")
    try:
        data = _run_holders_refresh(limit=limit, coingecko_id=None)
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


@router.get("/api/alpha/providers")
async def get_alpha_providers():
    """数据源密钥状态（不回传密钥本身；在 Railway Variables 配置）。"""
    from alpha_coingecko import alpha_providers_status

    return {"ok": True, "providers": alpha_providers_status()}


@router.get("/api/alpha/history")
async def get_alpha_history(limit: int = Query(40, ge=1, le=200)):
    """每期总结历史（默认保留 180 天）。"""
    from alpha_history import list_history

    try:
        return list_history(limit=limit)
    except Exception as e:
        logger.exception("alpha history list failed")
        raise HTTPException(status_code=500, detail=f"history_failed: {e}") from e


@router.get("/api/alpha/history/{period_id:path}")
async def get_alpha_history_period(period_id: str):
    """单期详情（含多次快照总结）。"""
    from alpha_history import get_period

    # FastAPI path 可能把 | 编码；兼容
    pid = period_id.replace("%7C", "|")
    row = get_period(pid)
    if not row:
        raise HTTPException(status_code=404, detail="period_not_found")
    return {"ok": True, "period": row}
