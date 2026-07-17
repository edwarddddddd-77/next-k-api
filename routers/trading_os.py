"""Trading OS automation API — cycle desk + wallets/alts/risk/alerts."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from utils.rate_limit import MinIntervalGuard
from utils.trading_os import (
    clear_cvdd_override,
    load_cvdd_override,
    load_journal,
    load_snapshot,
    refresh_snapshot,
    set_cvdd_override,
)
from utils.trading_os_desk import (
    add_wallet,
    compute_risk,
    list_alts,
    list_wallets,
    load_alts_snap,
    load_desk_bundle,
    load_wallets_snap,
    monitor_status,
    refresh_alts,
    refresh_desk_bundle,
    refresh_wallets,
    remove_wallet,
    set_alts,
)

router = APIRouter(prefix="/api/trading-os", tags=["trading-os"])

_refresh_cooldown = MinIntervalGuard("TRADING_OS_REFRESH_COOLDOWN_SEC", 60.0)
_desk_cooldown = MinIntervalGuard("TRADING_OS_DESK_REFRESH_COOLDOWN_SEC", 45.0)


class CvddOverrideBody(BaseModel):
    cvdd: float = Field(..., description="手动粘贴的 CVDD 数值，例如 46200")
    date: str = Field("", description="指标日期 YYYY-MM-DD，可空")
    note: str = Field("", description="备注，可空")
    source_label: str = Field("manual", description="来源标记")
    refresh: bool = Field(True, description="写入后是否立刻重算快照")


class WalletBody(BaseModel):
    address: str
    label: str = ""
    chain: str = ""  # btc | eth | auto


class AltsBody(BaseModel):
    symbols: list[str] = Field(default_factory=list)


@router.get("/snapshot")
async def trading_os_snapshot(
    refresh: bool = Query(False, description="true 时强制重拉（受冷却限制）"),
):
    try:
        if not refresh:
            return await run_in_threadpool(load_snapshot)
        allowed, wait = _refresh_cooldown.check_allow()
        if not allowed:
            raise HTTPException(status_code=429, detail=f"refresh_cooldown:{wait:.0f}s")
        out = await run_in_threadpool(lambda: refresh_snapshot(force=True))
        _refresh_cooldown.mark_used()
        return out
    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"trading_os_failed:{e}") from e


@router.get("/journal")
async def trading_os_journal(limit: int = Query(30, ge=1, le=200)):
    try:
        return await run_in_threadpool(lambda: load_journal(limit=limit))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"journal_failed:{e}") from e


@router.get("/cvdd-override")
async def get_cvdd_override():
    ov = await run_in_threadpool(load_cvdd_override)
    return {"ok": True, "override": ov}


@router.post("/cvdd-override")
async def post_cvdd_override(body: CvddOverrideBody):
    try:
        ov = await run_in_threadpool(
            lambda: set_cvdd_override(
                body.cvdd,
                date=body.date,
                note=body.note,
                source_label=body.source_label,
            )
        )
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="cvdd_out_of_range (expect 1000 < cvdd <= 1000000)",
        ) from None

    snap = None
    if body.refresh:
        try:
            snap = await run_in_threadpool(lambda: refresh_snapshot(force=True))
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"override_saved_but_refresh_failed:{e}",
            ) from e
    return {"ok": True, "override": ov, "snapshot": snap}


@router.delete("/cvdd-override")
async def delete_cvdd_override(refresh: bool = Query(True)):
    await run_in_threadpool(clear_cvdd_override)
    snap = None
    if refresh:
        try:
            snap = await run_in_threadpool(lambda: refresh_snapshot(force=True))
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"cleared_but_refresh_failed:{e}",
            ) from e
    return {"ok": True, "cleared": True, "snapshot": snap}


@router.get("/monitor")
async def trading_os_monitor():
    """全自动监控状态（调度间隔 / 上次运行 / TG）。"""
    return await run_in_threadpool(monitor_status)


@router.get("/desk")
async def trading_os_desk(refresh: bool = Query(False)):
    try:
        if not refresh:
            return await run_in_threadpool(load_desk_bundle)
        allowed, wait = _desk_cooldown.check_allow()
        if not allowed:
            raise HTTPException(status_code=429, detail=f"refresh_cooldown:{wait:.0f}s")
        out = await run_in_threadpool(refresh_desk_bundle)
        _desk_cooldown.mark_used()
        return out
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"desk_failed:{e}") from e


@router.post("/wallets/seed-defaults")
async def seed_wallets(force: bool = Query(True, description="重新从CEX提币发现吸筹地址")):
    from utils.trading_os_desk import ensure_smart_watchlist

    return await run_in_threadpool(lambda: ensure_smart_watchlist(force=force))


@router.get("/wallets/discover")
async def wallets_discover():
    from utils.trading_os_desk import discover_accumulators

    return await run_in_threadpool(discover_accumulators)


@router.get("/wallets")
async def get_wallets():
    return await run_in_threadpool(list_wallets)


@router.post("/wallets")
async def post_wallet(body: WalletBody):
    try:
        return await run_in_threadpool(
            lambda: add_wallet(body.address, label=body.label, chain=body.chain)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.delete("/wallets")
async def delete_wallet(address: str = Query(...)):
    return await run_in_threadpool(lambda: remove_wallet(address))


@router.get("/wallets/status")
async def wallets_status(refresh: bool = Query(False)):
    if refresh:
        return await run_in_threadpool(refresh_wallets)
    return await run_in_threadpool(load_wallets_snap)


@router.get("/alts")
async def get_alts():
    return await run_in_threadpool(list_alts)


@router.put("/alts")
async def put_alts(body: AltsBody):
    return await run_in_threadpool(lambda: set_alts(body.symbols))


@router.get("/alts/status")
async def alts_status(refresh: bool = Query(False)):
    if refresh:
        return await run_in_threadpool(refresh_alts)
    return await run_in_threadpool(load_alts_snap)


@router.get("/risk")
async def get_risk(equity_usd: float | None = Query(None)):
    return await run_in_threadpool(lambda: compute_risk(equity_usd))
