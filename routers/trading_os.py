"""Trading OS automation API — auto snapshot of CVDD / taker / orderbook."""

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

router = APIRouter(prefix="/api/trading-os", tags=["trading-os"])

_refresh_cooldown = MinIntervalGuard("TRADING_OS_REFRESH_COOLDOWN_SEC", 60.0)


class CvddOverrideBody(BaseModel):
    cvdd: float = Field(..., description="手动粘贴的 CVDD 数值，例如 46200")
    date: str = Field("", description="指标日期 YYYY-MM-DD，可空")
    note: str = Field("", description="备注，可空")
    source_label: str = Field("manual", description="来源标记，如 glassnode / bmp")
    refresh: bool = Field(True, description="写入后是否立刻重算快照")


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
