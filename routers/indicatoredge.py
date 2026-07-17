"""IndicatorEdge best-indicator lookup proxy for the strategy desk."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from utils.indicatoredge import lookup_best_indicator, load_screener_flips, refresh_screener_flips
from utils.rate_limit import MinIntervalGuard

router = APIRouter(prefix="/api/indicatoredge", tags=["indicatoredge"])

_flips_cooldown = MinIntervalGuard("IE_FLIPS_REFRESH_COOLDOWN_SEC", 90.0)


@router.get("/best")
async def best_indicator(
    symbol: str = Query(..., min_length=1, max_length=32, description="e.g. SOL, ETH, BTCUSDT"),
):
    try:
        return lookup_best_indicator(symbol)
    except ValueError:
        raise HTTPException(status_code=400, detail="symbol_required") from None
    except LookupError:
        raise HTTPException(
            status_code=404,
            detail="asset_not_found",
        ) from None
    except RuntimeError as e:
        msg = str(e)
        if msg.startswith("upstream_"):
            raise HTTPException(status_code=502, detail=msg) from e
        raise HTTPException(status_code=502, detail=msg) from e


@router.get("/flips")
async def screener_just_flipped(
    refresh: bool = Query(False, description="true 时强制重拉 Screener（受冷却限制）"),
):
    """IndicatorEdge Screener「Just flipped」最新信号翻转列表。"""
    try:
        if not refresh:
            return load_screener_flips()
        allowed, wait = _flips_cooldown.check_allow()
        if not allowed:
            raise HTTPException(status_code=429, detail=f"refresh_cooldown:{wait:.0f}s")
        out = refresh_screener_flips(force=True)
        _flips_cooldown.mark_used()
        return out
    except HTTPException:
        raise
    except RuntimeError as e:
        msg = str(e)
        if msg.startswith("upstream_"):
            raise HTTPException(status_code=502, detail=msg) from e
        raise HTTPException(status_code=502, detail=msg) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"flips_failed:{e}") from e
