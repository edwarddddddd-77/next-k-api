"""IndicatorEdge best-indicator lookup proxy for the strategy desk."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from utils.indicatoredge import lookup_best_indicator

router = APIRouter(prefix="/api/indicatoredge", tags=["indicatoredge"])


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
