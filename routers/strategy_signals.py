"""策略发出信号 API（Trading ORB / ICT 2022 / Aberration）。"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from orb.vnpy.strategy_signals import (
    LANE_ABERRATION,
    LANE_ICT_2022,
    LANE_SKEW_NEUTRAL,
    LANE_TRADING_ORB,
    VALID_LANES,
    list_strategy_signals,
)

router = APIRouter(prefix="/api/strategy", tags=["strategy"])


@router.get("/signals")
async def strategy_signals(
    lane: str = Query(..., description="trading_orb、ict_2022、aberration 或 skew_neutral"),
    limit: int = Query(100, ge=1, le=500),
) -> Dict[str, Any]:
    lane_s = str(lane or "").strip()
    if lane_s not in VALID_LANES:
        raise HTTPException(
            status_code=400,
            detail=f"invalid_lane: use {LANE_TRADING_ORB}, {LANE_ICT_2022}, {LANE_ABERRATION}, or {LANE_SKEW_NEUTRAL}",
        )
    try:
        return await run_in_threadpool(list_strategy_signals, lane=lane_s, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"strategy_signals_error: {exc}") from exc
