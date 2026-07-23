"""策略开仓信号 API — 仅 AVAX F-MR。"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query
from starlette.concurrency import run_in_threadpool

from quant.engine.strategy_signals import LANE_AVAX_F_MR, list_strategy_signals

router = APIRouter(prefix="/api/strategy", tags=["strategy"])


@router.get("/signals")
async def strategy_signals(
    lane: str = Query(LANE_AVAX_F_MR, description="only avax_f_mr"),
    limit: int = Query(100, ge=1, le=500),
) -> Dict[str, Any]:
    lane_s = str(lane or LANE_AVAX_F_MR).strip() or LANE_AVAX_F_MR
    if lane_s != LANE_AVAX_F_MR:
        raise HTTPException(status_code=400, detail="only lane=avax_f_mr is supported")
    try:
        return await run_in_threadpool(list_strategy_signals, lane=LANE_AVAX_F_MR, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"strategy_signals_error: {exc}") from exc
