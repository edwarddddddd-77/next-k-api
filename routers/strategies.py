"""量化策略开关 API。"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from quant.engine.registry import list_strategy_switch_status

router = APIRouter(prefix="/api/strategies", tags=["strategies"])


@router.get("/switches")
async def strategy_switches() -> Dict[str, Any]:
    """列出所有已注册策略及其开关状态。"""
    return list_strategy_switch_status()
