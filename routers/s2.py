from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(tags=["s2"])


def _filter_s2_funding_signals_last_days(signals: List[Dict[str, Any]], days: int = 2) -> List[Dict[str, Any]]:
    """Keep entries with recorded_at within last `days` (Asia/Shanghai cutoff)."""
    cst = timezone(timedelta(hours=8))
    cutoff = datetime.now(cst) - timedelta(days=days)
    out: List[Dict[str, Any]] = []
    for row in signals:
        if not isinstance(row, dict):
            continue
        ts = row.get("recorded_at")
        if not ts or not isinstance(ts, str):
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=cst)
            if dt >= cutoff:
                out.append(row)
        except Exception:
            continue
    out.sort(key=lambda r: str(r.get("recorded_at", "")), reverse=True)
    return out


@router.get("/api/s2/funding-signals")
async def get_s2_funding_signals():
    """
    返回近 2 日「费率刚转负 + OI 涨」强信号（与 TG 同源）。
    持久化：accumulation.db 表 s2_funding_signals（原 JSON 由脚本启动时迁移）。
    """
    try:
        from s2_oi_funding_rate_scanner import get_s2_funding_signals_for_api

        return get_s2_funding_signals_for_api(2)
    except Exception as e:
        logger.warning("s2 funding signals read failed: %s", e)
        raise HTTPException(status_code=500, detail="s2_signals_db_error")
