from __future__ import annotations

from datetime import datetime, timezone

import os

from fastapi import APIRouter, Request

from app_state import state
from models.api_models import HealthResponse
import scheduler_config as sched_cfg
from utils.maintenance_auth import maintenance_token_configured

router = APIRouter(tags=["core"])


@router.get("/")
async def root():
    return {
        "name": "Next K API",
        "version": "2.0.0",
        "description": "OI / accumulation / ZCT VWAP API",
        "docs": "/docs",
        "health": "/api/health",
        "zct_vwap_dashboard": "/dashboard/zct-vwap",
        "supertrend_api": "/api/supertrend/signals",
    }


@router.get("/api/health", response_model=HealthResponse)
async def health(request: Request):
    uptime = (
        (datetime.now(timezone.utc) - state.startup_time).total_seconds()
        if state.startup_time
        else 0
    )
    embedded = os.getenv("NEXT_K_EMBED_SCHEDULER", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    sch = getattr(request.app.state, "accumulation_scheduler", None)
    return HealthResponse(
        status="healthy",
        crypto_connected=state.ccxt_exchange is not None,
        stocks_available=state.yfinance_available,
        forex_available=state.yfinance_available,
        version="2.0.0",
        uptime=uptime,
        maintenance_auth_required=maintenance_token_configured(),
        scheduler_embedded=embedded,
        scheduler_running=sch is not None,
        zct_vwap_scheduler_enabled=sched_cfg.ZCT_VWAP_SIGNAL_SCHEDULER_ENABLED,
        st_scheduler_enabled=sched_cfg.ST_SCHEDULER_ENABLED,
    )
