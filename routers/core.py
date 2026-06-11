from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request

from app_state import state
from models.api_models import HealthResponse

router = APIRouter(tags=["core"])


@router.get("/")
async def root():
    return {
        "name": "Next K API",
        "version": "2.0.0",
        "description": "OI / accumulation / ORB API",
        "docs": "/docs",
        "health": "/api/health",
    }


@router.get("/api/health", response_model=HealthResponse)
async def health(request: Request):
    from scheduler_config import embed_scheduler_enabled

    uptime = (
        (datetime.now(timezone.utc) - state.startup_time).total_seconds()
        if state.startup_time
        else 0
    )
    embedded = embed_scheduler_enabled()
    sch = getattr(request.app.state, "accumulation_scheduler", None)
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        uptime=uptime,
        scheduler_embedded=embedded,
        scheduler_running=sch is not None,
    )
