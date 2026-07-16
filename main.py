"""Next K API — OI 雷达、收筹看盘、Trading ORB vnpy API。"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager

from env_loader import load_env_oi

load_env_oi()

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app_state import state
import scheduler_config as sched_cfg
from scheduler_config import embed_scheduler_enabled
from routers import accumulation as accumulation_router
from routers import core as core_router
from routers import indicatoredge as indicatoredge_router
from routers import maintenance as maintenance_router
from routers import strategies as strategies_router
from routers import strategy_signals as strategy_signals_router
from routers import trading_orb as trading_orb_router
import worker_tasks as wt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from datetime import datetime, timezone

    logger.info("Starting Next K API...")
    state.startup_time = datetime.now(timezone.utc)

    if embed_scheduler_enabled():
        _start_embedded_scheduler(app)
    else:
        logger.info(
            "Embedded scheduler off (NEXT_K_EMBED_SCHEDULER=0); "
            "run: python scheduler_main.py"
        )

    try:
        from accumulation_radar import init_db

        conn = init_db()
        conn.close()
    except Exception as e:
        logger.warning("DB init on startup skipped: %s", e)

    try:
        from quant.engine.combined_supervisor import combined_vnpy_supervisor

        if combined_vnpy_supervisor.should_start():
            combined_vnpy_supervisor.start()
    except Exception as e:
        logger.warning("vnpy supervisor startup skipped: %s", e)

    yield

    try:
        from quant.engine.combined_supervisor import combined_vnpy_supervisor

        combined_vnpy_supervisor.stop()
    except Exception as e:
        logger.warning("vnpy supervisor shutdown skipped: %s", e)

    sch = getattr(app.state, "accumulation_scheduler", None)
    if sch is not None:
        sch.shutdown(wait=False)
        app.state.accumulation_scheduler = None
    logger.info("Shutting down...")


def _start_embedded_scheduler(app: FastAPI) -> None:
    import pytz

    tz = pytz.timezone("Asia/Shanghai")
    sch = BackgroundScheduler(timezone=tz)
    sched_cfg.register_scheduled_jobs(sch, wt)
    sch.start()
    app.state.accumulation_scheduler = sch
    logger.info("Embedded APScheduler started (Asia/Shanghai)")


app = FastAPI(
    title="Next K",
    description="OI radar, accumulation watchlists, Trading ORB vnpy.",
    version="2.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(core_router.router)
app.include_router(maintenance_router.router)
app.include_router(accumulation_router.router)
app.include_router(trading_orb_router.router)
app.include_router(strategies_router.router)
app.include_router(strategy_signals_router.router)
app.include_router(indicatoredge_router.router)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
