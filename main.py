"""
Next K API — OI 雷达、收筹看盘、ZCT VWAP 信号（无 Kronos 预测模块）。
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from env_loader import load_env_oi
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app_state import state
import scheduler_config as sched_cfg
from routers import accumulation as accumulation_router
from routers import core as core_router
from routers import maintenance as maintenance_router
from routers import radar as radar_router
from routers import s2_s6 as s2_s6_router
from routers import vp_regime as vp_regime_router
from routers import jiezhen as jiezhen_router
from routers import momentum as momentum_router
from routers import zct as zct_router
import worker_tasks as wt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

load_env_oi()

NEXT_K_EMBED_SCHEDULER = os.getenv("NEXT_K_EMBED_SCHEDULER", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from datetime import datetime, timezone

    logger.info("Starting Next K API...")
    state.startup_time = datetime.now(timezone.utc)

    try:
        import ccxt

        state.ccxt_exchange = ccxt.binance(
            {"enableRateLimit": True, "timeout": 15000}
        )
        await asyncio.get_running_loop().run_in_executor(
            None, state.ccxt_exchange.load_markets
        )
        logger.info("Binance (crypto) connection established")
    except Exception as e:
        logger.warning("Binance connection failed: %s", e)
        state.ccxt_exchange = None

    try:
        import yfinance as yf  # noqa: F401

        state.yfinance_available = True
        logger.info("yfinance (stocks/forex) available")
    except ImportError:
        logger.warning("yfinance not available")
        state.yfinance_available = False

    if NEXT_K_EMBED_SCHEDULER:
        _start_embedded_scheduler(app)
    else:
        logger.info(
            "Embedded scheduler off; run: python scheduler_main.py "
            "(or NEXT_K_EMBED_SCHEDULER=1)"
        )

    yield

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
    description="OI radar, accumulation watchlists, ZCT VWAP signals API.",
    version="2.0.0",
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
if sched_cfg.env_truthy("NEXT_K_RADAR_API_ENABLED"):
    app.include_router(radar_router.router)
app.include_router(accumulation_router.router)
app.include_router(zct_router.router)
app.include_router(momentum_router.router)
app.include_router(jiezhen_router.router)
app.include_router(vp_regime_router.router)
app.include_router(s2_s6_router.router)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
