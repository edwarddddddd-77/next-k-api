"""APScheduler 任务注册与开关（main 内嵌调度 / scheduler_main 共用）。"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def env_truthy(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def embed_scheduler_enabled() -> bool:
    """API 进程内嵌 APScheduler；未设 env 时默认开启。设 NEXT_K_EMBED_SCHEDULER=0 关闭。"""
    return env_truthy("NEXT_K_EMBED_SCHEDULER", default=True)


def register_scheduled_jobs(sch: Any, wt: Any) -> None:
    """向 BackgroundScheduler / BlockingScheduler 注册与 worker_tasks 对齐的 cron。"""
    sch.add_job(wt.run_pool_task, "cron", hour=10, minute=0, id="pool_daily")
    sch.add_job(
        wt.run_heat_watch_refresh_task,
        "cron",
        minute=7,
        id="heat_watch_refresh",
    )
    sch.add_job(wt.run_oi_task, "cron", minute=30, id="oi_hourly")
