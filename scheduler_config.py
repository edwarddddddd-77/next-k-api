"""APScheduler 任务注册与开关（main 内嵌调度 / scheduler_main 共用）。"""

from __future__ import annotations

import logging
import os
from typing import Any

from apscheduler.triggers.interval import IntervalTrigger

from orb.config import default_scan_interval_minutes


def env_truthy(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def embed_scheduler_enabled() -> bool:
    """API 进程内嵌 APScheduler；未设 env 时默认开启。设 NEXT_K_EMBED_SCHEDULER=0 关闭。"""
    return env_truthy("NEXT_K_EMBED_SCHEDULER", default=True)


ORB_SCHEDULER_ENABLED = env_truthy("ORB_SCHEDULER_ENABLED", default=True)


def _int_env_orb_scan_interval() -> int:
    raw = os.getenv("ORB_SCAN_INTERVAL_MINUTES")
    if raw is not None and str(raw).strip():
        try:
            return int(float(str(raw).strip()))
        except ValueError:
            logging.getLogger(__name__).warning(
                "Invalid ORB_SCAN_INTERVAL_MINUTES=%r, using default", raw
            )
    return default_scan_interval_minutes()


ORB_SCAN_INTERVAL_MINUTES = max(1, _int_env_orb_scan_interval())


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
    sch.add_job(wt.run_s2_oi_funding_task, "cron", minute=5, id="s2_oi_funding")
    if ORB_SCHEDULER_ENABLED:
        sch.add_job(
            wt.run_orb_scan_task,
            IntervalTrigger(minutes=ORB_SCAN_INTERVAL_MINUTES),
            id="orb_scanner",
        )
    try:
        from top_trader_config import top_trader_scheduler_enabled
    except ImportError:
        def top_trader_scheduler_enabled() -> bool:  # type: ignore[misc]
            return env_truthy("TOP_TRADER_SCHEDULER_ENABLED")

    if top_trader_scheduler_enabled():
        try:
            top_trader_minute = max(0, min(59, int(os.getenv("TOP_TRADER_CRON_MINUTE", "45") or 45)))
        except ValueError:
            top_trader_minute = 45
        sch.add_job(
            wt.run_top_trader_radar_task,
            "cron",
            minute=top_trader_minute,
            id="top_trader_radar",
            replace_existing=True,
        )
