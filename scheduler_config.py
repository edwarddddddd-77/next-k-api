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
    # Alpha 筹码：默认每 3 分钟自动对比 Top 持仓（NEXT_K_ALPHA_HOLDERS_INTERVAL_MIN 可改；0=关）
    alpha_min = int(os.getenv("NEXT_K_ALPHA_HOLDERS_INTERVAL_MIN", "3") or "3")
    if alpha_min > 0:
        from datetime import datetime, timezone

        sch.add_job(
            wt.run_alpha_holders_refresh_task,
            "interval",
            minutes=max(1, alpha_min),
            id="alpha_holders_refresh",
            max_instances=1,
            replace_existing=True,
            next_run_time=datetime.now(timezone.utc),
        )
        logger.info("Registered alpha holders refresh every %s min", max(1, alpha_min))

    # 跨所费率/价差警报（默认 2 分钟；0=关）
    xarb_min = int(os.getenv("NEXT_K_XARB_INTERVAL_MIN", "2") or "2")
    if xarb_min > 0:
        from datetime import datetime, timezone

        sch.add_job(
            wt.run_xarb_refresh_task,
            "interval",
            minutes=max(1, xarb_min),
            id="xarb_board_refresh",
            max_instances=1,
            replace_existing=True,
            next_run_time=datetime.now(timezone.utc),
        )
        logger.info("Registered xarb board refresh every %s min", max(1, xarb_min))
