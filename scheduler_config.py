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

    # IndicatorEdge Just flipped（默认 30 分钟；0=关）
    ie_min = int(os.getenv("NEXT_K_IE_FLIPS_INTERVAL_MIN", "30") or "30")
    if ie_min > 0:
        from datetime import datetime, timezone

        sch.add_job(
            wt.run_indicatoredge_flips_task,
            "interval",
            minutes=max(5, ie_min),
            id="indicatoredge_flips_refresh",
            max_instances=1,
            replace_existing=True,
            next_run_time=datetime.now(timezone.utc),
        )
        logger.info("Registered IndicatorEdge flips refresh every %s min", max(5, ie_min))

    # Trading OS：CVDD / taker / 订单簿快照（默认 30 分钟；0=关）
    tos_min = int(os.getenv("NEXT_K_TRADING_OS_INTERVAL_MIN", "30") or "30")
    if tos_min > 0:
        from datetime import datetime, timezone

        sch.add_job(
            wt.run_trading_os_task,
            "interval",
            minutes=max(10, tos_min),
            id="trading_os_refresh",
            max_instances=1,
            replace_existing=True,
            next_run_time=datetime.now(timezone.utc),
        )
        logger.info("Registered Trading OS refresh every %s min", max(10, tos_min))
