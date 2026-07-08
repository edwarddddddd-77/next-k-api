"""APScheduler 任务注册与开关（main 内嵌调度 / scheduler_main 共用）。"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import pytz

from orb.core.config import default_scan_interval_minutes

logger = logging.getLogger(__name__)

KK_SCAN_CRON_TZ = pytz.UTC
ORB_SCAN_CRON_TZ = KK_SCAN_CRON_TZ


def env_truthy(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def embed_scheduler_enabled() -> bool:
    """API 进程内嵌 APScheduler；未设 env 时默认开启。设 NEXT_K_EMBED_SCHEDULER=0 关闭。"""
    return env_truthy("NEXT_K_EMBED_SCHEDULER", default=True)


def _int_env_scan_interval(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is not None and str(raw).strip():
        try:
            return max(1, int(float(str(raw).strip())))
        except ValueError:
            logging.getLogger(__name__).warning(
                "Invalid %s=%r, using default %s", name, raw, default
            )
    return default


KK_SCAN_INTERVAL_MINUTES = _int_env_scan_interval(
    "KK_SCAN_INTERVAL_MINUTES",
    _int_env_scan_interval("ORB_SCAN_INTERVAL_MINUTES", default_scan_interval_minutes()),
)


def _int_env_kk_scan_cron_second() -> int:
    raw = os.getenv("KK_SCAN_CRON_SECOND") or os.getenv("ORB_SCAN_CRON_SECOND", "5")
    try:
        return max(0, min(59, int(float(str(raw).strip()))))
    except ValueError:
        logger.warning("Invalid KK_SCAN_CRON_SECOND=%r, using 5", raw)
        return 5


KK_SCAN_CRON_SECOND = _int_env_kk_scan_cron_second()


def kk_scan_cron_kwargs(interval_minutes: int, *, second: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Binance K 线 UTC 边界对齐的 cron 参数；interval 须能整除 60。"""
    n = max(1, int(interval_minutes))
    if 60 % n != 0:
        return None
    sec = KK_SCAN_CRON_SECOND if second is None else max(0, min(59, int(second)))
    if n == 1:
        return {"minute": "*", "second": sec, "timezone": KK_SCAN_CRON_TZ}
    return {"minute": f"*/{n}", "second": sec, "timezone": KK_SCAN_CRON_TZ}


# 兼容 ORB V2 调度测试 / 旧引用
orb_scan_cron_kwargs = kk_scan_cron_kwargs


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
