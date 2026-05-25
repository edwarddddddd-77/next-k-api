"""APScheduler 任务注册与开关（main 内嵌调度 / scheduler_main 共用）。"""

from __future__ import annotations

import os
from typing import Any

from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger


def env_truthy(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


S6_FUTURES_ALPHA_SCHEDULER_ENABLED = env_truthy("S6_FUTURES_ALPHA_SCHEDULER_ENABLED")
ZCT_VWAP_SIGNAL_SCHEDULER_ENABLED = env_truthy("ZCT_VWAP_SIGNAL_SCHEDULER_ENABLED")

ZCT_VWAP_SCAN_INTERVAL_MINUTES = max(
    1, int(os.getenv("ZCT_VWAP_SCAN_INTERVAL_MINUTES", "7") or 7)
)
ZCT_VWAP_RESOLVE_INTERVAL_MINUTES = max(
    0, int(os.getenv("ZCT_VWAP_RESOLVE_INTERVAL_MINUTES", "5") or 5)
)

MOM_SCHEDULER_ENABLED = env_truthy("MOM_SCHEDULER_ENABLED", default=True)

try:
    from powder_keg_config import POWDER_KEG_CRON_MINUTES, powder_keg_radar_enabled
except ImportError:
    POWDER_KEG_CRON_MINUTES = (2, 17, 32, 47)

    def powder_keg_radar_enabled() -> bool:  # type: ignore[misc]
        return env_truthy("POWDER_KEG_RADAR_ENABLED", default=True)


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
    from touch_pool_config import touch_pool_4h_cron_slots, touch_pool_scheduler_enabled

    if touch_pool_scheduler_enabled():
        for h, m in touch_pool_4h_cron_slots():
            jid = f"zct_touch_pool_4h_{h:02d}{m:02d}"
            sch.add_job(
                wt.run_zct_touch_pool_4h_task,
                "cron",
                hour=h,
                minute=m,
                id=jid,
            )
    if powder_keg_radar_enabled():
        for minute in POWDER_KEG_CRON_MINUTES:
            sch.add_job(
                wt.run_powder_keg_radar_task,
                CronTrigger(minute=int(minute)),
                id=f"powder_keg_radar_{int(minute):02d}",
                replace_existing=True,
            )
    if S6_FUTURES_ALPHA_SCHEDULER_ENABLED:
        sch.add_job(
            wt.run_s6_futures_alpha_task,
            "cron",
            minute=25,
            id="s6_futures_alpha",
        )
    if ZCT_VWAP_SIGNAL_SCHEDULER_ENABLED:
        sch.add_job(
            wt.run_zct_vwap_signal_task,
            IntervalTrigger(minutes=ZCT_VWAP_SCAN_INTERVAL_MINUTES),
            id="zct_vwap_signal_scanner",
        )
        if ZCT_VWAP_RESOLVE_INTERVAL_MINUTES > 0:
            sch.add_job(
                wt.run_zct_vwap_resolve_only_task,
                IntervalTrigger(minutes=ZCT_VWAP_RESOLVE_INTERVAL_MINUTES),
                id="zct_vwap_resolve_only",
            )
    if MOM_SCHEDULER_ENABLED:
        from momentum_config import (
            MOM_SCAN_INTERVAL_MINUTES,
            MOM_TRAIL_SCAN_INTERVAL_SEC,
            mom_trail_scheduler_enabled,
        )

        sch.add_job(
            wt.run_momentum_scan_task,
            IntervalTrigger(minutes=MOM_SCAN_INTERVAL_MINUTES),
            id="momentum_top_movers_scan",
            replace_existing=True,
        )
        if mom_trail_scheduler_enabled():
            sch.add_job(
                wt.run_momentum_trail_task,
                IntervalTrigger(seconds=MOM_TRAIL_SCAN_INTERVAL_SEC),
                id="momentum_trail_scan",
                replace_existing=True,
            )
    if env_truthy("JIEZHEN_SCHEDULER_ENABLED", default=True):
        from jiezhen_config import (
            JIEZHEN_SCAN_INTERVAL_SEC,
            JIEZHEN_TRAIL_SCAN_INTERVAL_SEC,
            jz_trail_scheduler_enabled,
        )

        sch.add_job(
            wt.run_jiezhen_scan_task,
            IntervalTrigger(seconds=JIEZHEN_SCAN_INTERVAL_SEC),
            id="jiezhen_spike_scan",
            replace_existing=True,
        )
        if jz_trail_scheduler_enabled():
            sch.add_job(
                wt.run_jiezhen_trail_task,
                IntervalTrigger(seconds=JIEZHEN_TRAIL_SCAN_INTERVAL_SEC),
                id="jiezhen_trail_scan",
                replace_existing=True,
            )