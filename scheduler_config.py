"""APScheduler 任务注册与开关（main 内嵌调度 / scheduler_main 共用）。"""

from __future__ import annotations

import logging
import os
from typing import Any

from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger


def env_truthy(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def embed_scheduler_enabled() -> bool:
    """API 进程内嵌 APScheduler；未设 env 时默认开启。设 NEXT_K_EMBED_SCHEDULER=0 关闭。"""
    return env_truthy("NEXT_K_EMBED_SCHEDULER", default=True)


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
    try:
        from moss_quant.config import (
            MOSS_QUANT_SCAN_INTERVAL_MINUTES,
            paper_scheduler_enabled,
        )
    except ImportError as e:
        logging.getLogger(__name__).warning(
            "moss_quant scheduler disabled: %s", e
        )
        paper_scheduler_enabled = lambda: False  # type: ignore[misc, assignment]
        MOSS_QUANT_SCAN_INTERVAL_MINUTES = 15

    if paper_scheduler_enabled():
        sch.add_job(
            wt.run_moss_quant_paper_task,
            IntervalTrigger(minutes=MOSS_QUANT_SCAN_INTERVAL_MINUTES),
            id="moss_quant_paper_scan",
            replace_existing=True,
        )
    try:
        from moss_quant.config import (
            daily_optimize_scheduler_enabled,
            parse_daily_optimize_utc,
        )
    except ImportError:
        daily_optimize_scheduler_enabled = lambda: False  # type: ignore[misc, assignment]
        parse_daily_optimize_utc = lambda: (6, 30)  # type: ignore[misc, assignment]

    try:
        from moss_quant.config import daily_optimize_bootstrap_enabled
    except ImportError:
        daily_optimize_bootstrap_enabled = lambda: False  # type: ignore[misc, assignment]

    if daily_optimize_scheduler_enabled():
        dh, dm = parse_daily_optimize_utc()
        sch.add_job(
            wt.run_moss_daily_optimize_task,
            "cron",
            hour=dh,
            minute=dm,
            id="moss_daily_optimize",
            replace_existing=True,
        )
    try:
        from datetime import datetime, timedelta, timezone

        from moss2.config import (
            MOSS2_AUTO_PROVISION_ON_START,
            MOSS2_AUTO_PROVISION_WEEKLY,
            MOSS2_CULL_SCHEDULER_WEEKLY,
            MOSS2_DATA_BOOTSTRAP_ON_START,
            MOSS2_DATA_BOOTSTRAP_WEEKLY,
            MOSS2_SCAN_INTERVAL_MINUTES,
            auto_provision_scheduler_enabled as moss2_auto_provision_scheduler_enabled,
            cull_scheduler_enabled as moss2_cull_scheduler_enabled,
            data_bootstrap_scheduler_enabled as moss2_data_bootstrap_scheduler_enabled,
            discipline_snapshot_scheduler_enabled as moss2_discipline_snapshot_enabled,
            evolve_scheduler_enabled as moss2_evolve_scheduler_enabled,
            paper_scheduler_enabled as moss2_paper_scheduler_enabled,
        )
    except ImportError:
        moss2_data_bootstrap_scheduler_enabled = lambda: False  # type: ignore[misc, assignment]
        moss2_paper_scheduler_enabled = lambda: False  # type: ignore[misc, assignment]
        moss2_evolve_scheduler_enabled = lambda: False  # type: ignore[misc, assignment]
        moss2_discipline_snapshot_enabled = lambda: False  # type: ignore[misc, assignment]
        MOSS2_DATA_BOOTSTRAP_ON_START = False
        MOSS2_DATA_BOOTSTRAP_WEEKLY = False
        MOSS2_AUTO_PROVISION_ON_START = False
        MOSS2_AUTO_PROVISION_WEEKLY = False
        MOSS2_CULL_SCHEDULER_WEEKLY = False
        MOSS2_SCAN_INTERVAL_MINUTES = 15
        moss2_auto_provision_scheduler_enabled = lambda: False  # type: ignore[misc, assignment]
        moss2_cull_scheduler_enabled = lambda: False  # type: ignore[misc, assignment]
        datetime = None  # type: ignore[misc, assignment]
        timedelta = None  # type: ignore[misc, assignment]
        timezone = None  # type: ignore[misc, assignment]

    if moss2_data_bootstrap_scheduler_enabled():
        if MOSS2_DATA_BOOTSTRAP_ON_START:
            run_at = datetime.now(timezone.utc) + timedelta(seconds=90)  # type: ignore[union-attr]
            sch.add_job(
                wt.run_moss2_data_bootstrap_task,
                "date",
                run_date=run_at,
                kwargs={"context": "startup"},
                id="moss2_data_bootstrap_startup",
                replace_existing=True,
                misfire_grace_time=3600,
            )
        if MOSS2_DATA_BOOTSTRAP_WEEKLY:
            sch.add_job(
                wt.run_moss2_data_bootstrap_task,
                "cron",
                day_of_week="sun",
                hour=4,
                minute=0,
                kwargs={"context": "weekly"},
                id="moss2_data_bootstrap_weekly",
                replace_existing=True,
                misfire_grace_time=3600,
            )
    if moss2_auto_provision_scheduler_enabled():
        if MOSS2_AUTO_PROVISION_ON_START:
            prov_at = datetime.now(timezone.utc) + timedelta(minutes=5)  # type: ignore[union-attr]
            sch.add_job(
                wt.run_moss2_auto_provision_task,
                "date",
                run_date=prov_at,
                id="moss2_auto_provision_startup",
                replace_existing=True,
                misfire_grace_time=7200,
            )
        if MOSS2_AUTO_PROVISION_WEEKLY:
            sch.add_job(
                wt.run_moss2_auto_provision_task,
                "cron",
                day_of_week="sun",
                hour=4,
                minute=45,
                id="moss2_auto_provision_weekly",
                replace_existing=True,
                misfire_grace_time=7200,
            )
    if moss2_paper_scheduler_enabled():
        sch.add_job(
            wt.run_moss2_paper_task,
            IntervalTrigger(minutes=MOSS2_SCAN_INTERVAL_MINUTES),
            id="moss2_paper_scan",
            replace_existing=True,
        )
    if moss2_evolve_scheduler_enabled():
        sch.add_job(
            wt.run_moss2_evolve_task,
            "cron",
            day_of_week="sun",
            hour=5,
            minute=0,
            id="moss2_evolve",
            replace_existing=True,
        )
    if moss2_discipline_snapshot_enabled():
        sch.add_job(
            wt.run_moss2_discipline_snapshot_task,
            "cron",
            day_of_week="sun",
            hour=6,
            minute=0,
            id="moss2_discipline_snapshot",
            replace_existing=True,
        )
    if moss2_cull_scheduler_enabled():
        sch.add_job(
            wt.run_moss2_cull_task,
            "cron",
            day_of_week="sun",
            hour=6,
            minute=30,
            id="moss2_cull_weekly",
            replace_existing=True,
            misfire_grace_time=7200,
        )
    if daily_optimize_bootstrap_enabled():
        schedule_bootstrap = True
        try:
            from accumulation_radar import init_db
            from moss_quant.daily_optimize_service import needs_daily_optimize_bootstrap

            conn = init_db()
            try:
                schedule_bootstrap = needs_daily_optimize_bootstrap(conn)
            finally:
                conn.close()
        except Exception as e:
            logging.getLogger(__name__).warning(
                "Moss bootstrap schedule check failed, will still register: %s", e
            )
        if schedule_bootstrap:
            from datetime import datetime, timedelta

            from moss_quant.config import MOSS_QUANT_DAILY_OPTIMIZE_BOOTSTRAP_DELAY_SEC

            run_at = datetime.now(sch.timezone) + timedelta(
                seconds=MOSS_QUANT_DAILY_OPTIMIZE_BOOTSTRAP_DELAY_SEC
            )
            sch.add_job(
                wt.run_moss_daily_optimize_bootstrap_task,
                "date",
                run_date=run_at,
                id="moss_daily_optimize_bootstrap",
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