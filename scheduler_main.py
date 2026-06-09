#!/usr/bin/env python3
"""
独立 Worker：仅运行 APScheduler 定时任务（与 Web API 进程解耦）。

用法:
  cd next-k-api
  python scheduler_main.py

Web 默认内嵌调度器（NEXT_K_EMBED_SCHEDULER 未设或为 1）。仅当设 NEXT_K_EMBED_SCHEDULER=0 时需本脚本。
"""

from __future__ import annotations

import logging
import sys

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler

from env_loader import load_env_oi

load_env_oi()

import scheduler_config as sched_cfg
import worker_tasks as wt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("scheduler_main")


def main() -> None:
    tz = pytz.timezone("Asia/Shanghai")
    sch = BlockingScheduler(timezone=tz)
    sched_cfg.register_scheduled_jobs(sch, wt)
    logger.info("scheduler_main 已启动 (Asia/Shanghai)")
    try:
        sch.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("scheduler_main 退出")


if __name__ == "__main__":
    main()
