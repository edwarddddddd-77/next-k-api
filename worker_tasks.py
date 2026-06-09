#!/usr/bin/env python3
"""后台定时子进程任务：与 FastAPI 解耦，供 main / scheduler_main 共用。"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_API_DIR = Path(__file__).resolve().parent
_RADAR_SCRIPT = _API_DIR / "accumulation_radar.py"
_S2_FUNDING_SCRIPT = _API_DIR / "s2_oi_funding_rate_scanner.py"
_ORB_SCRIPT = _API_DIR / "orb_scanner.py"

_subprocess_locks: Dict[str, threading.Lock] = {
    "accumulation_pool": threading.Lock(),
    "accumulation_oi": threading.Lock(),
    "s2_funding": threading.Lock(),
    "orb_scan": threading.Lock(),
}
_heat_watch_refresh_lock = threading.Lock()


def _run_subprocess_locked(lock_key: str, argv: list[str], *, cwd: Path, env: dict | None = None) -> None:
    lk = _subprocess_locks.get(lock_key)
    if lk is None:
        subprocess.run(argv, cwd=str(cwd), env=env, check=False)
        return
    if not lk.acquire(blocking=False):
        logger.warning(
            "跳过 %s：上一轮子进程仍在运行（本轮未执行）",
            lock_key,
        )
        return
    try:
        subprocess.run(argv, cwd=str(cwd), env=env, check=False)
    except Exception as e:
        logger.exception("%s subprocess failed: %s", lock_key, e)
    finally:
        lk.release()


def run_accumulation_radar_subprocess(mode: str) -> None:
    logger.info("Starting accumulation_radar subprocess mode=%s", mode)
    key = "accumulation_pool" if mode == "pool" else "accumulation_oi"
    _run_subprocess_locked(
        key,
        [sys.executable, str(_RADAR_SCRIPT), mode],
        cwd=_RADAR_SCRIPT.parent,
    )


def run_pool_task() -> None:
    logger.info("开始执行每日收筹池扫描...")
    run_accumulation_radar_subprocess("pool")


def run_oi_task() -> None:
    logger.info("开始执行每小时 OI 异动扫描...")
    run_accumulation_radar_subprocess("oi")


def refresh_heat_accum_watch_full_once() -> Dict[str, Any]:
    return _refresh_heat_accum_watch_full_once()


def _refresh_heat_accum_watch_full_once() -> Dict[str, Any]:
    from accumulation_radar import init_db, refresh_all_heat_accum_watch_full

    conn = init_db()
    try:
        return refresh_all_heat_accum_watch_full(conn)
    finally:
        conn.close()


def run_heat_watch_refresh_task() -> None:
    if not _heat_watch_refresh_lock.acquire(blocking=False):
        logger.info("热度看盘整表刷新跳过：已有任务在执行")
        return
    try:
        logger.info("开始执行热度看盘整表刷新...")
        data = _refresh_heat_accum_watch_full_once()
        logger.info(
            "热度看盘整表刷新完成: prices=%s",
            data.get("recalculated_prices"),
        )
    except Exception as e:
        logger.exception("heat watch refresh failed: %s", e)
    finally:
        _heat_watch_refresh_lock.release()


def run_s2_oi_funding_subprocess() -> None:
    logger.info("Starting s2_oi_funding_rate_scanner subprocess")
    _run_subprocess_locked(
        "s2_funding",
        [sys.executable, str(_S2_FUNDING_SCRIPT)],
        cwd=_S2_FUNDING_SCRIPT.parent,
    )


def run_s2_oi_funding_task() -> None:
    logger.info("开始执行 s2 OI+费率转负扫描...")
    run_s2_oi_funding_subprocess()


def _orb_scan_enabled() -> bool:
    from scheduler_config import ORB_SCHEDULER_ENABLED

    return bool(ORB_SCHEDULER_ENABLED)


def run_orb_scan_subprocess() -> None:
    logger.info("Starting orb_scanner subprocess")
    _run_subprocess_locked(
        "orb_scan",
        [sys.executable, str(_ORB_SCRIPT)],
        cwd=_ORB_SCRIPT.parent,
    )


def run_orb_scan_task() -> None:
    if not _orb_scan_enabled():
        logger.info("ORB_SCHEDULER_ENABLED=0，跳过 ORB 纸面扫描")
        return
    run_orb_scan_subprocess()


_top_trader_radar_lock = threading.Lock()


def run_top_trader_radar_task(*, force: bool = False) -> None:
    """大户多空 + Taker：公开 fapi/futures/data → top_trader_snapshots + JSON。"""
    if not _top_trader_radar_lock.acquire(blocking=False):
        logger.warning("跳过 top_trader_radar：上一轮仍在运行")
        return
    try:
        from top_trader_config import top_trader_scheduler_enabled
        from top_trader_radar import run_top_trader_radar_once

        if not force and not top_trader_scheduler_enabled():
            logger.info("TOP_TRADER_SCHEDULER_ENABLED=0，跳过大户多空雷达")
            return
        logger.info("开始执行大户多空 + Taker 雷达…")
        out = run_top_trader_radar_once(quiet=True)
        if not out.get("ok"):
            logger.warning(
                "大户多空雷达结束(未成功) error=%s universe=%s msg=%s",
                out.get("error"),
                out.get("universe"),
                out.get("message"),
            )
            return
        logger.info(
            "大户多空雷达完成 universe=%s captured=%s/%s period=%s elapsed=%ss",
            out.get("universe"),
            out.get("captured"),
            out.get("requested"),
            out.get("period"),
            out.get("elapsed_sec"),
        )
    except Exception as e:
        logger.exception("top_trader_radar failed: %s", e)
    finally:
        _top_trader_radar_lock.release()


def heat_watch_refresh_lock() -> threading.Lock:
    return _heat_watch_refresh_lock
