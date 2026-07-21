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

_subprocess_locks: Dict[str, threading.Lock] = {
    "accumulation_pool": threading.Lock(),
    "accumulation_oi": threading.Lock(),
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


def heat_watch_refresh_lock() -> threading.Lock:
    return _heat_watch_refresh_lock


_alpha_holders_lock = threading.Lock()


def run_alpha_holders_refresh_task() -> None:
    """定时自动刷日历币 Top 持仓，对比余额变动并落盘。"""
    if not _alpha_holders_lock.acquire(blocking=False):
        logger.info("Alpha 持仓监控跳过：已有任务在执行")
        return
    try:
        from alpha_holders import watch_calendar_tokens
        from alpha_radar import _load_calendar, patch_board_snapshot_chip_watch

        logger.info("开始执行 Alpha 筹码自动监控...")
        cal = _load_calendar()
        payload = watch_calendar_tokens(cal, limit=15)
        try:
            patch_board_snapshot_chip_watch()
        except Exception:
            logger.exception("patch board after auto holders refresh failed")
        n = len((payload or {}).get("watches") or [])
        logger.info("Alpha 筹码自动监控完成: watches=%s", n)
        try:
            from alpha_paper import auto_manage_from_watches

            auto = auto_manage_from_watches((payload or {}).get("watches") or [])
            if auto.get("enabled"):
                logger.info(
                    "Alpha 纸面自动: opened=%s closed=%s",
                    len(auto.get("opened") or []),
                    len(auto.get("closed") or []),
                )
        except Exception:
            logger.exception("alpha paper auto manage failed")
    except Exception as e:
        logger.exception("alpha holders auto refresh failed: %s", e)
    finally:
        _alpha_holders_lock.release()


_ie_flips_lock = threading.Lock()


def run_indicatoredge_flips_task() -> None:
    """定时轮询 IndicatorEdge Screener「Just flipped」。"""
    if not _ie_flips_lock.acquire(blocking=False):
        logger.info("IE flips 刷新跳过：已有任务在执行")
        return
    try:
        from utils.indicatoredge import refresh_screener_flips

        logger.info("开始轮询 IndicatorEdge Just flipped...")
        snap = refresh_screener_flips(force=True)
        logger.info(
            "IE Just flipped: total=%s new=%s source_updated=%s",
            snap.get("count"),
            snap.get("new_count"),
            snap.get("source_updated"),
        )
    except Exception as e:
        logger.exception("indicatoredge flips refresh failed: %s", e)
    finally:
        _ie_flips_lock.release()
