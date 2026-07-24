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


_hl_desk_candidates_lock = threading.Lock()


def run_hl_desk_candidates_task() -> None:
    """每周桌面跟单候选池（flat 日内/波段 ready），写入 hl_desk_candidates.json。"""
    if not _hl_desk_candidates_lock.acquire(blocking=False):
        logger.info("HL desk candidates 跳过：已有任务在执行")
        return
    try:
        from utils.hl_desk_candidates import build_candidate_pool

        logger.info("开始构建 HL desk candidate pool…")
        board = build_candidate_pool()
        logger.info(
            "HL desk candidates 完成: ready=%s watch=%s bound=%s scanned=%s",
            board.get("ready_count"),
            board.get("watch_count"),
            board.get("bound_count"),
            board.get("scanned"),
        )
    except Exception as e:
        logger.exception("hl desk candidates task failed: %s", e)
    finally:
        _hl_desk_candidates_lock.release()
