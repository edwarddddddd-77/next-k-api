#!/usr/bin/env python3
"""后台定时子进程任务：与 FastAPI 解耦，供 main / scheduler_main 共用。"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_API_DIR = Path(__file__).resolve().parent
_RADAR_SCRIPT = _API_DIR / "accumulation_radar.py"
_S2_FUNDING_SCRIPT = _API_DIR / "s2_oi_funding_rate_scanner.py"
_S6_ALPHA_SCRIPT = _API_DIR / "s6_futures_alpha_autonomous_trading_v1.py"
_ZCT_VWAP_SCRIPT = _API_DIR / "zct_vwap_signal_scanner.py"
_ZCT_TOUCH_POOL_JOB = _API_DIR / "zct_vwap_asset_pool_daily_job.py"
_ZCT_TOUCH_POOL_PRUNE = _API_DIR / "zct_touch_pool_intraday_prune.py"

_subprocess_locks: Dict[str, threading.Lock] = {
    "accumulation_pool": threading.Lock(),
    "accumulation_oi": threading.Lock(),
    "s2_funding": threading.Lock(),
    "s6_alpha": threading.Lock(),
    "zct_vwap_scan": threading.Lock(),
    "zct_vwap_resolve": threading.Lock(),
    "zct_touch_pool": threading.Lock(),
}
_heat_watch_refresh_lock = threading.Lock()
_powder_keg_radar_lock = threading.Lock()


def _run_subprocess_locked(lock_key: str, argv: list[str], *, cwd: Path, env: dict | None = None) -> None:
    lk = _subprocess_locks.get(lock_key)
    if lk is None:
        subprocess.run(argv, cwd=str(cwd), env=env, check=False)
        return
    if not lk.acquire(blocking=False):
        logger.warning(
            "跳过 %s：上一轮子进程仍在运行（本轮未执行，触轨池可能 stale）",
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
    from accumulation_radar import init_db, refresh_all_heat_accum_watch_full, refresh_all_worth_watch_bpc_states

    conn = init_db()
    try:
        out = refresh_all_heat_accum_watch_full(conn)
        w = refresh_all_worth_watch_bpc_states(conn)
        out.update(w)
        return out
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
            "热度看盘整表刷新完成: prices=%s bpc=%s",
            data.get("recalculated_prices"),
            data.get("bpc_recalculated"),
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


def run_s6_futures_alpha_subprocess() -> None:
    logger.info("Starting s6_futures_alpha subprocess")
    _run_subprocess_locked(
        "s6_alpha",
        [sys.executable, str(_S6_ALPHA_SCRIPT)],
        cwd=_S6_ALPHA_SCRIPT.parent,
    )


def run_s6_futures_alpha_task() -> None:
    logger.info("开始执行 s6 期货 Alpha 自主扫描...")
    run_s6_futures_alpha_subprocess()


def _zct_touch_pool_child_env() -> dict:
    from touch_pool_config import apply_touch_pool_default_env

    return apply_touch_pool_default_env(os.environ.copy())


def run_zct_vwap_signal_subprocess() -> None:
    logger.info("Starting zct_vwap_signal_scanner subprocess")
    _run_subprocess_locked(
        "zct_vwap_scan",
        [sys.executable, str(_ZCT_VWAP_SCRIPT)],
        cwd=_ZCT_VWAP_SCRIPT.parent,
        env=_zct_touch_pool_child_env(),
    )


def run_zct_vwap_signal_task() -> None:
    run_zct_vwap_signal_subprocess()
    # Push any new signals to the Binance live-trading bridge (push model, not polling).
    if os.getenv("BINANCE_ENABLED", "").strip().lower() in ("1", "true", "yes", "on"):
        try:
            from binance_bridge.signal_bridge import on_scan_complete
            on_scan_complete()
        except Exception as e:
            logger.warning("binance_bridge signal_bridge on_scan_complete failed: %s", e)


def run_zct_vwap_resolve_only_subprocess() -> None:
    logger.info("Starting zct_vwap_signal_scanner --resolve-only subprocess")
    _run_subprocess_locked(
        "zct_vwap_resolve",
        [sys.executable, str(_ZCT_VWAP_SCRIPT), "--resolve-only"],
        cwd=_ZCT_VWAP_SCRIPT.parent,
        env=_zct_touch_pool_child_env(),
    )


def run_zct_vwap_resolve_only_task() -> None:
    run_zct_vwap_resolve_only_subprocess()


def run_zct_touch_pool_4h_subprocess() -> None:
    logger.info(
        "Starting zct_vwap_asset_pool_daily_job --once --worth-watch-plus-default-22 (6h walk 2h-cron)"
    )
    _run_subprocess_locked(
        "zct_touch_pool",
        [
            sys.executable,
            str(_ZCT_TOUCH_POOL_JOB),
            "--once",
            "--worth-watch-plus-default-22",
        ],
        cwd=_ZCT_TOUCH_POOL_JOB.parent,
        env=_zct_touch_pool_child_env(),
    )


def run_zct_touch_pool_4h_task() -> None:
    logger.info("开始执行 ZCT 触轨池每 2h 全量筛选（6h walk + 全表重写）...")
    run_zct_touch_pool_4h_subprocess()


def run_zct_touch_pool_daily_subprocess() -> None:
    run_zct_touch_pool_4h_subprocess()


def run_zct_touch_pool_daily_task() -> None:
    run_zct_touch_pool_4h_task()


def run_zct_touch_pool_intraday_prune_subprocess() -> None:
    run_zct_touch_pool_4h_subprocess()


def run_zct_touch_pool_intraday_prune_task() -> None:
    run_zct_touch_pool_4h_task()


def run_powder_keg_radar_task() -> None:
    """火药桶宏观雷达：收筹池内 OI+费率+横盘 → powder_keg_watchlist（每 15 分钟）。"""
    if not _powder_keg_radar_lock.acquire(blocking=False):
        logger.warning("跳过 powder_keg_radar：上一轮仍在运行")
        return
    try:
        from powder_keg_config import powder_keg_radar_enabled
        from powder_keg_radar import run_powder_keg_radar_once

        if not powder_keg_radar_enabled():
            logger.info("POWDER_KEG_RADAR_ENABLED=0，跳过火药桶雷达")
            return
        logger.info("开始执行火药桶宏观雷达…")
        out = run_powder_keg_radar_once(quiet=True)
        n = int((out.get("watchlist") or {}).get("count") or 0)
        logger.info(
            "火药桶雷达完成 matched=%s 入库=%s",
            out.get("matched"),
            n,
        )
    except Exception as e:
        logger.exception("powder_keg_radar failed: %s", e)
    finally:
        _powder_keg_radar_lock.release()


def heat_watch_refresh_lock() -> threading.Lock:
    return _heat_watch_refresh_lock
