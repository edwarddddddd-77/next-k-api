#!/usr/bin/env python3
"""后台定时子进程任务：与 FastAPI 解耦，供 main / scheduler_main 共用。"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import subprocess
import sys
import threading
import urllib.request
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
_momentum_lane_lock = threading.Lock()
_jiezhen_lane_lock = threading.Lock()


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


def _zct_vwap_scan_child_env() -> dict:
    """实盘 VWAP 扫描：默认火药桶 universe（非触轨池）。"""
    from powder_keg_config import apply_powder_keg_scan_env

    return apply_powder_keg_scan_env(os.environ.copy())


def run_zct_vwap_signal_subprocess() -> None:
    logger.info("Starting zct_vwap_signal_scanner subprocess (powder_keg universe)")
    _run_subprocess_locked(
        "zct_vwap_scan",
        [sys.executable, str(_ZCT_VWAP_SCRIPT)],
        cwd=_ZCT_VWAP_SCRIPT.parent,
        env=_zct_vwap_scan_child_env(),
    )


def _push_signals_to_protocol() -> None:
    """Read new ZCT signals from accumulation.db and POST to Next-k-protocol."""
    proto_url = os.getenv("PROTOCOL_API_URL", "").strip().rstrip("/")
    proto_token = os.getenv("PROTOCOL_MAINTENANCE_TOKEN", "").strip()
    if not proto_url:
        return

    from accumulation_radar import init_db

    conn = init_db()
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT id, symbol, play, side, entry_price, sl_price, tp_price,
                      confidence, regime, virtual_notional_usdt
               FROM zct_vwap_signals
               WHERE outcome IS NULL
                 AND sl_price IS NOT NULL
                 AND tp_price IS NOT NULL
                 AND side IN ('LONG','SHORT')
               ORDER BY id ASC"""
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        logger.debug("_push_signals_to_protocol: no new signals")
        return

    signals = []
    for r in rows:
        logger.info(
            "_push_signals: id=%s symbol=%s side=%s play=%s entry=%s sl=%s tp=%s",
            r["id"], r["symbol"], r["side"], r["play"],
            r["entry_price"], r["sl_price"], r["tp_price"],
        )
        signals.append({
            "source": "zct_vwap",
            "api_signal_id": str(r["id"]),
            "symbol": r["symbol"],
            "play": r["play"],
            "side": r["side"],
            "entry_price": r["entry_price"],
            "sl_price": r["sl_price"],
            "tp_price": r["tp_price"],
            "confidence": r["confidence"],
            "regime": r["regime"],
            "notional_usdt": r["virtual_notional_usdt"],
        })

    body = json.dumps({"signals": signals}).encode("utf-8")
    url = f"{proto_url}/api/binance/signals/ingest"
    headers = {"Content-Type": "application/json"}
    if proto_token:
        headers["X-Maintenance-Token"] = proto_token

    try:
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        logger.info(
            "_push_signals_to_protocol: scanned=%d traded=%d skipped=%d errors=%d",
            result.get("scanned", 0),
            result.get("traded", 0),
            result.get("skipped", 0),
            result.get("errors", 0),
        )
    except Exception as e:
        logger.warning("_push_signals_to_protocol failed: %s", e)


def _zct_vwap_scan_enabled() -> bool:
    from scheduler_config import ZCT_VWAP_SIGNAL_SCHEDULER_ENABLED

    return bool(ZCT_VWAP_SIGNAL_SCHEDULER_ENABLED)


def run_zct_vwap_signal_task() -> None:
    if not _zct_vwap_scan_enabled():
        logger.info("ZCT_VWAP_SIGNAL_SCHEDULER_ENABLED=0，跳过 ZCT VWAP 扫描与 protocol 推送")
        return
    run_zct_vwap_signal_subprocess()
    _push_signals_to_protocol()


def run_zct_vwap_resolve_only_subprocess() -> None:
    logger.info("Starting zct_vwap_signal_scanner --resolve-only subprocess")
    _run_subprocess_locked(
        "zct_vwap_resolve",
        [sys.executable, str(_ZCT_VWAP_SCRIPT), "--resolve-only"],
        cwd=_ZCT_VWAP_SCRIPT.parent,
        env=_zct_touch_pool_child_env(),
    )


def run_zct_vwap_resolve_only_task() -> None:
    if not _zct_vwap_scan_enabled():
        logger.info("ZCT_VWAP_SIGNAL_SCHEDULER_ENABLED=0，跳过 ZCT VWAP 结算")
        return
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


def run_momentum_scan_task() -> None:
    """动量多一空一：topMovers 纸面调仓（默认每 15 分钟）。"""
    if not _momentum_lane_lock.acquire(blocking=False):
        logger.warning("跳过 momentum_scan：动量 lane 上一轮仍在运行")
        return
    try:
        from momentum_config import momentum_scheduler_enabled
        from momentum_scanner import run_scan

        if not momentum_scheduler_enabled():
            logger.info("MOM_SCHEDULER_ENABLED=0，跳过动量扫描")
            return
        logger.info("开始执行动量 topMovers 纸面扫描…")
        stats = run_scan(notify=True)
        logger.info(
            "动量扫描完成 long=%s short=%s opens=%s closes=%s skipped=%s",
            stats.get("long_target"),
            stats.get("short_target"),
            stats.get("opens"),
            stats.get("closes"),
            stats.get("skipped"),
        )
    except Exception as e:
        logger.exception("momentum_scan failed: %s", e)
    finally:
        _momentum_lane_lock.release()


def run_momentum_trail_task() -> None:
    """动量持仓：分档移动止盈 / 止损（默认每 1 分钟，独立于 topMovers 调仓）。"""
    if not _momentum_lane_lock.acquire(blocking=False):
        logger.warning("跳过 momentum_trail：动量 lane 上一轮仍在运行")
        return
    try:
        from momentum_config import mom_trail_scheduler_enabled
        from momentum_scanner import run_trail_checks

        if not mom_trail_scheduler_enabled():
            return
        stats = run_trail_checks(notify=True)
        if stats.get("closes") or stats.get("skipped"):
            logger.info(
                "动量止盈检查完成 closes=%s skipped=%s events=%s",
                stats.get("closes"),
                stats.get("skipped"),
                stats.get("events"),
            )
    except Exception as e:
        logger.exception("momentum_trail failed: %s", e)
    finally:
        _momentum_lane_lock.release()


def run_jiezhen_scan_task() -> None:
    """接针：热度+OI 标的池纸面扫描（默认每 60 秒）。"""
    if not _jiezhen_lane_lock.acquire(blocking=False):
        logger.warning("跳过 jiezhen_scan：接针 lane 上一轮仍在运行")
        return
    try:
        from jiezhen_config import jiezhen_scheduler_enabled
        from jiezhen_scanner import run_scan

        if not jiezhen_scheduler_enabled():
            logger.info("JIEZHEN_SCHEDULER_ENABLED=0，跳过接针扫描")
            return
        logger.info("开始执行接针（热度+OI）纸面扫描…")
        stats = run_scan(notify=True)
        logger.info(
            "接针扫描完成 universe=%s opens=%s closes=%s skipped=%s",
            len(stats.get("universe") or []),
            stats.get("opens"),
            stats.get("closes"),
            stats.get("skipped"),
        )
    except Exception as e:
        logger.exception("jiezhen_scan failed: %s", e)
    finally:
        _jiezhen_lane_lock.release()


def run_jiezhen_trail_task() -> None:
    """接针策略（独立 lane）：分档移动止盈 / 止损（阈值 MOM_TRAIL_*，开关 JIEZHEN_TRAIL_*）。"""
    if not _jiezhen_lane_lock.acquire(blocking=False):
        logger.warning("跳过 jiezhen_trail：接针 lane 上一轮仍在运行")
        return
    try:
        from jiezhen_config import jz_trail_scheduler_enabled
        from jiezhen_scanner import run_trail_checks

        if not jz_trail_scheduler_enabled():
            return
        stats = run_trail_checks(notify=True)
        if stats.get("closes") or stats.get("skipped"):
            logger.info(
                "接针止盈检查完成 closes=%s skipped=%s events=%s",
                stats.get("closes"),
                stats.get("skipped"),
                stats.get("events"),
            )
    except Exception as e:
        logger.exception("jiezhen_trail failed: %s", e)
    finally:
        _jiezhen_lane_lock.release()


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
        if not out.get("ok"):
            logger.warning(
                "火药桶雷达结束(未成功) error=%s watchlist=%s msg=%s",
                out.get("error"),
                out.get("watchlist_count"),
                out.get("message"),
            )
            return
        n = int((out.get("watchlist") or {}).get("count") or 0)
        stats = out.get("scan_stats") or {}
        persist = out.get("persist") or {}
        logger.info(
            "火药桶雷达完成 pool=%s 币安ticker=%s pre=%s depth_oi=%s/%s "
            "matched=%s 本轮入库=%s 表内总数=%s api=%s 耗时=%ss",
            out.get("watchlist_count"),
            stats.get("ticker_rows"),
            out.get("scanned_pre"),
            stats.get("oi_fetched"),
            stats.get("depth_scanned"),
            out.get("matched"),
            persist.get("inserted"),
            n,
            out.get("api_mode"),
            out.get("elapsed_sec"),
        )
    except Exception as e:
        logger.exception("powder_keg_radar failed: %s", e)
    finally:
        _powder_keg_radar_lock.release()


def heat_watch_refresh_lock() -> threading.Lock:
    return _heat_watch_refresh_lock
