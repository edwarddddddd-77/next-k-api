from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query

import worker_tasks as wt
from models.api_models import ClearWatchTablesBody, TriggerCronBody
from utils.maintenance_auth import require_maintenance_token
from utils.rate_limit import MinIntervalGuard
from worker_tasks import refresh_heat_accum_watch_full_once

logger = logging.getLogger(__name__)

router = APIRouter(tags=["accumulation"])

_oi_radar_refresh_lock = threading.Lock()
_oi_radar_refresh_cooldown = MinIntervalGuard("OI_RADAR_REFRESH_COOLDOWN_SEC", 120.0)
_heat_watch_refresh_lock = wt.heat_watch_refresh_lock()

_CRON_TASK_FUNCS: Dict[str, Any] = {
    "pool": wt.run_pool_task,
    "heat_watch": wt.run_heat_watch_refresh_task,
    "heat_zones": wt.run_heat_watch_refresh_task,
    "heat_bpc": wt.run_heat_watch_refresh_task,
    "oi": wt.run_oi_task,
    "s2_funding": wt.run_s2_oi_funding_task,
    "touch_pool": wt.run_zct_touch_pool_4h_task,
    "touch_pool_4h": wt.run_zct_touch_pool_4h_task,
    "s6_alpha": wt.run_s6_futures_alpha_task,
    "zct_vwap": wt.run_zct_vwap_signal_task,
    "zct_vwap_resolve": wt.run_zct_vwap_resolve_only_task,
    "zct_hot_oi": wt.run_zct_vwap_signal_task,
    "zct_hot_oi_resolve": wt.run_zct_vwap_resolve_only_task,
    "powder_keg": wt.run_powder_keg_radar_task,
    "powder_keg_radar": wt.run_powder_keg_radar_task,
    "mom_scan": wt.run_momentum_scan_task,
    "momentum_scan": wt.run_momentum_scan_task,
    "mom_trail": wt.run_momentum_trail_task,
    "momentum_trail": wt.run_momentum_trail_task,
}


def _oi_radar_snapshot_path() -> Path:
    """与 accumulation_radar 的 DATA_DIR / accumulation.db 同目录。"""
    db_dir = Path(os.getenv("DATA_DIR", str(Path(__file__).resolve().parent.parent)))
    return db_dir / "oi_radar_snapshot.json"


def _run_refresh_heat_watch_background() -> None:
    try:
        logger.info("manual refresh heat watch (full) accepted")
        data = refresh_heat_accum_watch_full_once()
        logger.info(
            "manual refresh heat watch done: prices=%s bpc=%s",
            data.get("recalculated_prices"),
            data.get("bpc_recalculated"),
        )
    except Exception:
        logger.exception("manual refresh heat watch failed")
    finally:
        _heat_watch_refresh_lock.release()


@router.get("/api/accumulation/powder-keg")
async def get_powder_keg_watchlist():
    """当前火药桶监控名单（收筹池；按 symbol 去重，每币保留最新一条）。"""
    from accumulation_radar import init_db
    from powder_keg_radar import load_powder_keg_watchlist

    conn = init_db()
    try:
        return load_powder_keg_watchlist(conn)
    except Exception as e:
        logger.warning("powder_keg watchlist read failed: %s", e)
        raise HTTPException(status_code=500, detail="powder_keg_read_error")
    finally:
        conn.close()


@router.get("/api/accumulation/oi-radar")
async def get_accumulation_oi_radar():
    """
    返回磁盘上的最新 OI 雷达 JSON（由定时任务或 POST refresh 写入），响应极快，避免
    Railway/浏览器对长连接（完整扫描 1–2 分钟）超时导致「Failed to fetch」。

    本接口不触发 Telegram；每小时 :30 子进程仍会推送。
    """
    path = _oi_radar_snapshot_path()
    if not path.is_file():
        return {
            "ok": False,
            "error": "no_snapshot",
            "message": "尚无快照。请等待整点 :30 定时扫描写入，或点击前端「刷新」触发后台扫描（约 1–2 分钟后再次加载本接口）。",
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("snapshot root must be object")
        data["snapshot_source"] = "disk"
        return data
    except Exception as e:
        logger.warning("OI radar snapshot read failed: %s", e)
        raise HTTPException(status_code=500, detail="snapshot_corrupt")


@router.get("/api/accumulation/heat-accum-watch")
async def get_heat_accum_watch():
    """热度+收筹独立看盘：读写 accumulation.db 表 heat_accum_watch；含生成日与 2 日保留。"""
    try:
        from accumulation_radar import init_db, load_heat_accum_watchlist_from_db

        conn = init_db()
        try:
            data = load_heat_accum_watchlist_from_db(conn)
        finally:
            conn.close()
        if not data.get("items"):
            data.setdefault(
                "message",
                "尚无归档，请等待整点 :30 扫描或点击「刷新」后重试。",
            )
        return data
    except Exception as e:
        logger.warning("heat_accum watchlist read failed: %s", e)
        raise HTTPException(status_code=500, detail="watchlist_db_error")


@router.get("/api/accumulation/ambush-watch")
async def get_ambush_watch():
    """埋伏榜内 🎯 暗流 / 💎 低市值+OI：表 ambush_watch；含生成日与 2 日保留。"""
    try:
        from accumulation_radar import init_db, load_ambush_watchlist_from_db

        conn = init_db()
        try:
            data = load_ambush_watchlist_from_db(conn)
        finally:
            conn.close()
        if not data.get("items"):
            data.setdefault(
                "message",
                "尚无归档，请等待整点 :30 扫描或点击「刷新」后重试。",
            )
        return data
    except Exception as e:
        logger.warning("ambush watchlist read failed: %s", e)
        raise HTTPException(status_code=500, detail="ambush_watch_db_error")


@router.get("/api/accumulation/focus-watch")
async def get_focus_watch():
    """👑 重点关注（逼空/天量/暗流 + 否决）：表 focus_watch；含生成日与 2 日保留。"""
    try:
        from accumulation_radar import init_db, load_focus_watchlist_from_db

        conn = init_db()
        try:
            data = load_focus_watchlist_from_db(conn)
        finally:
            conn.close()
        if not data.get("items"):
            data.setdefault(
                "message",
                "尚无归档，请等待整点 :30 扫描或点击「刷新」后重试。",
            )
        return data
    except Exception as e:
        logger.warning("focus_watch read failed: %s", e)
        raise HTTPException(status_code=500, detail="focus_watch_db_error")


@router.get("/api/accumulation/patrick-core-watch")
async def get_patrick_core_watch():
    """📍 Patrick 核心：收筹池 + OI 异动；表 patrick_core_watch；含生成日与 2 日保留。"""
    try:
        from accumulation_radar import init_db, load_patrick_core_watchlist_from_db

        conn = init_db()
        try:
            data = load_patrick_core_watchlist_from_db(conn)
        finally:
            conn.close()
        if not data.get("items"):
            data.setdefault(
                "message",
                "尚无归档，请等待整点 :30 扫描或点击「刷新」后重试。",
            )
        return data
    except Exception as e:
        logger.warning("patrick_core watchlist read failed: %s", e)
        raise HTTPException(status_code=500, detail="patrick_core_watch_db_error")


@router.get("/api/accumulation/worth-watch")
async def get_worth_watch(category: Optional[str] = Query(None, description="可选：heat_accum / patrick_core / …")):
    """值得关注七类归档：七张独立表 worth_watch_*；每类每轮动态门槛+至多 5 条入库；保留 7 日（含当日，见 WORTH_WATCH_RETENTION_DAYS）；各行含 bpc。可选 ?category=heat_accum。"""
    try:
        from accumulation_radar import (
            WORTH_HIGHLIGHT_CATEGORY_ORDER,
            init_db,
            load_worth_highlight_watchlist_from_db,
        )

        if category is not None and str(category).strip():
            cat = str(category).strip()
            if cat not in set(WORTH_HIGHLIGHT_CATEGORY_ORDER):
                raise HTTPException(status_code=400, detail=f"unknown category: {cat}")
        else:
            cat = None

        conn = init_db()
        try:
            data = load_worth_highlight_watchlist_from_db(conn, category=cat)
        finally:
            conn.close()
        if not data.get("items"):
            data.setdefault(
                "message",
                "尚无归档，请等待整点 :30 扫描或点击「刷新」后重试。",
            )
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("worth_highlight watchlist read failed: %s", e)
        raise HTTPException(status_code=500, detail="worth_watch_db_error")
@router.post("/api/accumulation/maintenance/clear-watch-tables")
async def post_clear_watch_tables(
    body: ClearWatchTablesBody,
    _: None = Depends(require_maintenance_token),
):
    """
    清空看盘 SQLite 表。

    清库后请再调一次「OI 刷新」或等整点扫描，以按新规则写回数据。
    """
    from accumulation_radar import WORTH_WATCH_TABLE_BY_CATEGORY

    _worth_tables = set(WORTH_WATCH_TABLE_BY_CATEGORY.values())
    allowed = {
        "watchlist",
        "focus_watch",
        "ambush_watch",
        "heat_accum_watch",
        "patrick_core_watch",
        "worth_highlight_watch",
        "worth_watch_all",
        *_worth_tables,
    }
    tables = [t.strip() for t in body.tables if t and str(t).strip()]
    if not tables:
        tables = ["ambush_watch"]
    unknown = [t for t in tables if t not in allowed]
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown tables: {unknown}")

    try:
        from accumulation_radar import (
            clear_all_worth_watch_category_tables,
            clear_ambush_watch_table,
            clear_heat_accum_watch_table,
            clear_one_worth_watch_category_table,
            clear_patrick_core_watch_table,
            clear_focus_watch_table,
            clear_watchlist_table,
            init_db,
            patch_oi_radar_snapshot_after_watchlist_clear,
            patch_oi_radar_snapshot_watchlists_from_db,
        )

        conn = init_db()
        try:
            cleared: Dict[str, Any] = {}
            if "watchlist" in tables:
                cleared["watchlist"] = clear_watchlist_table(conn)
            if "focus_watch" in tables:
                cleared["focus_watch"] = clear_focus_watch_table(conn)
            if "ambush_watch" in tables:
                cleared["ambush_watch"] = clear_ambush_watch_table(conn)
            if "heat_accum_watch" in tables:
                cleared["heat_accum_watch"] = clear_heat_accum_watch_table(conn)
            if "patrick_core_watch" in tables:
                cleared["patrick_core_watch"] = clear_patrick_core_watch_table(conn)
            worth_tbls = {t for t in tables if t in set(WORTH_WATCH_TABLE_BY_CATEGORY.values())}
            worth_all = (
                "worth_watch_all" in tables
                or "worth_highlight_watch" in tables
            )
            if worth_all:
                cleared.update(clear_all_worth_watch_category_tables(conn))
            else:
                for t in sorted(worth_tbls):
                    cleared[t] = clear_one_worth_watch_category_table(conn, t)
            try:
                if "watchlist" in tables:
                    patch_oi_radar_snapshot_after_watchlist_clear(conn)
                else:
                    patch_oi_radar_snapshot_watchlists_from_db(conn)
            except Exception:
                logger.exception("patch oi_radar snapshot after clear failed")
            logger.warning(
                "maintenance clear-watch-tables tables=%s cleared=%s",
                tables,
                cleared,
            )
            return {"ok": True, "cleared_rows": cleared}
        finally:
            conn.close()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("clear watch tables failed: %s", e)
        raise HTTPException(status_code=500, detail="clear_failed")

@router.post("/api/accumulation/maintenance/trigger-cron")
async def post_trigger_accumulation_cron(
    body: TriggerCronBody,
    _: None = Depends(require_maintenance_token),
):
    """
    在后台线程执行与定时任务相同的逻辑（子进程跑脚本），HTTP 立即返回。

    - pool: accumulation_radar pool（定时每日 10:00 CST）
    - heat_watch: 热度看盘整表（现价/摘要 + 1h BPC，定时每小时 xx:07）
    - heat_zones / heat_bpc: 与 heat_watch 相同（兼容旧 task 名）
    - oi: accumulation_radar oi（定时每小时 :30）
    - s2_funding: s2_oi_funding_rate_scanner（定时每时 :05）
    - touch_pool / touch_pool_4h: ZCT 触轨池每 2h 全量 walk 入库（6h 窗口，偶数整点 :07 上海，可 env 覆盖）
    - s6_alpha: s6 期货 Alpha（定时每时 :25，与 S6_FUTURES_ALPHA_SCHEDULER_ENABLED 无关可手动跑）
    - zct_vwap: ZCT VWAP 全量扫描（与定时同源子进程，间隔见 ZCT_VWAP_SCAN_INTERVAL_MINUTES）
    - zct_vwap_resolve: 仅纸面结算（--resolve-only，与定时 ZCT_VWAP_RESOLVE_INTERVAL_MINUTES 同源）
    - zct_hot_oi / zct_hot_oi_resolve: 与 zct_vwap / zct_vwap_resolve 相同（兼容旧 task 名；已统一到 zct_vwap_* 表）
    - powder_keg / powder_keg_radar: 火药桶雷达（仅收筹池 watchlist，每 15 分钟）
    - mom_scan / momentum_scan: 动量 topMovers 纸面调仓（MOM_SCAN_INTERVAL_MINUTES，默认 15 分钟）
    - mom_trail / momentum_trail: 动量移动止盈检查（MOM_TRAIL_SCAN_INTERVAL_SEC，默认 20 秒）
    """
    key = (body.task or "").strip()
    fn = _CRON_TASK_FUNCS.get(key)
    if fn is None:
        raise HTTPException(
            status_code=400,
            detail=f"unknown task {key!r}; allowed: {sorted(_CRON_TASK_FUNCS.keys())}",
        )

    def _work() -> None:
        try:
            fn()
        except Exception:
            logger.exception("manual trigger-cron task=%s failed", key)

    threading.Thread(target=_work, daemon=True).start()
    return {"accepted": True, "task": key}


@router.post("/api/accumulation/oi-radar/refresh")
async def post_accumulation_oi_radar_refresh():
    """
    在后台线程执行完整扫描并写入 `oi_radar_snapshot.json`，立即返回，避免 HTTP 超时。
    与 GET 快照配合：前端轮询 GET 直至 `ok` 为 true。

    无需维护令牌（主界面「刷新」可用）；通过并发锁 + OI_RADAR_REFRESH_COOLDOWN_SEC 防滥用。
    清库 / trigger-cron 等仍须 NEXT_K_MAINTENANCE_TOKEN。
    """
    if not _oi_radar_refresh_lock.acquire(blocking=False):
        return {"accepted": False, "busy": True, "message": "已有扫描任务在执行中"}

    allowed, retry_after = _oi_radar_refresh_cooldown.check_allow()
    if not allowed:
        _oi_radar_refresh_lock.release()
        wait = max(1, int(retry_after + 0.5))
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limited",
                "retry_after_sec": round(retry_after, 1),
                "message": f"OI 刷新过于频繁，请约 {wait} 秒后再试",
            },
        )

    def _work():
        try:
            _oi_radar_refresh_cooldown.mark_used()
            from accumulation_radar import init_db, run_oi_hourly_radar

            conn = init_db()
            try:
                run_oi_hourly_radar(conn, notify=False)
            finally:
                conn.close()
        except Exception:
            logger.exception("OI radar background refresh failed")
        finally:
            _oi_radar_refresh_lock.release()

    threading.Thread(target=_work, daemon=True).start()
    return {"accepted": True, "busy": False}


@router.post("/api/accumulation/maintenance/refresh-heat-watch")
async def post_refresh_heat_watch(
    _: None = Depends(require_maintenance_token),
):
    """热度看盘整表：现价/摘要 + 1h BPC；并刷新 worth_watch_* 七表各行 1H BPC（后台线程）。与定时 heat_watch_refresh 同源。"""
    if not _heat_watch_refresh_lock.acquire(blocking=False):
        return {"accepted": False, "busy": True, "message": "已有热度看盘刷新任务在执行中"}
    threading.Thread(target=_run_refresh_heat_watch_background, daemon=True).start()
    return {"accepted": True, "busy": False}


@router.post("/api/accumulation/maintenance/refresh-heat-zones")
async def post_refresh_heat_zones(
    _: None = Depends(require_maintenance_token),
):
    """兼容旧路径：等同 refresh-heat-watch（现价 + BPC）。"""
    if not _heat_watch_refresh_lock.acquire(blocking=False):
        return {"accepted": False, "busy": True, "message": "已有热度看盘刷新任务在执行中"}
    threading.Thread(target=_run_refresh_heat_watch_background, daemon=True).start()
    return {"accepted": True, "busy": False}


@router.post("/api/accumulation/maintenance/refresh-heat-bpc")
async def post_refresh_heat_bpc(
    _: None = Depends(require_maintenance_token),
):
    """兼容旧路径：等同 refresh-heat-watch（现价 + BPC + 值得关注七表 BPC）。"""
    if not _heat_watch_refresh_lock.acquire(blocking=False):
        return {"accepted": False, "busy": True, "message": "已有热度看盘刷新任务在执行中"}
    threading.Thread(target=_run_refresh_heat_watch_background, daemon=True).start()
    return {"accepted": True, "busy": False}
