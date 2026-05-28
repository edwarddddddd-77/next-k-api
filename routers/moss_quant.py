"""Moss 量化 lane API（回测 / 进化 / 纸面）。"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from utils.maintenance_auth import require_maintenance_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/moss-quant", tags=["moss-quant"])


class ProfileCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    symbol: str
    template: str = "balanced"
    enabled: bool = False
    virtual_equity_usdt: Optional[float] = None
    evolution_enabled: bool = True
    param_overrides: Optional[dict] = None


class ProfileFromDailyCreate(BaseModel):
    symbol: str
    name: Optional[str] = None
    enabled: bool = True
    update_existing: bool = True


class ProfilePatch(BaseModel):
    name: Optional[str] = None
    enabled: Optional[bool] = None
    tactical_params: Optional[dict] = None
    virtual_equity_usdt: Optional[float] = None


class BacktestRequest(BaseModel):
    profile_id: Optional[int] = None
    symbol: Optional[str] = None
    params: Optional[dict] = None
    template: str = "balanced"
    capital: Optional[float] = None
    refresh_klines: bool = False


class EvolveBaselineRequest(BaseModel):
    profile_id: Optional[int] = None
    symbol: Optional[str] = None
    params: Optional[dict] = None
    template: str = "balanced"
    segment_bars: Optional[int] = None
    capital: Optional[float] = None
    refresh_klines: bool = True


class EvolveReflectRequest(BaseModel):
    baseline_run_id: int


class EvolveRunRequest(BaseModel):
    baseline_run_id: int
    schedule: Optional[List[dict]] = None


class ApplyFinalParamsRequest(BaseModel):
    """将进化 final_params 写入 Profile 的 tactical_params（纸面生效）。"""
    params: Optional[dict] = None
    run_id: Optional[int] = None


class OptimizeRequest(BaseModel):
    """网格搜索模板 + 战术参数（按回测收益排序）。"""
    profile_id: Optional[int] = None
    symbol: Optional[str] = None
    capital: Optional[float] = None
    refresh_klines: bool = False
    top_n: int = Field(15, ge=1, le=50)
    max_combinations: int = Field(96, ge=4, le=200)
    apply_best_tactical_to_profile_id: Optional[int] = None


class DailyOptimizeRunRequest(BaseModel):
    capital: Optional[float] = None
    refresh_klines: Optional[bool] = None
    apply_profiles: Optional[bool] = None


def _conn():
    from accumulation_radar import init_db

    c = init_db()
    c.row_factory = sqlite3.Row
    return c


def _resolve_symbol_params(body_symbol, body_params, body_template, profile_id):
    from moss_quant import config as cfg
    from moss_quant.db import get_profile
    from moss_quant.params import build_initial_params
    from moss_quant.universe import is_research_symbol_allowed, normalize_usdt_perp_symbol

    if profile_id:
        conn = _conn()
        try:
            prof = get_profile(conn, int(profile_id))
        finally:
            conn.close()
        if not prof:
            raise HTTPException(404, "profile_not_found")
        sym = prof["symbol"]
        params = dict(prof["initial_params"])
        params.update(prof.get("tactical_params") or {})
        return sym, params, prof
    sym = normalize_usdt_perp_symbol(body_symbol or "")
    if not sym or not is_research_symbol_allowed(sym):
        raise HTTPException(
            400,
            "symbol_not_allowed: 需为合法 XXXUSDT 代码",
        )
    params = build_initial_params(
        template=body_template or "balanced",
        overrides=body_params,
    )
    return sym, params, None


@router.get("/universe")
async def get_universe(refresh: bool = False):
    from moss_quant.kline_cache import catalog_entry, load_cached
    from moss_quant.universe import list_universe

    items = list_universe()
    if refresh:
        for it in items[:8]:
            try:
                df = load_cached(it["symbol"], refresh=True)
                conn = _conn()
                try:
                    from moss_quant.kline_cache import update_kline_meta

                    update_kline_meta(conn, it["symbol"], df)
                    conn.commit()
                    it.update(catalog_entry(it["symbol"], df))
                finally:
                    conn.close()
            except Exception as e:
                it["cache_error"] = str(e)
    return {"symbols": items, "count": len(items)}


@router.get("/profiles")
async def list_profiles():
    from moss_quant.db import row_to_profile

    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT * FROM moss_profiles ORDER BY id DESC"
        ).fetchall()
        return {"profiles": [row_to_profile(r) for r in rows]}
    finally:
        conn.close()


@router.post("/profiles/from-daily")
async def create_profile_from_daily(body: ProfileFromDailyCreate):
    """从最近一次每日寻优结果加入纸面 Profile（不自动删除、启用由用户决定）。"""
    from moss_quant.daily_optimize_service import import_profile_from_daily

    conn = _conn()
    try:
        try:
            prof = import_profile_from_daily(
                conn,
                body.symbol,
                enabled=body.enabled,
                name=body.name,
                update_existing=body.update_existing,
            )
        except ValueError as e:
            code = str(e)
            if code in (
                "symbol_not_allowed",
                "daily_item_not_found",
                "profile_already_exists",
                "max_active_profiles_reached",
                "symbol_already_active",
            ):
                raise HTTPException(400, code) from e
            raise HTTPException(400, "import_failed") from e
        return {"ok": True, "profile": prof}
    finally:
        conn.close()


@router.post("/profiles")
async def create_profile(body: ProfileCreate):
    from moss_quant import config as cfg
    from moss_quant.db import _utc_now, count_enabled_profiles
    from moss_quant.params import build_initial_params
    from moss_quant.universe import active_symbols_taken, is_symbol_allowed

    sym = body.symbol.strip().upper()
    if not is_symbol_allowed(sym):
        raise HTTPException(400, "symbol_not_allowed")

    conn = _conn()
    try:
        if body.enabled:
            if count_enabled_profiles(conn) >= cfg.MOSS_QUANT_MAX_ACTIVE_PROFILES:
                raise HTTPException(400, "max_active_profiles_reached")
            if sym in active_symbols_taken(conn):
                raise HTTPException(400, "symbol_already_active")

        initial = build_initial_params(
            template=body.template,
            overrides=body.param_overrides,
        )
        now = _utc_now()
        equity = float(
            body.virtual_equity_usdt or cfg.MOSS_QUANT_DEFAULT_CAPITAL
        )
        cur = conn.execute(
            """INSERT INTO moss_profiles(
                name, symbol, template, enabled, profile_source,
                initial_params_json, tactical_params_json,
                virtual_equity_usdt, evolution_enabled, created_at_utc, updated_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                body.name,
                sym,
                body.template,
                1 if body.enabled else 0,
                "manual",
                json.dumps(initial, ensure_ascii=False),
                json.dumps(initial, ensure_ascii=False),
                equity,
                1 if body.evolution_enabled else 0,
                now,
                now,
            ),
        )
        conn.commit()
        pid = int(cur.lastrowid)
        from moss_quant.db import get_profile

        return {"ok": True, "profile": get_profile(conn, pid)}
    finally:
        conn.close()


@router.patch("/profiles/{profile_id}")
async def patch_profile(profile_id: int, body: ProfilePatch):
    from moss_quant import config as cfg
    from moss_quant.db import _utc_now, count_enabled_profiles, get_profile
    from moss_quant.universe import active_symbols_taken

    conn = _conn()
    try:
        prof = get_profile(conn, profile_id)
        if not prof:
            raise HTTPException(404, "profile_not_found")
        if body.enabled is True and not prof["enabled"]:
            if count_enabled_profiles(conn) >= cfg.MOSS_QUANT_MAX_ACTIVE_PROFILES:
                raise HTTPException(400, "max_active_profiles_reached")
            taken = active_symbols_taken(conn, exclude_profile_id=profile_id)
            if prof["symbol"] in taken:
                raise HTTPException(400, "symbol_already_active")
        sets = ["updated_at_utc=?"]
        vals: list[Any] = [_utc_now()]
        if body.name is not None:
            sets.append("name=?")
            vals.append(body.name)
        if body.enabled is not None:
            sets.append("enabled=?")
            vals.append(1 if body.enabled else 0)
        if body.virtual_equity_usdt is not None:
            sets.append("virtual_equity_usdt=?")
            vals.append(float(body.virtual_equity_usdt))
        if body.tactical_params is not None:
            sets.append("tactical_params_json=?")
            vals.append(json.dumps(body.tactical_params, ensure_ascii=False))
        vals.append(profile_id)
        conn.execute(
            f"UPDATE moss_profiles SET {', '.join(sets)} WHERE id=?", vals
        )
        conn.commit()
        return {"ok": True, "profile": get_profile(conn, profile_id)}
    finally:
        conn.close()


@router.delete("/profiles/{profile_id}")
async def delete_profile(profile_id: int):
    from moss_quant.db import delete_profile as db_delete_profile

    conn = _conn()
    try:
        try:
            deleted = db_delete_profile(conn, profile_id)
        except ValueError as e:
            if str(e) == "profile_has_open_position":
                raise HTTPException(400, "profile_has_open_position") from e
            raise
        if not deleted:
            raise HTTPException(404, "profile_not_found")
        conn.commit()
        return {"ok": True, **deleted}
    finally:
        conn.close()


@router.post("/profiles/{profile_id}/apply-final-params")
async def post_apply_final_params(profile_id: int, body: ApplyFinalParamsRequest):
    from moss_quant.db import _utc_now, get_profile
    from moss_quant.params import extract_tactical_params

    conn = _conn()
    try:
        prof = get_profile(conn, profile_id)
        if not prof:
            raise HTTPException(404, "profile_not_found")
        initial = dict(prof.get("initial_params") or {})
        final = body.params
        if body.run_id:
            row = conn.execute(
                "SELECT result_json, mode FROM moss_backtest_runs WHERE id=?",
                (int(body.run_id),),
            ).fetchone()
            if not row:
                raise HTTPException(404, "run_not_found")
            if str(row["mode"] or "") not in ("evolve_final", "evolve_baseline"):
                raise HTTPException(400, "run_has_no_final_params")
            result = json.loads(row["result_json"] or "{}")
            final = result.get("final_params") or final
        if not final or not isinstance(final, dict):
            raise HTTPException(
                400,
                "final_params_missing_pass_params_or_evolve_final_run_id",
            )
        tactical = extract_tactical_params(final, initial)
        now = _utc_now()
        conn.execute(
            """UPDATE moss_profiles SET tactical_params_json=?, updated_at_utc=? WHERE id=?""",
            (json.dumps(tactical, ensure_ascii=False), now, profile_id),
        )
        conn.commit()
        updated = get_profile(conn, profile_id)
        return {
            "ok": True,
            "profile_id": profile_id,
            "tactical_params": tactical,
            "entry_threshold": tactical.get("entry_threshold"),
            "profile": updated,
        }
    finally:
        conn.close()


@router.post("/backtest")
async def post_backtest(body: BacktestRequest):
    from moss_quant.backtest_service import run_full_backtest
    from moss_quant.db import _utc_now

    sym, params, prof = _resolve_symbol_params(
        body.symbol, body.params, body.template, body.profile_id
    )
    result = run_full_backtest(
        symbol=sym,
        params=params,
        capital=body.capital,
        refresh_klines=body.refresh_klines,
    )
    conn = _conn()
    try:
        cur = conn.execute(
            """INSERT INTO moss_backtest_runs(
                profile_id, mode, symbol, segment_bars, initial_params_json,
                result_json, summary_json, created_at_utc)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                body.profile_id,
                "full",
                sym,
                0,
                json.dumps(params, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
                json.dumps(result.get("summary"), ensure_ascii=False),
                _utc_now(),
            ),
        )
        conn.commit()
        run_id = int(cur.lastrowid)
    finally:
        conn.close()
    return {"ok": True, "run_id": run_id, **result}


@router.post("/optimize")
async def post_optimize(body: OptimizeRequest):
    """遍历 4 模板 × 战术网格，返回收益最高的策略+参数列表。"""
    from moss_quant.optimize_service import run_strategy_optimize
    from moss_quant.db import _utc_now, get_profile
    from moss_quant.universe import is_research_symbol_allowed, normalize_usdt_perp_symbol

    if body.profile_id:
        sym, _, _prof = _resolve_symbol_params(
            None, None, None, body.profile_id
        )
    else:
        sym = normalize_usdt_perp_symbol(body.symbol or "")
        if not sym or not is_research_symbol_allowed(sym):
            raise HTTPException(
                400,
                "symbol_not_allowed: 需为合法 XXXUSDT 代码",
            )

    try:
        out = run_strategy_optimize(
            symbol=sym,
            capital=body.capital,
            refresh_klines=body.refresh_klines,
            top_n=body.top_n,
            max_combinations=body.max_combinations,
        )
    except Exception as e:
        logger.exception("moss optimize failed")
        raise HTTPException(500, f"optimize_failed: {e}") from e

    applied = None
    pid = body.apply_best_tactical_to_profile_id
    best = out.get("best")
    if pid and best and best.get("tactical_params"):
        conn = _conn()
        try:
            if not get_profile(conn, int(pid)):
                raise HTTPException(404, "profile_not_found")
            now = _utc_now()
            conn.execute(
                """UPDATE moss_profiles SET tactical_params_json=?, updated_at_utc=? WHERE id=?""",
                (
                    json.dumps(best["tactical_params"], ensure_ascii=False),
                    now,
                    int(pid),
                ),
            )
            conn.commit()
            applied = {
                "profile_id": int(pid),
                "tactical_params": best["tactical_params"],
                "note": "template_unchanged_create_new_profile_if_needed",
                "suggested_template": best.get("template"),
            }
        finally:
            conn.close()

    return {**out, "applied": applied}


@router.post("/evolve/baseline")
async def post_evolve_baseline(body: EvolveBaselineRequest):
    from moss_quant import config as cfg
    from moss_quant.evolve_service import run_segmented_evolve
    from moss_quant.db import _utc_now

    sym, params, _ = _resolve_symbol_params(
        body.symbol, body.params, body.template, body.profile_id
    )
    out = run_segmented_evolve(
        symbol=sym,
        initial_params=params,
        evolution_schedule=None,
        segment_bars=body.segment_bars,
        capital=body.capital,
        refresh_klines=body.refresh_klines,
    )
    conn = _conn()
    try:
        cur = conn.execute(
            """INSERT INTO moss_backtest_runs(
                profile_id, mode, symbol, segment_bars, initial_params_json,
                result_json, evolution_log_json, summary_json, created_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                body.profile_id,
                "evolve_baseline",
                sym,
                body.segment_bars or cfg.MOSS_QUANT_SEGMENT_BARS,
                json.dumps(params, ensure_ascii=False),
                json.dumps(
                    {
                        "backtest_result": out.get("backtest_result"),
                        "equity_curve": out.get("equity_curve"),
                        "final_params": out.get("final_params"),
                    },
                    ensure_ascii=False,
                ),
                json.dumps(out.get("evolution_log"), ensure_ascii=False),
                json.dumps(out.get("summary"), ensure_ascii=False),
                _utc_now(),
            ),
        )
        conn.commit()
        run_id = int(cur.lastrowid)
    finally:
        conn.close()
    return {"ok": True, "run_id": run_id, **out}


@router.post("/evolve/reflect")
async def post_evolve_reflect(body: EvolveReflectRequest):
    from moss_quant.reflect import generate_evolution_schedule

    conn = _conn()
    try:
        row = conn.execute(
            "SELECT * FROM moss_backtest_runs WHERE id=?",
            (body.baseline_run_id,),
        ).fetchone()
        if not row:
            raise HTTPException(404, "run_not_found")
        if str(row["mode"] or "") != "evolve_baseline":
            raise HTTPException(400, "reflect_requires_evolve_baseline_run")
        initial = json.loads(row["initial_params_json"] or "{}")
        evo_log = json.loads(row["evolution_log_json"] or "[]")
        if not evo_log:
            raise HTTPException(400, "evolution_log_empty")
        n_seg = len(evo_log)
    finally:
        conn.close()

    try:
        schedule = generate_evolution_schedule(
            initial_params=initial,
            evolution_log=evo_log,
            n_segments=n_seg,
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    except ValueError as e:
        raise HTTPException(502, str(e)) from e

    conn = _conn()
    try:
        conn.execute(
            "UPDATE moss_backtest_runs SET schedule_json=? WHERE id=?",
            (json.dumps(schedule, ensure_ascii=False), body.baseline_run_id),
        )
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, "baseline_run_id": body.baseline_run_id, "schedule": schedule}


@router.post("/evolve/run")
async def post_evolve_run(body: EvolveRunRequest):
    from moss_quant.evolve_service import run_segmented_evolve
    from moss_quant.db import _utc_now

    conn = _conn()
    try:
        row = conn.execute(
            "SELECT * FROM moss_backtest_runs WHERE id=?",
            (body.baseline_run_id,),
        ).fetchone()
        if not row:
            raise HTTPException(404, "run_not_found")
        if str(row["mode"] or "") != "evolve_baseline":
            raise HTTPException(400, "run_requires_evolve_baseline_run")
        sym = str(row["symbol"])
        profile_id = row["profile_id"]
        initial = json.loads(row["initial_params_json"] or "{}")
        if body.schedule:
            schedule = body.schedule
        else:
            schedule = json.loads(row["schedule_json"] or "null")
        if not schedule:
            raise HTTPException(400, "schedule_missing_run_reflect_first")
        seg_bars = int(row["segment_bars"] or 672)
    finally:
        conn.close()

    out = run_segmented_evolve(
        symbol=sym,
        initial_params=initial,
        evolution_schedule=schedule,
        segment_bars=seg_bars,
        refresh_klines=False,
    )
    conn = _conn()
    try:
        cur = conn.execute(
            """INSERT INTO moss_backtest_runs(
                profile_id, mode, symbol, segment_bars, initial_params_json,
                result_json, evolution_log_json, schedule_json, summary_json, created_at_utc)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                profile_id,
                "evolve_final",
                sym,
                seg_bars,
                json.dumps(initial, ensure_ascii=False),
                json.dumps(
                    {
                        "backtest_result": out.get("backtest_result"),
                        "equity_curve": out.get("equity_curve"),
                        "final_params": out.get("final_params"),
                    },
                    ensure_ascii=False,
                ),
                json.dumps(out.get("evolution_log"), ensure_ascii=False),
                json.dumps(schedule, ensure_ascii=False),
                json.dumps(out.get("summary"), ensure_ascii=False),
                _utc_now(),
            ),
        )
        conn.commit()
        run_id = int(cur.lastrowid)
    finally:
        conn.close()
    return {"ok": True, "run_id": run_id, **out}


@router.get("/backtests/{run_id}")
async def get_backtest(run_id: int):
    conn = _conn()
    try:
        row = conn.execute(
            "SELECT * FROM moss_backtest_runs WHERE id=?", (run_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "run_not_found")
        d = dict(row)
        for k in ("initial_params_json", "result_json", "evolution_log_json", "schedule_json", "summary_json"):
            if d.get(k):
                d[k.replace("_json", "")] = json.loads(d[k])
                del d[k]
        return d
    finally:
        conn.close()


@router.get("/summary")
async def get_summary():
    conn = _conn()
    try:
        cur = conn.cursor()
        open_n = int(
            cur.execute(
                """SELECT COUNT(*) FROM moss_signals
                   WHERE outcome IS NULL AND side IN ('LONG','SHORT')"""
            ).fetchone()[0]
            or 0
        )
        settled = int(
            cur.execute("SELECT COUNT(*) FROM moss_settlements").fetchone()[0] or 0
        )
        pnl_row = cur.execute(
            "SELECT COALESCE(SUM(pnl_usdt),0) FROM moss_settlements"
        ).fetchone()
        total_pnl = float(pnl_row[0] or 0)
        profiles = int(
            cur.execute("SELECT COUNT(*) FROM moss_profiles WHERE enabled=1").fetchone()[0]
            or 0
        )
        from moss_quant import config as mq_cfg

        running = False
        try:
            from moss_quant.daily_optimize_service import is_daily_optimize_in_progress

            running = is_daily_optimize_in_progress(conn)
        except Exception:
            pass
        return {
            "ok": True,
            "lane": "moss_quant",
            "open_positions": open_n,
            "settled_count": settled,
            "total_pnl_usdt": total_pnl,
            "enabled_profiles": profiles,
            "max_active_profiles": mq_cfg.MOSS_QUANT_MAX_ACTIVE_PROFILES,
            "data_source": mq_cfg.MOSS_QUANT_DATA_SOURCE,
            "data_source_label": mq_cfg.data_source_label(),
            "kline_limit": mq_cfg.MOSS_QUANT_KLINE_LIMIT,
            "daily_optimize_utc": mq_cfg.MOSS_QUANT_DAILY_OPTIMIZE_UTC,
            "daily_optimize_enabled": mq_cfg.MOSS_QUANT_DAILY_OPTIMIZE_ENABLED,
            "daily_optimize_apply_profiles": mq_cfg.MOSS_QUANT_DAILY_OPTIMIZE_APPLY_PROFILES,
            "daily_optimize_running": running,
        }
    except sqlite3.OperationalError:
        from moss_quant import config as mq_cfg

        return {
            "ok": True,
            "lane": "moss_quant",
            "open_positions": 0,
            "settled_count": 0,
            "total_pnl_usdt": 0.0,
            "enabled_profiles": 0,
            "max_active_profiles": mq_cfg.MOSS_QUANT_MAX_ACTIVE_PROFILES,
            "data_source": mq_cfg.MOSS_QUANT_DATA_SOURCE,
            "data_source_label": mq_cfg.data_source_label(),
            "kline_limit": mq_cfg.MOSS_QUANT_KLINE_LIMIT,
            "daily_optimize_utc": mq_cfg.MOSS_QUANT_DAILY_OPTIMIZE_UTC,
            "daily_optimize_enabled": mq_cfg.MOSS_QUANT_DAILY_OPTIMIZE_ENABLED,
            "daily_optimize_apply_profiles": mq_cfg.MOSS_QUANT_DAILY_OPTIMIZE_APPLY_PROFILES,
            "daily_optimize_running": False,
        }
    finally:
        conn.close()


@router.get("/paper-scan/latest")
async def get_paper_scan_latest():
    """最近一次 15m 纸面扫描摘要（与 Railway `[moss]` 日志同风格）。"""
    from moss_quant.paper_scanner import (
        append_missing_open_position_details,
        enrich_scan_details_with_positions,
        refresh_live_open_signals,
        scan_detail_lines,
    )

    conn = _conn()
    try:
        open_map = refresh_live_open_signals(conn)
        open_hold_count = len(open_map)
        row = conn.execute(
            """SELECT id, ran_at_utc, profiles_scanned, opens, closes, detail_json
               FROM moss_paper_runs ORDER BY id DESC LIMIT 1"""
        ).fetchone()
        if not row:
            details = append_missing_open_position_details(conn, [], open_map)
            details = enrich_scan_details_with_positions(details, open_map)
            return {
                "ok": True,
                "has_run": False,
                "has_open_positions": bool(open_map),
                "ran_at_utc": None,
                "profiles_scanned": 0,
                "opens": 0,
                "closes": 0,
                "lines": scan_detail_lines(details),
                "details": details,
                "open_positions": list(open_map.values()),
                "open_hold_count": open_hold_count,
            }
        details: List[Dict[str, Any]] = []
        raw = row["detail_json"]
        if raw:
            try:
                details = json.loads(raw)
            except json.JSONDecodeError:
                details = []
        details = append_missing_open_position_details(conn, details, open_map)
        details = enrich_scan_details_with_positions(details, open_map)
        lines = scan_detail_lines(details)
        return {
            "ok": True,
            "has_run": True,
            "run_id": int(row["id"]),
            "ran_at_utc": row["ran_at_utc"],
            "profiles_scanned": int(row["profiles_scanned"] or 0),
            "opens": int(row["opens"] or 0),
            "closes": int(row["closes"] or 0),
            "lines": lines,
            "details": details,
            "open_positions": list(open_map.values()),
            "open_hold_count": open_hold_count,
        }
    except sqlite3.OperationalError:
        return {
            "ok": True,
            "has_run": False,
            "ran_at_utc": None,
            "profiles_scanned": 0,
            "opens": 0,
            "closes": 0,
            "lines": [],
            "details": [],
            "open_positions": [],
        }
    finally:
        conn.close()


@router.get("/signals")
async def get_signals(profile_id: Optional[int] = None):
    from moss_quant.paper_scanner import (
        refresh_live_open_signals,
        serialize_signal_rows,
    )

    conn = _conn()
    try:
        try:
            refresh_live_open_signals(conn)
        except Exception as e:
            logger.warning("[moss] signals refresh marks: %s", e)
        if profile_id:
            rows = conn.execute(
                """SELECT * FROM moss_signals WHERE profile_id=?
                   ORDER BY recorded_at_utc DESC LIMIT 200""",
                (profile_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM moss_signals
                   ORDER BY CASE WHEN outcome IS NULL THEN 0 ELSE 1 END,
                            recorded_at_utc DESC LIMIT 200"""
            ).fetchall()
        return {"signals": serialize_signal_rows(conn, rows)}
    except sqlite3.OperationalError:
        return {"signals": []}
    finally:
        conn.close()


@router.get("/daily-optimize/latest")
async def get_daily_optimize_latest():
    """最近一次每日全市场寻优批次（含 23 标的明细）。"""
    from moss_quant.daily_optimize_service import (
        get_latest_daily_batch,
        reconcile_stale_daily_batches,
    )

    conn = _conn()
    try:
        reconcile_stale_daily_batches(conn)
        batch = get_latest_daily_batch(conn)
        if not batch:
            return {"ok": True, "has_batch": False, "batch": None}
        return {"ok": True, "has_batch": True, "batch": batch}
    except sqlite3.OperationalError:
        return {"ok": True, "has_batch": False, "batch": None}
    finally:
        conn.close()


@router.post("/daily-optimize/run")
async def post_daily_optimize_run(body: DailyOptimizeRunRequest = DailyOptimizeRunRequest()):
    """后台触发全宇宙寻优（约 10–20 分钟，勿重复点击）。"""
    import worker_tasks as wt

    from moss_quant import config as cfg
    from moss_quant.daily_optimize_service import is_daily_optimize_in_progress

    if not cfg.MOSS_QUANT_ENABLED:
        raise HTTPException(503, "moss_quant_disabled")

    if wt.moss_daily_optimize_busy():
        return {
            "ok": True,
            "started": False,
            "already_running": True,
            "message": "daily_optimize_already_running",
        }

    conn = _conn()
    try:
        if is_daily_optimize_in_progress(conn):
            return {
                "ok": True,
                "started": False,
                "already_running": True,
                "message": "daily_optimize_already_running",
            }
    finally:
        conn.close()

    threading.Thread(
        target=wt.run_moss_daily_optimize_task,
        kwargs={
            "capital": body.capital,
            "refresh_klines": body.refresh_klines,
            "apply_profiles": body.apply_profiles,
        },
        daemon=True,
    ).start()
    return {
        "ok": True,
        "started": True,
        "already_running": False,
        "message": "daily_optimize_started",
    }


@router.post("/daily-optimize/apply-profiles/{batch_id}")
async def post_daily_optimize_apply_profiles(batch_id: int):
    """为指定批次写入达标/不达标标注（不创建纸面 Profile）。"""
    from moss_quant.daily_optimize_service import annotate_daily_batch_items

    conn = _conn()
    try:
        row = conn.execute(
            "SELECT id, status FROM moss_daily_optimize_batches WHERE id=?",
            (int(batch_id),),
        ).fetchone()
        if not row:
            raise HTTPException(404, "batch_not_found")
        if str(row["status"]) == "running":
            raise HTTPException(409, "batch_still_running")
        stats = annotate_daily_batch_items(conn, int(batch_id))
        return {"ok": True, "batch_id": int(batch_id), "annotate": stats}
    finally:
        conn.close()


@router.post("/maintenance/clear-db")
async def clear_db(_: None = Depends(require_maintenance_token)):
    conn = _conn()
    deleted: dict[str, int] = {}
    tables = (
        ("moss_settlements", "deleted_moss_settlements"),
        ("moss_signals", "deleted_moss_signals"),
        ("moss_paper_runs", "deleted_moss_paper_runs"),
        ("moss_backtest_runs", "deleted_moss_backtest_runs"),
        ("moss_kline_meta", "deleted_moss_kline_meta"),
        ("moss_profiles", "deleted_moss_profiles"),
    )
    try:
        for table, key in tables:
            try:
                n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                conn.execute(f"DELETE FROM {table}")
                deleted[key] = int(n or 0)
            except sqlite3.OperationalError:
                deleted[key] = 0
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, **deleted}
