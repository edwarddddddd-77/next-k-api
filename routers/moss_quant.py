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


class DailyCoreSymbolAdd(BaseModel):
    symbol: str = Field(..., min_length=2, max_length=24)
    note: Optional[str] = Field(None, max_length=128)


class McapScanRunRequest(BaseModel):
    capital: Optional[float] = None
    refresh_klines: Optional[bool] = None


def _conn():
    from accumulation_radar import init_db

    c = init_db()
    c.row_factory = sqlite3.Row
    return c


def _summarize_protocol_moss(
    account: Dict[str, Any],
    positions: List[Dict[str, Any]],
    enabled_profiles: int,
) -> Dict[str, Any]:
    open_rows = [
        p for p in positions or [] if str(p.get("status") or "").lower() == "open"
    ]
    closed_rows = [
        p for p in positions or [] if str(p.get("status") or "").lower() == "closed"
    ]
    total_pnl = round(sum(float(p.get("pnl_usdt") or 0) for p in closed_rows), 4)

    per_profile_map: Dict[int, Dict[str, Any]] = {}
    for row in closed_rows:
        pid = row.get("profile_id")
        if pid is None:
            continue
        pid_i = int(pid)
        item = per_profile_map.setdefault(
            pid_i,
            {
                "profile_id": pid_i,
                "symbol": str(row.get("symbol") or "").upper(),
                "settled_count": 0,
                "total_pnl_usdt": 0.0,
            },
        )
        if not item.get("symbol") and row.get("symbol"):
            item["symbol"] = str(row.get("symbol") or "").upper()
        item["settled_count"] += 1
        item["total_pnl_usdt"] = round(
            float(item["total_pnl_usdt"]) + float(row.get("pnl_usdt") or 0),
            4,
        )

    open_profile_map: Dict[int, Dict[str, Any]] = {}
    for row in open_rows:
        pid = row.get("profile_id")
        if pid is None:
            continue
        pid_i = int(pid)
        item = open_profile_map.setdefault(
            pid_i,
            {
                "profile_id": pid_i,
                "symbol": str(row.get("symbol") or "").upper(),
                "open_count": 0,
                "unrealized_pnl_usdt": 0.0,
            },
        )
        if not item.get("symbol") and row.get("symbol"):
            item["symbol"] = str(row.get("symbol") or "").upper()
        item["open_count"] += 1
        item["unrealized_pnl_usdt"] = round(
            float(item["unrealized_pnl_usdt"]) + _position_unrealized_pnl(row),
            4,
        )

    wallet_balance = float(account.get("wallet_balance_usdt") or 0)
    profile_capital = (
        round(wallet_balance / int(enabled_profiles), 4)
        if int(enabled_profiles or 0) > 0
        else None
    )
    return {
        "ok": True,
        "mode": "live",
        "lane": "moss_quant",
        "open_positions": len(open_rows),
        "settled_count": len(closed_rows),
        "total_pnl_usdt": total_pnl,
        "wallet_initial_usdt": round(wallet_balance - total_pnl, 4),
        "wallet_balance_usdt": wallet_balance,
        "available_balance_usdt": float(account.get("available_balance_usdt") or 0),
        "profile_capital_usdt": profile_capital,
        "enabled_profiles": int(enabled_profiles or 0),
        "per_profile": [
            per_profile_map[k] for k in sorted(per_profile_map.keys())
        ],
        "open_by_profile": [
            open_profile_map[k] for k in sorted(open_profile_map.keys())
        ],
        "protocol_moss": account.get("moss_quant") or {},
    }


def _moss_optimize_policy(mq_cfg) -> Dict[str, Any]:
    return {
        "train_ratio": mq_cfg.MOSS_QUANT_OPTIMIZE_TRAIN_RATIO,
        "require_validation": mq_cfg.MOSS_QUANT_OPTIMIZE_REQUIRE_VALIDATION,
        "min_train_trades": mq_cfg.MOSS_QUANT_OPTIMIZE_MIN_TRAIN_TRADES,
        "min_val_trades": mq_cfg.MOSS_QUANT_OPTIMIZE_MIN_VAL_TRADES,
        "max_train_drawdown": mq_cfg.MOSS_QUANT_OPTIMIZE_MAX_TRAIN_DRAWDOWN,
        "max_val_drawdown": mq_cfg.MOSS_QUANT_OPTIMIZE_MAX_VAL_DRAWDOWN,
        "validation_top_k": mq_cfg.MOSS_QUANT_OPTIMIZE_VALIDATION_TOP_K,
        "full_risk_slots": mq_cfg.MOSS_QUANT_OPTIMIZE_FULL_RISK_SLOTS,
        "mcap_observation_days": mq_cfg.MOSS_QUANT_MCAP_OBSERVATION_DAYS,
    }


def _moss_runtime_fields(conn, mq_cfg) -> Dict[str, Any]:
    running = False
    mcap_running = False
    daily_pools: dict = {}
    try:
        from moss_quant.daily_optimize_service import (
            is_daily_optimize_in_progress,
            summarize_latest_daily_pools,
        )
        from moss_quant.mcap_scan_service import is_mcap_scan_in_progress

        running = is_daily_optimize_in_progress(conn)
        mcap_running = is_mcap_scan_in_progress(conn)
        daily_pools = summarize_latest_daily_pools(conn)
    except Exception:
        pass
    return {
        "max_active_profiles": mq_cfg.MOSS_QUANT_MAX_ACTIVE_PROFILES,
        "data_source": mq_cfg.MOSS_QUANT_DATA_SOURCE,
        "data_source_label": mq_cfg.data_source_label(),
        "kline_limit": mq_cfg.MOSS_QUANT_KLINE_LIMIT,
        "daily_optimize_utc": mq_cfg.MOSS_QUANT_DAILY_OPTIMIZE_UTC,
        "daily_optimize_enabled": mq_cfg.MOSS_QUANT_DAILY_OPTIMIZE_ENABLED,
        "daily_optimize_apply_profiles": mq_cfg.MOSS_QUANT_DAILY_OPTIMIZE_APPLY_PROFILES,
        "daily_optimize_running": running,
        "mcap_scan_running": mcap_running,
        "mcap_scan_pool_limit": mq_cfg.MOSS_QUANT_MCAP_SCAN_POOL_LIMIT,
        "optimize_policy": _moss_optimize_policy(mq_cfg),
        "daily_optimize_pools": daily_pools,
        "pool_governance": _pool_governance_summary(conn),
    }


def _moss_live_unavailable_summary(
    conn,
    mq_cfg,
    *,
    reason: str,
    enabled_profiles: int = 0,
) -> Dict[str, Any]:
    return {
        "ok": True,
        "mode": "live_unavailable",
        "lane": "moss_quant",
        "protocol_error": reason,
        "open_positions": 0,
        "settled_count": 0,
        "total_pnl_usdt": 0.0,
        "wallet_initial_usdt": None,
        "wallet_balance_usdt": None,
        "available_balance_usdt": None,
        "profile_capital_usdt": None,
        "enabled_profiles": int(enabled_profiles or 0),
        "per_profile": [],
        "per_symbol": [],
        "open_by_profile": [],
        "protocol_moss": {},
        **_moss_runtime_fields(conn, mq_cfg),
    }


def _position_unrealized_pnl(p: Dict[str, Any]) -> float:
    for key in ("unrealized_pnl_usdt", "upnl"):
        if key in p and p.get(key) is not None:
            return float(p.get(key) or 0)
    if str(p.get("status") or "").lower() == "open" and "pnl_usdt" in p:
        return float(p.get("pnl_usdt") or 0)
    return 0.0


def _position_mark_price(p: Dict[str, Any]) -> Any:
    for key in ("mark_price", "close_price", "entry_price"):
        if key in p and p.get(key) is not None:
            return p.get(key)
    return None


def _position_outcome_fields(p: Dict[str, Any]) -> Dict[str, Any]:
    status = str(p.get("status") or "").lower()
    close_reason = p.get("close_reason")
    if status == "open":
        return {"outcome": None, "outcome_at_utc": None, "exit_rule": None}
    if status == "pending_entry":
        return {
            "outcome": "pending_entry",
            "outcome_at_utc": None,
            "exit_rule": None,
        }
    if status == "cancelled_pending":
        return {
            "outcome": close_reason or "cancelled_pending",
            "outcome_at_utc": p.get("closed_at"),
            "exit_rule": close_reason,
        }
    if status == "closed":
        return {
            "outcome": close_reason or "closed",
            "outcome_at_utc": p.get("closed_at"),
            "exit_rule": close_reason,
        }
    return {
        "outcome": close_reason or status or None,
        "outcome_at_utc": p.get("closed_at"),
        "exit_rule": close_reason,
    }


def _position_to_moss_signal_row(p: Dict[str, Any]) -> Dict[str, Any]:
    outcome_fields = _position_outcome_fields(p)
    return {
        "id": p.get("id"),
        "profile_id": p.get("profile_id"),
        "recorded_at_utc": p.get("opened_at"),
        "side": p.get("side"),
        "symbol": p.get("symbol"),
        "entry_price": p.get("entry_price"),
        "virtual_notional_usdt": p.get("notional_usdt"),
        "mark_price": _position_mark_price(p),
        "unrealized_pnl_usdt": _position_unrealized_pnl(p),
        "outcome": outcome_fields["outcome"],
        "outcome_at_utc": outcome_fields["outcome_at_utc"],
        "exit_price": p.get("close_price"),
        "pnl_usdt": p.get("pnl_usdt"),
        "exit_rule": outcome_fields["exit_rule"],
        "leverage": p.get("leverage"),
        "client_ref": p.get("client_ref"),
        "position_id": p.get("id"),
        "source": p.get("source"),
    }


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


@router.get("/daily-core-universe")
async def get_daily_core_universe():
    """每日寻优必扫标的（moss_daily_core_symbols，默认 25：主板 23 + ICP + TON）。"""
    from moss_quant.db import list_daily_core_symbols
    from moss_quant.universe import list_daily_core_universe

    conn = _conn()
    try:
        rows = list_daily_core_symbols(conn)
        items = list_daily_core_universe(conn)
        return {
            "ok": True,
            "count": len(items),
            "items": items,
            "rows": rows,
        }
    finally:
        conn.close()


@router.post("/daily-core-symbols")
async def post_daily_core_symbol(body: DailyCoreSymbolAdd):
    """将标的加入每日寻优表 moss_daily_core_symbols（扩展寻优看盘可点选）。"""
    from moss_quant.db import add_symbol_to_daily_core, list_daily_core_symbols
    from moss_quant.universe import is_research_symbol_allowed, normalize_usdt_perp_symbol

    sym = normalize_usdt_perp_symbol(body.symbol)
    if not sym or not is_research_symbol_allowed(sym):
        raise HTTPException(400, "symbol_not_allowed")

    conn = _conn()
    try:
        try:
            out = add_symbol_to_daily_core(
                conn,
                sym,
                note=(body.note or "from_ui"),
            )
        except ValueError as e:
            code = str(e)
            if code in ("invalid_symbol", "symbol_not_on_binance_perp"):
                raise HTTPException(400, code) from e
            raise HTTPException(400, "add_daily_core_failed") from e
        enabled = [r["symbol"] for r in list_daily_core_symbols(conn)]
        return {
            **out,
            "daily_core_count": len(enabled),
            "daily_core_symbols": enabled,
        }
    finally:
        conn.close()


@router.get("/universe")
async def get_universe(refresh: bool = False):
    from moss_quant.kline_cache import catalog_entry, load_cached
    from moss_quant.universe import list_universe

    conn = _conn()
    try:
        items = list_universe(conn)
        if refresh:
            for it in items[:8]:
                try:
                    df = load_cached(it["symbol"], refresh=True)
                    from moss_quant.kline_cache import update_kline_meta

                    update_kline_meta(conn, it["symbol"], df)
                    conn.commit()
                    it.update(catalog_entry(it["symbol"], df))
                except Exception as e:
                    it["cache_error"] = str(e)
        return {"symbols": items, "count": len(items)}
    finally:
        conn.close()


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

    conn = _conn()
    try:
        sym = body.symbol.strip().upper()
        if not is_symbol_allowed(sym, conn=conn):
            raise HTTPException(400, "symbol_not_allowed")
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
            body.virtual_equity_usdt or cfg.MOSS_QUANT_PROFILE_CAPITAL
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
            if body.enabled is False:
                sets.append("governance_manual_lock=?")
                vals.append(1)
            elif body.enabled is True:
                sets.append("governance_manual_lock=?")
                vals.append(0)
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


@router.post("/maintenance/reconcile-wallet")
async def reconcile_wallet():
    """回补缺失结算行并重算全局纸面钱包（含已删 Profile 的历史盈亏）。"""
    from moss_quant.db import reconcile_moss_wallet

    conn = _conn()
    try:
        out = reconcile_moss_wallet(conn)
        conn.commit()
        return {"ok": True, **out}
    finally:
        conn.close()


def _pool_governance_summary(conn) -> dict:
    try:
        from moss_quant.pool_governance import summarize_pool_governance

        return summarize_pool_governance(conn)
    except Exception:
        return {}


@router.get("/summary")
async def get_summary():
    conn = _conn()
    try:
        from moss_quant import config as mq_cfg

        try:
            from moss_quant.db import count_enabled_profiles
            from moss_quant.protocol_client import ProtocolClient

            protocol = ProtocolClient.from_env()
            enabled_profile_count = count_enabled_profiles(conn)
            if protocol.enabled():
                account = protocol.get_account_summary()
                positions = protocol.get_moss_positions(status=None, limit=1000)
                summary = _summarize_protocol_moss(
                    account=account,
                    positions=positions,
                    enabled_profiles=enabled_profile_count,
                )
                return {
                    **summary,
                    **_moss_runtime_fields(conn, mq_cfg),
                }
            return _moss_live_unavailable_summary(
                conn,
                mq_cfg,
                reason="protocol_api_url_missing",
                enabled_profiles=enabled_profile_count,
            )
        except Exception as e:
            logger.warning("[moss] live protocol summary failed: %s", e)
            try:
                from moss_quant.db import count_enabled_profiles

                enabled_profile_count = count_enabled_profiles(conn)
            except Exception:
                enabled_profile_count = 0
            return _moss_live_unavailable_summary(
                conn,
                mq_cfg,
                reason=str(e),
                enabled_profiles=enabled_profile_count,
            )
    except sqlite3.OperationalError:
        from moss_quant import config as mq_cfg

        return {
            "ok": True,
            "mode": "live_unavailable",
            "lane": "moss_quant",
            "protocol_error": "local_db_unavailable",
            "open_positions": 0,
            "settled_count": 0,
            "total_pnl_usdt": 0.0,
            "wallet_initial_usdt": None,
            "wallet_balance_usdt": None,
            "available_balance_usdt": None,
            "profile_capital_usdt": None,
            "enabled_profiles": 0,
            "max_active_profiles": mq_cfg.MOSS_QUANT_MAX_ACTIVE_PROFILES,
            "data_source": mq_cfg.MOSS_QUANT_DATA_SOURCE,
            "data_source_label": mq_cfg.data_source_label(),
            "kline_limit": mq_cfg.MOSS_QUANT_KLINE_LIMIT,
            "daily_optimize_utc": mq_cfg.MOSS_QUANT_DAILY_OPTIMIZE_UTC,
            "daily_optimize_enabled": mq_cfg.MOSS_QUANT_DAILY_OPTIMIZE_ENABLED,
            "daily_optimize_apply_profiles": mq_cfg.MOSS_QUANT_DAILY_OPTIMIZE_APPLY_PROFILES,
            "daily_optimize_running": False,
            "mcap_scan_running": False,
            "mcap_scan_pool_limit": mq_cfg.MOSS_QUANT_MCAP_SCAN_POOL_LIMIT,
            "optimize_policy": _moss_optimize_policy(mq_cfg),
            "daily_optimize_pools": {},
            "pool_governance": {},
        }
    finally:
        conn.close()


@router.get("/mcap-scan/candidates")
async def get_mcap_scan_candidates():
    """预览市值扩展寻优候选（未跑回测）。"""
    from moss_quant.binance_mcap_universe import build_mcap_scan_candidates
    from moss_quant import config as mq_cfg

    try:
        candidates = build_mcap_scan_candidates(
            mcap_limit=mq_cfg.MOSS_QUANT_MCAP_SCAN_POOL_LIMIT
        )
    except Exception as e:
        raise HTTPException(503, f"mcap_candidates_failed: {e}") from e
    return {
        "ok": True,
        "count": len(candidates),
        "pool_limit": mq_cfg.MOSS_QUANT_MCAP_SCAN_POOL_LIMIT,
        "candidates": candidates,
    }


@router.get("/mcap-scan/latest")
async def get_mcap_scan_latest():
    from moss_quant.mcap_scan_service import (
        get_latest_mcap_scan_batch,
        reconcile_stale_mcap_batches,
    )

    from moss_quant.db import list_daily_core_symbols

    conn = _conn()
    try:
        reconcile_stale_mcap_batches(conn)
        batch = get_latest_mcap_scan_batch(conn)
        daily_core_symbols = [
            str(r["symbol"]).upper()
            for r in list_daily_core_symbols(conn)
            if int(r.get("enabled") or 0) and r.get("symbol")
        ]
        if not batch:
            return {
                "ok": True,
                "has_batch": False,
                "batch": None,
                "daily_core_symbols": daily_core_symbols,
            }
        return {
            "ok": True,
            "has_batch": True,
            "batch": batch,
            "daily_core_symbols": daily_core_symbols,
        }
    except sqlite3.OperationalError:
        return {"ok": True, "has_batch": False, "batch": None, "daily_core_symbols": []}
    finally:
        conn.close()


@router.post("/mcap-scan/run")
async def post_mcap_scan_run(body: McapScanRunRequest = McapScanRunRequest()):
    import worker_tasks as wt

    from moss_quant import config as mq_cfg
    from moss_quant.mcap_scan_service import is_mcap_scan_in_progress

    if not mq_cfg.MOSS_QUANT_ENABLED:
        raise HTTPException(503, "moss_quant_disabled")

    if wt.moss_mcap_scan_busy():
        return {
            "ok": True,
            "started": False,
            "already_running": True,
            "message": "mcap_scan_already_running",
        }

    conn = _conn()
    try:
        if is_mcap_scan_in_progress(conn):
            return {
                "ok": True,
                "started": False,
                "already_running": True,
                "message": "mcap_scan_already_running",
            }
    finally:
        conn.close()

    threading.Thread(
        target=wt.run_moss_mcap_scan_task,
        kwargs={
            "capital": body.capital,
            "refresh_klines": body.refresh_klines,
        },
        daemon=True,
    ).start()
    return {
        "ok": True,
        "started": True,
        "already_running": False,
        "message": "mcap_scan_started",
    }


@router.get("/paper-scan/latest")
async def get_paper_scan_latest():
    """最近一次 15m 纸面扫描摘要（与 Railway `[moss]` 日志同风格）。"""
    from moss_quant.paper_scanner import (
        append_missing_open_position_details,
        enrich_scan_details_with_positions,
        latest_protocol_open_positions,
        refresh_live_open_signals,
        scan_detail_lines,
    )

    protocol_open_positions: Optional[List[Dict[str, Any]]] = None
    try:
        protocol_open_positions = latest_protocol_open_positions()
    except Exception as e:
        logger.warning("[moss] latest protocol positions failed, fallback local: %s", e)

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
                "mode": "live" if protocol_open_positions is not None else "paper",
                "has_run": False,
                "has_open_positions": bool(open_map),
                "ran_at_utc": None,
                "profiles_scanned": 0,
                "opens": 0,
                "closes": 0,
                "lines": scan_detail_lines(details),
                "details": details,
                "open_positions": (
                    protocol_open_positions
                    if protocol_open_positions is not None
                    else list(open_map.values())
                ),
                "open_hold_count": (
                    len(protocol_open_positions)
                    if protocol_open_positions is not None
                    else open_hold_count
                ),
                "paper_open_positions": list(open_map.values()),
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
            "mode": "live" if protocol_open_positions is not None else "paper",
            "has_run": True,
            "run_id": int(row["id"]),
            "ran_at_utc": row["ran_at_utc"],
            "profiles_scanned": int(row["profiles_scanned"] or 0),
            "opens": int(row["opens"] or 0),
            "closes": int(row["closes"] or 0),
            "lines": lines,
            "details": details,
            "open_positions": (
                protocol_open_positions
                if protocol_open_positions is not None
                else list(open_map.values())
            ),
            "open_hold_count": (
                len(protocol_open_positions)
                if protocol_open_positions is not None
                else open_hold_count
            ),
            "paper_open_positions": list(open_map.values()),
        }
    except sqlite3.OperationalError:
        return {
            "ok": True,
            "mode": "live" if protocol_open_positions is not None else "paper",
            "has_run": False,
            "ran_at_utc": None,
            "profiles_scanned": 0,
            "opens": 0,
            "closes": 0,
            "lines": [],
            "details": [],
            "open_positions": protocol_open_positions or [],
            "paper_open_positions": [],
        }
    finally:
        conn.close()


@router.get("/signals")
async def get_signals(profile_id: Optional[int] = None):
    try:
        from moss_quant.protocol_client import ProtocolClient

        protocol = ProtocolClient.from_env()
        if protocol.enabled():
            positions = protocol.get_moss_positions(status=None, limit=1000)
            if profile_id is not None:
                positions = [
                    p
                    for p in positions
                    if p.get("profile_id") is not None
                    and int(p.get("profile_id")) == int(profile_id)
                ]
            return {
                "mode": "live",
                "signals": [_position_to_moss_signal_row(p) for p in positions],
            }
    except Exception as e:
        logger.warning("[moss] live protocol signals failed, fallback local: %s", e)

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
    """最近一次每日核心寻优批次（moss_daily_core_symbols，默认 25 标的）。"""
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
    """为指定批次写入达标标注，并将已启用/有持仓 Profile 同步为本批次最优策略（有仓则触发纸面扫描）。"""
    from moss_quant.daily_optimize_service import (
        annotate_daily_batch_items,
        sync_enabled_profiles_from_batch,
    )
    from moss_quant.pool_governance import apply_pool_governance

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
        bid = int(batch_id)
        annotate_stats = annotate_daily_batch_items(conn, bid)
        sync_stats = sync_enabled_profiles_from_batch(conn, bid)
        governance_stats = apply_pool_governance(conn, bid)
        return {
            "ok": True,
            "batch_id": bid,
            "annotate": annotate_stats,
            "sync_profiles": sync_stats,
            "pool_governance": governance_stats,
        }
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
        ("moss_mcap_scan_items", "deleted_moss_mcap_scan_items"),
        ("moss_mcap_scan_batches", "deleted_moss_mcap_scan_batches"),
        ("moss_daily_optimize_items", "deleted_moss_daily_optimize_items"),
        ("moss_daily_optimize_batches", "deleted_moss_daily_optimize_batches"),
    )
    try:
        for table, key in tables:
            try:
                n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                conn.execute(f"DELETE FROM {table}")
                deleted[key] = int(n or 0)
            except sqlite3.OperationalError:
                deleted[key] = 0
        from moss_quant.db import reset_moss_wallet

        reset_moss_wallet(conn)
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, **deleted}
