"""Moss2 API — 与 /api/moss-quant 隔离，复刻 factory HL/EN 回测与纸面。"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from utils.maintenance_auth import require_maintenance_token

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/moss2", tags=["moss2"])


class ProfileCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    symbol: str
    variant: Optional[str] = Field(None, pattern="^(hl|en)$")
    template: Optional[str] = None
    enabled: bool = False
    virtual_equity_usdt: Optional[float] = None
    param_overrides: Optional[dict] = None


class ProfilePatch(BaseModel):
    name: Optional[str] = None
    enabled: Optional[bool] = None
    tactical_params: Optional[dict] = None


class BacktestRequest(BaseModel):
    profile_id: Optional[int] = None
    symbol: Optional[str] = None
    variant: Optional[str] = Field(None, pattern="^(hl|en)$")
    template: Optional[str] = None
    params: Optional[dict] = None
    capital: Optional[float] = None
    limit_bars: Optional[int] = Field(None, ge=100)


def _conn():
    from accumulation_radar import init_db

    return init_db()


@router.get("/summary")
def moss2_summary() -> Dict[str, Any]:
    from moss2 import config as c
    from moss2.dataset import list_en_catalog, list_hl_catalog
    from moss2.db import summarize_lane

    conn = _conn()
    try:
        base = summarize_lane(conn)
    finally:
        conn.close()
    base["runtime"] = c.moss2_runtime_snapshot()
    base["hl_datasets"] = len(list_hl_catalog()) if c.MOSS2_HL_ENABLED else 0
    base["en_datasets"] = len(list_en_catalog())
    base["ops_variant"] = c.MOSS2_OPS_VARIANT
    base["protocol_venue"] = c.MOSS2_PROTOCOL_VENUE
    base["default_variant"] = c.MOSS2_DEFAULT_VARIANT
    base["default_template"] = c.MOSS2_DEFAULT_TEMPLATE
    base["scan_interval_minutes"] = c.MOSS2_SCAN_INTERVAL_MINUTES
    return base


@router.get("/paper-scan/latest")
def moss2_paper_scan_latest() -> Dict[str, Any]:
    from moss2.db import latest_paper_run, list_open_signals

    conn = _conn()
    try:
        run = latest_paper_run(conn)
        open_pos = list_open_signals(conn)
        if not run:
            return {
                "ok": True,
                "lane": "moss2",
                "has_run": False,
                "open_positions": open_pos,
                "details": [],
                "lines": [],
            }
        details = run.get("details") or []
        lines = []
        for d in details:
            if not isinstance(d, dict):
                continue
            label = d.get("label") or ""
            act = d.get("action") or ""
            extra = d.get("reason") or d.get("side") or d.get("rule") or ""
            lines.append(f"{label} {act} {extra}".strip())
        return {
            "ok": True,
            "lane": "moss2",
            "has_run": True,
            **run,
            "lines": lines,
            "open_positions": open_pos,
        }
    finally:
        conn.close()


@router.get("/signals")
def moss2_signals(profile_id: Optional[int] = None, limit: int = 120) -> Dict[str, Any]:
    from moss2.db import list_open_signals, list_signals

    conn = _conn()
    try:
        open_pos = list_open_signals(conn)
        rows = list_signals(conn, profile_id=profile_id, limit=min(limit, 500))
        return {"ok": True, "lane": "moss2", "signals": rows, "open_positions": open_pos}
    finally:
        conn.close()


@router.delete("/profiles/{profile_id}")
def delete_moss2_profile(profile_id: int) -> Dict[str, Any]:
    from moss2.db import delete_profile

    conn = _conn()
    try:
        if not delete_profile(conn, profile_id):
            raise HTTPException(404, "profile not found")
        return {"ok": True, "deleted": profile_id}
    finally:
        conn.close()


@router.get("/onboarding/suggest")
def moss2_onboarding_suggest(
    symbol: str,
    lookback_bars: Optional[int] = None,
    backtest_bars: Optional[int] = None,
    capital: Optional[float] = None,
) -> Dict[str, Any]:
    """Moss2 创建 Profile 建议：近期 regime + 可选四模板 discipline 回测。"""
    from moss2.onboarding import suggest_profile

    if not symbol.strip():
        raise HTTPException(400, "symbol required")
    return suggest_profile(
        symbol,
        lookback_bars=lookback_bars,
        backtest_bars=backtest_bars,
        capital=capital,
    )


@router.get("/onboarding/tradeable")
def moss2_tradeable_symbols() -> Dict[str, Any]:
    """当前 data_cache 中已有 EN CSV 的 symbol（Moss2 可回测范围）。"""
    from moss2.onboarding import list_tradeable_symbols

    syms = list_tradeable_symbols()
    return {
        "lane": "moss2",
        "variant": "en",
        "count": len(syms),
        "symbols": syms,
    }


@router.get("/catalog")
def moss2_catalog() -> Dict[str, Any]:
    from moss2 import config as c
    from moss2.dataset import list_en_catalog, list_hl_catalog

    en = list_en_catalog()
    hl = list_hl_catalog() if c.MOSS2_HL_ENABLED else []
    return {
        "ops_variant": c.MOSS2_OPS_VARIANT,
        "protocol_venue": c.MOSS2_PROTOCOL_VENUE,
        "hl_enabled": c.MOSS2_HL_ENABLED,
        "hl": hl,
        "en": en,
    }


@router.get("/params-schema")
def moss2_params_schema(variant: Optional[str] = None) -> dict:
    from moss2 import config as c
    from moss2.params import load_params_schema

    try:
        v = c.effective_variant(variant)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return load_params_schema(v)


@router.get("/defaults")
def moss2_defaults(
    variant: Optional[str] = None,
    template: Optional[str] = None,
) -> Dict[str, Any]:
    """factory params_schema 全字段默认 + 模板权重 + 性格/战术拆分。"""
    from moss2 import config as c
    from moss2.params import default_params_bundle

    try:
        v = c.effective_variant(variant)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    t = (template or c.MOSS2_DEFAULT_TEMPLATE or "balanced").strip().lower()
    return default_params_bundle(template=t, variant=v)  # type: ignore[arg-type]


@router.get("/profiles")
def list_moss2_profiles(enabled_only: bool = False) -> List[dict]:
    from moss2.db import list_profiles

    conn = _conn()
    try:
        return list_profiles(conn, enabled_only=enabled_only)
    finally:
        conn.close()


@router.post("/profiles")
def create_moss2_profile(body: ProfileCreate) -> dict:
    from moss2 import config as c
    from moss2.db import create_profile, get_profile
    from moss2.params import build_initial_params, split_profile_params

    try:
        variant = c.effective_variant(body.variant)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    template = (body.template or c.MOSS2_DEFAULT_TEMPLATE or "balanced").strip().lower()
    from moss2.dataset import normalize_symbol

    symbol = normalize_symbol(body.symbol, variant=variant)
    merged = build_initial_params(
        template, body.param_overrides, variant=variant  # type: ignore[arg-type]
    )
    initial, tactical = split_profile_params(merged, variant=variant)  # type: ignore[arg-type]
    conn = _conn()
    try:
        try:
            pid = create_profile(
                conn,
                name=body.name,
                symbol=symbol,
                variant=variant,
                template=template,
                enabled=body.enabled,
                initial_params=initial,
                tactical_params=tactical,
                virtual_equity_usdt=float(
                    body.virtual_equity_usdt or c.MOSS2_PROFILE_CAPITAL
                ),
            )
        except sqlite3.IntegrityError as e:
            if body.enabled:
                raise HTTPException(
                    409,
                    "该 symbol+variant 已有启用的 Profile，请先停用或改标的",
                ) from e
            raise HTTPException(409, "Profile 创建冲突") from e
        prof = get_profile(conn, pid)
    finally:
        conn.close()
    return prof or {}


@router.patch("/profiles/{profile_id}")
def patch_moss2_profile(profile_id: int, body: ProfilePatch) -> dict:
    from moss2.db import get_profile, patch_profile

    conn = _conn()
    try:
        if not get_profile(conn, profile_id):
            raise HTTPException(404, "profile not found")
        patch_profile(
            conn,
            profile_id,
            **{k: v for k, v in body.model_dump().items() if v is not None},
        )
        return get_profile(conn, profile_id) or {}
    finally:
        conn.close()


@router.post("/backtest")
def moss2_backtest(body: BacktestRequest) -> Dict[str, Any]:
    from moss2 import config as c
    from moss2.backtest_service import run_factory_backtest, run_profile_backtest
    from moss2.db import get_profile, insert_backtest_run
    from moss2.params import build_initial_params

    conn = _conn()
    try:
        if body.profile_id:
            prof = get_profile(conn, body.profile_id)
            if not prof:
                raise HTTPException(404, "profile not found")
            try:
                c.profile_variant(prof)
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
            out = run_profile_backtest(
                prof, capital=body.capital, limit_bars=body.limit_bars
            )
            pid = body.profile_id
            variant = prof["variant"]
            symbol = prof["symbol"]
        else:
            if not body.symbol:
                raise HTTPException(400, "symbol or profile_id required")
            try:
                variant = c.effective_variant(body.variant)
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
            template = (
                body.template or c.MOSS2_DEFAULT_TEMPLATE or "balanced"
            ).strip().lower()
            from moss2.dataset import normalize_symbol

            symbol = normalize_symbol(body.symbol, variant=variant)
            params = body.params or build_initial_params(
                template, variant=variant  # type: ignore[arg-type]
            )
            out = run_factory_backtest(
                symbol=symbol,
                params=params,
                variant=variant,
                capital=body.capital,
                limit_bars=body.limit_bars,
            )
            pid = None
        summ = dict(out["summary"])
        if out.get("discipline"):
            summ["discipline"] = out["discipline"]
        run_id = insert_backtest_run(
            conn,
            profile_id=pid,
            variant=out["variant"],
            symbol=symbol,
            data_csv=out.get("data_csv"),
            initial_params=out["initial_params"],
            result=out,
            summary=summ,
        )
        if pid and out.get("discipline"):
            from moss2.db import insert_discipline_snapshot
            from moss2.versioning import effective_version

            prof = get_profile(conn, pid)
            insert_discipline_snapshot(
                conn,
                profile_id=pid,
                symbol=symbol,
                variant=str(out["variant"]),
                template=str((prof or {}).get("template") or ""),
                params_version=effective_version(prof or {}),
                data_csv=out.get("data_csv"),
                discipline=out["discipline"],
            )
        out["run_id"] = run_id
        return out
    finally:
        conn.close()


@router.post("/profiles/{profile_id}/evolve")
def moss2_evolve_profile(profile_id: int, force: bool = False) -> Dict[str, Any]:
    from moss2.evolve_service import run_profile_evolve

    conn = _conn()
    try:
        return run_profile_evolve(conn, profile_id, force=force)
    finally:
        conn.close()


@router.post("/profiles/{profile_id}/approve-candidate")
def moss2_approve_candidate(profile_id: int) -> Dict[str, Any]:
    from moss2.evolve_service import approve_candidate

    conn = _conn()
    try:
        out = approve_candidate(conn, profile_id)
        if not out.get("ok"):
            raise HTTPException(400, out.get("reason", "approve_failed"))
        return out
    finally:
        conn.close()


@router.get("/profiles/{profile_id}/discipline/history")
def moss2_discipline_history(profile_id: int, limit: int = 12) -> List[dict]:
    from moss2.db import list_discipline_snapshots

    conn = _conn()
    try:
        return list_discipline_snapshots(conn, profile_id, limit=limit)
    finally:
        conn.close()


@router.post("/evolve/run")
def moss2_evolve_lane() -> Dict[str, Any]:
    from moss2.evolve_service import run_lane_evolve

    conn = _conn()
    try:
        return run_lane_evolve(conn)
    finally:
        conn.close()


@router.post("/paper-scan")
def moss2_paper_scan_trigger() -> Dict[str, Any]:
    from moss2.paper_scanner import run_paper_scan

    conn = _conn()
    try:
        return run_paper_scan(conn)
    finally:
        conn.close()


@router.post("/maintenance/bootstrap-data")
async def moss2_bootstrap_data(
    force: bool = False,
    _: None = Depends(require_maintenance_token),
) -> Dict[str, Any]:
    """手动触发 25 核心币 CSV 拉取（写入 data/moss2_en_data_cache）。"""
    from moss2.data_bootstrap import bootstrap_seed_data

    return bootstrap_seed_data(force=force)


@router.post("/maintenance/cull")
async def moss2_cull_profiles(_: None = Depends(require_maintenance_token)) -> Dict[str, Any]:
    """手动触发 Moss2 淘汰体检（不过关停用；可配置淘汰前重赛四模板）。"""
    from moss2.cull_service import run_lane_cull

    conn = _conn()
    try:
        return run_lane_cull(conn)
    finally:
        conn.close()


@router.post("/maintenance/auto-provision")
async def moss2_auto_provision(
    force_evolve: bool = False,
    _: None = Depends(require_maintenance_token),
) -> Dict[str, Any]:
    """手动触发 Moss2 全自动 Profile 运维（25 核心 suggest→创建→进化→启用）。"""
    from moss2.auto_provision import run_lane_auto_provision

    conn = _conn()
    try:
        return run_lane_auto_provision(conn, force_evolve=force_evolve)
    finally:
        conn.close()


@router.post("/maintenance/clear-db")
async def moss2_clear_db(_: None = Depends(require_maintenance_token)) -> Dict[str, Any]:
    """清空 moss2_* 表（不影响 moss_quant）。"""
    conn = _conn()
    deleted: Dict[str, int] = {}
    tables = (
        ("moss2_signals", "deleted_moss2_signals"),
        ("moss2_paper_runs", "deleted_moss2_paper_runs"),
        ("moss2_discipline_snapshots", "deleted_moss2_discipline_snapshots"),
        ("moss2_backtest_runs", "deleted_moss2_backtest_runs"),
        ("moss2_profiles", "deleted_moss2_profiles"),
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
        logger.warning("moss2 clear-db: %s", deleted)
    finally:
        conn.close()
    return {"ok": True, "lane": "moss2", **deleted}
