"""Moss2 全自动 Profile 运维：seed 币对 → onboarding 建议 → 创建/进化 → 启用。

与 Moss1 daily-optimize 解耦；仅操作 MOSS2_SEED_BASES + suggest_profile 逻辑。
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, List, Optional

from moss2 import config as cfg
from moss2.dataset import normalize_symbol
from moss2.db import create_profile, get_profile, list_profiles, patch_profile
from moss2.evolve_service import _approve_candidate, approve_candidate, run_profile_evolve
from moss2.onboarding import suggest_profile
from moss2.params import build_initial_params, split_profile_params
from moss2.selection import passes_backtest_gates

logger = logging.getLogger(__name__)


def find_profiles_for_symbol(
    conn: sqlite3.Connection, symbol: str, *, variant: Optional[str] = None
) -> List[dict]:
    variant = variant or cfg.MOSS2_OPS_VARIANT
    sym = normalize_symbol(symbol, variant=variant)
    return [
        p
        for p in list_profiles(conn)
        if str(p.get("symbol")).upper() == sym and str(p.get("variant")) == variant
    ]


def _enabled_profile(profiles: List[dict]) -> Optional[dict]:
    for p in profiles:
        if p.get("enabled"):
            return p
    return None


def _pick_work_profile(profiles: List[dict]) -> Optional[dict]:
    if not profiles:
        return None
    enabled = _enabled_profile(profiles)
    if enabled:
        return enabled
    return max(profiles, key=lambda x: int(x.get("id") or 0))


def _needs_full_evolve(suggestion: Dict[str, Any], *, force_evolve: bool) -> bool:
    if force_evolve:
        return True
    if suggestion.get("reason") != "backtest_selection_pass":
        return True
    if not suggestion.get("recommended_params"):
        return True
    return False


def _apply_selection_candidate(
    conn: sqlite3.Connection, profile_id: int, suggestion: Dict[str, Any]
) -> Dict[str, Any]:
    """suggest 已选好参时直接发布，避免重复跑四模板竞赛。"""
    from datetime import datetime, timezone

    best = suggestion.get("selection_best") or {}
    params = suggestion["recommended_params"]
    template = str(suggestion.get("recommended_template") or cfg.MOSS2_DEFAULT_TEMPLATE)
    initial, tactical = split_profile_params(params, variant=cfg.MOSS2_OPS_VARIANT)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ver = f"v{now[:10].replace('-', '')}-{template}"
    disc = best.get("discipline") or {}
    summ = best.get("summary") or {}
    if not summ and (suggestion.get("template_scores") or []):
        for row in suggestion["template_scores"]:
            if row.get("template") == template:
                disc = row.get("discipline") or disc
                summ = row.get("summary") or summ
                break
    candidate = {
        "params_version": ver,
        "template": template,
        "initial_params": initial,
        "tactical_params": tactical,
        "discipline": disc,
        "summary": summ,
    }
    status = "pending"
    if cfg.MOSS2_EVOLVE_AUTO_APPROVE:
        _approve_candidate(conn, profile_id, candidate)
        status = "approved"
    from moss2.db import patch_profile_evolution
    import json

    patch_profile_evolution(
        conn,
        profile_id,
        candidate_params_json=json.dumps(candidate, ensure_ascii=False),
        evolution_status=status,
        last_evolve_at_utc=now,
    )
    return {
        "ok": True,
        "profile_id": profile_id,
        "status": status,
        "candidate": candidate,
        "skipped_full_evolve": True,
    }


def should_auto_enable(
    suggestion: Dict[str, Any], evolve_out: Optional[Dict[str, Any]]
) -> bool:
    if not cfg.MOSS2_AUTO_ENABLE_PROFILES:
        return False
    if suggestion.get("reason") == "backtest_selection_pass":
        return True
    if not evolve_out or not evolve_out.get("ok"):
        return False
    if evolve_out.get("status") == "approved" and cfg.MOSS2_AUTO_ENABLE_ON_APPROVED:
        return True
    cand = evolve_out.get("candidate") or {}
    summ = cand.get("summary") or {}
    disc = cand.get("discipline") or {}
    if evolve_out.get("status") == "approved" and passes_backtest_gates(summ, disc):
        return True
    if cand and passes_backtest_gates(summ, disc):
        return bool(cfg.MOSS2_EVOLVE_AUTO_APPROVE)
    return False


def _finalize_force_evolve(suggestion: Dict[str, Any], *, force_evolve: bool) -> bool:
    """创建/更新时：已选优过关则不重复全量 evolve。"""
    if force_evolve:
        return True
    return _needs_full_evolve(suggestion, force_evolve=False)


def sync_enable_approved_profiles(conn: sqlite3.Connection) -> int:
    """补救：DB 里已 approved 但仍未 enabled 的 Profile（如旧版逻辑未打开关）。"""
    if not cfg.MOSS2_AUTO_ENABLE_PROFILES:
        return 0
    n = 0
    for p in list_profiles(conn):
        if p.get("enabled"):
            continue
        if str(p.get("evolution_status") or "") != "approved":
            continue
        try:
            patch_profile(conn, int(p["id"]), enabled=True)
            n += 1
        except sqlite3.IntegrityError as e:
            logger.warning(
                "[moss2] sync_enable skip profile=%s: %s",
                p.get("id"),
                e,
            )
    return n


def _create_from_suggestion(
    conn: sqlite3.Connection, suggestion: Dict[str, Any]
) -> int:
    sym = str(suggestion["symbol"])
    template = str(suggestion.get("recommended_template") or cfg.MOSS2_DEFAULT_TEMPLATE)
    name = str(suggestion.get("recommended_name") or f"{sym.lower()}-en-{template}")
    raw_params = suggestion.get("recommended_params")
    if raw_params:
        initial, tactical = split_profile_params(
            raw_params, variant=cfg.MOSS2_OPS_VARIANT
        )
    else:
        merged = build_initial_params(template, variant=cfg.MOSS2_OPS_VARIANT)
        initial, tactical = split_profile_params(merged, variant=cfg.MOSS2_OPS_VARIANT)
    return create_profile(
        conn,
        name=name,
        symbol=sym,
        variant=cfg.MOSS2_OPS_VARIANT,
        template=template,
        enabled=False,
        initial_params=initial,
        tactical_params=tactical,
        virtual_equity_usdt=float(cfg.MOSS2_PROFILE_CAPITAL),
    )


def _sync_template_from_suggestion(
    conn: sqlite3.Connection, profile_id: int, suggestion: Dict[str, Any]
) -> None:
    template = str(suggestion.get("recommended_template") or "")
    if not template:
        return
    prof = get_profile(conn, profile_id) or {}
    if str(prof.get("template") or "") == template:
        return
    raw_params = suggestion.get("recommended_params")
    if raw_params:
        initial, tactical = split_profile_params(
            raw_params, variant=cfg.MOSS2_OPS_VARIANT
        )
    else:
        merged = build_initial_params(template, variant=cfg.MOSS2_OPS_VARIANT)
        initial, tactical = split_profile_params(merged, variant=cfg.MOSS2_OPS_VARIANT)
    patch_profile(
        conn,
        profile_id,
        template=template,
        initial_params=initial,
        tactical_params=tactical,
    )


def _finalize_profile(
    conn: sqlite3.Connection,
    profile_id: int,
    suggestion: Dict[str, Any],
    *,
    force_evolve: bool,
) -> Dict[str, Any]:
    if _finalize_force_evolve(suggestion, force_evolve=force_evolve):
        evolve_out = run_profile_evolve(conn, profile_id, force=True)
    else:
        evolve_out = _apply_selection_candidate(conn, profile_id, suggestion)
    if (
        evolve_out.get("ok")
        and evolve_out.get("status") == "pending"
        and cfg.MOSS2_EVOLVE_AUTO_APPROVE
    ):
        approve_candidate(conn, profile_id)
        evolve_out = {**evolve_out, "status": "approved", "auto_approved": True}

    enable = should_auto_enable(suggestion, evolve_out)
    if enable:
        try:
            patch_profile(conn, profile_id, enabled=True)
        except sqlite3.IntegrityError as e:
            logger.warning(
                "[moss2] auto_enable conflict profile=%s symbol=%s: %s",
                profile_id,
                suggestion.get("symbol"),
                e,
            )
            enable = False

    prof = get_profile(conn, profile_id)
    return {
        "profile_id": profile_id,
        "evolve": evolve_out,
        "enabled": bool(prof and prof.get("enabled")),
        "auto_enabled": enable,
    }


def provision_symbol(
    conn: sqlite3.Connection,
    symbol: str,
    *,
    force_evolve: bool = False,
) -> Dict[str, Any]:
    """单标的全自动：建议 → 创建或更新 → evolve（+auto approve）→ 按需启用。"""
    sym = normalize_symbol(symbol, variant=cfg.MOSS2_OPS_VARIANT)
    row: Dict[str, Any] = {"symbol": sym, "action": "skip"}

    if not cfg.MOSS2_AUTO_PROVISION_ENABLED:
        row["reason"] = "auto_provision_disabled"
        return row

    suggestion = suggest_profile(
        sym,
        backtest_bars=int(cfg.MOSS2_AUTO_PROVISION_BACKTEST_BARS),
        min_trades=int(cfg.MOSS2_AUTO_PROVISION_MIN_TRADES),
    )
    row["suggest_reason"] = suggestion.get("reason")
    row["recommended_template"] = suggestion.get("recommended_template")

    if not suggestion.get("ok"):
        row["reason"] = suggestion.get("reason") or "suggest_failed"
        return row

    existing = find_profiles_for_symbol(conn, sym)
    enabled = _enabled_profile(existing)

    if enabled:
        row["action"] = "maintain"
        row["profile_id"] = int(enabled["id"])
        if force_evolve or cfg.MOSS2_AUTO_REPROVISION_EXISTING:
            fin = _finalize_profile(
                conn,
                int(enabled["id"]),
                suggestion,
                force_evolve=force_evolve,
            )
            row.update(fin)
        else:
            row["reason"] = "already_enabled"
        return row

    work = _pick_work_profile(existing)
    if work:
        row["action"] = "update"
        pid = int(work["id"])
        row["profile_id"] = pid
        _sync_template_from_suggestion(conn, pid, suggestion)
        fin = _finalize_profile(
            conn, pid, suggestion, force_evolve=_finalize_force_evolve(suggestion, force_evolve=force_evolve)
        )
        row.update(fin)
        return row

    row["action"] = "create"
    pid = _create_from_suggestion(conn, suggestion)
    row["profile_id"] = pid
    fin = _finalize_profile(
        conn, pid, suggestion, force_evolve=_finalize_force_evolve(suggestion, force_evolve=force_evolve)
    )
    row.update(fin)
    return row


def run_lane_auto_provision(
    conn: sqlite3.Connection,
    *,
    bases: Optional[List[str]] = None,
    force_evolve: bool = False,
) -> Dict[str, Any]:
    """对 MOSS2_SEED_BASES（默认 25 核心）执行全自动 Profile 运维。"""
    if not cfg.MOSS2_AUTO_PROVISION_ENABLED:
        return {"ok": False, "reason": "auto_provision_disabled", "lane": "moss2"}

    bases = list(bases or cfg.MOSS2_SEED_BASES)
    results: List[Dict[str, Any]] = []
    created = updated = maintained = skipped = enabled_n = 0

    for base in bases:
        sym = normalize_symbol(base, variant=cfg.MOSS2_OPS_VARIANT)
        try:
            r = provision_symbol(conn, sym, force_evolve=force_evolve)
        except Exception as e:
            logger.exception("[moss2] auto_provision %s failed", sym)
            r = {"symbol": sym, "action": "error", "reason": str(e)}
        results.append(r)
        act = r.get("action")
        if act == "create":
            created += 1
        elif act == "update":
            updated += 1
        elif act == "maintain":
            maintained += 1
        else:
            skipped += 1
        if r.get("auto_enabled") or r.get("enabled"):
            enabled_n += 1

    synced = sync_enable_approved_profiles(conn)
    return {
        "ok": True,
        "lane": "moss2",
        "bases": len(bases),
        "created": created,
        "updated": updated,
        "maintained": maintained,
        "skipped": skipped,
        "enabled_profiles": enabled_n,
        "sync_enabled_approved": synced,
        "auto_approve": cfg.MOSS2_EVOLVE_AUTO_APPROVE,
        "results": results,
    }
