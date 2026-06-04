"""Moss2 L2 慢进化：4 模板窄搜 + discipline 闸门。"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from moss2 import config as cfg
from moss2.db import get_profile, patch_profile_evolution
from moss2.params import split_profile_params
from moss2.selection import compete_templates
from moss2.versioning import params_hash

logger = logging.getLogger(__name__)


def _needs_evolve(conn: sqlite3.Connection, profile_id: int) -> bool:
    prof = get_profile(conn, profile_id)
    if not prof or not prof.get("enabled"):
        return False
    last = prof.get("last_evolve_at_utc") or ""
    if not last:
        return True
    try:
        t0 = datetime.fromisoformat(last.replace("Z", "+00:00"))
        age_days = (datetime.now(timezone.utc) - t0).days
        return age_days >= cfg.MOSS2_EVOLVE_INTERVAL_DAYS
    except Exception:
        return True


def _run_compete_evolve(
    conn: sqlite3.Connection,
    profile_id: int,
    prof: dict,
    *,
    limit_bars: int,
    min_trades: int,
) -> Dict[str, Any]:
    """对单 Profile 跑四模板竞赛并写库；无过关候选则 evolution_status=no_candidate。"""
    from moss2.config import profile_variant

    variant = profile_variant(prof)
    symbol = str(prof["symbol"])
    capital = float(prof.get("virtual_equity_usdt") or cfg.MOSS2_PROFILE_CAPITAL)
    comp = compete_templates(
        symbol,
        variant=variant,
        capital=capital,
        limit_bars=limit_bars,
        optimize_tactical=True,
        min_trades=min_trades,
    )
    best = comp.get("best")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not best:
        logger.warning(
            "[moss2] evolve no_candidate profile=%s %s bars=%s min_trades=%s",
            profile_id,
            symbol,
            limit_bars,
            min_trades,
        )
        patch_profile_evolution(
            conn,
            profile_id,
            evolution_status="no_candidate",
            last_evolve_at_utc=now,
        )
        return {
            "ok": True,
            "profile_id": profile_id,
            "candidate": None,
            "limit_bars": limit_bars,
        }

    initial, tactical = split_profile_params(best["params"], variant=variant)  # type: ignore
    ver = f"v{now[:10].replace('-', '')}-{best['template']}"
    candidate = {
        "params_version": ver,
        "template": best["template"],
        "initial_params": initial,
        "tactical_params": tactical,
        "discipline": best["discipline"],
        "summary": best["summary"],
    }
    status = "pending"
    if cfg.MOSS2_EVOLVE_AUTO_APPROVE:
        status = "approved"
        _approve_candidate(conn, profile_id, candidate)

    patch_profile_evolution(
        conn,
        profile_id,
        candidate_params_json=json.dumps(candidate, ensure_ascii=False),
        evolution_status=status,
        last_evolve_at_utc=now,
    )
    summ = best.get("summary") or {}
    logger.info(
        "[moss2] evolve ok profile=%s %s tpl=%s status=%s bars=%s score=%s sharpe=%s trades=%s",
        profile_id,
        symbol,
        best["template"],
        status,
        limit_bars,
        best.get("score"),
        summ.get("sharpe"),
        summ.get("total_trades"),
    )
    return {
        "ok": True,
        "profile_id": profile_id,
        "candidate": candidate,
        "status": status,
        "limit_bars": limit_bars,
    }


def run_profile_evolve(
    conn: sqlite3.Connection,
    profile_id: int,
    *,
    force: bool = False,
    limit_bars: Optional[int] = None,
    min_trades: Optional[int] = None,
    retry_long_window: bool = False,
) -> Dict[str, Any]:
    if not cfg.MOSS2_EVOLVE_ENABLED and not force:
        return {"ok": False, "reason": "evolve_disabled"}
    if not force and not _needs_evolve(conn, profile_id):
        logger.info("[moss2] evolve skip profile=%s reason=interval", profile_id)
        return {"ok": True, "skipped": True, "reason": "interval"}

    prof = get_profile(conn, profile_id)
    if not prof:
        return {"ok": False, "reason": "profile_not_found"}

    from moss2.config import is_ops_variant, profile_variant

    if not is_ops_variant(prof.get("variant")):
        return {
            "ok": False,
            "reason": "variant_disabled",
            "variant": prof.get("variant"),
        }
    bars_primary = int(limit_bars or cfg.MOSS2_EVOLVE_LIMIT_BARS)
    mt = int(min_trades if min_trades is not None else cfg.MOSS2_SELECTION_MIN_TRADES)
    out = _run_compete_evolve(
        conn, profile_id, prof, limit_bars=bars_primary, min_trades=mt
    )
    if (
        retry_long_window
        and not out.get("candidate")
        and bars_primary < int(cfg.MOSS2_EVOLVE_LIMIT_BARS)
    ):
        logger.info(
            "[moss2] evolve retry long window profile=%s %s→%s bars",
            profile_id,
            bars_primary,
            cfg.MOSS2_EVOLVE_LIMIT_BARS,
        )
        out = _run_compete_evolve(
            conn,
            profile_id,
            prof,
            limit_bars=int(cfg.MOSS2_EVOLVE_LIMIT_BARS),
            min_trades=mt,
        )
        out["retried_long_window"] = True
    return out


def _approve_candidate(conn: sqlite3.Connection, profile_id: int, candidate: dict) -> None:
    from moss2.db import patch_profile

    patch_profile(
        conn,
        profile_id,
        template=candidate["template"],
        initial_params=candidate["initial_params"],
        tactical_params=candidate["tactical_params"],
        approved_params_version=candidate["params_version"],
        params_version=candidate["params_version"],
        canary_scale=1.0,
        evolution_status="approved",
    )
    prof = get_profile(conn, profile_id)
    if prof:
        patch_profile_evolution(
            conn,
            profile_id,
            params_hash=params_hash(prof),
        )


def approve_candidate(conn: sqlite3.Connection, profile_id: int) -> Dict[str, Any]:
    prof = get_profile(conn, profile_id)
    if not prof:
        return {"ok": False, "reason": "not_found"}
    raw = prof.get("candidate_params_json")
    if not raw:
        return {"ok": False, "reason": "no_candidate"}
    candidate = json.loads(raw) if isinstance(raw, str) else raw
    _approve_candidate(conn, profile_id, candidate)
    return {"ok": True, "profile_id": profile_id, "version": candidate.get("params_version")}


def run_lane_evolve(conn: sqlite3.Connection) -> Dict[str, Any]:
    from moss2.db import list_profiles

    results = []
    for p in list_profiles(conn, enabled_only=True):
        results.append(run_profile_evolve(conn, int(p["id"])))
    ok_n = sum(1 for r in results if r.get("ok"))
    skip_n = sum(1 for r in results if r.get("reason") == "interval")
    no_cand = sum(1 for r in results if r.get("reason") == "no_candidate")
    logger.info(
        "[moss2] evolve lane done profiles=%s ok=%s interval_skip=%s no_candidate=%s",
        len(results),
        ok_n,
        skip_n,
        no_cand,
    )
    return {"lane": "moss2", "results": results}
