"""Moss2 Profile 淘汰：实盘恶化或回测不再过关 → 自动停用。"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from moss2 import config as cfg
from moss2.db import get_profile, list_profiles, patch_profile, patch_profile_evolution
from moss2.discipline.gates import recent_settled_trades
from moss2.discipline.metrics import trade_stats_from_rows
from moss2.params import merge_profile_params
from moss2.selection import compete_templates, passes_backtest_gates

logger = logging.getLogger(__name__)


def _live_fitness(conn: sqlite3.Connection, profile_id: int) -> Tuple[bool, str, Dict[str, Any]]:
    debug: Dict[str, Any] = {}
    trades = recent_settled_trades(conn, profile_id, limit=50)
    debug["settled_count"] = len(trades)
    min_n = int(cfg.MOSS2_CULL_LIVE_MIN_TRADES)
    if len(trades) < min_n:
        return True, "insufficient_live_history", debug

    stats = trade_stats_from_rows(trades)
    debug.update(stats)
    ev = float(stats.get("ev_per_trade_pct") or 0)
    if ev < float(cfg.MOSS2_CULL_LIVE_EV_FLOOR):
        return False, "live_ev_below_floor", debug
    if int(stats.get("max_consecutive_losses") or 0) >= int(
        cfg.MOSS2_CULL_LIVE_MAX_CONSEC_LOSS
    ):
        return False, "live_max_consec_loss", debug
    return True, "live_ok", debug


def _backtest_fitness(conn: sqlite3.Connection, profile: dict) -> Tuple[bool, str, Dict[str, Any]]:
    debug: Dict[str, Any] = {}
    if not cfg.MOSS2_CULL_REBACKTEST_ENABLED:
        return True, "rebacktest_skipped", debug

    symbol = str(profile["symbol"])
    variant = cfg.MOSS2_OPS_VARIANT
    params = merge_profile_params(profile)
    from moss2.backtest_service import run_factory_backtest
    from moss2.discipline.report import build_discipline_report

    try:
        out = run_factory_backtest(
            symbol=symbol,
            params=params,
            variant=variant,  # type: ignore[arg-type]
            capital=float(profile.get("virtual_equity_usdt") or cfg.MOSS2_PROFILE_CAPITAL),
            limit_bars=cfg.MOSS2_EVOLVE_LIMIT_BARS,
        )
    except Exception as e:
        return False, f"rebacktest_error:{e}", debug

    summ = out.get("summary") or {}
    trades = out.get("trades") or []
    disc = build_discipline_report(
        summary=summ,
        trades=trades,
        template=str(profile.get("template") or "balanced"),
    )
    debug["sharpe"] = summ.get("sharpe")
    debug["max_drawdown"] = summ.get("max_drawdown")
    debug["ev_per_trade_pct"] = (disc.get("ev") or {}).get("ev_per_trade_pct")
    if not passes_backtest_gates(summ, disc):
        return False, "rebacktest_gates_fail", debug
    return True, "rebacktest_ok", debug


def evaluate_profile(conn: sqlite3.Connection, profile_id: int) -> Dict[str, Any]:
    prof = get_profile(conn, profile_id)
    if not prof:
        return {"ok": False, "reason": "not_found"}
    live_ok, live_reason, live_dbg = _live_fitness(conn, profile_id)
    bt_ok, bt_reason, bt_dbg = _backtest_fitness(conn, prof)
    keep = live_ok and bt_ok
    return {
        "ok": True,
        "profile_id": profile_id,
        "symbol": prof.get("symbol"),
        "enabled": bool(prof.get("enabled")),
        "keep": keep,
        "live_ok": live_ok,
        "live_reason": live_reason,
        "backtest_ok": bt_ok,
        "backtest_reason": bt_reason,
        "debug": {"live": live_dbg, "backtest": bt_dbg},
    }


def cull_profile(conn: sqlite3.Connection, profile_id: int) -> Dict[str, Any]:
    """不过关则停用并标记 culled。"""
    ev = evaluate_profile(conn, profile_id)
    if not ev.get("ok"):
        return ev
    if ev.get("keep") or not ev.get("enabled"):
        ev["action"] = "keep" if ev.get("keep") else "already_disabled"
        return ev

    if not cfg.MOSS2_CULL_AUTO_DISABLE:
        ev["action"] = "would_cull"
        return ev

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    reason = ev.get("live_reason") if not ev.get("live_ok") else ev.get("backtest_reason")
    patch_profile(conn, profile_id, enabled=False)
    patch_profile_evolution(
        conn,
        profile_id,
        evolution_status="culled",
        last_evolve_at_utc=now,
    )
    logger.warning(
        "[moss2] culled profile=%s symbol=%s reason=%s",
        profile_id,
        ev.get("symbol"),
        reason,
    )
    ev["action"] = "culled"
    ev["cull_reason"] = reason
    return ev


def recompete_and_refresh(conn: sqlite3.Connection, profile_id: int) -> Dict[str, Any]:
    """淘汰前最后一搏：四模板+窄搜；有赢家则换参并保留启用。"""
    from moss2.evolve_service import _approve_candidate

    prof = get_profile(conn, profile_id)
    if not prof:
        return {"ok": False, "reason": "not_found"}

    comp = compete_templates(
        str(prof["symbol"]),
        limit_bars=cfg.MOSS2_EVOLVE_LIMIT_BARS,
        optimize_tactical=True,
    )
    best = comp.get("best")
    if not best:
        return {"ok": True, "refreshed": False, "reason": "no_winner"}

    from moss2.params import split_profile_params

    initial, tactical = split_profile_params(
        best["params"], variant=cfg.MOSS2_OPS_VARIANT
    )
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    ver = f"v{now[:10].replace('-', '')}-{best['template']}"
    candidate = {
        "params_version": ver,
        "template": best["template"],
        "initial_params": initial,
        "tactical_params": tactical,
        "discipline": best["discipline"],
        "summary": best["summary"],
    }
    _approve_candidate(conn, profile_id, candidate)
    patch_profile_evolution(
        conn,
        profile_id,
        evolution_status="approved",
        last_evolve_at_utc=now,
    )
    return {
        "ok": True,
        "refreshed": True,
        "template": best["template"],
        "score": best.get("score"),
    }


def run_lane_cull(conn: sqlite3.Connection) -> Dict[str, Any]:
    if not cfg.MOSS2_CULL_ENABLED:
        return {"ok": False, "reason": "cull_disabled", "lane": "moss2"}

    results: List[Dict[str, Any]] = []
    culled = refreshed = kept = 0

    for p in list_profiles(conn, enabled_only=True):
        pid = int(p["id"])
        if cfg.MOSS2_CULL_RECOMPETE_BEFORE_DISABLE:
            rc = recompete_and_refresh(conn, pid)
            if rc.get("refreshed"):
                ev = evaluate_profile(conn, pid)
                if ev.get("keep"):
                    refreshed += 1
                    results.append(
                        {"profile_id": pid, "action": "refreshed", **rc, "fitness": ev}
                    )
                    kept += 1
                    continue
                results.append(
                    {
                        "profile_id": pid,
                        "action": "refresh_still_fail",
                        "recompete": rc,
                        "fitness": ev,
                    }
                )
        row = cull_profile(conn, pid)
        results.append(row)
        act = row.get("action")
        if act == "culled":
            culled += 1
        else:
            kept += 1

    return {
        "ok": True,
        "lane": "moss2",
        "culled": culled,
        "refreshed": refreshed,
        "kept": kept,
        "results": results,
    }
