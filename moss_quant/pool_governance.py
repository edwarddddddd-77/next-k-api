"""每日寻优后纸面 Profile 池子治理：连续轮次防抖 + TopN 自动补位。"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from moss_quant import config as cfg
from moss_quant.daily_optimize_service import (
    _batch_items_by_symbol,
    _build_initial_for_sync,
    _run_paper_scan_after_param_sync,
    _sync_rank_index,
    import_profile_from_batch_item,
)
from moss_quant.db import (
    GOVERNANCE_PROFILE_SOURCE,
    _utc_now,
    count_enabled_profiles,
    get_profile_by_symbol,
    get_symbol_pool_streak,
    insert_pool_governance_log,
    list_enabled_profiles,
    upsert_symbol_pool_streak,
)
from moss_quant.optimize_policy import can_sync_profile_params, enrich_summary

logger = logging.getLogger(__name__)


def _pool_tier(summary: Optional[Dict[str, Any]]) -> str:
    if not summary or summary.get("error"):
        return "C"
    s = summary if summary.get("pool_tier") else enrich_summary(dict(summary))
    return str(s.get("pool_tier") or "C").upper()


def _is_upgrade_round(summary: Optional[Dict[str, Any]]) -> bool:
    if not summary or summary.get("error"):
        return False
    s = summary if summary.get("pool_tier") else enrich_summary(dict(summary))
    return bool(can_sync_profile_params(s))


def _next_streaks(
    prev: Optional[Dict[str, Any]],
    *,
    upgrade_round: bool,
    pool_tier: str,
) -> Tuple[int, int]:
    """升级连击仅「可同步」日累加；降级连击仅 B/C 池日累加（A 但不同步不打断 B 观测）。"""
    prev_deg = int((prev or {}).get("degrade_streak") or 0)
    prev_up = int((prev or {}).get("upgrade_streak") or 0)
    if upgrade_round:
        return 0, prev_up + 1
    tier = str(pool_tier or "C").upper()
    if tier in ("B", "C"):
        return prev_deg + 1, 0
    return prev_deg, 0


def auto_enable_eligible_symbols(
    items_by_sym: Dict[str, Dict[str, Any]],
    streak_by_sym: Dict[str, Dict[str, Any]],
    *,
    top_n: Optional[int] = None,
    upgrade_streak_need: Optional[int] = None,
) -> set[str]:
    """与 apply_pool_governance 补位逻辑一致：可同步 TopN 且升级连击已满。"""
    need = int(
        upgrade_streak_need
        if upgrade_streak_need is not None
        else cfg.MOSS_QUANT_POOL_UPGRADE_STREAK
    )
    n = int(top_n if top_n is not None else cfg.MOSS_QUANT_POOL_AUTO_ADD_TOP_N)
    eligible: set[str] = set()
    for sym, _, _ in _ranked_a_pool_candidates(items_by_sym, top_n=n):
        st = streak_by_sym.get(sym) or {}
        if int(st.get("upgrade_streak") or 0) >= need:
            eligible.add(sym)
    return eligible


def _ranked_a_pool_candidates(
    items_by_sym: Dict[str, Dict[str, Any]], *, top_n: int
) -> List[Tuple[str, Dict[str, Any], int]]:
    ranked: List[Tuple[str, float, Dict[str, Any]]] = []
    for sym, item in items_by_sym.items():
        summary = item.get("summary") or {}
        if not _is_upgrade_round(summary):
            continue
        score = float(summary.get("val_sharpe") or item.get("score") or 0)
        ranked.append((sym, score, item))
    ranked.sort(key=lambda x: -x[1])
    out: List[Tuple[str, Dict[str, Any], int]] = []
    for i, (sym, _, item) in enumerate(ranked[: max(1, int(top_n))]):
        out.append((sym, item, i))
    return out


def _should_auto_disable(tier: str, degrade_streak: int) -> bool:
    if tier == "C" and degrade_streak >= cfg.MOSS_QUANT_POOL_DEGRADE_STREAK_C:
        return True
    if tier == "B" and degrade_streak >= cfg.MOSS_QUANT_POOL_DEGRADE_STREAK_B:
        return True
    return False


def _disable_profile(
    conn,
    profile_id: int,
    *,
    batch_id: int,
    symbol: str,
    tier: str,
    degrade_streak: int,
    upgrade_streak: int,
) -> None:
    now = _utc_now()
    conn.execute(
        "UPDATE moss_profiles SET enabled=0, updated_at_utc=? WHERE id=?",
        (now, int(profile_id)),
    )
    upsert_symbol_pool_streak(
        conn,
        symbol,
        last_pool_tier=tier,
        last_batch_id=batch_id,
        degrade_streak=degrade_streak,
        upgrade_streak=upgrade_streak,
        last_action="auto_disabled",
        last_action_at_utc=now,
        commit=False,
    )
    insert_pool_governance_log(
        conn,
        batch_id=batch_id,
        symbol=symbol,
        action="auto_disabled",
        pool_tier=tier,
        degrade_streak=degrade_streak,
        upgrade_streak=upgrade_streak,
        profile_id=int(profile_id),
        detail={"tier": tier},
    )


def _sync_profile_from_item(
    conn,
    profile_id: int,
    item: Dict[str, Any],
    items_by_sym: Dict[str, Dict[str, Any]],
) -> None:
    sym = str(item.get("symbol") or "").strip().upper()
    template = str(item.get("template") or "balanced")
    tactical = dict(item.get("tactical_params") or {})
    rank_idx = _sync_rank_index(items_by_sym, sym)
    from moss_quant.optimize_policy import risk_scale_for_rank

    pool_sharpes = [
        float((items_by_sym[s].get("summary") or {}).get("val_sharpe") or 0)
        for s in items_by_sym
        if can_sync_profile_params(items_by_sym[s].get("summary") or {})
    ]
    pool_max = max(pool_sharpes) if pool_sharpes else 0.0
    val_sh = float((item.get("summary") or {}).get("val_sharpe") or 0)
    risk_scale = risk_scale_for_rank(
        rank_idx, val_sharpe=val_sh, pool_max_val_sharpe=pool_max
    )
    initial = _build_initial_for_sync(template, risk_scale)
    conn.execute(
        """UPDATE moss_profiles SET
           template=?, initial_params_json=?, tactical_params_json=?,
           updated_at_utc=?
           WHERE id=?""",
        (
            template,
            json.dumps(initial, ensure_ascii=False),
            json.dumps(tactical, ensure_ascii=False),
            _utc_now(),
            int(profile_id),
        ),
    )


def summarize_pool_governance(conn) -> Dict[str, Any]:
    """供 summary API：治理配置 + 各 Profile/标的 streak。"""
    from moss_quant.daily_optimize_service import get_latest_daily_batch
    from moss_quant.db import list_recent_pool_governance_logs, list_symbol_pool_streaks

    eligible_syms: set[str] = set()
    latest_batch = get_latest_daily_batch(conn)
    if latest_batch and latest_batch.get("id"):
        items_by_sym = _batch_items_by_symbol(conn, int(latest_batch["id"]))
        streak_by_sym = {
            str(s["symbol"]).upper(): s for s in list_symbol_pool_streaks(conn)
        }
        eligible_syms = auto_enable_eligible_symbols(items_by_sym, streak_by_sym)

    profiles = []
    profile_syms: set[str] = set()
    try:
        rows = conn.execute(
            "SELECT id, symbol, enabled, governance_manual_lock, profile_source FROM moss_profiles"
        ).fetchall()
    except Exception:
        rows = conn.execute(
            "SELECT id, symbol, enabled, profile_source FROM moss_profiles"
        ).fetchall()
    streak_map = {
        str(s["symbol"]).upper(): s for s in list_symbol_pool_streaks(conn)
    }
    for row in rows:
        if len(row) >= 5:
            pid, sym, enabled, manual_lock, src = row
        else:
            pid, sym, enabled, src = row
            manual_lock = 0
        sym_u = str(sym).upper()
        profile_syms.add(sym_u)
        st = streak_map.get(sym_u) or {}
        profiles.append(
            {
                "profile_id": int(pid),
                "symbol": sym_u,
                "enabled": bool(enabled),
                "governance_manual_lock": bool(manual_lock),
                "profile_source": str(src or "manual"),
                "last_pool_tier": st.get("last_pool_tier"),
                "degrade_streak": int(st.get("degrade_streak") or 0),
                "upgrade_streak": int(st.get("upgrade_streak") or 0),
                "last_action": st.get("last_action"),
                "auto_enable_eligible": sym_u in eligible_syms,
            }
        )
    symbol_streaks = [
        {
            "symbol": sym_u,
            "last_pool_tier": s.get("last_pool_tier"),
            "degrade_streak": int(s.get("degrade_streak") or 0),
            "upgrade_streak": int(s.get("upgrade_streak") or 0),
            "last_action": s.get("last_action"),
            "auto_enable_eligible": sym_u in eligible_syms,
        }
        for s in streak_map.values()
        for sym_u in [str(s["symbol"]).upper()]
        if sym_u not in profile_syms
    ]
    return {
        "enabled": cfg.pool_governance_enabled(),
        "auto_disable": cfg.MOSS_QUANT_POOL_AUTO_DISABLE,
        "auto_enable": cfg.MOSS_QUANT_POOL_AUTO_ENABLE,
        "max_auto_enabled": min(
            cfg.MOSS_QUANT_MAX_ACTIVE_PROFILES,
            cfg.MOSS_QUANT_POOL_MAX_AUTO_ENABLED,
        ),
        "auto_add_top_n": cfg.MOSS_QUANT_POOL_AUTO_ADD_TOP_N,
        "degrade_streak_b": cfg.MOSS_QUANT_POOL_DEGRADE_STREAK_B,
        "degrade_streak_c": cfg.MOSS_QUANT_POOL_DEGRADE_STREAK_C,
        "upgrade_streak": cfg.MOSS_QUANT_POOL_UPGRADE_STREAK,
        "respect_manual_disable": cfg.MOSS_QUANT_POOL_RESPECT_MANUAL_DISABLE,
        "auto_disable_on_paper_loss": cfg.MOSS_QUANT_POOL_AUTO_DISABLE_ON_PAPER_LOSS,
        "profiles": profiles,
        "symbol_streaks": symbol_streaks,
        "recent_actions": list_recent_pool_governance_logs(conn, limit=20),
    }


def apply_pool_governance(
    conn, batch_id: int, *, trigger_paper_scan: bool = True
) -> Dict[str, Any]:
    """每日寻优 annotate+sync 之后调用：更新 streak、自动停/启/补位。"""
    if not cfg.pool_governance_enabled():
        return {"enabled": False, "skipped": True}

    items_by_sym = _batch_items_by_symbol(conn, int(batch_id))
    stats: Dict[str, Any] = {
        "enabled": True,
        "batch_id": int(batch_id),
        "streak_updates": 0,
        "disabled": 0,
        "enabled_auto": 0,
        "added": 0,
        "skipped_manual_lock": 0,
        "skipped_upgrade_streak": 0,
        "skipped_at_cap": 0,
        "actions": [],
    }

    streak_by_sym: Dict[str, Dict[str, Any]] = {}
    mutated = False
    param_updates = 0

    for sym, item in items_by_sym.items():
        summary = item.get("summary") or {}
        tier = _pool_tier(summary)
        prev = get_symbol_pool_streak(conn, sym)
        upgrade_round = _is_upgrade_round(summary)
        deg, up = _next_streaks(prev, upgrade_round=upgrade_round, pool_tier=tier)
        streak = upsert_symbol_pool_streak(
            conn,
            sym,
            last_pool_tier=tier,
            last_batch_id=int(batch_id),
            degrade_streak=deg,
            upgrade_streak=up,
            commit=False,
        )
        streak_by_sym[sym] = streak
        stats["streak_updates"] += 1
    mutated = stats["streak_updates"] > 0

    if cfg.MOSS_QUANT_POOL_AUTO_DISABLE:
        from moss_quant.optimize_policy import paper_recent_pnl_block_reason

        for prof in list_enabled_profiles(conn):
            sym = str(prof.get("symbol") or "").upper()
            item = items_by_sym.get(sym)
            if not item:
                continue
            summary = item.get("summary") or {}
            tier = _pool_tier(summary)
            st = streak_by_sym.get(sym) or {}
            deg = int(st.get("degrade_streak") or 0)
            pid = int(prof["id"])

            if cfg.MOSS_QUANT_POOL_AUTO_DISABLE_ON_PAPER_LOSS:
                paper_reason = paper_recent_pnl_block_reason(
                    conn,
                    pid,
                    profile_capital=float(prof.get("virtual_equity_usdt") or 0) or None,
                )
                if paper_reason:
                    _disable_profile(
                        conn,
                        pid,
                        batch_id=int(batch_id),
                        symbol=sym,
                        tier=tier or "paper",
                        degrade_streak=deg,
                        upgrade_streak=int(st.get("upgrade_streak") or 0),
                    )
                    stats["disabled"] += 1
                    mutated = True
                    stats["actions"].append(
                        {
                            "action": "auto_disabled_paper_loss",
                            "symbol": sym,
                            "detail": paper_reason,
                        }
                    )
                    logger.info(
                        "[moss] governance paper-loss disable %s: %s",
                        sym,
                        paper_reason,
                    )
                    continue

            if not _should_auto_disable(tier, deg):
                continue
            _disable_profile(
                conn,
                pid,
                batch_id=int(batch_id),
                symbol=sym,
                tier=tier,
                degrade_streak=deg,
                upgrade_streak=int(st.get("upgrade_streak") or 0),
            )
            stats["disabled"] += 1
            mutated = True
            stats["actions"].append(
                {"action": "auto_disabled", "symbol": sym, "tier": tier}
            )
            logger.info(
                "[moss] governance auto-disable %s tier=%s degrade=%s",
                sym,
                tier,
                deg,
            )

    if cfg.MOSS_QUANT_POOL_AUTO_ENABLE:
        max_cap = min(
            cfg.MOSS_QUANT_MAX_ACTIVE_PROFILES,
            cfg.MOSS_QUANT_POOL_MAX_AUTO_ENABLED,
        )
        candidates = _ranked_a_pool_candidates(
            items_by_sym, top_n=cfg.MOSS_QUANT_POOL_AUTO_ADD_TOP_N
        )
        enabled_n = count_enabled_profiles(conn)

        for sym, item, _rank in candidates:
            if enabled_n >= max_cap:
                stats["skipped_at_cap"] += 1
                break
            st = streak_by_sym.get(sym) or {}
            if int(st.get("upgrade_streak") or 0) < cfg.MOSS_QUANT_POOL_UPGRADE_STREAK:
                stats["skipped_upgrade_streak"] += 1
                continue
            existing = get_profile_by_symbol(conn, sym)
            if existing and existing.get("enabled"):
                continue
            if (
                existing
                and cfg.MOSS_QUANT_POOL_RESPECT_MANUAL_DISABLE
                and existing.get("governance_manual_lock")
            ):
                stats["skipped_manual_lock"] += 1
                continue

            try:
                if existing:
                    pid = int(existing["id"])
                    _sync_profile_from_item(conn, pid, item, items_by_sym)
                    conn.execute(
                        """UPDATE moss_profiles SET
                           enabled=1, profile_source=?, governance_manual_lock=0,
                           updated_at_utc=?
                           WHERE id=?""",
                        (
                            GOVERNANCE_PROFILE_SOURCE,
                            _utc_now(),
                            pid,
                        ),
                    )
                    conn.execute(
                        """UPDATE moss_daily_optimize_items SET profile_id=?
                           WHERE batch_id=? AND symbol=?""",
                        (pid, int(batch_id), sym),
                    )
                    param_updates += 1
                    stats["enabled_auto"] += 1
                    action = "auto_enabled"
                else:
                    prof = import_profile_from_batch_item(
                        conn,
                        item,
                        enabled=True,
                        profile_source=GOVERNANCE_PROFILE_SOURCE,
                        update_existing=True,
                        commit=False,
                    )
                    pid = int(prof["id"])
                    _sync_profile_from_item(conn, pid, item, items_by_sym)
                    param_updates += 1
                    stats["added"] += 1
                    stats["enabled_auto"] += 1
                    action = "auto_added"
            except ValueError as e:
                logger.warning("[moss] governance enable %s failed: %s", sym, e)
                continue

            now = _utc_now()
            upsert_symbol_pool_streak(
                conn,
                sym,
                last_pool_tier="A",
                last_batch_id=int(batch_id),
                degrade_streak=int(st.get("degrade_streak") or 0),
                upgrade_streak=int(st.get("upgrade_streak") or 0),
                last_action=action,
                last_action_at_utc=now,
                commit=False,
            )
            insert_pool_governance_log(
                conn,
                batch_id=int(batch_id),
                symbol=sym,
                action=action,
                pool_tier="A",
                degrade_streak=int(st.get("degrade_streak") or 0),
                upgrade_streak=int(st.get("upgrade_streak") or 0),
                profile_id=pid,
            )
            stats["actions"].append({"action": action, "symbol": sym})
            enabled_n += 1
            mutated = True
            logger.info("[moss] governance %s %s profile_id=%s", action, sym, pid)

    if mutated:
        conn.commit()

    if trigger_paper_scan and (param_updates or stats["disabled"]):
        stats["paper_scan"] = _run_paper_scan_after_param_sync(
            conn, param_updates + stats["disabled"]
        )

    logger.info(
        "[moss] pool governance batch=%s disabled=%s added=%s enabled=%s",
        batch_id,
        stats["disabled"],
        stats["added"],
        stats["enabled_auto"],
    )
    return stats
