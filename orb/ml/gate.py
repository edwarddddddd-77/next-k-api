#!/usr/bin/env python3
"""实盘开单闸门：突破当下打分，过线且未超日限额才允许开单。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from orb.ml.ranker import BreakoutRanker
from orb.ml.features import extract_features, label_is_true_breakout
from orb.ml.profiles import load_profiles

from orb.ml.paths import CONFIG_V1, CONFIG_V2, PROJECT_ROOT

DEFAULT_GATE_CONFIG = (
    CONFIG_V2 / "live_gate.json"
    if (CONFIG_V2 / "live_gate.json").is_file()
    else CONFIG_V1 / "live_gate.json"
    if (CONFIG_V1 / "live_gate.json").is_file()
    else PROJECT_ROOT / "config" / "orb_live_gate.json"
)


@dataclass
class LiveGateConfig:
    max_opens_per_day: int = 8
    min_p_true: float = 0.35
    min_goal_true: int = 1
    target_goal_true: int = 2
    early_trap_minutes: float = 20.0
    early_trap_sync_min: int = 3
    early_trap_sync_max: int = 14
    day_abort_after_signals: int = 8
    day_abort_median_p_max: float = 0.32
    tier_c_extra_min_p: float = 0.05
    day_abort_enabled: bool = False
    early_trap_bypass_min_p: float = 0.0
    robot_reuse_after_exit: bool = False

    @classmethod
    def from_json(cls, path: Optional[Path] = None) -> "LiveGateConfig":
        p = path or DEFAULT_GATE_CONFIG
        if not p.is_file():
            return cls()
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls(
            max_opens_per_day=int(d.get("max_opens_per_day", 8)),
            min_p_true=float(d.get("min_p_true", 0.35)),
            min_goal_true=int(d.get("min_goal_true", 1)),
            target_goal_true=int(d.get("target_goal_true", 2)),
            early_trap_minutes=float(d.get("early_trap_minutes", 20)),
            early_trap_sync_min=int(d.get("early_trap_sync_min", 3)),
            early_trap_sync_max=int(d.get("early_trap_sync_max", 14)),
            day_abort_after_signals=int(d.get("day_abort_after_signals", 8)),
            day_abort_median_p_max=float(d.get("day_abort_median_p_max", 0.32)),
            tier_c_extra_min_p=float(d.get("tier_c_extra_min_p", 0.05)),
            day_abort_enabled=bool(d.get("day_abort_enabled", False)),
            early_trap_bypass_min_p=float(d.get("early_trap_bypass_min_p", 0) or 0),
            robot_reuse_after_exit=bool(d.get("robot_reuse_after_exit", False)),
        )


@dataclass
class LiveGateDayState:
    opens: int = 0
    scored_signals: int = 0
    recent_p: List[float] = field(default_factory=list)
    day_aborted: bool = False
    opened: List[Dict[str, Any]] = field(default_factory=list)
    skipped: List[Dict[str, Any]] = field(default_factory=list)


def is_early_trap(
    feat: Dict[str, float],
    sync: int,
    *,
    early_minutes: float,
    sync_min: int,
    sync_max: int,
) -> bool:
    mins = float(feat.get("minutes_after_or", 0) or 0)
    return mins < early_minutes and sync_min <= int(sync) <= sync_max


def _effective_min_p(symbol: str, cfg: LiveGateConfig, profiles: Dict[str, Any]) -> float:
    base = float(cfg.min_p_true)
    key = str(symbol or "").upper()
    if not key.endswith("USDT"):
        key += "USDT"
    prof = (profiles.get("profiles") or {}).get(key)
    if prof and str(prof.get("tier") or "") == "C":
        return base + float(cfg.tier_c_extra_min_p)
    return base


def should_open(
    *,
    p_true: float,
    symbol: str,
    feat: Dict[str, float],
    sync: int,
    state: LiveGateDayState,
    gate: LiveGateConfig,
    profiles: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """突破当下调用：是否允许开这一单。"""
    if state.day_aborted:
        return False, "day_aborted"
    if not gate.robot_reuse_after_exit and state.opens >= gate.max_opens_per_day:
        return False, "max_opens_reached"
    if is_early_trap(
        feat,
        sync,
        early_minutes=gate.early_trap_minutes,
        sync_min=gate.early_trap_sync_min,
        sync_max=gate.early_trap_sync_max,
    ):
        bypass = float(gate.early_trap_bypass_min_p or 0)
        if bypass <= 0 or float(p_true) < bypass:
            return False, "early_sync_trap"

    profiles = profiles or load_profiles()
    min_p = _effective_min_p(symbol, gate, profiles)
    if float(p_true) < min_p:
        return False, f"p_true<{min_p:.2f}"

    return True, "open_ok"


def record_scored_signal(
    state: LiveGateDayState,
    *,
    p_true: float,
    gate: LiveGateConfig,
) -> None:
    state.scored_signals += 1
    state.recent_p.append(float(p_true))
    if not gate.day_abort_enabled:
        return
    if state.scored_signals == gate.day_abort_after_signals and state.recent_p:
        med = sorted(state.recent_p)[len(state.recent_p) // 2]
        if med < gate.day_abort_median_p_max and state.opens == 0:
            state.day_aborted = True


def evaluate_open_decision(
    ranker: BreakoutRanker,
    *,
    symbol: str,
    feat: Dict[str, float],
    sync: int,
    state: LiveGateDayState,
    gate: LiveGateConfig,
    trade_row: Optional[Dict[str, Any]] = None,
    p_true: Optional[float] = None,
    p_fake: Optional[float] = None,
) -> Dict[str, Any]:
    """突破当下：打分 + 决策（供回测与将来 paper 共用）。"""
    p_true_v = float(p_true if p_true is not None else ranker.predict_true(feat, symbol=symbol))
    p_fake_v = float(p_fake if p_fake is not None else ranker.predict_fake(feat, symbol=symbol))
    record_scored_signal(state, p_true=p_true_v, gate=gate)
    ok, reason = should_open(
        p_true=p_true_v,
        symbol=symbol,
        feat=feat,
        sync=sync,
        state=state,
        gate=gate,
        profiles=ranker.profiles,
    )
    row: Dict[str, Any] = {
        "symbol": symbol,
        "p_true": p_true_v,
        "p_fake": p_fake_v,
        "sync_same_side": int(sync),
        "minutes_after_or": round(float(feat.get("minutes_after_or", 0) or 0), 1),
        "opened": ok,
        "reason": reason,
    }
    if trade_row:
        row.update(
            {
                "session_date": trade_row.get("session_date"),
                "scan_open_ms": trade_row.get("scan_open_ms"),
                "side": trade_row.get("side"),
                "entry": trade_row.get("entry"),
                "notional_usdt": trade_row.get("notional_usdt"),
                "pnl_usdt": trade_row.get("pnl_usdt"),
                "true_breakout": trade_row.get("true_breakout"),
                "outcome": trade_row.get("outcome"),
                "exit_ms": trade_row.get("exit_ms"),
                "robot_id": trade_row.get("robot_id"),
                "wallet_before": trade_row.get("wallet_before"),
                "wallet_after": trade_row.get("wallet_after"),
            }
        )
    if ok:
        state.opens += 1
        state.opened.append(row)
    else:
        state.skipped.append(row)
    return row


def rollback_open_decision(state: LiveGateDayState, *, symbol: str) -> None:
    """撤销一次已通过 gate 但未实际落库的开单计数（如无 robot slot）。"""
    sym = str(symbol or "").strip().upper()
    if not sym.endswith("USDT"):
        sym = f"{sym}USDT"
    if state.opened and str(state.opened[-1].get("symbol") or "").upper() == sym:
        state.opened.pop()
    else:
        state.opened = [r for r in state.opened if str(r.get("symbol") or "").upper() != sym]
    state.opens = max(0, state.opens - 1)


def summarize_live_gate_day(state: LiveGateDayState, gate: LiveGateConfig) -> Dict[str, Any]:
    true_n = sum(1 for r in state.opened if r.get("true_breakout"))
    return {
        "opens": state.opens,
        "skipped": len(state.skipped),
        "true_opens": true_n,
        "goal_met_min": true_n >= gate.min_goal_true,
        "goal_met_target": true_n >= gate.target_goal_true,
        "day_aborted": state.day_aborted,
    }
