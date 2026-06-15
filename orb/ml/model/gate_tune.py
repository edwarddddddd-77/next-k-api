"""Promote 后 Gate 自动建议 / 有条件自动应用 min_p_true。"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from orb.ml.gate import LiveGateConfig
from orb.ml.gate_replay import replay_sessions, sweep_min_p_grid, timeline_score_stats
from orb.ml.live_gate_sim import cached_symbols, run_gate_eval_sessions, trading_dates_from_samples
from orb.ml.model.auto_config import MlAutoConfig
from orb.ml.model.bundle import BreakoutModelBundle
from orb.ml.model.paths import (
    ARCHIVE_DIR,
    GATE_SUGGESTION_JSON,
    MANIFEST_JSON,
    ensure_model_dirs,
    resolve_train_symbols_path,
)
from orb.v2.paths import resolve_gate_config_path
from orb.v2.robots import init_robot_wallets, robot_count_from_env, robot_equity_from_env


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _min_p_grid(cfg: MlAutoConfig) -> List[float]:
    step = float(cfg.gate_sweep_step)
    lo = float(cfg.gate_sweep_min)
    hi = float(cfg.gate_sweep_max)
    out: List[float] = []
    p = lo
    while p <= hi + 1e-9:
        out.append(round(p, 4))
        p += step
    return out


def _previous_train_report() -> dict:
    manifest = _read_json(MANIFEST_JSON)
    prev = manifest.get("monthly_report") or {}
    return dict(prev.get("train") or manifest.get("train_report") or {})


def _previous_gate_tune() -> dict:
    manifest = _read_json(MANIFEST_JSON)
    prev = manifest.get("monthly_report") or {}
    return dict(prev.get("gate_tune") or {})


def _detect_triggers(
    *,
    current_train: dict,
    previous_train: dict,
    previous_tune: dict,
    current_stats: dict,
    cfg: MlAutoConfig,
) -> List[str]:
    triggers: List[str] = []
    cur_sep = float(current_train.get("holdout_separation") or 0)
    prev_sep = float(previous_train.get("holdout_separation") or 0)
    if prev_sep > 0 and cur_sep < prev_sep * (1.0 - float(cfg.gate_trigger_sep_drop)):
        triggers.append("holdout_separation_drop")
    if cur_sep > 0 and cur_sep < float(cfg.gate_separation_warn):
        triggers.append("low_holdout_separation")

    prev_med = float(previous_tune.get("score_stats", {}).get("median_p") or 0)
    cur_med = float(current_stats.get("median_p") or 0)
    if prev_med > 0 and abs(cur_med - prev_med) >= float(cfg.gate_trigger_median_shift):
        triggers.append("median_p_shift")

    prev_opens = float(previous_tune.get("current_replay", {}).get("avg_opens") or 0)
    cur_opens = float(current_stats.get("avg_opens") or 0)
    if prev_opens > 0:
        delta = abs(cur_opens - prev_opens) / prev_opens
        if delta >= float(cfg.gate_trigger_opens_change):
            triggers.append("avg_opens_change")
    elif cur_opens > 0 and cur_opens < float(cfg.gate_avg_opens_min):
        triggers.append("avg_opens_low")
    return triggers


def _pick_candidate(
    *,
    rows: Sequence[dict],
    current_min_p: float,
    current_replay: dict,
    cfg: MlAutoConfig,
    tighten_only: bool,
) -> tuple[Optional[dict], List[str]]:
    reasons: List[str] = []
    cur_pnl = float(current_replay.get("total_pnl") or 0)
    best = None
    for row in rows:
        min_p = float(row["min_p"])
        if abs(min_p - current_min_p) > float(cfg.gate_min_p_delta) + 1e-9:
            continue
        if tighten_only and min_p < current_min_p - 1e-9:
            continue
        if float(row.get("avg_opens") or 0) < float(cfg.gate_avg_opens_min):
            continue
        if float(row.get("avg_opens") or 0) > float(cfg.gate_avg_opens_max):
            continue
        if cur_pnl > 0 and float(row.get("total_pnl") or 0) < cur_pnl * float(cfg.gate_min_pnl_ratio):
            continue
        if cur_pnl <= 0 and float(row.get("total_pnl") or 0) < cur_pnl:
            continue
        if best is None or float(row["total_pnl"]) > float(best["total_pnl"]):
            best = dict(row)
    if best is None:
        reasons.append("no_safe_candidate")
    elif abs(float(best["min_p"]) - current_min_p) < 0.005:
        reasons.append("same_as_current")
        best = None
    return best, reasons


def _backup_gate(gate_path: Path, tag: str) -> Path:
    ensure_model_dirs()
    stamp = tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = ARCHIVE_DIR / f"gate_pre_apply_{stamp}.json"
    shutil.copy2(gate_path, dest)
    return dest


def apply_gate_min_p(
    *,
    gate_path: Path,
    new_min_p: float,
    tag: str = "",
) -> Dict[str, Any]:
    payload = _read_json(gate_path)
    if not payload:
        raise FileNotFoundError(str(gate_path))
    old = float(payload.get("min_p_true") or 0.35)
    backup = _backup_gate(gate_path, tag)
    payload["min_p_true"] = round(float(new_min_p), 4)
    payload["gate_tune_applied_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    gate_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"gate_path": str(gate_path), "backup": str(backup), "old_min_p_true": old, "new_min_p_true": payload["min_p_true"]}


def tune_gate_after_promote(
    *,
    train_report: dict,
    archive_tag: str = "",
    cfg: MlAutoConfig | None = None,
    gate_path: Path | None = None,
) -> Dict[str, Any]:
    """L1 写 gate_suggestion.json；L2 满足条件时自动改 min_p_true。"""
    c = cfg or MlAutoConfig.from_env()
    ensure_model_dirs()
    gate_file = gate_path or resolve_gate_config_path()
    gate = LiveGateConfig.from_json(gate_file)
    current_min_p = float(gate.min_p_true)

    out: Dict[str, Any] = {
        "action": "skipped",
        "reason": "auto_gate_suggest_disabled",
        "gate_path": str(gate_file),
        "current_min_p_true": current_min_p,
    }
    if not c.auto_gate_suggest:
        GATE_SUGGESTION_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        return out

    sym_file = resolve_train_symbols_path()
    syms = cached_symbols(parse_symbol_list(sym_file.read_text(encoding="utf-8")))
    dates = trading_dates_from_samples(last_sessions=int(c.gate_eval_sessions))
    if not syms or not dates:
        out.update({"action": "skipped", "reason": "missing_symbols_or_dates", "symbols": len(syms), "dates": len(dates)})
        GATE_SUGGESTION_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        return out

    ranker = BreakoutModelBundle.load_production().ranker
    robots = init_robot_wallets(count=robot_count_from_env(), equity_usdt=robot_equity_from_env())
    grid = _min_p_grid(c)
    sweep_floor = min(grid + [current_min_p])

    floor_gate = LiveGateConfig.from_json(gate_file)
    floor_gate.min_p_true = float(sweep_floor)
    print(f"[gate_tune] eval {len(dates)} sessions, timeline floor min_p={sweep_floor:.2f}", flush=True)
    days = run_gate_eval_sessions(
        dates=dates,
        symbols=syms,
        gate=floor_gate,
        ranker=ranker,
        respect_env_filters=True,
        robot_wallets=robots,
        label="timeline",
    )

    score_stats = timeline_score_stats(days)
    current_replay = replay_sessions(days, min_p=current_min_p, gate=gate)
    score_stats["avg_opens"] = current_replay.get("avg_opens", 0)

    previous_train = _previous_train_report()
    previous_tune = _previous_gate_tune()
    triggers = _detect_triggers(
        current_train=train_report,
        previous_train=previous_train,
        previous_tune=previous_tune,
        current_stats=score_stats,
        cfg=c,
    )

    out = {
        "action": "no_change",
        "gate_path": str(gate_file),
        "current_min_p_true": current_min_p,
        "eval_sessions": len(dates),
        "score_stats": score_stats,
        "current_replay": current_replay,
        "triggers": triggers,
        "train_holdout_separation": train_report.get("holdout_separation"),
    }

    tighten_only = float(train_report.get("holdout_separation") or 0) < float(c.gate_separation_warn)
    if tighten_only:
        out["tighten_only"] = True

    if not triggers and not c.gate_always_sweep:
        out["reason"] = "no_trigger"
        GATE_SUGGESTION_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        return out

    sweep_rows = sweep_min_p_grid(days, gate=gate, grid=grid)
    candidate, pick_reasons = _pick_candidate(
        rows=sweep_rows,
        current_min_p=current_min_p,
        current_replay=current_replay,
        cfg=c,
        tighten_only=tighten_only,
    )
    out["sweep_top5"] = sweep_rows[:5]
    out["pick_reasons"] = pick_reasons

    if candidate is None:
        out["action"] = "suggest_hold"
        out["reason"] = "no_safe_candidate"
        GATE_SUGGESTION_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        return out

    suggested_min_p = float(candidate["min_p"])
    out["action"] = "suggest_change"
    out["suggested_min_p_true"] = suggested_min_p
    out["suggested_replay"] = candidate
    out["delta_pnl"] = round(float(candidate["total_pnl"]) - float(current_replay.get("total_pnl") or 0), 1)

    if c.auto_gate_apply:
        applied = apply_gate_min_p(gate_path=gate_file, new_min_p=suggested_min_p, tag=archive_tag)
        out["action"] = "applied"
        out["applied"] = applied
    else:
        out["reason"] = "apply_disabled"

    GATE_SUGGESTION_JSON.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
