"""Bot 参数模板与进化约束。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from moss_quant.core.decision import DecisionParams

PERSONALITY_FIELDS = [
    "long_bias",
    "base_leverage",
    "max_leverage",
    "risk_per_trade",
    "max_position_pct",
    "rolling_enabled",
    "rolling_trigger_pct",
    "rolling_reinvest_pct",
    "rolling_max_times",
    "rolling_move_stop",
    "trend_weight",
    "momentum_weight",
    "mean_revert_weight",
    "volume_weight",
    "volatility_weight",
]

TACTICAL_FLOAT_FIELDS = [
    "entry_threshold",
    "exit_threshold",
    "sl_atr_mult",
    "tp_rr_ratio",
    "trailing_activation_pct",
    "trailing_distance_atr",
    "regime_sensitivity",
    "supertrend_mult",
    "trend_strength_min",
]

MAX_DRIFT_PCT = 0.30

# 无杠杆默认仓位（1x 满仓；名义 ≈ 单 Profile 钱包，对齐原 10x+10%risk 尺度）
SPOT_PERSONALITY_DEFAULTS: Dict[str, Any] = {
    "base_leverage": 1.0,
    "max_leverage": 1.0,
    "risk_per_trade": 1.0,
    "max_position_pct": 1.0,
}

_TEMPLATES: Dict[str, Dict[str, float]] = {
    "balanced": {
        "trend_weight": 0.30,
        "momentum_weight": 0.25,
        "mean_revert_weight": 0.15,
        "volume_weight": 0.15,
        "volatility_weight": 0.15,
    },
    "momentum": {
        "trend_weight": 0.20,
        "momentum_weight": 0.50,
        "mean_revert_weight": 0.05,
        "volume_weight": 0.15,
        "volatility_weight": 0.10,
    },
    "trend": {
        "trend_weight": 0.50,
        "momentum_weight": 0.20,
        "mean_revert_weight": 0.05,
        "volume_weight": 0.15,
        "volatility_weight": 0.10,
    },
    "mean_revert": {
        "trend_weight": 0.15,
        "momentum_weight": 0.15,
        "mean_revert_weight": 0.45,
        "volume_weight": 0.15,
        "volatility_weight": 0.10,
    },
}


def resolve_params_dict(raw: Optional[dict]) -> dict:
    raw = raw or {}
    clean = {k: v for k, v in raw.items() if v is not None}
    p = DecisionParams.from_dict(clean)
    p.normalize_weights()
    return p.to_dict()


def apply_spot_personality(params: dict) -> dict:
    """无杠杆满仓：写入性格参数并归一化。"""
    merged = copy.deepcopy(params or {})
    merged.update(SPOT_PERSONALITY_DEFAULTS)
    return resolve_params_dict(merged)


def build_initial_params(
    *,
    template: str = "balanced",
    overrides: Optional[dict] = None,
) -> dict:
    key = (template or "balanced").strip().lower()
    weights = _TEMPLATES.get(key, _TEMPLATES["balanced"])
    base = DecisionParams()
    d = base.to_dict()
    d.update(weights)
    d.update(SPOT_PERSONALITY_DEFAULTS)
    if overrides:
        d.update({k: v for k, v in overrides.items() if v is not None})
    return resolve_params_dict(d)


def clamp_tactical_drift(current: dict, initial: dict) -> dict:
    result = copy.deepcopy(current)
    for field in TACTICAL_FLOAT_FIELDS:
        if field not in initial or field not in result:
            continue
        init_val = initial[field]
        curr_val = result[field]
        if isinstance(init_val, (int, float)) and init_val != 0:
            lo = init_val * (1 - MAX_DRIFT_PCT)
            hi = init_val * (1 + MAX_DRIFT_PCT)
            if lo > hi:
                lo, hi = hi, lo
            result[field] = max(lo, min(float(curr_val), hi))
    return result


def lock_personality(current: dict, initial: dict) -> dict:
    result = copy.deepcopy(current)
    for field in PERSONALITY_FIELDS:
        if field in initial:
            result[field] = initial[field]
    return result


def extract_tactical_params(final_params: dict, initial_params: dict) -> dict:
    """进化 final_params → 仅战术字段（性格锁定为 initial）。"""
    merged = lock_personality(final_params, initial_params)
    merged = clamp_tactical_drift(merged, initial_params)
    return {k: v for k, v in merged.items() if k not in PERSONALITY_FIELDS}


def cap_leverage_for_symbol(params: dict, symbol: str) -> dict:
    from moss_quant.leverage_caps_bn import cap_params_for_symbol

    return cap_params_for_symbol(params, symbol)


def load_params_schema() -> dict:
    path = Path(__file__).resolve().parent / "params_schema.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_schedule_round(params: dict, initial: dict) -> dict:
    p = lock_personality(params, initial)
    p = clamp_tactical_drift(p, initial)
    return resolve_params_dict(p)
