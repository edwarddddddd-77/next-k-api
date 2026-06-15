"""ML 自动化配置（env）。"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _truthy(name: str, *, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


def _float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


@dataclass
class MlAutoConfig:
    """完全自动化流水线开关与验收阈值。"""

    auto_fetch_klines: bool
    kline_days: float
    kline_skip_existing: bool
    auto_validate: bool
    auto_promote: bool
    min_holdout_n: int
    min_holdout_accuracy: float
    min_holdout_separation: float
    min_train_samples: int
    max_train_brier: float
    fail_on_kline_errors: bool
    allow_promote_without_validate: bool
    auto_gate_suggest: bool
    auto_gate_apply: bool
    gate_eval_sessions: int
    gate_min_p_delta: float
    gate_min_pnl_ratio: float
    gate_avg_opens_min: float
    gate_avg_opens_max: float
    gate_sweep_min: float
    gate_sweep_max: float
    gate_sweep_step: float
    gate_trigger_sep_drop: float
    gate_separation_warn: float
    gate_trigger_median_shift: float
    gate_trigger_opens_change: float
    gate_always_sweep: bool

    @classmethod
    def from_env(cls) -> "MlAutoConfig":
        return cls(
            auto_fetch_klines=_truthy("ORB_ML_AUTO_FETCH_KLINES", default=True),
            kline_days=_float("ORB_ML_KLINE_DAYS", 180.0),
            kline_skip_existing=_truthy("ORB_ML_KLINE_SKIP_EXISTING", default=False),
            auto_validate=_truthy("ORB_ML_AUTO_VALIDATE", default=False),
            auto_promote=_truthy("ORB_ML_AUTO_PROMOTE", default=False),
            min_holdout_n=_int("ORB_ML_MIN_HOLDOUT_N", 5),
            min_holdout_accuracy=_float("ORB_ML_MIN_HOLDOUT_ACCURACY", 0.52),
            min_holdout_separation=_float("ORB_ML_MIN_HOLDOUT_SEPARATION", 0.02),
            min_train_samples=_int("ORB_ML_MIN_TRAIN_SAMPLES", 200),
            max_train_brier=_float("ORB_ML_MAX_TRAIN_BRIER", 0.22),
            fail_on_kline_errors=_truthy("ORB_ML_FAIL_ON_KLINE_ERRORS", default=True),
            allow_promote_without_validate=_truthy("ORB_ML_ALLOW_PROMOTE_WITHOUT_VALIDATE", default=False),
            auto_gate_suggest=_truthy("ORB_ML_AUTO_GATE_SUGGEST", default=False),
            auto_gate_apply=_truthy("ORB_ML_AUTO_GATE_APPLY", default=False),
            gate_eval_sessions=_int("ORB_ML_GATE_EVAL_SESSIONS", 60),
            gate_min_p_delta=_float("ORB_ML_GATE_MIN_P_DELTA", 0.04),
            gate_min_pnl_ratio=_float("ORB_ML_GATE_MIN_PNL_RATIO", 0.95),
            gate_avg_opens_min=_float("ORB_ML_GATE_AVG_OPENS_MIN", 1.5),
            gate_avg_opens_max=_float("ORB_ML_GATE_AVG_OPENS_MAX", 8.0),
            gate_sweep_min=_float("ORB_ML_GATE_SWEEP_MIN", 0.30),
            gate_sweep_max=_float("ORB_ML_GATE_SWEEP_MAX", 0.45),
            gate_sweep_step=_float("ORB_ML_GATE_SWEEP_STEP", 0.02),
            gate_trigger_sep_drop=_float("ORB_ML_GATE_TRIGGER_SEP_DROP", 0.30),
            gate_separation_warn=_float("ORB_ML_GATE_SEPARATION_WARN", 0.025),
            gate_trigger_median_shift=_float("ORB_ML_GATE_TRIGGER_MEDIAN_SHIFT", 0.05),
            gate_trigger_opens_change=_float("ORB_ML_GATE_TRIGGER_OPENS_CHANGE", 0.30),
            gate_always_sweep=_truthy("ORB_ML_GATE_ALWAYS_SWEEP", default=False),
        )
