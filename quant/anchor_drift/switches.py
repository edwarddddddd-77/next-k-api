"""Anchor Drift 策略开关。"""

from __future__ import annotations

from quant.common.strategy_switch import StrategySwitchSpec

ANCHOR_DRIFT_SWITCH = StrategySwitchSpec(
    lane="anchor_drift",
    title="Anchor Drift",
    enabled_keys=(
        "STRATEGY_ANCHOR_DRIFT_ENABLED",
        "ANCHOR_DRIFT_VNPY_ENABLED",
    ),
    live_keys=(
        "STRATEGY_ANCHOR_DRIFT_LIVE",
        "ANCHOR_DRIFT_VNPY_LIVE_ENABLED",
    ),
    shadow_keys=(
        "STRATEGY_ANCHOR_DRIFT_SHADOW",
        "ANCHOR_DRIFT_VNPY_SHADOW",
    ),
    default_enabled=False,
    default_live=False,
    default_shadow=False,
)
