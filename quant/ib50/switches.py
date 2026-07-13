"""IB50 策略开关。"""

from __future__ import annotations

from quant.common.strategy_switch import StrategySwitchSpec

IB50_SWITCH = StrategySwitchSpec(
    lane="ib50",
    title="IB50 Initial Balance",
    enabled_keys=(
        "STRATEGY_IB50_ENABLED",
        "IB50_VNPY_ENABLED",
        "IB50_ENABLED",
    ),
    live_keys=(
        "STRATEGY_IB50_LIVE",
        "IB50_VNPY_LIVE_ENABLED",
        "IB50_LIVE_ENABLED",
    ),
    shadow_keys=(
        "STRATEGY_IB50_SHADOW",
        "IB50_VNPY_SHADOW",
        "IB50_SHADOW",
    ),
    default_enabled=False,
    default_live=False,
    default_shadow=False,
)
