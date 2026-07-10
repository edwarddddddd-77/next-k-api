"""MtfMomo2xA 策略开关。"""

from __future__ import annotations

from quant.common.strategy_switch import StrategySwitchSpec

MTFMOMO_SWITCH = StrategySwitchSpec(
    lane="mtfmomo",
    title="MtfMomo2xA",
    enabled_keys=(
        "STRATEGY_MTFMOMO_ENABLED",
        "MTFMOMO_VNPY_ENABLED",
    ),
    live_keys=(
        "STRATEGY_MTFMOMO_LIVE",
        "MTFMOMO_VNPY_LIVE_ENABLED",
    ),
    shadow_keys=(
        "STRATEGY_MTFMOMO_SHADOW",
        "MTFMOMO_VNPY_SHADOW",
    ),
    default_enabled=False,
    default_live=False,
    default_shadow=False,
)
