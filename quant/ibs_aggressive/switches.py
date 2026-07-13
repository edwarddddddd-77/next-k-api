"""IBS 激进 lane 开关。"""

from __future__ import annotations

from quant.common.strategy_switch import StrategySwitchSpec

IBS_AGGRESSIVE_SWITCH = StrategySwitchSpec(
    lane="ibs_aggressive",
    title="IBS Aggressive",
    enabled_keys=(
        "STRATEGY_IBS_AGGRESSIVE_ENABLED",
        "IBS_AGGRESSIVE_VNPY_ENABLED",
    ),
    live_keys=(
        "STRATEGY_IBS_AGGRESSIVE_LIVE",
        "IBS_AGGRESSIVE_VNPY_LIVE_ENABLED",
    ),
    shadow_keys=(
        "STRATEGY_IBS_AGGRESSIVE_SHADOW",
        "IBS_AGGRESSIVE_VNPY_SHADOW",
    ),
    default_enabled=False,
    default_live=False,
    default_shadow=False,
)
