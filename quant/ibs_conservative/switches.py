"""IBS 保守 lane 开关。"""

from __future__ import annotations

from quant.common.strategy_switch import StrategySwitchSpec

IBS_CONSERVATIVE_SWITCH = StrategySwitchSpec(
    lane="ibs_conservative",
    title="IBS Conservative",
    enabled_keys=(
        "STRATEGY_IBS_CONSERVATIVE_ENABLED",
        "IBS_CONSERVATIVE_VNPY_ENABLED",
    ),
    live_keys=(
        "STRATEGY_IBS_CONSERVATIVE_LIVE",
        "IBS_CONSERVATIVE_VNPY_LIVE_ENABLED",
    ),
    shadow_keys=(
        "STRATEGY_IBS_CONSERVATIVE_SHADOW",
        "IBS_CONSERVATIVE_VNPY_SHADOW",
    ),
    default_enabled=False,
    default_live=False,
    default_shadow=False,
)
