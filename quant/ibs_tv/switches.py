"""IBS TV lane 开关。"""

from __future__ import annotations

from quant.common.strategy_switch import StrategySwitchSpec

IBS_TV_SWITCH = StrategySwitchSpec(
    lane="ibs_tv",
    title="IBS TradingView",
    enabled_keys=(
        "STRATEGY_IBS_TV_ENABLED",
        "IBS_TV_VNPY_ENABLED",
    ),
    live_keys=(
        "STRATEGY_IBS_TV_LIVE",
        "IBS_TV_VNPY_LIVE_ENABLED",
    ),
    shadow_keys=(
        "STRATEGY_IBS_TV_SHADOW",
        "IBS_TV_VNPY_SHADOW",
    ),
    default_enabled=False,
    default_live=False,
    default_shadow=False,
)
