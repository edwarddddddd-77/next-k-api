"""KAMA Trend 策略开关。"""

from __future__ import annotations

from quant.common.strategy_switch import StrategySwitchSpec

KAMA_TREND_SWITCH = StrategySwitchSpec(
    lane="kama_trend",
    title="KAMA Trend",
    enabled_keys=(
        "STRATEGY_KAMA_TREND_ENABLED",
        "KAMA_TREND_VNPY_ENABLED",
    ),
    live_keys=(
        "STRATEGY_KAMA_TREND_LIVE",
        "KAMA_TREND_VNPY_LIVE_ENABLED",
    ),
    shadow_keys=(
        "STRATEGY_KAMA_TREND_SHADOW",
        "KAMA_TREND_VNPY_SHADOW",
    ),
    default_enabled=False,
    default_live=False,
    default_shadow=False,
)
