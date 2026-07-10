"""Trading ORB 策略开关。"""

from __future__ import annotations

from quant.common.strategy_switch import StrategySwitchSpec

TRADING_ORB_SWITCH = StrategySwitchSpec(
    lane="trading_orb",
    title="Trading ORB",
    enabled_keys=(
        "STRATEGY_TRADING_ORB_ENABLED",
        "ORB_VNPY_ENABLED",
        "ORB_ENABLED",
    ),
    live_keys=(
        "STRATEGY_TRADING_ORB_LIVE",
        "ORB_VNPY_LIVE_ENABLED",
        "ORB_LIVE_ENABLED",
    ),
    shadow_keys=(
        "STRATEGY_TRADING_ORB_SHADOW",
        "ORB_VNPY_SHADOW",
        "ORB_SHADOW",
    ),
    default_enabled=False,
    default_live=False,
    default_shadow=False,
)
