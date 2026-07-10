"""Smart Breakout 策略开关。"""

from __future__ import annotations

from quant.common.strategy_switch import StrategySwitchSpec

SQUEEZE_BREAKOUT_SWITCH = StrategySwitchSpec(
    lane="squeeze_breakout",
    title="Smart Breakout Targets",
    enabled_keys=(
        "STRATEGY_SQUEEZE_BREAKOUT_ENABLED",
        "SQUEEZE_BREAKOUT_VNPY_ENABLED",
    ),
    live_keys=(
        "STRATEGY_SQUEEZE_BREAKOUT_LIVE",
        "SQUEEZE_BREAKOUT_VNPY_LIVE_ENABLED",
    ),
    shadow_keys=(
        "STRATEGY_SQUEEZE_BREAKOUT_SHADOW",
        "SQUEEZE_BREAKOUT_VNPY_SHADOW",
    ),
    default_enabled=False,
    default_live=False,
    default_shadow=False,
)
