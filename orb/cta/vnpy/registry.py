"""vnpy_ctastrategy 官方策略注册表（BacktestingEngine 用）。"""

from __future__ import annotations

from typing import Any, Dict, Type

from orb.vnpy.bootstrap import ensure_vnpy_path

ensure_vnpy_path()

from vnpy_ctastrategy import CtaTemplate  # noqa: E402
from vnpy_ctastrategy.strategies.atr_rsi_strategy import AtrRsiStrategy  # noqa: E402
from vnpy_ctastrategy.strategies.boll_channel_strategy import BollChannelStrategy  # noqa: E402
from vnpy_ctastrategy.strategies.double_ma_strategy import DoubleMaStrategy  # noqa: E402
from vnpy_ctastrategy.strategies.dual_thrust_strategy import DualThrustStrategy  # noqa: E402
from vnpy_ctastrategy.strategies.king_keltner_strategy import KingKeltnerStrategy  # noqa: E402
from vnpy_ctastrategy.strategies.turtle_signal_strategy import TurtleSignalStrategy  # noqa: E402

from orb.gtl.vnpy.strategy import GtlBreakoutStrategy  # noqa: E402


VNPY_CTA_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "double_ma": {
        "title": "双均线金叉死叉",
        "class": DoubleMaStrategy,
        "uses_fixed_size": False,
    },
    "atr_rsi": {
        "title": "ATR放大 + RSI极端",
        "class": AtrRsiStrategy,
        "uses_fixed_size": True,
    },
    "boll_channel": {
        "title": "布林通道 + CCI + ATR止损",
        "class": BollChannelStrategy,
        "uses_fixed_size": True,
    },
    "king_keltner": {
        "title": "肯特纳通道突破",
        "class": KingKeltnerStrategy,
        "uses_fixed_size": True,
    },
    "dual_thrust": {
        "title": "Dual Thrust 日内突破",
        "class": DualThrustStrategy,
        "uses_fixed_size": True,
    },
    "turtle": {
        "title": "海龟唐奇安通道",
        "class": TurtleSignalStrategy,
        "uses_fixed_size": True,
        "compound": False,
    },
    "gtl_birth_break": {
        "title": "GTL 诞生预测+突破确认（文章语义）",
        "class": GtlBreakoutStrategy,
        "uses_fixed_size": True,
        "gtl": True,
        "default_setting": {"trade_mode": "birth_break", "force_flat_on_stop": True},
    },
    "gtl_birth_break_honest": {
        "title": "GTL birth_break 诚实研究（20bar+反向突破平仓）",
        "class": GtlBreakoutStrategy,
        "uses_fixed_size": True,
        "gtl": True,
        "default_setting": {
            "trade_mode": "birth_break",
            "max_hold_bars": 20,
            "exit_on_opposite_break": True,
            "force_flat_on_stop": True,
        },
    },
    "gtl_break": {
        "title": "GTL 冻结结构突破",
        "class": GtlBreakoutStrategy,
        "uses_fixed_size": True,
        "gtl": True,
        "default_setting": {"trade_mode": "break"},
    },
    "gtl_signal": {
        "title": "GTL 四道门信号",
        "class": GtlBreakoutStrategy,
        "uses_fixed_size": True,
        "gtl": True,
        "default_setting": {"trade_mode": "signal"},
    },
    "gtl_signal_break": {
        "title": "GTL 信号+突破确认",
        "class": GtlBreakoutStrategy,
        "uses_fixed_size": True,
        "gtl": True,
        "default_setting": {"trade_mode": "signal_break"},
    },
}


def list_vnpy_strategies() -> list[str]:
    return list(VNPY_CTA_STRATEGIES.keys())


def get_vnpy_strategy_class(key: str) -> Type[CtaTemplate]:
    meta = VNPY_CTA_STRATEGIES[key]
    return meta["class"]
