"""每日寻优后 Profile 纸面自动开/关判定（内置默认规则，无需 env 配置）。"""

from __future__ import annotations

from typing import Any, Dict

# 内置默认门槛（与 MOSS_QUANT_DAILY_OPTIMIZE_APPLY 同步生效）
MIN_RETURN = 0.0          # 收益须 > 0
MIN_TRADES = 8            # 回测回合 ≥ 8
MAX_DRAWDOWN = 0.45       # 最大回撤 ≤ 45%
REQUIRE_NO_BLOWUP = True  # 无回测爆仓


def evaluate_profile_auto_enable(summary: Dict[str, Any]) -> Dict[str, Any]:
    """根据回测摘要决定是否启用纸面 Profile。"""
    if summary.get("error"):
        return _pack(False, "寻优失败")

    ret = float(summary.get("total_return") or 0)
    trades = int(summary.get("total_trades") or 0)
    wr = float(summary.get("win_rate") or 0)
    mdd = abs(float(summary.get("max_drawdown") or 0))
    blow = int(summary.get("blowup_count") or 0)

    fails = []
    if ret <= MIN_RETURN:
        fails.append("收益≤%.1f%%" % (MIN_RETURN * 100))
    if trades < MIN_TRADES:
        fails.append("回合<%d" % MIN_TRADES)
    if mdd > MAX_DRAWDOWN:
        fails.append("回撤>%.0f%%" % (MAX_DRAWDOWN * 100))
    if REQUIRE_NO_BLOWUP and blow > 0:
        fails.append("回测爆仓")

    if fails:
        return _pack(False, "；".join(fails))

    detail = "收益%.1f%%·%d笔" % (ret * 100, trades)
    if trades > 0:
        detail += "·胜率%.0f%%" % (wr * 100)
    return _pack(True, detail)


def _pack(enabled: bool, reason: str) -> Dict[str, Any]:
    return {
        "auto_enabled": bool(enabled),
        "auto_enable_label": "开" if enabled else "关",
        "auto_enable_reason": reason,
    }


def profile_enabled_from_gate(gate: Dict[str, Any]) -> bool:
    return bool(gate.get("auto_enabled"))
