"""分档移动止盈（纸面）— 逻辑来自 buou_trail，与交易所解耦。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


TIER_NONE = "none"
TIER_LOW = "low"
TIER_TIER1 = "tier1"
TIER_TIER2 = "tier2"

TIER_LABELS = {
    TIER_NONE: "无",
    TIER_LOW: "低档保护止盈",
    TIER_TIER1: "第一档移动止盈",
    TIER_TIER2: "第二档移动止盈",
}


@dataclass(frozen=True)
class TrailConfig:
    enabled: bool
    stop_loss_pct: float
    low_trail_stop_loss_pct: float
    trail_stop_loss_pct: float
    higher_trail_stop_loss_pct: float
    low_trail_profit_threshold: float
    first_trail_profit_threshold: float
    second_trail_profit_threshold: float


@dataclass
class TrailEval:
    profit_pct: float
    peak_profit_pct: float
    trail_tier: str
    exit_rule: Optional[str] = None

    @property
    def tier_label(self) -> str:
        return TIER_LABELS.get(self.trail_tier, self.trail_tier)


def profit_pct(side: str, entry: float, mark: float) -> float:
    if entry <= 0 or mark <= 0:
        return 0.0
    s = side.upper()
    if s == "LONG":
        return (mark - entry) / entry * 100.0
    if s == "SHORT":
        return (entry - mark) / entry * 100.0
    return 0.0


def tier_from_peak(peak: float, cfg: TrailConfig) -> str:
    if peak >= cfg.second_trail_profit_threshold:
        return TIER_TIER2
    if peak >= cfg.first_trail_profit_threshold:
        return TIER_TIER1
    if peak >= cfg.low_trail_profit_threshold:
        return TIER_LOW
    return TIER_NONE


def evaluate_trail(
    *,
    side: str,
    entry: float,
    mark: float,
    peak_profit_pct: float,
    cfg: TrailConfig,
) -> TrailEval:
    """
    更新峰值并判断是否触发平仓。

    回撤比例语义与 buou 一致：第一档止盈线 = peak * (1 - trail_stop_loss_pct)，
    其中 trail_stop_loss_pct=0.2 表示从峰值利润回撤 20%。
    """
    profit = profit_pct(side, entry, mark)
    peak = max(float(peak_profit_pct or 0.0), profit)
    tier = tier_from_peak(peak, cfg) if cfg.enabled else TIER_NONE
    out = TrailEval(profit_pct=profit, peak_profit_pct=peak, trail_tier=tier)

    # 硬止损与分档移动止盈解耦：关闭 MOM_TRAIL_ENABLED 时仍执行 -2% 硬止损
    if cfg.stop_loss_pct > 0 and profit <= -cfg.stop_loss_pct:
        out.exit_rule = "trail_stop"
        return out

    if not cfg.enabled:
        return out

    if tier == TIER_LOW and profit <= cfg.low_trail_stop_loss_pct:
        out.exit_rule = "trail_low"
        return out

    if tier == TIER_TIER1:
        floor = peak * (1.0 - cfg.trail_stop_loss_pct)
        if profit <= floor:
            out.exit_rule = "trail_tier1"
            return out

    if tier == TIER_TIER2:
        floor = peak * (1.0 - cfg.higher_trail_stop_loss_pct)
        if profit <= floor:
            out.exit_rule = "trail_tier2"
            return out

    return out
