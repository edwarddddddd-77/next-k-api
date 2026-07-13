"""IBS 保守 / 激进默认参数（对齐 CazSyd / Pagonidis）。"""

from __future__ import annotations

from dataclasses import dataclass

PROFILE_CONSERVATIVE = "conservative"
PROFILE_AGGRESSIVE = "aggressive"
PROFILE_TV = "tv"


@dataclass(frozen=True)
class IbsProfileDefaults:
    entry_threshold: float
    exit_threshold: float
    position_pct: float
    risk_pct: float
    stop_loss_pct: float
    sma_period: int
    trend_ma_type: str
    trend_ma_period: int
    trend_price_mode: str
    min_entry_distance_pct: float
    max_trade_duration_days: int
    daily_bar_source: str
    execute_at_next_open: bool
    exec_after_open_minutes: int
    init_bar_days: int
    trade_type: str = "long_only"
    product_type: str = "spot"


PROFILE_DEFAULTS: dict[str, IbsProfileDefaults] = {
    PROFILE_CONSERVATIVE: IbsProfileDefaults(
        entry_threshold=0.20,
        exit_threshold=0.50,
        position_pct=0.10,
        risk_pct=0.01,
        stop_loss_pct=0.0,
        sma_period=200,
        trend_ma_type="sma",
        trend_ma_period=200,
        trend_price_mode="prev_close",
        min_entry_distance_pct=0.0,
        max_trade_duration_days=0,
        daily_bar_source="session_5m",
        execute_at_next_open=True,
        exec_after_open_minutes=1,
        init_bar_days=400,
    ),
    PROFILE_AGGRESSIVE: IbsProfileDefaults(
        entry_threshold=0.19,
        exit_threshold=0.95,
        position_pct=0.30,
        risk_pct=0.02,
        stop_loss_pct=0.0,
        sma_period=0,
        trend_ma_type="none",
        trend_ma_period=0,
        trend_price_mode="none",
        min_entry_distance_pct=0.0,
        max_trade_duration_days=0,
        daily_bar_source="session_5m",
        execute_at_next_open=True,
        exec_after_open_minutes=1,
        init_bar_days=400,
    ),
    PROFILE_TV: IbsProfileDefaults(
        entry_threshold=0.09,
        exit_threshold=0.985,
        position_pct=1.0,
        risk_pct=0.01,
        stop_loss_pct=0.0,
        sma_period=0,
        trend_ma_type="ema",
        trend_ma_period=220,
        trend_price_mode="current",
        min_entry_distance_pct=0.0,
        max_trade_duration_days=14,
        daily_bar_source="session_5m",
        execute_at_next_open=True,
        exec_after_open_minutes=1,
        init_bar_days=400,
        trade_type="long_short",
        product_type="perp",
    ),
}
