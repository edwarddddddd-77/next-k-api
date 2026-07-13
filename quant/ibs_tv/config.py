"""IBS TV lane — 对齐 TradingView AlgoTradeKit v0.5（按 SPY/QQQ 推荐参数）。"""

from __future__ import annotations

import os
from dataclasses import replace

from quant.ibs.config import IbsLaneConfig, _float_env, _int_env
from quant.ibs.profile import PROFILE_TV
from quant.ibs_tv.paths import resolve_ibs_tv_symbols_path
from quant.ibs_tv.switches import IBS_TV_SWITCH
from quant.ibs_tv.symbol_params import resolve_tv_symbol_params


def _env_is_set(name: str) -> bool:
    return bool(str(os.getenv(name, "")).strip())


class IbsTvConfig(IbsLaneConfig):
    lane: str = "ibs_tv"
    profile: str = PROFILE_TV

    @classmethod
    def from_env(cls) -> "IbsTvConfig":
        base = IbsLaneConfig.from_env(
            lane="ibs_tv",
            profile=PROFILE_TV,
            switch=IBS_TV_SWITCH,
            resolve_symbols_path=resolve_ibs_tv_symbols_path,
            env_prefix="IBS_TV",
        )
        return cls(**base.__dict__)

    def lane_config_for_symbol(self, symbol: str) -> IbsLaneConfig:
        """合并 TV 按标的推荐值；显式 env 覆盖 symbol 默认。"""
        sp = resolve_tv_symbol_params(symbol)
        prefix = "IBS_TV"
        entry = (
            _float_env(f"{prefix}_VNPY_ENTRY_THRESHOLD", sp.entry_threshold)
            if _env_is_set(f"{prefix}_VNPY_ENTRY_THRESHOLD")
            else sp.entry_threshold
        )
        exit_ = (
            _float_env(f"{prefix}_VNPY_EXIT_THRESHOLD", sp.exit_threshold)
            if _env_is_set(f"{prefix}_VNPY_EXIT_THRESHOLD")
            else sp.exit_threshold
        )
        trend_ma_period = (
            max(0, _int_env(f"{prefix}_VNPY_TREND_MA_PERIOD", sp.trend_ma_period))
            if _env_is_set(f"{prefix}_VNPY_TREND_MA_PERIOD")
            else sp.trend_ma_period
        )
        min_entry_distance_pct = (
            _float_env(f"{prefix}_VNPY_MIN_ENTRY_DISTANCE_PCT", sp.min_entry_distance_pct)
            if _env_is_set(f"{prefix}_VNPY_MIN_ENTRY_DISTANCE_PCT")
            else sp.min_entry_distance_pct
        )
        max_trade_duration_days = (
            max(1, _int_env(f"{prefix}_VNPY_MAX_TRADE_DURATION_DAYS", sp.max_trade_duration_days))
            if _env_is_set(f"{prefix}_VNPY_MAX_TRADE_DURATION_DAYS")
            else sp.max_trade_duration_days
        )
        return replace(
            self,
            entry_threshold=entry,
            exit_threshold=exit_,
            trend_ma_period=trend_ma_period,
            min_entry_distance_pct=min_entry_distance_pct,
            max_trade_duration_days=max_trade_duration_days,
        )
