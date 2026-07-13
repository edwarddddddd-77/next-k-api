"""AlgoTradeKit IBS v0.5 按标的推荐参数（TV 源码注释）。

//Setting for QQQ: 0.09, 0.985, 220, 0, 14
//Setting for SPY:  0.11, 0.995, 200, 0, 12
"""

from __future__ import annotations

from dataclasses import dataclass

from quant.common.kline_cache import norm_symbol

TV_SYMBOL_ALIASES: dict[str, str] = {
    "SPY": "SPY",
    "SPYUSDT": "SPY",
    "SPX": "SPY",
    "QQQ": "QQQ",
    "QQQUSDT": "QQQ",
    "NDQ": "QQQ",
}


@dataclass(frozen=True)
class IbsTvSymbolParams:
    entry_threshold: float
    exit_threshold: float
    trend_ma_period: int
    min_entry_distance_pct: float
    max_trade_duration_days: int


TV_SYMBOL_PARAMS: dict[str, IbsTvSymbolParams] = {
    "SPY": IbsTvSymbolParams(
        entry_threshold=0.11,
        exit_threshold=0.995,
        trend_ma_period=200,
        min_entry_distance_pct=0.0,
        max_trade_duration_days=12,
    ),
    "QQQ": IbsTvSymbolParams(
        entry_threshold=0.09,
        exit_threshold=0.985,
        trend_ma_period=220,
        min_entry_distance_pct=0.0,
        max_trade_duration_days=14,
    ),
}


def tv_symbol_key(raw: str) -> str:
    s = norm_symbol(str(raw or "").strip())
    if not s:
        return ""
    if s in TV_SYMBOL_ALIASES:
        return TV_SYMBOL_ALIASES[s]
    base = s[:-4] if s.endswith("USDT") else s
    return TV_SYMBOL_ALIASES.get(base, base)


def resolve_tv_symbol_params(raw: str, *, fallback: str = "QQQ") -> IbsTvSymbolParams:
    key = tv_symbol_key(raw)
    if key in TV_SYMBOL_PARAMS:
        return TV_SYMBOL_PARAMS[key]
    return TV_SYMBOL_PARAMS[fallback]
