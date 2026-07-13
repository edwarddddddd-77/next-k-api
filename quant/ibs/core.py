"""IBS 信号核心 — CazSyd / Pagonidis / TradingView AlgoTradeKit。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

IbsAction = Literal["BUY", "SELL", "SHORT", "COVER", "HOLD"]
TrendPriceMode = Literal["prev_close", "current", "none"]
VALID_TRADE_TYPES = ("long_only", "short_only", "long_short")


@dataclass(frozen=True)
class SessionDailyBar:
    session_day: str
    open_ms: int
    open: float
    high: float
    low: float
    close: float

    @property
    def ibs(self) -> float:
        return compute_ibs(self.high, self.low, self.close)


@dataclass(frozen=True)
class IbsSignalContext:
    """收盘评估用的 session 日线上下文。"""

    prev_bar: SessionDailyBar
    ibs_closes: tuple[float, ...]
    ma_closes: tuple[float, ...]
    trend_price: float | None


def compute_ibs(high: float, low: float, close: float) -> float:
    rng = float(high) - float(low)
    if rng <= 0:
        return 0.5
    return (float(close) - float(low)) / rng


def normalize_trade_type(raw: str) -> str:
    value = str(raw or "long_only").strip().lower().replace(" ", "_").replace("-", "_")
    if value in ("long", "longonly"):
        return "long_only"
    if value in ("short", "shortonly"):
        return "short_only"
    if value in ("long_short", "longandshort", "long+short", "both"):
        return "long_short"
    if value in VALID_TRADE_TYPES:
        return value
    return "long_only"


def select_signal_context(
    daily: Sequence[SessionDailyBar],
    *,
    trend_price_mode: str,
    current_price: float,
    ma_excludes_last_bar: bool = False,
) -> IbsSignalContext | None:
    """对齐原策略：IBS 用上一根完整 session 日 K；MA 序列按 profile 选取。"""
    if len(daily) < 2:
        return None
    prev_bar = daily[-2]
    ibs_closes = tuple(float(b.close) for b in daily[:-1])
    mode = (trend_price_mode or "prev_close").strip().lower()
    if mode == "current":
        ma_source = daily[:-1] if ma_excludes_last_bar else daily
        ma_closes = tuple(float(b.close) for b in ma_source)
        trend_price = float(current_price)
    else:
        ma_closes = ibs_closes
        trend_price = None if mode in ("", "none", "prev_close") else float(current_price)
    return IbsSignalContext(
        prev_bar=prev_bar,
        ibs_closes=ibs_closes,
        ma_closes=ma_closes,
        trend_price=trend_price,
    )


def sma_last(closes: Sequence[float], period: int) -> float | None:
    p = int(period)
    if p <= 0 or len(closes) < p:
        return None
    window = closes[-p:]
    return sum(float(x) for x in window) / float(p)


def ema_last(closes: Sequence[float], period: int) -> float | None:
    p = int(period)
    if p <= 0 or len(closes) < p:
        return None
    alpha = 2.0 / (float(p) + 1.0)
    ema = float(closes[0])
    for px in closes[1:]:
        ema = alpha * float(px) + (1.0 - alpha) * ema
    return ema


def trend_ma_last(closes: Sequence[float], period: int, ma_type: str) -> float | None:
    kind = (ma_type or "none").strip().lower()
    if kind in ("", "none", "off"):
        return None
    if kind == "ema":
        return ema_last(closes, period)
    return sma_last(closes, period)


def _resolve_position_side(*, position_side: int | None, in_position: bool) -> int:
    if position_side is not None:
        if position_side > 0:
            return 1
        if position_side < 0:
            return -1
        return 0
    return 1 if in_position else 0


def _ma_period(sma_period: int, trend_ma_period: int) -> int:
    return int(trend_ma_period) if int(trend_ma_period) > 0 else int(sma_period)


def _ma_kind(trend_ma_type: str, period: int) -> str:
    kind = (trend_ma_type or "").strip().lower()
    if not kind and period > 0:
        return "sma"
    return kind


def _long_ma_ok(
    *,
    px: float,
    period: int,
    kind: str,
    daily_closes: Sequence[float],
    ma_closes: Sequence[float] | None,
) -> bool:
    if period <= 0 or kind in ("", "none", "off"):
        return True
    series = ma_closes if ma_closes is not None else daily_closes
    ma = trend_ma_last(series, period, kind)
    return ma is None or px > float(ma)


def _short_ma_ok(
    *,
    px: float,
    period: int,
    kind: str,
    daily_closes: Sequence[float],
    ma_closes: Sequence[float] | None,
) -> bool:
    if period <= 0 or kind in ("", "none", "off"):
        return True
    series = ma_closes if ma_closes is not None else daily_closes
    ma = trend_ma_last(series, period, kind)
    return ma is None or px < float(ma)


def _long_entry_distance_ok(px: float, last_entry_price: float, min_entry_distance_pct: float) -> bool:
    min_dist = float(min_entry_distance_pct or 0.0)
    if min_dist <= 0 or float(last_entry_price or 0.0) <= 0:
        return True
    return px <= float(last_entry_price) * (1.0 - min_dist / 100.0)


def _short_entry_distance_ok(px: float, last_entry_price: float, min_entry_distance_pct: float) -> bool:
    min_dist = float(min_entry_distance_pct or 0.0)
    if min_dist <= 0 or float(last_entry_price or 0.0) <= 0:
        return True
    return px >= float(last_entry_price) * (1.0 + min_dist / 100.0)


def evaluate_signal(
    *,
    prev_bar: SessionDailyBar,
    in_position: bool = False,
    position_side: int | None = None,
    trade_type: str = "long_only",
    entry_threshold: float,
    exit_threshold: float,
    daily_closes: Sequence[float],
    sma_period: int = 0,
    trend_ma_type: str = "sma",
    trend_ma_period: int = 0,
    trend_price: float | None = None,
    ma_closes: Sequence[float] | None = None,
    holding_days: int = 0,
    max_trade_duration_days: int = 0,
    last_entry_price: float = 0.0,
    min_entry_distance_pct: float = 0.0,
) -> IbsAction:
    """上一完整 session 日 K 的 IBS 决定动作。"""
    mode = normalize_trade_type(trade_type)
    side = _resolve_position_side(position_side=position_side, in_position=in_position)
    ibs = prev_bar.ibs
    px = float(trend_price if trend_price is not None else prev_bar.close)
    period = _ma_period(sma_period, trend_ma_period)
    kind = _ma_kind(trend_ma_type, period)

    if side > 0:
        if int(max_trade_duration_days) > 0 and int(holding_days) >= int(max_trade_duration_days):
            return "SELL"
        if ibs > float(exit_threshold):
            return "SELL"
        return "HOLD"

    if side < 0:
        if int(max_trade_duration_days) > 0 and int(holding_days) >= int(max_trade_duration_days):
            return "COVER"
        if ibs < float(entry_threshold):
            return "COVER"
        return "HOLD"

    actions: list[IbsAction] = []
    if mode in ("long_only", "long_short") and ibs < float(entry_threshold):
        if _long_ma_ok(px=px, period=period, kind=kind, daily_closes=daily_closes, ma_closes=ma_closes):
            if _long_entry_distance_ok(px, last_entry_price, min_entry_distance_pct):
                actions.append("BUY")
    if mode in ("short_only", "long_short") and ibs > float(exit_threshold):
        if _short_ma_ok(px=px, period=period, kind=kind, daily_closes=daily_closes, ma_closes=ma_closes):
            if _short_entry_distance_ok(px, last_entry_price, min_entry_distance_pct):
                actions.append("SHORT")
    if "BUY" in actions:
        return "BUY"
    if "SHORT" in actions:
        return "SHORT"
    return "HOLD"


def evaluate_signal_context(
    ctx: IbsSignalContext,
    *,
    in_position: bool = False,
    position_side: int | None = None,
    trade_type: str = "long_only",
    entry_threshold: float,
    exit_threshold: float,
    sma_period: int = 0,
    trend_ma_type: str = "sma",
    trend_ma_period: int = 0,
    holding_days: int = 0,
    max_trade_duration_days: int = 0,
    last_entry_price: float = 0.0,
    min_entry_distance_pct: float = 0.0,
) -> IbsAction:
    return evaluate_signal(
        prev_bar=ctx.prev_bar,
        in_position=in_position,
        position_side=position_side,
        trade_type=trade_type,
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
        daily_closes=ctx.ibs_closes,
        sma_period=sma_period,
        trend_ma_type=trend_ma_type,
        trend_ma_period=trend_ma_period,
        trend_price=ctx.trend_price,
        ma_closes=ctx.ma_closes,
        holding_days=holding_days,
        max_trade_duration_days=max_trade_duration_days,
        last_entry_price=last_entry_price,
        min_entry_distance_pct=min_entry_distance_pct,
    )


def stop_loss_hit(*, side: int, entry_price: float, close: float, stop_loss_pct: float) -> bool:
    if side == 0 or float(stop_loss_pct) <= 0:
        return False
    px = float(entry_price)
    if px <= 0:
        return False
    if side > 0:
        return float(close) <= px * (1.0 - float(stop_loss_pct))
    return float(close) >= px * (1.0 + float(stop_loss_pct))
