"""KAMA Trend 信号核心（对齐 Jesse kama-trendfollowing）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from quant.common.jesse_indicators import (
    adx_last,
    atr_last,
    bollinger_bands_width_pct,
    chop_last,
    kama_last,
)

BarOhlc = tuple[int, float, float, float, float]


@dataclass(frozen=True)
class KamaSnapshot:
    kama: float
    long_kama: float
    atr: float
    adx: float
    chop: float
    bb_width_pct: float


def compute_snapshot(
    bars: Sequence[BarOhlc],
    *,
    long_tf_closes: Sequence[float] | None = None,
    kama_period: int = 14,
    adx_period: int = 14,
    chop_period: int = 14,
    bb_period: int = 20,
) -> KamaSnapshot | None:
    """15m 信号周期指标 + Jesse get_candles('4h') 长周期 KAMA。"""
    if len(bars) < max(60, bb_period + 5):
        return None
    closes = [float(b[4]) for b in bars]
    highs = [float(b[2]) for b in bars]
    lows = [float(b[3]) for b in bars]
    kama = kama_last(closes, period=kama_period)
    atr = atr_last(highs, lows, closes, adx_period)
    adx = adx_last(highs, lows, closes, adx_period)
    chop = chop_last(highs, lows, closes, chop_period)
    bbw = bollinger_bands_width_pct(closes, bb_period)
    if None in (kama, atr, adx, chop, bbw) or atr <= 0:
        return None
    lt_closes = list(long_tf_closes or [])
    long_kama = kama_last(lt_closes, period=kama_period) if len(lt_closes) >= kama_period + 1 else None
    if long_kama is None:
        return None
    return KamaSnapshot(
        kama=float(kama),
        long_kama=float(long_kama),
        atr=float(atr),
        adx=float(adx),
        chop=float(chop),
        bb_width_pct=float(bbw),
    )


def entry_signal(
    close: float,
    snap: KamaSnapshot,
    *,
    adx_min: float = 50.0,
    chop_max: float = 50.0,
    bb_width_max_pct: float = 7.0,
    bars_since_trade: int,
    cooldown_bars: int = 10,
) -> int:
    if bars_since_trade < cooldown_bars:
        return 0
    if snap.adx <= adx_min or snap.chop >= chop_max or snap.bb_width_pct >= bb_width_max_pct:
        return 0
    px = float(close)
    if px > snap.kama and px > snap.long_kama:
        return 1
    if px < snap.kama and px < snap.long_kama:
        return -1
    return 0


def stop_tp_prices(entry: float, side: int, atr: float, *, stop_atr: float, tp_atr: float) -> tuple[float, float]:
    if side > 0:
        return entry - stop_atr * atr, entry + tp_atr * atr
    return entry + stop_atr * atr, entry - tp_atr * atr


def bar_hits_stop_tp(
    *,
    side: int,
    high: float,
    low: float,
    stop: float,
    tp: float,
    prev_high: float | None = None,
    prev_low: float | None = None,
) -> str | None:
    if side > 0:
        sl_hit = low <= stop if prev_low is None else (low <= stop and prev_low > stop)
        tp_hit = high >= tp if prev_high is None else (high >= tp and prev_high < tp)
    else:
        sl_hit = high >= stop if prev_high is None else (high >= stop and prev_high < stop)
        tp_hit = low <= tp if prev_low is None else (low <= tp and prev_low > tp)
    if sl_hit and tp_hit:
        return "stop"
    if sl_hit:
        return "stop"
    if tp_hit:
        return "tp"
    return None
