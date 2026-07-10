"""Smart Breakout Targets 信号核心（对齐 WillyAlgoTrader v1.5）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

BarOhlc = tuple[int, float, float, float, float]
BarOhlcv = tuple[int, float, float, float, float, float]

# WillyAlgoTrader Smart Breakout — 成交量 SMA 周期（Pine 默认 20）
VOLUME_SMA_PERIOD = 20


@dataclass(frozen=True)
class BreakoutSignal:
    side: int
    entry: float
    stop: float
    tp1: float
    tp2: float
    tp3: float
    strength: int
    range_top: float
    range_bottom: float
    atr: float
    squeeze_bars: int


@dataclass
class RangeBox:
    top: float
    bottom: float
    squeeze_bars: int
    start_ts: int
    end_ts: int


@dataclass
class BreakoutEngineState:
    ranges: list[RangeBox] = field(default_factory=list)
    squeeze_active: bool = False
    squeeze_high: float = 0.0
    squeeze_low: float = 0.0
    squeeze_count: int = 0
    prev_squeeze: bool = False


def _sma(arr: np.ndarray, period: int) -> float | None:
    if len(arr) < period:
        return None
    return float(np.mean(arr[-period:]))


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> float | None:
    series = _atr_series(high, low, close, period)
    if series.size == 0:
        return None
    return float(series[-1])


def _atr_series(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    if len(close) < period + 1:
        return np.array([])
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    if len(tr) < period:
        return np.array([])
    out = np.zeros(len(tr), dtype=float)
    out[period - 1] = float(np.mean(tr[:period]))
    for i in range(period, len(tr)):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def _atr_sma_last(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    *,
    atr_period: int,
    sma_period: int,
) -> float | None:
    h = np.asarray(highs, dtype=float)
    l = np.asarray(lows, dtype=float)
    c = np.asarray(closes, dtype=float)
    series = _atr_series(h, l, c, atr_period)
    if len(series) < sma_period:
        return None
    return float(np.mean(series[-sma_period:]))


def bb_width_last(closes: Sequence[float], *, length: int, mult: float) -> float | None:
    arr = np.asarray(closes, dtype=float)
    if len(arr) < length:
        return None
    window = arr[-length:]
    basis = float(np.mean(window))
    if basis <= 0:
        return None
    std = float(np.std(window, ddof=0))
    upper = basis + mult * std
    lower = basis - mult * std
    return (upper - lower) / basis


def bb_width_sma(closes: Sequence[float], *, length: int, mult: float) -> float | None:
    arr = np.asarray(closes, dtype=float)
    if len(arr) < length * 2 - 1:
        return None
    widths: list[float] = []
    for i in range(length - 1, len(arr)):
        w = bb_width_last(arr[: i + 1], length=length, mult=mult)
        if w is not None:
            widths.append(w)
    if len(widths) < length:
        return None
    return float(np.mean(widths[-length:]))


def is_squeeze_bar(
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    *,
    squeeze_length: int,
    bb_mult: float,
    squeeze_threshold: float,
    atr_compress_ratio: float,
) -> tuple[bool, float | None]:
    """双引擎压缩：BB Width + ATR 同时低于各自 SMA 阈值。"""
    min_bars = max(squeeze_length * 2, squeeze_length + 5)
    if len(closes) < min_bars:
        return False, None
    bbw = bb_width_last(closes, length=squeeze_length, mult=bb_mult)
    bbw_sma = bb_width_sma(closes, length=squeeze_length, mult=bb_mult)
    atr_period = max(2, squeeze_length // 2)
    h = np.asarray(highs, dtype=float)
    l = np.asarray(lows, dtype=float)
    c = np.asarray(closes, dtype=float)
    atr_val = _atr(h, l, c, atr_period)
    atr_sma = _atr_sma_last(highs, lows, closes, atr_period=atr_period, sma_period=squeeze_length)
    if None in (bbw, bbw_sma, atr_val, atr_sma) or bbw_sma <= 0 or atr_sma <= 0:
        return False, atr_val
    bb_ok = bbw < bbw_sma * squeeze_threshold
    atr_ok = atr_val < atr_sma * atr_compress_ratio
    return bool(bb_ok and atr_ok), atr_val


def _clamp_range(top: float, bottom: float, atr: float) -> tuple[float, float]:
    height = top - bottom
    if atr <= 0 or height <= 6.0 * atr:
        return top, bottom
    center = (top + bottom) / 2.0
    return center + 3.0 * atr, center - 3.0 * atr


def _ranges_overlap(a: RangeBox, bottom: float, top: float) -> bool:
    return not (top < a.bottom or bottom > a.top)


def _signal_strength(
    *,
    body: float,
    atr: float,
    impulse_mult: float,
    squeeze_bars: int,
    min_squeeze_bars: int,
    volume: float,
    vol_sma: float | None,
    volume_filter: bool,
    volume_mult: float,
) -> int:
    score = 0
    if body >= atr * impulse_mult * 1.5:
        score += 1
    if vol_sma is not None and vol_sma > 0 and volume > vol_sma * volume_mult:
        score += 1
    if squeeze_bars >= 2 * min_squeeze_bars:
        score += 1
    if body >= atr * impulse_mult:
        score += 1
    return min(4, score)


def breakout_levels(
    entry: float,
    side: int,
    range_top: float,
    range_bottom: float,
    atr: float,
    *,
    sl_atr_buffer: float,
    tp1_rr: float,
    tp2_rr: float,
    tp3_rr: float,
) -> tuple[float, float, float, float, float]:
    if side > 0:
        stop = range_bottom - atr * sl_atr_buffer
        risk = abs(entry - stop)
        return stop, entry + risk * tp1_rr, entry + risk * tp2_rr, entry + risk * tp3_rr, risk
    stop = range_top + atr * sl_atr_buffer
    risk = abs(entry - stop)
    return stop, entry - risk * tp1_rr, entry - risk * tp2_rr, entry - risk * tp3_rr, risk


def _qualified_breakout(
    *,
    close: float,
    open_: float,
    range_top: float,
    range_bottom: float,
    atr: float,
    impulse_mult: float,
) -> int:
    body = abs(close - open_)
    if atr <= 0 or body < atr * impulse_mult:
        return 0
    if close > range_top and close > open_:
        return 1
    if close < range_bottom and close < open_:
        return -1
    return 0


def scan_range_breakouts(
    *,
    close: float,
    open_: float,
    ranges: list[RangeBox],
    atr: float,
    impulse_mult: float,
    squeeze_bars_for_strength: int,
    min_squeeze_bars: int,
    sl_atr_buffer: float,
    tp1_rr: float,
    tp2_rr: float,
    tp3_rr: float,
    volume: float = 0.0,
    vol_sma: float | None = None,
    volume_filter: bool = False,
    volume_mult: float = 1.5,
) -> tuple[list[RangeBox], BreakoutSignal | None]:
    """扫描现有区间；过期刺破（无冲动 K）会静默消耗区间。"""
    kept: list[RangeBox] = []
    signal: BreakoutSignal | None = None
    for rng in ranges:
        beyond = close > rng.top or close < rng.bottom
        if not beyond:
            kept.append(rng)
            continue
        side = _qualified_breakout(
            close=close,
            open_=open_,
            range_top=rng.top,
            range_bottom=rng.bottom,
            atr=atr,
            impulse_mult=impulse_mult,
        )
        if side == 0 or signal is not None:
            continue
        if volume_filter and (vol_sma is None or vol_sma <= 0 or volume <= vol_sma * volume_mult):
            continue
        stop, tp1, tp2, tp3, _ = breakout_levels(
            close,
            side,
            rng.top,
            rng.bottom,
            atr,
            sl_atr_buffer=sl_atr_buffer,
            tp1_rr=tp1_rr,
            tp2_rr=tp2_rr,
            tp3_rr=tp3_rr,
        )
        strength = _signal_strength(
            body=abs(close - open_),
            atr=atr,
            impulse_mult=impulse_mult,
            squeeze_bars=rng.squeeze_bars,
            min_squeeze_bars=min_squeeze_bars,
            volume=volume,
            vol_sma=vol_sma,
            volume_filter=volume_filter,
            volume_mult=volume_mult,
        )
        signal = BreakoutSignal(
            side=side,
            entry=close,
            stop=stop,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            strength=strength,
            range_top=rng.top,
            range_bottom=rng.bottom,
            atr=atr,
            squeeze_bars=rng.squeeze_bars,
        )
    if signal is not None:
        return kept, signal
    return kept, None


def update_squeeze_state(
    state: BreakoutEngineState,
    *,
    ts: int,
    high: float,
    low: float,
    is_squeeze: bool,
    atr: float,
    min_squeeze_bars: int,
    prevent_overlap: bool,
) -> BreakoutEngineState:
    """更新压缩计数；压缩结束时创建区间 box。"""
    if is_squeeze and not state.prev_squeeze:
        state.squeeze_active = True
        state.squeeze_high = float(high)
        state.squeeze_low = float(low)
        state.squeeze_count = 1
    elif is_squeeze and state.prev_squeeze:
        state.squeeze_high = max(state.squeeze_high, float(high))
        state.squeeze_low = min(state.squeeze_low, float(low))
        state.squeeze_count += 1
    elif not is_squeeze and state.prev_squeeze:
        if state.squeeze_count >= min_squeeze_bars and atr is not None and atr > 0:
            top, bottom = _clamp_range(state.squeeze_high, state.squeeze_low, atr)
            new_box = RangeBox(
                top=top,
                bottom=bottom,
                squeeze_bars=state.squeeze_count,
                start_ts=ts,
                end_ts=ts,
            )
            overlap = prevent_overlap and any(
                _ranges_overlap(r, new_box.bottom, new_box.top) for r in state.ranges
            )
            if not overlap:
                state.ranges.append(new_box)
        state.squeeze_active = False
        state.squeeze_count = 0
    state.prev_squeeze = is_squeeze
    return state


def process_signal_bar(
    bars: Sequence[BarOhlcv],
    state: BreakoutEngineState,
    *,
    squeeze_length: int = 20,
    bb_mult: float = 2.0,
    squeeze_threshold: float = 0.6,
    atr_compress_ratio: float = 0.75,
    min_squeeze_bars: int = 5,
    impulse_mult: float = 0.8,
    sl_atr_buffer: float = 0.5,
    tp1_rr: float = 1.0,
    tp2_rr: float = 2.0,
    tp3_rr: float = 3.0,
    prevent_overlap: bool = True,
    volume_filter: bool = False,
    volume_mult: float = 1.5,
) -> tuple[BreakoutEngineState, BreakoutSignal | None]:
    """处理一根已确认信号 K：先扫突破，再更新压缩状态。"""
    if len(bars) < max(50, squeeze_length * 2):
        return state, None
    ts, o, h, l, c, vol = bars[-1]
    closes = [float(b[4]) for b in bars]
    highs = [float(b[2]) for b in bars]
    lows = [float(b[3]) for b in bars]
    volumes = [float(b[5]) for b in bars]
    squeezed, atr = is_squeeze_bar(
        closes,
        highs,
        lows,
        squeeze_length=squeeze_length,
        bb_mult=bb_mult,
        squeeze_threshold=squeeze_threshold,
        atr_compress_ratio=atr_compress_ratio,
    )
    atr_f = float(atr or 0.0)
    vol_sma = _sma(np.asarray(volumes, dtype=float), VOLUME_SMA_PERIOD) if volumes else None
    kept, signal = scan_range_breakouts(
        close=float(c),
        open_=float(o),
        ranges=state.ranges,
        atr=max(atr_f, 1e-9),
        impulse_mult=impulse_mult,
        squeeze_bars_for_strength=min_squeeze_bars,
        min_squeeze_bars=min_squeeze_bars,
        sl_atr_buffer=sl_atr_buffer,
        tp1_rr=tp1_rr,
        tp2_rr=tp2_rr,
        tp3_rr=tp3_rr,
        volume=float(vol),
        vol_sma=vol_sma,
        volume_filter=volume_filter,
        volume_mult=volume_mult,
    )
    state.ranges = kept
    update_squeeze_state(
        state,
        ts=int(ts),
        high=float(h),
        low=float(l),
        is_squeeze=squeezed,
        atr=atr_f,
        min_squeeze_bars=min_squeeze_bars,
        prevent_overlap=prevent_overlap,
    )
    return state, signal


def replay_engine_state(
    bars: Sequence[BarOhlcv],
    *,
    squeeze_length: int = 20,
    bb_mult: float = 2.0,
    squeeze_threshold: float = 0.6,
    atr_compress_ratio: float = 0.75,
    min_squeeze_bars: int = 5,
    impulse_mult: float = 0.8,
    sl_atr_buffer: float = 0.5,
    tp1_rr: float = 1.0,
    tp2_rr: float = 2.0,
    tp3_rr: float = 3.0,
    prevent_overlap: bool = True,
    volume_filter: bool = False,
    volume_mult: float = 1.5,
    warmup: int | None = None,
) -> BreakoutEngineState:
    """从历史 K 线重放引擎状态（trim 后重建区间）。"""
    state = BreakoutEngineState()
    start = int(warmup if warmup is not None else max(50, squeeze_length * 2))
    for i in range(start, len(bars)):
        state, _ = process_signal_bar(
            bars[: i + 1],
            state,
            squeeze_length=squeeze_length,
            bb_mult=bb_mult,
            squeeze_threshold=squeeze_threshold,
            atr_compress_ratio=atr_compress_ratio,
            min_squeeze_bars=min_squeeze_bars,
            impulse_mult=impulse_mult,
            sl_atr_buffer=sl_atr_buffer,
            tp1_rr=tp1_rr,
            tp2_rr=tp2_rr,
            tp3_rr=tp3_rr,
            prevent_overlap=prevent_overlap,
            volume_filter=volume_filter,
            volume_mult=volume_mult,
        )
    return state


def bar_exit_reason(
    *,
    side: int,
    high: float,
    low: float,
    stop: float,
    tp1: float,
    tp2: float,
    tp3: float,
    prev_high: float,
    prev_low: float,
    tp1_hit: bool = False,
    tp2_hit: bool = False,
) -> str | None:
    """TP3 / SL 首次触碰；同 bar 双触 → SL（v1.5）。TP1/TP2 仅里程碑，不平仓。"""
    if side > 0:
        sl_hit = low <= stop and prev_low > stop
        tp_hit = high >= tp3 and prev_high < tp3
    else:
        sl_hit = high >= stop and prev_high < stop
        tp_hit = low <= tp3 and prev_low > tp3
    if sl_hit and tp_hit:
        return "sl"
    if sl_hit:
        return "sl"
    if tp_hit:
        return "tp3"
    return None


def bar_tp_milestones(
    *,
    side: int,
    high: float,
    low: float,
    tp1: float,
    tp2: float,
    prev_high: float,
    prev_low: float,
    tp1_hit: bool,
    tp2_hit: bool,
) -> list[str]:
    """TV v1.5：TP1/TP2 命中仅标记，不触发平仓。"""
    hits: list[str] = []
    if side > 0:
        if not tp1_hit and high >= tp1 and prev_high < tp1:
            hits.append("tp1")
        if not tp2_hit and high >= tp2 and prev_high < tp2:
            hits.append("tp2")
    else:
        if not tp1_hit and low <= tp1 and prev_low > tp1:
            hits.append("tp1")
        if not tp2_hit and low <= tp2 and prev_low > tp2:
            hits.append("tp2")
    return hits
