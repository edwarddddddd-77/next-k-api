"""ICT Fractal T-Spot + CISD signal core (Pine FRACTAL.pine port)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

BarOhlc = tuple[int, float, float, float, float]  # open_time_ms, o, h, l, c


@dataclass(frozen=True)
class TSpotSetup:
    """Confirmed HTF T-Spot when C2 closes and C3 opens."""

    side: int  # 1 long, -1 short
    c2_level: float
    c2_bar_idx: int
    sweep_mid: float
    close_level: float
    htf_open_ms: int
    pattern: str  # c2_sweep | c4


@dataclass
class ActiveSetup:
    setup: TSpotSetup
    start_ltf_idx: int
    expire_ltf_idx: int
    touched: bool = False
    cisd_triggered: bool = False
    cisd_trigger_level: float = 0.0


def log_midpoint(high: float, low: float, open_: float, close: float) -> float:
    """Pine calculateLogMidpoint."""
    if min(high, low, open_, close) <= 0:
        return (high + low) / 2.0
    log_high = math.log(high)
    log_low = math.log(low)
    log_open = math.log(open_)
    log_close = math.log(close)
    body_size = abs(log_close - log_open)
    upper_wick = log_high - max(log_open, log_close)
    lower_wick = min(log_open, log_close) - log_low
    if max(upper_wick, lower_wick) > body_size:
        if upper_wick > lower_wick:
            log_mid = log_high - upper_wick / 2.0
        else:
            log_mid = log_low + lower_wick / 2.0
    else:
        log_mid = (log_high + log_low) / 2.0
    return math.exp(log_mid)


def _inside_bar(h: float, l: float, c: float, prev_h: float, prev_l: float) -> bool:
    return h > prev_h and l < prev_l and c > prev_l and c < prev_h


def _bias_ok(side: int, bias: str) -> bool:
    b = bias.lower()
    if b == "bullish":
        return side > 0
    if b == "bearish":
        return side < 0
    return True


def _candle_extremes(
    o: float, h: float, l: float, c: float, *, use_body: bool
) -> tuple[float, float]:
    if use_body:
        return max(o, c), min(o, c)
    return h, l


def detect_tspot(
    htf_bars: Sequence[BarOhlc],
    *,
    bias: str = "none",
    use_body: bool = True,
) -> TSpotSetup | None:
    """
    Detect T-Spot on HTF candle roll (C2 closed, C3 opens).
    Uses bars [-4:-1] as prev_prev, prev, last_closed; current bar is forming.
    """
    if len(htf_bars) < 4:
        return None

    _, o0, h0, l0, c0 = htf_bars[-1]
    _, o1, h1, l1, c1 = htf_bars[-2]
    _, o2, h2, l2, c2 = htf_bars[-3]
    _, _o3, _h3, _l3, _c3 = htf_bars[-4]
    htf_open_ms = int(htf_bars[-1][0])

    # Primary bearish C2 sweep
    if (
        h1 > h2
        and c1 < h2
        and _bias_ok(-1, bias)
        and not _inside_bar(h1, l1, c1, h2, l2)
    ):
        sweep_mid = log_midpoint(h1, l1, o1, c1)
        if c1 < sweep_mid:
            return TSpotSetup(
                side=-1,
                c2_level=h1,
                c2_bar_idx=len(htf_bars) - 2,
                sweep_mid=sweep_mid,
                close_level=c1,
                htf_open_ms=htf_open_ms,
                pattern="c2_sweep",
            )

    # Primary bullish C2 sweep
    if (
        l1 < l2
        and c1 > l2
        and _bias_ok(1, bias)
        and not _inside_bar(h1, l1, c1, h2, l2)
    ):
        sweep_mid = log_midpoint(h1, l1, o1, c1)
        if c1 > sweep_mid:
            return TSpotSetup(
                side=1,
                c2_level=l1,
                c2_bar_idx=len(htf_bars) - 2,
                sweep_mid=sweep_mid,
                close_level=c1,
                htf_open_ms=htf_open_ms,
                pattern="c2_sweep",
            )

    # C4 bearish variant
    prev_mid = log_midpoint(h2, l2, o2, c2)
    sweep_mid = log_midpoint(h0, l0, o0, c0)
    c4_bear_ok = (
        h2 > _h3
        and c0 < max(o2, c2)
        and _bias_ok(-1, bias)
        and not _inside_bar(h0, l0, c0, h2, l2)
        and (
            c2 >= prev_mid
            or c2 >= _h3
            or _inside_bar(h2, l2, c2, _h3, _l3)
        )
        and c0 < sweep_mid
    )
    if c4_bear_ok:
        return TSpotSetup(
            side=-1,
            c2_level=h2,
            c2_bar_idx=len(htf_bars) - 3,
            sweep_mid=sweep_mid,
            close_level=c0,
            htf_open_ms=htf_open_ms,
            pattern="c4",
        )

    # C4 bullish variant
    c4_bull_ok = (
        l2 < _l3
        and c0 > min(o2, c2)
        and _bias_ok(1, bias)
        and not _inside_bar(h0, l0, c0, h2, l2)
        and (
            c2 <= prev_mid
            or c2 <= _l3
            or _inside_bar(h2, l2, c2, _h3, _l3)
        )
        and c0 > sweep_mid
    )
    if c4_bull_ok:
        return TSpotSetup(
            side=1,
            c2_level=l2,
            c2_bar_idx=len(htf_bars) - 3,
            sweep_mid=sweep_mid,
            close_level=c0,
            htf_open_ms=htf_open_ms,
            pattern="c4",
        )

    return None


def build_cisd_series(
    ltf_bars: Sequence[BarOhlc],
    c2_ltf_idx: int,
    *,
    is_bullish: bool,
    use_body: bool = True,
    max_lookback: int = 20,
) -> tuple[float, float] | None:
    """Build opposite-color series ending at C2 (Pine detectCISDAndProjections)."""
    if c2_ltf_idx < 0 or c2_ltf_idx >= len(ltf_bars):
        return None

    _, o, h, l, c = ltf_bars[c2_ltf_idx]
    series_high, series_low = _candle_extremes(o, h, l, c, use_body=use_body)

    for i in range(1, max_lookback + 1):
        idx = c2_ltf_idx - i
        if idx < 0:
            break
        _, o_i, h_i, l_i, c_i = ltf_bars[idx]
        bull = c_i > o_i
        if is_bullish:
            if not bull:
                ch, cl = _candle_extremes(o_i, h_i, l_i, c_i, use_body=use_body)
                series_high = max(series_high, ch)
                series_low = min(series_low, cl)
            else:
                break
        else:
            if bull:
                ch, cl = _candle_extremes(o_i, h_i, l_i, c_i, use_body=use_body)
                series_high = max(series_high, ch)
                series_low = min(series_low, cl)
            else:
                break

    return series_high, series_low


def check_cisd_break(
    ltf_bars: Sequence[BarOhlc],
    *,
    from_idx: int,
    to_idx: int,
    series_high: float,
    series_low: float,
    is_bullish: bool,
) -> int | None:
    """Return LTF index where CISD confirms (close breaks series)."""
    for idx in range(from_idx, to_idx + 1):
        if idx < 0 or idx >= len(ltf_bars):
            continue
        _, _o, _h, _l, c = ltf_bars[idx]
        if is_bullish and c > series_high:
            return idx
        if not is_bullish and c < series_low:
            return idx
    return None


def fractal_touch_level(setup: TSpotSetup) -> float:
    top = max(setup.sweep_mid, setup.close_level)
    bottom = min(setup.sweep_mid, setup.close_level)
    return bottom if setup.side > 0 else top


def check_fractal_touch(
    o: float,
    h: float,
    l: float,
    c: float,
    setup: TSpotSetup,
) -> bool:
    """Pine fractal_touched logic on LTF bar."""
    touch = fractal_touch_level(setup)
    if setup.side > 0:
        return (l < touch or o < touch) and c > touch
    return (h > touch or o > touch) and c < touch


def find_pivot(
    ltf_bars: Sequence[BarOhlc],
    upto_idx: int,
    *,
    pivot_high: bool,
    left: int = 2,
    right: int = 2,
    use_body: bool = True,
) -> tuple[float, int] | None:
    """Simple pivot for C3 confirmation."""
    start = left
    end = upto_idx - right
    if end < start:
        return None

    best_val = None
    best_idx = -1
    for i in range(start, end + 1):
        _, o, h, l, c = ltf_bars[i]
        if use_body:
            val = max(o, c) if pivot_high else min(o, c)
        else:
            val = h if pivot_high else l
        ok = True
        for j in range(i - left, i + right + 1):
            if j == i:
                continue
            _, oj, hj, lj, cj = ltf_bars[j]
            if use_body:
                cmp = max(oj, cj) if pivot_high else min(oj, cj)
            else:
                cmp = hj if pivot_high else lj
            if pivot_high and cmp >= val:
                ok = False
                break
            if not pivot_high and cmp <= val:
                ok = False
                break
        if ok:
            if pivot_high:
                if best_val is None or val > best_val:
                    best_val, best_idx = val, i
            else:
                if best_val is None or val < best_val:
                    best_val, best_idx = val, i
    if best_idx < 0:
        return None
    return float(best_val), best_idx


def entry_signal_cisd(
    ltf_bars: Sequence[BarOhlc],
    active: ActiveSetup,
    ltf_idx: int,
    *,
    use_body: bool = True,
    require_touch: bool = False,
) -> int | None:
    """CISD C2 confirmed entry. Returns entry bar index or None."""
    if active.cisd_triggered:
        return None
    if ltf_idx <= active.start_ltf_idx:
        return None

    _, o, h, l, c = ltf_bars[ltf_idx]
    if require_touch and not active.touched:
        if check_fractal_touch(o, h, l, c, active.setup):
            active.touched = True
        return None

    series = build_cisd_series(
        ltf_bars,
        active.setup.c2_bar_idx,
        is_bullish=active.setup.side > 0,
        use_body=use_body,
    )
    if series is None:
        return None
    series_high, series_low = series
    break_idx = check_cisd_break(
        ltf_bars,
        from_idx=active.start_ltf_idx,
        to_idx=ltf_idx,
        series_high=series_high,
        series_low=series_low,
        is_bullish=active.setup.side > 0,
    )
    if break_idx is None:
        return None
    active.cisd_triggered = True
    active.cisd_trigger_level = series_high if active.setup.side > 0 else series_low
    return break_idx


def entry_signal_c3(
    ltf_bars: Sequence[BarOhlc],
    active: ActiveSetup,
    ltf_idx: int,
    *,
    use_body: bool = True,
) -> int | None:
    """C3 touch + pivot break (continuation OB style)."""
    if ltf_idx <= active.start_ltf_idx:
        return None

    _, o, h, l, c = ltf_bars[ltf_idx]
    if not active.touched:
        if check_fractal_touch(o, h, l, c, active.setup):
            active.touched = True
        return None

    touch_idx = ltf_idx
    pivot = find_pivot(ltf_bars, ltf_idx, pivot_high=active.setup.side > 0, use_body=use_body)
    if pivot is None:
        return None
    pivot_level, pivot_idx = pivot
    if pivot_idx >= touch_idx:
        return None

    touch_level = fractal_touch_level(active.setup)
    if active.setup.side > 0:
        if c > pivot_level and o < pivot_level and pivot_level > touch_level:
            return ltf_idx
    else:
        if c < pivot_level and o > pivot_level and pivot_level < touch_level:
            return ltf_idx
    return None


def stop_tp_from_setup(
    setup: TSpotSetup,
    entry_price: float,
    *,
    rr_ratio: float = 2.0,
    trigger_level: float | None = None,
) -> tuple[float, float]:
    """Stop at C2 extreme; TP = RR * risk from trigger level."""
    stop = setup.c2_level
    trig = trigger_level if trigger_level is not None else entry_price
    risk = abs(trig - stop)
    if risk <= 0:
        risk = abs(entry_price - stop)
    if setup.side > 0:
        return stop, entry_price + rr_ratio * risk
    return stop, entry_price - rr_ratio * risk


def bar_hits_stop_tp(
    *,
    side: int,
    high: float,
    low: float,
    stop: float,
    tp: float,
) -> str | None:
    if side > 0:
        sl_hit = low <= stop
        tp_hit = high >= tp
    else:
        sl_hit = high >= stop
        tp_hit = low <= tp
    if sl_hit and tp_hit:
        return "stop"
    if sl_hit:
        return "stop"
    if tp_hit:
        return "tp"
    return None


def position_qty(
    entry: float,
    stop: float,
    *,
    equity: float,
    risk_pct: float,
) -> float:
    risk_usd = equity * risk_pct / 100.0
    per_unit = abs(entry - stop)
    if per_unit <= 0:
        return 0.0
    return risk_usd / per_unit
