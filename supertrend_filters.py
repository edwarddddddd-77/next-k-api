"""Supertrend 开仓过滤（横盘减磨损）：ADX / HTF / 箱体 / 确认 K / 翻转冷却等。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

import supertrend_config as cfg
from supertrend_indicator import compute_supertrend


@dataclass(frozen=True)
class EntryFilterContext:
    symbol: str
    side: str  # LONG | SHORT
    st_df: pd.DataFrame
    close_px: float
    st_atr: float
    st_up: float
    st_dn: float
    trend: int
    bar_open_ms: int
    htf_trend: Optional[int] = None
    adx: Optional[float] = None
    range_pct: Optional[float] = None
    atr_pct: Optional[float] = None
    flip_count: int = 0


def closed_bars_df(st_df: pd.DataFrame, *, timeframe_ms: int, now_ms: Optional[int] = None) -> pd.DataFrame:
    if st_df is None or st_df.empty or "open_time" not in st_df.columns:
        return pd.DataFrame()
    now_ms = int(now_ms if now_ms is not None else time.time() * 1000)
    close_times = st_df["open_time"].astype(np.int64) + int(timeframe_ms) - 1
    return st_df.loc[close_times <= now_ms].copy()


def compute_adx_series(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder ADX；与 st_df 同索引。"""
    if df is None or len(df) < period + 2:
        return pd.Series(dtype=float)
    high = df["high"].astype(float).to_numpy()
    low = df["low"].astype(float).to_numpy()
    close = df["close"].astype(float).to_numpy()
    n = len(close)
    up_move = np.zeros(n)
    dn_move = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        up_move[i] = up if up > dn and up > 0 else 0.0
        dn_move[i] = dn if dn > up and dn > 0 else 0.0
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]

    def wilder_smooth(x: np.ndarray, p: int) -> np.ndarray:
        out = np.full(n, np.nan)
        if n < p:
            return out
        out[p - 1] = np.sum(x[1:p])
        for i in range(p, n):
            out[i] = out[i - 1] - (out[i - 1] / p) + x[i]
        return out

    atr = wilder_smooth(tr, period)
    plus_dm = wilder_smooth(up_move, period)
    minus_dm = wilder_smooth(dn_move, period)
    plus_di = np.where(atr > 0, 100.0 * plus_dm / atr, 0.0)
    minus_di = np.where(atr > 0, 100.0 * minus_dm / atr, 0.0)
    denom = plus_di + minus_di
    with np.errstate(divide="ignore", invalid="ignore"):
        dx = np.where(denom > 0, 100.0 * np.abs(plus_di - minus_di) / denom, 0.0)
    dx = np.nan_to_num(dx, nan=0.0)
    adx = np.full(n, np.nan)
    start = period * 2 - 1
    if n <= start:
        return pd.Series(adx, index=df.index)
    adx[start] = np.nanmean(dx[period : start + 1])
    for i in range(start + 1, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return pd.Series(adx, index=df.index)


def last_adx(closed: pd.DataFrame, period: int) -> Optional[float]:
    s = compute_adx_series(closed, period)
    if s.empty:
        return None
    v = s.iloc[-1]
    return None if pd.isna(v) else float(v)


def htf_trend_for_symbol(
    symbol: str,
    *,
    fetch_klines_fn,
    klines_to_df_fn,
    timeframe_ms: int,
) -> Optional[int]:
    tf = (cfg.ST_HTF_TIMEFRAME or "").strip()
    if not tf:
        return None
    rows = fetch_klines_fn(symbol, tf, cfg.ST_KLINE_LIMIT)
    if len(rows) < cfg.ST_ATR_PERIOD + 5:
        return None
    df = klines_to_df_fn(rows)
    st = compute_supertrend(
        df,
        period=cfg.ST_ATR_PERIOD,
        multiplier=cfg.ST_ATR_MULTIPLIER,
        source=cfg.ST_SOURCE,
        atr_method=cfg.ST_ATR_METHOD,
    )
    closed = closed_bars_df(st, timeframe_ms=timeframe_ms)
    if closed.empty:
        return None
    return int(closed.iloc[-1]["st_trend"])


def range_pct(closed: pd.DataFrame, lookback: int) -> Optional[float]:
    if closed.empty or lookback <= 0:
        return None
    tail = closed.tail(lookback)
    if tail.empty:
        return None
    hi = float(tail["high"].max())
    lo = float(tail["low"].min())
    mid = float(tail["close"].iloc[-1])
    if mid <= 0:
        return None
    return (hi - lo) / mid


def bars_since_last_signal(closed: pd.DataFrame, column: str) -> int:
    """距最近一根信号 K 的偏移：0=当根，1=上一根，无信号=999。"""
    if closed.empty or column not in closed.columns:
        return 999
    flags = closed[column].astype(bool)
    for i in range(len(closed) - 1, -1, -1):
        if bool(flags.iloc[i]):
            return len(closed) - 1 - i
    return 999


def compute_entry_intent(
    *,
    trend: int,
    buy: bool,
    sell: bool,
    closed: pd.DataFrame,
    open_row: Optional[Any],
) -> Tuple[bool, bool]:
    """
    入场意图：flip 当根 + 翻转后窗口内同向（解决「确认 K 需 2 根但仅 flip 根下单」）。
    """
    window = cfg.ST_ENTRY_WINDOW_BARS
    has_long = open_row is not None and str(open_row["side"]) == "LONG"
    has_short = open_row is not None and str(open_row["side"]) == "SHORT"

    since_buy = bars_since_last_signal(closed, "buy_signal")
    since_sell = bars_since_last_signal(closed, "sell_signal")

    def in_window(since: int) -> bool:
        if window <= 0:
            return since == 0
        return since <= window

    want_long = (
        not sell
        and not has_long
        and trend == 1
        and (buy or in_window(since_buy))
    )
    want_short = (
        not buy
        and not has_short
        and trend == -1
        and (sell or in_window(since_sell))
    )
    return want_long, want_short


def flip_signal_count(closed: pd.DataFrame, lookback: int) -> int:
    if closed.empty or lookback <= 0:
        return 0
    tail = closed.tail(lookback)
    buys = tail.get("buy_signal", pd.Series(False, index=tail.index)).astype(bool)
    sells = tail.get("sell_signal", pd.Series(False, index=tail.index)).astype(bool)
    return int((buys | sells).sum())


def entry_confirm_ok(closed: pd.DataFrame, side: str, bars: int) -> bool:
    if bars <= 0 or closed.empty:
        return True
    tail = closed.tail(bars)
    if len(tail) < bars:
        return False
    if side == "LONG":
        return bool((tail["st_trend"] == 1).all() and (tail["close"] > tail["st_up"]).all())
    if side == "SHORT":
        return bool((tail["st_trend"] == -1).all() and (tail["close"] < tail["st_dn"]).all())
    return False


def min_dist_ok(side: str, close_px: float, st_up: float, st_dn: float, st_atr: float) -> bool:
    mult = cfg.ST_MIN_DIST_ATR
    if mult <= 0 or st_atr <= 0:
        return True
    need = mult * st_atr
    if side == "LONG":
        return close_px - st_up >= need
    if side == "SHORT":
        return st_dn - close_px >= need
    return True


def build_filter_context(
    symbol: str,
    side: str,
    st_df: pd.DataFrame,
    last_bar: pd.Series,
    *,
    timeframe_ms: int,
    htf_trend: Optional[int] = None,
) -> EntryFilterContext:
    closed = closed_bars_df(st_df, timeframe_ms=timeframe_ms)
    close_px = float(last_bar["close"])
    st_atr = float(last_bar["st_atr"]) if not pd.isna(last_bar.get("st_atr")) else 0.0
    adx = last_adx(closed, cfg.ST_ADX_PERIOD) if cfg.ST_ADX_MIN > 0 else None
    rp = range_pct(closed, cfg.ST_RANGE_LOOKBACK) if cfg.ST_MAX_RANGE_PCT > 0 else None
    atr_pct = (st_atr / close_px) if close_px > 0 and st_atr > 0 else None
    flips = (
        flip_signal_count(closed, cfg.ST_CHOP_LOOKBACK)
        if cfg.ST_FILTER_ENABLED and cfg.ST_CHOP_MAX_FLIPS > 0
        else 0
    )
    return EntryFilterContext(
        symbol=symbol,
        side=side,
        st_df=st_df,
        close_px=close_px,
        st_atr=st_atr,
        st_up=float(last_bar["st_up"]),
        st_dn=float(last_bar["st_dn"]),
        trend=int(last_bar["st_trend"]),
        bar_open_ms=int(last_bar["open_time"]),
        htf_trend=htf_trend,
        adx=adx,
        range_pct=rp,
        atr_pct=atr_pct,
        flip_count=flips,
    )


def evaluate_entry_filters(ctx: EntryFilterContext) -> Tuple[bool, str]:
    """返回 (允许开仓, 拒绝原因 code)。"""
    if not cfg.ST_FILTER_ENABLED:
        return True, ""

    side = ctx.side
    if cfg.ST_ADX_MIN > 0:
        if ctx.adx is None:
            return False, "adx_unavailable"
        if ctx.adx < cfg.ST_ADX_MIN:
            return False, f"adx_low:{ctx.adx:.1f}<{cfg.ST_ADX_MIN}"

    if cfg.ST_HTF_REQUIRE_ALIGN and (cfg.ST_HTF_TIMEFRAME or "").strip():
        if ctx.htf_trend is None:
            return False, "htf_unavailable"
        if side == "LONG" and ctx.htf_trend != 1:
            return False, "htf_not_bull"
        if side == "SHORT" and ctx.htf_trend != -1:
            return False, "htf_not_bear"

    if cfg.ST_MIN_ATR_PCT > 0:
        if ctx.atr_pct is None or ctx.atr_pct < cfg.ST_MIN_ATR_PCT:
            return False, "atr_pct_low"

    if cfg.ST_MAX_RANGE_PCT > 0:
        if ctx.range_pct is None:
            return False, "range_unavailable"
        if ctx.range_pct < cfg.ST_MAX_RANGE_PCT:
            return False, f"range_chop:{ctx.range_pct:.4f}"

    if cfg.ST_ENTRY_CONFIRM_BARS > 0:
        closed = closed_bars_df(ctx.st_df, timeframe_ms=cfg.st_timeframe_ms(cfg.ST_TIMEFRAME))
        if not entry_confirm_ok(closed, side, cfg.ST_ENTRY_CONFIRM_BARS):
            return False, "confirm_bars"

    if not min_dist_ok(side, ctx.close_px, ctx.st_up, ctx.st_dn, ctx.st_atr):
        return False, "min_dist"

    return True, ""


def chop_cooldown_until_bar(ctx: EntryFilterContext, timeframe_ms: int) -> Optional[int]:
    """翻转过密时，返回禁止新开仓直到的 bar_open_ms（含）。"""
    if not cfg.ST_FILTER_ENABLED:
        return None
    if cfg.ST_CHOP_MAX_FLIPS <= 0 or cfg.ST_CHOP_COOLDOWN_BARS <= 0:
        return None
    if ctx.flip_count < cfg.ST_CHOP_MAX_FLIPS:
        return None
    return ctx.bar_open_ms + cfg.ST_CHOP_COOLDOWN_BARS * timeframe_ms


def record_filter_reject(stats: Dict[str, Any], reason: str) -> None:
    key = (reason or "unknown").split(":")[0]
    rejects = stats.setdefault("filter_rejects", {})
    if isinstance(rejects, dict):
        rejects[key] = int(rejects.get(key, 0)) + 1
