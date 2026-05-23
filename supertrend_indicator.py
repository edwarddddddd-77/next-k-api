"""Supertrend（对齐 TradingView Pine v4：ATR Wilder / hl2 / 递推 up-dn / trend 状态机）。"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd

SourceKind = Literal["hl2", "close", "hlc3", "ohlc4"]


def _series_source(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
    kind: str,
) -> np.ndarray:
    k = (kind or "hl2").strip().lower()
    if k == "close":
        return close
    if k == "hlc3":
        return (high + low + close) / 3.0
    if k == "ohlc4":
        return (open_ + high + low + close) / 4.0
    return (high + low) / 2.0


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(close)
    tr = np.empty(n, dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    return tr


def _atr_wilder(tr: np.ndarray, period: int) -> np.ndarray:
    n = len(tr)
    out = np.full(n, np.nan, dtype=float)
    if n < period:
        return out
    out[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def _atr_sma(tr: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(tr).rolling(period, min_periods=period).mean().to_numpy()


def compute_supertrend(
    df: pd.DataFrame,
    *,
    period: int = 10,
    multiplier: float = 3.0,
    source: str = "hl2",
    atr_method: str = "wilder",
) -> pd.DataFrame:
    """
    输入列：open, high, low, close（可选 open）。
    输出追加：st_up, st_dn, st_trend, buy_signal, sell_signal。
    """
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    for c in ("open", "high", "low", "close"):
        if c not in work.columns:
            raise ValueError(f"missing column {c}")
        work[c] = work[c].astype(float)
    if "open" not in work.columns:
        work["open"] = work["close"]

    high = work["high"].to_numpy()
    low = work["low"].to_numpy()
    close = work["close"].to_numpy()
    open_ = work["open"].to_numpy()
    src = _series_source(high, low, close, open_, source)

    tr = _true_range(high, low, close)
    if (atr_method or "wilder").strip().lower() == "sma":
        atr = _atr_sma(tr, period)
    else:
        atr = _atr_wilder(tr, period)

    n = len(work)
    up = np.zeros(n, dtype=float)
    dn = np.zeros(n, dtype=float)
    trend = np.ones(n, dtype=int)

    for i in range(n):
        if np.isnan(atr[i]):
            up[i] = src[i] - multiplier * tr[i]
            dn[i] = src[i] + multiplier * tr[i]
            trend[i] = 1 if i == 0 else trend[i - 1]
            continue

        up_raw = src[i] - multiplier * atr[i]
        dn_raw = src[i] + multiplier * atr[i]
        up1 = up[i - 1] if i > 0 else up_raw
        dn1 = dn[i - 1] if i > 0 else dn_raw
        prev_close = close[i - 1] if i > 0 else close[i]

        if i > 0 and prev_close > up1:
            up[i] = max(up_raw, up1)
        else:
            up[i] = up_raw

        if i > 0 and prev_close < dn1:
            dn[i] = min(dn_raw, dn1)
        else:
            dn[i] = dn_raw

        prev_trend = trend[i - 1] if i > 0 else 1
        if prev_trend == -1 and close[i] > dn1:
            trend[i] = 1
        elif prev_trend == 1 and close[i] < up1:
            trend[i] = -1
        else:
            trend[i] = prev_trend

    prev_trend = np.roll(trend, 1)
    prev_trend[0] = trend[0]
    buy = (trend == 1) & (prev_trend == -1)
    sell = (trend == -1) & (prev_trend == 1)

    work["st_up"] = up
    work["st_dn"] = dn
    work["st_trend"] = trend
    work["st_atr"] = atr
    work["buy_signal"] = buy
    work["sell_signal"] = sell
    return work


def last_closed_bar_signals(
    df: pd.DataFrame,
    *,
    timeframe_ms: int,
    now_ms: Optional[int] = None,
) -> Tuple[Optional[pd.Series], pd.DataFrame]:
    """
    去掉未收盘的最后一根，返回 (最后一根已收盘 bar, 全量指标 df)。
  """
    if df is None or df.empty:
        return None, df
    work = df.copy()
    if "open_time" not in work.columns:
        return None, work
    now_ms = int(now_ms if now_ms is not None else __import__("time").time() * 1000)
    close_times = work["open_time"].astype(np.int64) + int(timeframe_ms) - 1
    closed = work[close_times <= now_ms]
    if closed.empty:
        return None, work
    return closed.iloc[-1], work
