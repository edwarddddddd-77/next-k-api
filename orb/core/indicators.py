"""ORB 指标：14 日 ATR（论文止损基准）。"""

from __future__ import annotations

from typing import Optional

import pandas as pd


def compute_atr_series(df: pd.DataFrame, *, period: int = 14) -> pd.Series:
    """Wilder 平滑 ATR。"""
    if df.empty or period < 1:
        return pd.Series(dtype=float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / float(period), adjust=False).mean()


def daily_atr_asof(
    daily_df: pd.DataFrame,
    asof_open_ms: int,
    *,
    period: int = 14,
    tz: str = "America/New_York",
) -> Optional[float]:
    """截至 asof 前一日收盘的 ATR（不含当日未完成 K 线）。"""
    if daily_df.empty:
        return None
    df = daily_df.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time")
    tz_name = tz or "UTC"
    asof_day = pd.Timestamp(int(asof_open_ms), unit="ms", tz=tz_name).normalize()
    day_ts = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_convert(tz_name).dt.normalize()
    completed = df[day_ts < asof_day]
    if len(completed) < period + 1:
        return None
    atr_s = compute_atr_series(completed, period=period)
    val = float(atr_s.iloc[-1])
    return val if val > 0 else None
