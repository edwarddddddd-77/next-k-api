"""Kline resample helpers (pandas 2.x safe)."""

from __future__ import annotations

import pandas as pd


def normalize_resample_rule(rule: str) -> str:
    r = rule.strip().lower()
    if r.endswith("min"):
        return r
    if r.endswith("m") and r[:-1].isdigit():
        return f"{r[:-1]}min"
    return r


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV; expects open_time in ms."""
    if df.empty:
        return df
    freq = normalize_resample_rule(rule)
    ts = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    out = df.assign(_ts=ts).set_index("_ts")
    agg = (
        out.resample(freq)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )
    agg["open_time"] = [int(ts.value // 1_000_000) for ts in agg.index]
    return agg.reset_index(drop=True)
