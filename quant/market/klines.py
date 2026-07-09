"""K 线 DataFrame 转换（交易所无关）。"""

from __future__ import annotations

from typing import Any, List

import pandas as pd


def klines_to_df(rows: List[List[Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[[0, 1, 2, 3, 4, 5]].copy()
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df["open_time"] = df["open_time"].astype("int64")
    return df


def interval_step_ms(interval: str) -> int:
    return {
        "1m": 60_000,
        "2m": 120_000,
        "3m": 180_000,
        "5m": 300_000,
        "15m": 900_000,
        "1h": 3_600_000,
        "1d": 86_400_000,
    }.get(interval.strip().lower(), 300_000)
