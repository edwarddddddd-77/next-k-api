"""Supertrend 与 Pine 递推一致性的参考序列（固定 OHLC，非 TV 导出）。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from supertrend_indicator import compute_supertrend


def _build_reference_df() -> pd.DataFrame:
    """单调上行后回落，period=3 mult=1 便于手算核对 trend 翻转。"""
    n = 24
    close = np.array(
        [
            10.0,
            10.2,
            10.5,
            10.8,
            11.0,
            11.3,
            11.6,
            11.4,
            11.0,
            10.6,
            10.2,
            9.8,
            9.5,
            9.3,
            9.6,
            10.0,
            10.4,
            10.7,
            10.5,
            10.2,
            9.9,
            9.6,
            9.4,
            9.2,
        ],
        dtype=float,
    )
    high = close + 0.15
    low = close - 0.15
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    return pd.DataFrame(
        {
            "open_time": np.arange(n) * 300_000,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )


def test_reference_flip_matches_trend_change():
    df = _build_reference_df()
    out = compute_supertrend(
        df, period=3, multiplier=1.0, source="close", atr_method="wilder"
    )
    trend = out["st_trend"].to_numpy()
    for i in range(1, len(trend)):
        if trend[i] == 1 and trend[i - 1] == -1:
            assert bool(out["buy_signal"].iloc[i])
        if trend[i] == -1 and trend[i - 1] == 1:
            assert bool(out["sell_signal"].iloc[i])
    assert (out["buy_signal"] & out["sell_signal"]).sum() == 0
    # 该序列应至少出现一次多→空翻转
    assert ((trend[1:] == -1) & (trend[:-1] == 1)).sum() >= 1
