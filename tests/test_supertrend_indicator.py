"""Supertrend 指标：递推与翻转信号基本性质。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from supertrend_indicator import compute_supertrend


def _sample_ohlcv(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0, 0.2, n)
    return pd.DataFrame(
        {
            "open_time": np.arange(n) * 300_000,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
        }
    )


def test_compute_supertrend_columns():
    df = _sample_ohlcv()
    out = compute_supertrend(df, period=10, multiplier=3.0, source="hl2", atr_method="wilder")
    assert not out.empty
    for col in ("st_up", "st_dn", "st_trend", "buy_signal", "sell_signal", "st_atr"):
        assert col in out.columns
    assert set(out["st_trend"].dropna().unique()).issubset({-1, 1})


def test_buy_sell_mutually_exclusive_on_same_bar():
    df = _sample_ohlcv(120)
    out = compute_supertrend(df)
    both = out["buy_signal"] & out["sell_signal"]
    assert not both.any()


def test_trend_flip_implies_signal():
    out = compute_supertrend(_sample_ohlcv(150))
    trend = out["st_trend"].to_numpy()
    for i in range(1, len(trend)):
        if trend[i] == 1 and trend[i - 1] == -1:
            assert bool(out["buy_signal"].iloc[i])
        if trend[i] == -1 and trend[i - 1] == 1:
            assert bool(out["sell_signal"].iloc[i])
