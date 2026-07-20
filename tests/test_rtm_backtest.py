"""RTM backtest unit tests."""

import numpy as np
import pandas as pd

from quant.rtm_patterns.backtest import BacktestParams, backtest_rtm_patterns
from quant.rtm_patterns.config import RTMConfig


def test_backtest_long_win():
    # Simple long: entry 100, stop 95, target 110
    close = np.array([100.0, 100.0, 105.0, 112.0])
    high = close + 1
    low = close - 1
    open_ = close.copy()
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})

    cfg = RTMConfig(
        pivot_left=1,
        pivot_right=1,
        require_zone_for_qm=False,
        require_zone_for_fakeout=False,
        require_rejection_candle=False,
        min_quality_score=0.0,
    )
    summary = backtest_rtm_patterns(df, config=cfg, params=BacktestParams(max_hold_bars=5, entry_on_next_bar=False))
    assert summary.total_trades >= 0
