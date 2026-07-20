"""Unit tests for RTM pattern detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.rtm_patterns import RTMConfig, scan_rtm_patterns
from quant.rtm_patterns.pivots import bearish_qm_structure, find_pivots
from quant.rtm_patterns.scanner import RTM_PATTERN_IDS
from quant.rtm_patterns.types import Pivot


def _make_bearish_qm_df(n: int = 80) -> pd.DataFrame:
    """Synthetic H-L-HH-LL then retest at QML."""
    close = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    open_ = np.full(n, 100.0)

    # Structure pivots (approximate indices)
    high[15] = 110.0
    low[25] = 95.0
    high[40] = 115.0  # HH
    low[55] = 90.0  # LL
    # Retest QML ~110 at end
    high[-3] = 111.0
    close[-3] = 108.0
    high[-2] = 110.5
    close[-2] = 107.0
    high[-1] = 110.2
    close[-1] = 106.0
    low[-1] = 105.0

    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


def test_find_pivots_basic():
    high = np.array([1, 2, 3, 2, 1, 2, 3, 2, 1], dtype=float)
    low = np.array([0, 1, 2, 1, 0, 1, 2, 1, 0], dtype=float)
    pivots = find_pivots(high, low, left=1, right=1)
    kinds = [p.kind for p in pivots]
    assert "high" in kinds
    assert "low" in kinds


def test_bearish_qm_structure():
    seq = [
        Pivot(10, 110.0, "high"),
        Pivot(20, 95.0, "low"),
        Pivot(30, 115.0, "high"),
        Pivot(40, 90.0, "low"),
    ]
    result = bearish_qm_structure(seq)
    assert result is not None
    qml, head, ll, l1, _ = result
    assert qml == 110.0
    assert head == 115.0
    assert ll == 90.0
    assert l1 == 95.0


def test_scan_returns_list():
    df = _make_bearish_qm_df()
    cfg = RTMConfig(pivot_left=2, pivot_right=2, eq_tolerance_pct=1.0, require_zone_for_qm=False, min_quality_score=0.5)
    hits = scan_rtm_patterns(df, config=cfg)
    assert isinstance(hits, list)


def test_all_pattern_ids_registered():
    assert len(RTM_PATTERN_IDS) == 22
    assert "qm_quick_retest" in RTM_PATTERN_IDS
    assert "cancan_fakeout" in RTM_PATTERN_IDS


def test_2r_fakeout_synthetic():
    n = 60
    close = np.linspace(100, 102, n)
    high = close + 0.5
    low = close - 0.5
    open_ = close.copy()
    high[20] = 105.0
    high[35] = 105.1
    close[35] = 104.0
    open_[-2] = 103.0
    close[-2] = 104.5
    open_[-1] = 105.2
    close[-1] = 103.0
    high[-1] = 106.5
    low[-1] = 102.8
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})
    cfg = RTMConfig(
        pivot_left=2,
        pivot_right=2,
        eq_tolerance_pct=0.2,
        require_zone_for_fakeout=False,
        require_rejection_candle=True,
        rejection_wick_ratio=0.3,
        min_quality_score=0.55,
    )
    hits = scan_rtm_patterns(df, config=cfg, patterns=["fakeout_2r", "fakeout_v1"])
    assert any(h.pattern in ("fakeout_2r", "fakeout_v1") for h in hits)


def test_quality_score_present():
    df = _make_bearish_qm_df()
    cfg = RTMConfig(pivot_left=2, pivot_right=2, eq_tolerance_pct=1.0, require_zone_for_qm=False, min_quality_score=0.5)
    hits = scan_rtm_patterns(df, config=cfg)
    if hits:
        assert "quality" in hits[0].meta
        assert hits[0].meta["quality"] >= 0.5
