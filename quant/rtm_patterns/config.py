"""RTM pattern detection configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RTMConfig:
    """Tunable thresholds for RTM pattern matching (v2 — ~80% cheat-sheet alignment)."""

    pivot_left: int = 3
    pivot_right: int = 3
    eq_tolerance_pct: float = 0.12
    qm_quick_retest_bars: int = 12
    qm_late_retest_bars: int = 50
    min_ruler_touches: int = 3
    max_ruler_touches: int = 5
    compression_lookback: int = 15
    compression_atr_ratio: float = 0.45
    compression_min_zigzags: int = 3
    three_drive_min_sep: int = 2
    cancan_min_zigzags: int = 5
    sr_flip_lookback: int = 80
    sr_flip_break_atr: float = 0.15
    sr_flip_max_retest_bars: int = 35
    double_ssr_min_flips: int = 2
    engulfing_body_ratio: float = 1.0
    zone_atr_mult: float = 0.35
    zone_max_age_bars: int = 80
    min_quality_score: float = 0.70
    require_zone_for_qm: bool = True
    require_zone_for_fakeout: bool = True
    require_rejection_candle: bool = True
    rejection_wick_ratio: float = 0.55
    displacement_atr_mult: float = 1.2
    mpl_engulf_atr: float = 0.1
