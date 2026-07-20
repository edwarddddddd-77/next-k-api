"""Unit tests for fractal_ict core."""

from quant.fractal_ict.core import detect_tspot, log_midpoint, build_cisd_series, check_cisd_break


def test_log_midpoint_positive_prices():
    mid = log_midpoint(110.0, 100.0, 105.0, 108.0)
    assert 100.0 < mid < 110.0


def test_detect_bearish_tspot():
    bars = [
        (0, 100, 105, 99, 104),
        (1, 104, 108, 103, 107),
        (2, 107, 112, 106, 108),  # sweeps prev high, closes below
        (3, 108, 109, 105, 106),  # C3 forming
    ]
    setup = detect_tspot(bars, bias="none")
    assert setup is not None
    assert setup.side == -1
    assert setup.pattern == "c2_sweep"


def test_cisd_break_bullish():
    bars = [
        (0, 100, 101, 99, 100.5),
        (1, 100.5, 101, 99.5, 100.2),
        (2, 100.2, 100.8, 99.8, 100.0),  # bear in series
        (3, 100.0, 101.5, 99.9, 101.2),  # break up
    ]
    series = build_cisd_series(bars, 2, is_bullish=True, use_body=True)
    assert series is not None
    sh, sl = series
    idx = check_cisd_break(bars, from_idx=2, to_idx=3, series_high=sh, series_low=sl, is_bullish=True)
    assert idx == 3
