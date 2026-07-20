"""Unit tests: HangukQuant toy Barra + factor mimicking + trend helper."""

from __future__ import annotations

import numpy as np

from factor_sector import (
    build_exposure,
    classify_sectors,
    compound_levels,
    constrained_wls,
    factor_mimicking_weights,
    trend_sign_trailing,
    walk_forward_factor_trend,
)


def test_hangukquant_toy_barra():
    y = np.array([0.04, 0.02, 0.08, 0.06])
    w = np.array([0.50, 0.25, 0.125, 0.125])
    X = np.array([
        [1, 1, 0],
        [1, 1, 0],
        [1, 0, 1],
        [1, 0, 1],
    ], dtype=float)
    c = np.array([0.0, 0.75, 0.25])
    f = constrained_wls(y, w, X, c)
    assert abs(f[0] - 0.0425) < 1e-6
    assert abs(f[1] - (-0.0091666667)) < 1e-5
    assert abs(f[2] - 0.0275) < 1e-6
    assert abs(c @ f) < 1e-9


def test_build_exposure_matches_article():
    X = build_exposure(["L1", "L1", "Meme", "Meme"], ["L1", "Meme"])
    assert X.shape == (4, 3)
    assert np.allclose(X[:, 0], 1.0)
    assert np.allclose(X[:, 1], [1, 1, 0, 0])
    assert np.allclose(X[:, 2], [0, 0, 1, 1])
    # collinearity: market = L1 + Meme
    assert np.allclose(X[:, 0], X[:, 1] + X[:, 2])


def test_factor_mimicking_l1():
    w = np.array([0.50, 0.25, 0.125, 0.125])
    sectors = ["L1", "L1", "Meme", "Meme"]
    p = factor_mimicking_weights(w, sectors, "L1")
    assert abs(p.sum()) < 1e-9
    assert abs(p[0] - 0.1666667) < 1e-4
    assert abs(p[2] + 0.125) < 1e-9
    y = np.array([0.04, 0.02, 0.08, 0.06])
    assert abs(float(p @ y) - (-0.0091666667)) < 1e-5


def test_trend_sign_trailing():
    up = compound_levels([0.01] * 30)
    down = compound_levels([-0.01] * 30)
    assert trend_sign_trailing(up, 20) == 1
    assert trend_sign_trailing(down, 20) == -1


def test_cascade_needs_both_windows():
    # 20d still up, but last 7d down → flat (fails 7d confirm)
    rets = [0.02] * 25 + [-0.005] * 7
    hist = [
        {"date": f"d{i}", "f_market": 0.0, "factors": {"L1": r, "Meme": -r, "Other": 0.0}}
        for i, r in enumerate(rets)
    ]
    st = classify_sectors(hist, ["L1", "Meme", "Other"], lookback=20, confirm_lookback=7)
    assert st["L1"]["trend_20"] == 1
    assert st["L1"]["trend_7"] == -1
    assert st["L1"]["trend"] == 0
    assert st["Meme"]["trend_20"] == -1
    assert st["Meme"]["trend_7"] == 1
    assert st["Meme"]["trend"] == 0

    # both windows up on L1 / down on Meme → confirmed
    rets2 = [0.01] * 30
    hist2 = [
        {"date": f"d{i}", "f_market": 0.0, "factors": {"L1": r, "Meme": -r, "Other": 0.0}}
        for i, r in enumerate(rets2)
    ]
    st2 = classify_sectors(hist2, ["L1", "Meme", "Other"], lookback=20, confirm_lookback=7)
    assert st2["L1"]["trend"] == 1
    assert st2["Meme"]["trend"] == -1


def test_walk_forward_trend_smoke():
    hist = []
    for i in range(50):
        # L1 drifts up, Meme drifts down → eventual bull L1 / bear Meme
        hist.append({
            "date": f"2026-01-{(i % 28) + 1:02d}",
            "f_market": 0.0,
            "factors": {
                "L1": 0.003,
                "Meme": -0.003,
                "Other": 0.0,
            },
        })
    out = walk_forward_factor_trend(hist, lookback=20, confirm_lookback=7, cost_bps=0.0)
    assert out["ok"]
    assert out["days"] > 0
    st = classify_sectors(hist, ["L1", "Meme", "Other"], lookback=20, confirm_lookback=7)
    assert st["L1"]["trend"] == 1
    assert st["Meme"]["trend"] == -1
    assert out["active_days"] > 0
