"""Unit tests for Barra constrained WLS (HangukQuant toy numbers)."""

from __future__ import annotations

import numpy as np

from factor_sector import constrained_wls, factor_mimicking_weights


def test_hangukquant_toy_barra():
    # BTC/ETH L1, DOGE/PEPE Meme — article worked example
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


def test_factor_mimicking_l1():
    w = np.array([0.50, 0.25, 0.125, 0.125])
    sectors = ["L1", "L1", "Meme", "Meme"]
    p = factor_mimicking_weights(w, sectors, "L1")
    assert abs(p.sum()) < 1e-9
    assert abs(p[0] - 0.1666667) < 1e-4
    assert abs(p[2] + 0.125) < 1e-9
    y = np.array([0.04, 0.02, 0.08, 0.06])
    assert abs(float(p @ y) - (-0.0091666667)) < 1e-5
