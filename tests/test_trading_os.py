"""Unit tests for Trading OS (no network)."""

from __future__ import annotations

import json
from pathlib import Path

from utils.trading_os import (
    PHASE_COPY,
    SCORE_MAX,
    _score,
    clear_cvdd_override,
    load_cvdd_override,
    parse_cvdd_from_figure,
    set_cvdd_override,
)


def test_score_bull_when_far_from_cvdd():
    out = _score(
        price=100_000,
        cvdd=45_000,
        taker={"1h": {"taker_buy_ratio_last": 0.4, "taker_buy_ratio_avg": 0.5}},
        book={"imbalance": 0.2},
    )
    assert out["phase"] == "bull"
    assert out["distance_pct"] > 55
    assert out["score_max"] == SCORE_MAX == 4


def test_score_approach_near_cvdd_with_weak_taker():
    out = _score(
        price=50_000,
        cvdd=46_000,
        taker={"1h": {"taker_buy_ratio_last": 0.42, "taker_buy_ratio_avg": 0.52}},
        book={"imbalance": 0.01},
    )
    assert out["signals"]["cvdd_near"] is True
    assert out["signals"]["taker_weak"] is True
    assert out["phase"] in ("approach", "confirmed")
    assert out["phase"] in PHASE_COPY


def test_score_very_near_maxes_cvdd_points():
    out = _score(
        price=48_000,
        cvdd=47_000,
        taker={"1h": {"taker_buy_ratio_last": 0.55, "taker_buy_ratio_avg": 0.50}},
        book={"imbalance": 0.0},
    )
    assert out["signals"]["cvdd_very_near"] is True
    assert out["score"] == 2  # only cvdd points


def test_parse_cvdd_from_figure():
    fig = {
        "data": [
            {"name": "BTC Price", "x": ["2026-01-01"], "y": [90000]},
            {"name": "CVDD", "x": ["2026-01-02 00:00:00"], "y": [12345.6, 48423.7]},
        ]
    }
    out = parse_cvdd_from_figure(fig)
    assert out["cvdd"] == 48423.7
    assert out["date"] == "2026-01-02"
    assert out["primary"] == "bmp_dash_cvdd"


def test_parse_cvdd_missing_raises():
    try:
        parse_cvdd_from_figure({"data": [{"name": "BTC Price", "y": [1]}]})
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "missing" in str(e)


def test_override_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    # re-import path helpers resolve at call time via resolve_data_dir, so OK
    from utils import trading_os as tos

    monkeypatch.setattr(tos, "resolve_data_dir", lambda: Path(tmp_path))
    clear_cvdd_override()
    assert load_cvdd_override() is None
    set_cvdd_override(46200, source_label="glassnode", note="test")
    ov = load_cvdd_override()
    assert ov is not None
    assert ov["cvdd"] == 46200
    assert ov["source_label"] == "glassnode"
    clear_cvdd_override()
    assert load_cvdd_override() is None
