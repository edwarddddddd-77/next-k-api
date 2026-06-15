"""orb.ml.model 大模型包测试。"""

from __future__ import annotations

from orb.ml.model import BreakoutModelBundle, resolve_gbm_path, resolve_profiles_path


def test_model_bundle_loads_when_artifacts_exist():
    bundle = BreakoutModelBundle.load()
    status = bundle.status()
    if resolve_gbm_path().is_file() or resolve_profiles_path().is_file():
        assert status["gbm_path"]
        assert status["profiles_path"]
    if bundle.is_ready:
        p = bundle.predict_true(
            {
                "or_width_pct": 2.0,
                "vol_ratio": 1.2,
                "side_long": 1.0,
                "vwap_dist_pct": 0.1,
                "risk_frac_pct": 0.5,
                "minutes_after_or": 30.0,
                "gap_pct": 0.0,
                "pm_rvol": 0.0,
                "pm_regime_go": 0.0,
                "pm_regime_fade": 0.0,
                "atr_pct": 4.0,
            },
            symbol="TSLAUSDT",
        )
        assert 0.0 <= p <= 1.0


def test_model_paths_resolve():
    assert str(resolve_gbm_path()).endswith(".pkl")
    assert str(resolve_profiles_path()).endswith(".json")


def test_model_layout_under_data_orb_ml():
    from orb.ml.model import ML_DATA_ROOT, layout_status

    norm = str(ML_DATA_ROOT).replace("\\", "/")
    assert norm.endswith("data/orb/ml")
    st = layout_status()
    assert "data/orb/ml" in st["ml_data_root"].replace("\\", "/")
    assert "live_bundle_root" in st
