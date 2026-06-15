"""ML 自动化（staging / 验收 / promote）测试。"""

from __future__ import annotations

import json
from pathlib import Path

from orb.ml.model.auto_config import MlAutoConfig
from orb.ml.model.paths import ensure_model_dirs, staging_train_report_path
from orb.ml.model.validate import validate_staging_artifacts


def test_ml_auto_config_defaults(monkeypatch):
    monkeypatch.delenv("ORB_ML_AUTO_FETCH_KLINES", raising=False)
    cfg = MlAutoConfig.from_env()
    assert cfg.auto_fetch_klines is True
    assert cfg.auto_validate is False
    assert cfg.auto_promote is False
    assert cfg.auto_gate_suggest is False
    assert cfg.kline_days == 180.0


def test_validate_staging_fails_on_missing_files(tmp_path: Path):
    cfg = MlAutoConfig.from_env()
    passed, reasons, detail = validate_staging_artifacts(
        gbm_pkl=tmp_path / "missing.pkl",
        gbm_meta=tmp_path / "missing.json",
        profiles=tmp_path / "profiles.json",
        samples=tmp_path / "samples.json",
        train_report=tmp_path / "report.json",
        cfg=cfg,
    )
    assert passed is False
    assert any(r.startswith("missing_") for r in reasons)
    assert detail.get("skipped_metrics") is True


def test_resolve_train_symbols_path_prefers_v2_env(monkeypatch, tmp_path: Path):
    from orb.ml.model.paths import resolve_train_symbols_path

    custom = tmp_path / "custom_universe.txt"
    custom.write_text("TSLA\n", encoding="utf-8")
    monkeypatch.setenv("ORB_V2_SYMBOLS_FILE", str(custom))
    assert resolve_train_symbols_path() == custom


def test_validate_staging_fails_on_low_samples(tmp_path: Path):
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"samples_total": 10, "brier": 0.1, "holdout_n": 0}),
        encoding="utf-8",
    )
    for name in ("gbm.pkl", "gbm.json", "profiles.json", "samples.json"):
        (tmp_path / name).write_text("{}", encoding="utf-8")

    cfg = MlAutoConfig.from_env()
    passed, reasons, _detail = validate_staging_artifacts(
        gbm_pkl=tmp_path / "gbm.pkl",
        gbm_meta=tmp_path / "gbm.json",
        profiles=tmp_path / "profiles.json",
        samples=tmp_path / "samples.json",
        train_report=report,
        cfg=cfg,
    )
    assert passed is False
    assert any("samples_total" in r for r in reasons)


def test_staging_dirs_created():
    ensure_model_dirs()
    assert staging_train_report_path().parent.is_dir()
