"""训练产物验收（通过后自动 promote）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from orb.ml.model.auto_config import MlAutoConfig
from orb.ml.model.bundle import BreakoutModelBundle


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def validate_staging_artifacts(
    *,
    gbm_pkl: Path,
    gbm_meta: Path,
    profiles: Path,
    samples: Path,
    train_report: Path,
    cfg: MlAutoConfig | None = None,
) -> Tuple[bool, List[str], dict]:
    """返回 (passed, reasons, detail)。"""
    c = cfg or MlAutoConfig.from_env()
    reasons: List[str] = []
    detail: Dict[str, Any] = {}

    required = [
        ("gbm_pkl", gbm_pkl),
        ("gbm_meta", gbm_meta),
        ("profiles", profiles),
        ("samples", samples),
        ("train_report", train_report),
    ]
    for name, p in required:
        if not p.is_file():
            reasons.append(f"missing_{name}")

    if reasons:
        detail["skipped_metrics"] = True
        return False, reasons, detail

    report = _read_json(train_report)
    detail["train_report"] = report
    samples_n = int(report.get("samples_total") or report.get("samples") or 0)
    if samples_n < c.min_train_samples:
        reasons.append(f"samples_total<{c.min_train_samples}")

    brier = report.get("brier")
    if brier is not None and float(brier) > c.max_train_brier:
        reasons.append(f"brier>{c.max_train_brier}")

    holdout_n = int(report.get("holdout_n") or 0)
    if holdout_n >= c.min_holdout_n:
        acc = float(report.get("holdout_accuracy") or 0)
        sep = float(report.get("holdout_separation") or 0)
        detail["holdout"] = {"n": holdout_n, "accuracy": acc, "separation": sep}
        if acc < c.min_holdout_accuracy:
            reasons.append(f"holdout_accuracy<{c.min_holdout_accuracy}")
        if sep < c.min_holdout_separation:
            reasons.append(f"holdout_separation<{c.min_holdout_separation}")
    else:
        detail["holdout"] = {"n": holdout_n, "skipped_gate": True}

    try:
        bundle = BreakoutModelBundle.load(gbm_path=gbm_pkl, profiles_path=profiles)
        if not bundle.is_ready:
            reasons.append("model_not_ready")
        else:
            p = bundle.predict_true(
                {
                    "or_width_pct": 2.0,
                    "vol_ratio": 1.0,
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
            detail["smoke_predict"] = p
            if not (0.0 <= p <= 1.0):
                reasons.append("smoke_predict_out_of_range")
    except Exception as exc:
        reasons.append(f"smoke_load_failed:{exc}")

    passed = len(reasons) == 0
    return passed, reasons, detail
