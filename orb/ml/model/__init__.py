"""ORB 突破排序大模型 — 路径、加载、训练统一入口。

持久化产物在 ``data/orb/ml/``（模型、样本、universe）；``output/orb/v2/eval/`` 仅放回测报告。

Example::

    from orb.ml.model import BreakoutModelBundle, run_training_pipeline

    bundle = BreakoutModelBundle.load_production()
    p = bundle.predict_true(feat, symbol="TSLAUSDT")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from orb.ml.model.paths import (
    ARCHIVE_DIR,
    ARTIFACT_DIR,
    GBM_META,
    GBM_PKL,
    GBM_TRAIN_REPORT,
    LOGISTIC_TRUE_JSON,
    MANIFEST_JSON,
    ML_DATA_ROOT,
    ML_MODELS_DIR,
    ML_SAMPLES_DIR,
    ML_SYMBOLS_DIR,
    PROFILES_JSON,
    SAMPLES_JSON,
    SYMBOLS_UNIVERSE,
    ensure_model_dirs,
    layout_status,
    resolve_gbm_meta_path,
    resolve_gbm_path,
    resolve_logistic_true_path,
    resolve_profiles_path,
    resolve_samples_path,
    resolve_symbols_path,
    resolve_train_report_path,
)

if TYPE_CHECKING:
    from orb.ml.model.bundle import BreakoutModelBundle
    from orb.ml.model.train import TrainingValidationError

_LAZY = {
    "BreakoutModelBundle": ("orb.ml.model.bundle", "BreakoutModelBundle"),
    "archive_snapshot": ("orb.ml.model.manifest", "archive_snapshot"),
    "write_manifest": ("orb.ml.model.manifest", "write_manifest"),
    "bootstrap_from_legacy": ("orb.ml.model.train", "bootstrap_from_legacy"),
    "run_training_pipeline": ("orb.ml.model.train", "run_training_pipeline"),
    "TrainingValidationError": ("orb.ml.model.train", "TrainingValidationError"),
}

__all__ = [
    "BreakoutModelBundle",
    "ARCHIVE_DIR",
    "ARTIFACT_DIR",
    "GBM_META",
    "GBM_PKL",
    "GBM_TRAIN_REPORT",
    "LOGISTIC_TRUE_JSON",
    "MANIFEST_JSON",
    "ML_DATA_ROOT",
    "ML_MODELS_DIR",
    "ML_SAMPLES_DIR",
    "ML_SYMBOLS_DIR",
    "PROFILES_JSON",
    "SAMPLES_JSON",
    "SYMBOLS_UNIVERSE",
    "TrainingValidationError",
    "archive_snapshot",
    "bootstrap_from_legacy",
    "ensure_model_dirs",
    "layout_status",
    "resolve_gbm_meta_path",
    "resolve_gbm_path",
    "resolve_logistic_true_path",
    "resolve_profiles_path",
    "resolve_samples_path",
    "resolve_symbols_path",
    "resolve_train_report_path",
    "run_training_pipeline",
    "write_manifest",
]


def __getattr__(name: str):
    if name in _LAZY:
        mod, attr = _LAZY[name]
        import importlib

        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
