"""ORB 突破排序大模型 — 路径、加载、训练统一入口。

持久化产物在 ``data/orb/ml/``（模型、样本、43 标 universe）；``output/orb/v2/eval/`` 仅放回测报告。

Example::

    from orb.ml.model import BreakoutModelBundle, run_training_pipeline

    bundle = BreakoutModelBundle.load_production()
    p = bundle.predict_true(feat, symbol="TSLAUSDT")
"""

from orb.ml.model.bundle import BreakoutModelBundle
from orb.ml.model.manifest import archive_snapshot, write_manifest
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
    SYMBOLS_UNIVERSE_NO_COIN,
    ensure_model_dirs,
    layout_status,
    resolve_gbm_meta_path,
    resolve_gbm_path,
    resolve_logistic_true_path,
    resolve_profiles_path,
    resolve_samples_path,
    resolve_symbols_no_coin_path,
    resolve_symbols_path,
    resolve_train_report_path,
)
from orb.ml.model.train import bootstrap_from_legacy, run_training_pipeline, TrainingValidationError

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
    "SYMBOLS_UNIVERSE_NO_COIN",
    "archive_snapshot",
    "bootstrap_from_legacy",
    "ensure_model_dirs",
    "layout_status",
    "resolve_gbm_meta_path",
    "resolve_gbm_path",
    "resolve_logistic_true_path",
    "resolve_profiles_path",
    "resolve_samples_path",
    "resolve_symbols_no_coin_path",
    "resolve_symbols_path",
    "resolve_train_report_path",
    "run_training_pipeline",
    "TrainingValidationError",
    "write_manifest",
]
