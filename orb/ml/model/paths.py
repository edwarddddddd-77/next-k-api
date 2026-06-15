"""ORB 突破排序大模型 — 产物路径（data/orb/ml，非 output 回测目录）。"""

from __future__ import annotations

import os
from pathlib import Path

from orb.ml.paths import (
    CONFIG_V2,
    DEFAULT_GBM,
    DEFAULT_GBM_META,
    DEFAULT_GBM_TRAIN_REPORT,
    DEFAULT_PROFILES,
    DEFAULT_SAMPLES,
    LEGACY_OUTPUT,
    LEGACY_V2_OUTPUT,
    PROJECT_ROOT,
    V1_OUTPUT,
    V2_OUTPUT,
)
from orb.ml.live_bundle import (
    live_gbm_meta,
    live_gbm_pkl,
    live_profiles_json,
    live_train_report,
)

# 持久化 ML 根目录（可用 ORB_ML_DATA_ROOT 覆盖）
ML_DATA_ROOT = Path(os.getenv("ORB_ML_DATA_ROOT", "") or (PROJECT_ROOT / "data" / "orb" / "ml"))
ML_SYMBOLS_DIR = ML_DATA_ROOT / "symbols"
ML_MODELS_DIR = ML_DATA_ROOT / "models"
ML_SAMPLES_DIR = ML_DATA_ROOT / "samples"
ARCHIVE_DIR = ML_DATA_ROOT / "archive"
STAGING_MODELS_DIR = ML_DATA_ROOT / "staging" / "models"
STAGING_SAMPLES_DIR = ML_DATA_ROOT / "staging" / "samples"

# 规范路径（训练写入 / Live 加载默认）
SYMBOLS_UNIVERSE = ML_SYMBOLS_DIR / "universe.txt"
SYMBOLS_UNIVERSE_NO_COIN = ML_SYMBOLS_DIR / "universe_no_coin.txt"

GBM_PKL = ML_MODELS_DIR / "breakout_gbm.pkl"
GBM_META = ML_MODELS_DIR / "breakout_gbm.json"
GBM_TRAIN_REPORT = ML_MODELS_DIR / "breakout_gbm_train_report.json"
PROFILES_JSON = ML_MODELS_DIR / "symbol_breakout_profiles.json"
GATE_SUGGESTION_JSON = ML_MODELS_DIR / "gate_suggestion.json"
MANIFEST_JSON = ML_MODELS_DIR / "model_manifest.json"
LOGISTIC_TRUE_JSON = ML_MODELS_DIR / "logistic_true_breakout.json"
LOGISTIC_FAKE_JSON = ML_MODELS_DIR / "logistic_fake_breakout.json"

SAMPLES_JSON = ML_SAMPLES_DIR / "breakout_samples.json"

STAGING_GBM_PKL = STAGING_MODELS_DIR / "breakout_gbm.pkl"
STAGING_GBM_META = STAGING_MODELS_DIR / "breakout_gbm.json"
STAGING_GBM_TRAIN_REPORT = STAGING_MODELS_DIR / "breakout_gbm_train_report.json"
STAGING_PROFILES_JSON = STAGING_MODELS_DIR / "symbol_breakout_profiles.json"
STAGING_SAMPLES_JSON = STAGING_SAMPLES_DIR / "breakout_samples.json"

# 兼容旧 layout
OLD_V2_ARTIFACT = V2_OUTPUT
TRAIN_SYMBOLS_FILE = SYMBOLS_UNIVERSE
LEGACY_GBM = LEGACY_OUTPUT / "orb_shared_breakout_gbm.pkl"
LEGACY_PROFILES = LEGACY_OUTPUT / "symbol_breakout_profiles.json"
LEGACY_SAMPLES = LEGACY_OUTPUT / "orb_shared_breakout_samples.json"
LEGACY_LOGISTIC_TRUE = LEGACY_OUTPUT / "orb_shared_true_breakout_model.json"

# 向后兼容别名
ARTIFACT_DIR = ML_MODELS_DIR


def _first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.is_file():
            return p
    return paths[0]


def ensure_model_dirs() -> None:
    for d in (
        ML_DATA_ROOT,
        ML_SYMBOLS_DIR,
        ML_MODELS_DIR,
        ML_SAMPLES_DIR,
        ARCHIVE_DIR,
        STAGING_MODELS_DIR,
        STAGING_SAMPLES_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def staging_gbm_pkl_path() -> Path:
    return STAGING_GBM_PKL


def staging_gbm_meta_path() -> Path:
    return STAGING_GBM_META


def staging_train_report_path() -> Path:
    return STAGING_GBM_TRAIN_REPORT


def staging_profiles_path() -> Path:
    return STAGING_PROFILES_JSON


def staging_samples_path() -> Path:
    return STAGING_SAMPLES_JSON


def resolve_symbols_path() -> Path:
    return _first_existing(
        SYMBOLS_UNIVERSE,
        CONFIG_V2 / "symbols.txt",
        PROJECT_ROOT / "config" / "orb_shared_train_symbols.txt",
    )


def resolve_train_symbols_path() -> Path:
    """训练 / K 线刷新标的：优先 ORB_V2_SYMBOLS_FILE（与 Live 一致）。"""
    raw = (os.getenv("ORB_V2_SYMBOLS_FILE") or "").strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_file():
            return p
    return resolve_symbols_path()


def resolve_symbols_no_coin_path() -> Path:
    return _first_existing(
        SYMBOLS_UNIVERSE_NO_COIN,
        CONFIG_V2 / "symbols_no_coin.txt",
    )


def resolve_gbm_path() -> Path:
    return _first_existing(
        live_gbm_pkl(),
        GBM_PKL,
        OLD_V2_ARTIFACT / "breakout_gbm.pkl",
        LEGACY_V2_OUTPUT / "breakout_gbm.pkl",
        DEFAULT_GBM,
        LEGACY_GBM,
    )


def resolve_gbm_meta_path() -> Path:
    return _first_existing(
        live_gbm_meta(),
        GBM_META,
        OLD_V2_ARTIFACT / "breakout_gbm.json",
        LEGACY_V2_OUTPUT / "breakout_gbm.json",
        DEFAULT_GBM_META,
    )


def resolve_profiles_path() -> Path:
    return _first_existing(
        live_profiles_json(),
        PROFILES_JSON,
        OLD_V2_ARTIFACT / "symbol_breakout_profiles.json",
        LEGACY_V2_OUTPUT / "symbol_breakout_profiles.json",
        DEFAULT_PROFILES,
        LEGACY_PROFILES,
    )


def resolve_samples_path() -> Path:
    return _first_existing(
        SAMPLES_JSON,
        OLD_V2_ARTIFACT / "breakout_samples.json",
        DEFAULT_SAMPLES,
        LEGACY_SAMPLES,
    )


def resolve_train_report_path() -> Path:
    return _first_existing(
        live_train_report(),
        GBM_TRAIN_REPORT,
        OLD_V2_ARTIFACT / "breakout_gbm_train_report.json",
        DEFAULT_GBM_TRAIN_REPORT,
    )


def resolve_logistic_true_path() -> Path:
    return _first_existing(
        LOGISTIC_TRUE_JSON,
        V1_OUTPUT / "true_breakout_model.json",
        LEGACY_LOGISTIC_TRUE,
        LEGACY_OUTPUT / "orb_true_breakout_model.json",
    )


def layout_status() -> dict:
    """当前 ML / Live 目录布局摘要（运维 / 自检）。"""
    from orb.ml.live_bundle import bundle_status

    st = bundle_status()
    st.update(
        {
            "ml_data_root": str(ML_DATA_ROOT),
            "symbols_universe": str(resolve_symbols_path()),
            "samples": str(resolve_samples_path()),
            "manifest": str(MANIFEST_JSON),
        }
    )
    return st
