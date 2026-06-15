"""ORB 突破排序大模型 — 训练产物 data/orb/ml/；实盘参数 orb_live/。"""

from __future__ import annotations

import os
from pathlib import Path

from orb.ml.paths import CONFIG_V2, PROJECT_ROOT
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

SYMBOLS_UNIVERSE = ML_SYMBOLS_DIR / "universe.txt"
SYMBOLS_UNIVERSE_NO_COIN = ML_SYMBOLS_DIR / "universe_no_coin.txt"
CONFIG_SYMBOLS = CONFIG_V2 / "symbols.txt"
CONFIG_SYMBOLS_NO_COIN = CONFIG_V2 / "symbols_no_coin.txt"

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

TRAIN_SYMBOLS_FILE = CONFIG_SYMBOLS
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


def _symbols_env_override_issue() -> str:
    """ORB_V2_SYMBOLS_FILE 指向 data/ 时给出修复提示。"""
    raw = (os.getenv("ORB_V2_SYMBOLS_FILE") or "").strip()
    if not raw:
        return ""
    norm = raw.replace("\\", "/").rstrip("/").lower()
    if "data/orb" in norm or norm.startswith("data/"):
        return (
            f"ORB_V2_SYMBOLS_FILE={raw} 指向 data/（Volume 会盖住标的文件）。"
            "请删除该环境变量，默认读 config/orb/v2/symbols.txt。"
        )
    return ""


def resolve_symbols_path() -> Path:
    """生产标的池：config/orb/v2/symbols.txt（git 部署，不受 DATA_DIR Volume 影响）。"""
    raw = (os.getenv("ORB_V2_SYMBOLS_FILE") or "").strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_file():
            return p
    return CONFIG_SYMBOLS


def resolve_train_symbols_path() -> Path:
    """训练 / K 线刷新标的（与 Live 扫描同源）。"""
    return resolve_symbols_path()


def resolve_symbols_no_coin_path() -> Path:
    raw = (os.getenv("ORB_V2_SYMBOLS_NO_COIN_FILE") or "").strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_file():
            return p
    return CONFIG_SYMBOLS_NO_COIN


def symbols_status(*, relative_paths: bool = False) -> dict:
    from orb.ml.samples import parse_symbol_list

    path = resolve_symbols_path()
    exists = path.is_file()
    count = 0
    if exists:
        count = len(parse_symbol_list(path.read_text(encoding="utf-8")))
    rel = str(path.relative_to(PROJECT_ROOT)) if relative_paths else str(path)
    issue = _symbols_env_override_issue()
    ok = exists and count > 0
    return {
        "ok": ok,
        "path": rel,
        "exists": exists,
        "symbol_count": count,
        "env_issue": issue,
    }


def log_symbols_startup() -> None:
    import logging

    log = logging.getLogger(__name__)
    st = symbols_status(relative_paths=True)
    sev = "ok" if st["ok"] else "missing"
    log.info(
        "ORB symbols [%s] path=%s count=%s env=%s",
        sev,
        st["path"],
        st["symbol_count"],
        (os.getenv("ORB_V2_SYMBOLS_FILE") or "").strip() or "(default config/orb/v2/symbols.txt)",
    )
    if st.get("env_issue"):
        log.warning("ORB symbols: %s", st["env_issue"])
    if not st["ok"]:
        log.warning("ORB symbols: universe file missing or empty — scans will skip (orb_v2_no_symbols)")


def resolve_gbm_path() -> Path:
    """实盘只读 orb_live/（训练产物在 data/orb/ml/models/）。"""
    return live_gbm_pkl()


def resolve_gbm_meta_path() -> Path:
    return live_gbm_meta()


def resolve_profiles_path() -> Path:
    return live_profiles_json()


def resolve_samples_path() -> Path:
    return _first_existing(
        SAMPLES_JSON,
        STAGING_SAMPLES_JSON,
    )


def resolve_train_report_path() -> Path:
    return live_train_report()


def resolve_logistic_true_path() -> Path:
    return LOGISTIC_TRUE_JSON


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
