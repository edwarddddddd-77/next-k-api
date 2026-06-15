"""staging → production 原子发布。"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from orb.ml.model.paths import (
    ARCHIVE_DIR,
    GBM_META,
    GBM_PKL,
    GBM_TRAIN_REPORT,
    PROFILES_JSON,
    SAMPLES_JSON,
    STAGING_MODELS_DIR,
    STAGING_SAMPLES_DIR,
    ensure_model_dirs,
    staging_gbm_meta_path,
    staging_gbm_pkl_path,
    staging_profiles_path,
    staging_samples_path,
    staging_train_report_path,
)


def _atomic_copy(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(str(src))
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)


def archive_production(tag: str) -> Path:
    ensure_model_dirs()
    stamp = tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = ARCHIVE_DIR / f"pre_promote_{stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    for src in (GBM_PKL, GBM_META, GBM_TRAIN_REPORT, PROFILES_JSON, SAMPLES_JSON):
        if src.is_file():
            shutil.copy2(src, dest / src.name)
    return dest


def promote_staging_to_production(*, tag: str = "") -> Dict[str, str]:
    """验收通过后，将 staging 覆盖到 production。"""
    ensure_model_dirs()
    pairs = [
        (staging_gbm_pkl_path(), GBM_PKL),
        (staging_gbm_meta_path(), GBM_META),
        (staging_train_report_path(), GBM_TRAIN_REPORT),
        (staging_profiles_path(), PROFILES_JSON),
        (staging_samples_path(), SAMPLES_JSON),
    ]
    for src, _dst in pairs:
        if not src.is_file():
            raise FileNotFoundError(f"staging artifact missing: {src}")

    archive_production(tag)
    promoted: Dict[str, str] = {}
    for src, dst in pairs:
        _atomic_copy(src, dst)
        promoted[src.name] = str(dst)
    return promoted


def clear_staging() -> None:
    for d in (STAGING_MODELS_DIR, STAGING_SAMPLES_DIR):
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


def ensure_staging_dirs() -> None:
    STAGING_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    STAGING_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
