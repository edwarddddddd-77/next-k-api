"""模型 manifest 写入与月度归档。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from orb.ml.model.paths import (
    ARCHIVE_DIR,
    GBM_META,
    GBM_PKL,
    GBM_TRAIN_REPORT,
    MANIFEST_JSON,
    ML_DATA_ROOT,
    PROFILES_JSON,
    SAMPLES_JSON,
    ensure_model_dirs,
    resolve_symbols_path,
)


def write_manifest(*, extra: Optional[Dict[str, Any]] = None) -> Path:
    ensure_model_dirs()
    payload: Dict[str, Any] = {
        "version": 3,
        "ml_data_root": str(ML_DATA_ROOT),
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "symbols": str(resolve_symbols_path()),
        "gbm": str(GBM_PKL),
        "profiles": str(PROFILES_JSON),
        "samples": str(SAMPLES_JSON),
    }
    if GBM_META.is_file():
        try:
            payload["gbm_meta"] = json.loads(GBM_META.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    if GBM_TRAIN_REPORT.is_file():
        try:
            payload["train_report"] = json.loads(GBM_TRAIN_REPORT.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    if extra:
        payload.update(extra)
    MANIFEST_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return MANIFEST_JSON


def archive_snapshot(tag: str) -> Path:
    ensure_model_dirs()
    stamp = tag or datetime.now(timezone.utc).strftime("%Y%m")
    dest = ARCHIVE_DIR / stamp
    dest.mkdir(parents=True, exist_ok=True)
    for src in (GBM_PKL, GBM_META, GBM_TRAIN_REPORT, PROFILES_JSON, MANIFEST_JSON, SAMPLES_JSON):
        if src.is_file():
            (dest / src.name).write_bytes(src.read_bytes())
    return dest
