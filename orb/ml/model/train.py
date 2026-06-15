"""大模型训练流水线（K 线 → 样本 → hold30 → GBM → 验收 → promote）。"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from orb.data.kline_fetch import fetch_universe_klines
from orb.ml.model.auto_config import MlAutoConfig
from orb.ml.model.manifest import archive_snapshot, write_manifest
from orb.ml.model.paths import (
    GBM_META,
    GBM_PKL,
    GBM_TRAIN_REPORT,
    LOGISTIC_FAKE_JSON,
    LOGISTIC_TRUE_JSON,
    MANIFEST_JSON,
    PROFILES_JSON,
    SAMPLES_JSON,
    ensure_model_dirs,
    resolve_train_symbols_path,
    staging_gbm_meta_path,
    staging_gbm_pkl_path,
    staging_profiles_path,
    staging_samples_path,
    staging_train_report_path,
)
from orb.ml.model.promote import clear_staging, ensure_staging_dirs, promote_staging_to_production
from orb.ml.model.validate import validate_staging_artifacts
from orb.ml.paths import CONFIG_V2, PROJECT_ROOT

_LEGACY_OUTPUT = PROJECT_ROOT / "output"
_LEGACY_V2 = _LEGACY_OUTPUT / "orb" / "v2"

_TOOLS_ML = PROJECT_ROOT / "tools" / "orb" / "ml"


class TrainingValidationError(RuntimeError):
    """验收未通过，production 模型保持不变。"""

    def __init__(self, reasons: list[str], detail: dict):
        super().__init__("validation failed: " + ", ".join(reasons))
        self.reasons = reasons
        self.detail = detail


def _run(cmd: list[str], *, label: str) -> None:
    print(f"[orb_model] {label} ...", flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def bootstrap_from_legacy() -> Dict[str, Any]:
    """首次部署：从旧 output/config 复制到 data/orb/ml/。"""
    ensure_model_dirs()
    copied = []
    pairs = [
        (_LEGACY_V2 / "breakout_samples.json", SAMPLES_JSON),
        (_LEGACY_V2 / "breakout_gbm.pkl", GBM_PKL),
        (_LEGACY_V2 / "breakout_gbm.json", GBM_META),
        (_LEGACY_V2 / "symbol_breakout_profiles.json", PROFILES_JSON),
        (_LEGACY_V2 / "model_manifest.json", MANIFEST_JSON),
        (_LEGACY_V2 / "breakout_gbm_train_report.json", GBM_TRAIN_REPORT),
        (_LEGACY_OUTPUT / "orb_shared_true_breakout_model.json", LOGISTIC_TRUE_JSON),
        (_LEGACY_OUTPUT / "orb_true_breakout_model.json", LOGISTIC_TRUE_JSON),
        (_LEGACY_OUTPUT / "orb_shared_fake_breakout_model.json", LOGISTIC_FAKE_JSON),
        (_LEGACY_OUTPUT / "orb_shared_breakout_samples.json", SAMPLES_JSON),
        (_LEGACY_OUTPUT / "orb_shared_breakout_gbm.pkl", GBM_PKL),
        (_LEGACY_OUTPUT / "symbol_breakout_profiles.json", PROFILES_JSON),
    ]
    for src, dst in pairs:
        if src.is_file() and not dst.is_file():
            shutil.copy2(src, dst)
            copied.append(str(dst.relative_to(PROJECT_ROOT)))
    return {"bootstrapped": copied}


def _persist_monthly_report(report: Dict[str, Any], *, archive_tag: str = "") -> str:
    promoted = report.get("promoted")
    auto_promoted = isinstance(promoted, dict)
    manifest_path = write_manifest(
        extra={
            "monthly_tag": archive_tag or report.get("archive_tag", ""),
            "auto_promoted": auto_promoted,
            "monthly_report_status": (
                "validation_failed"
                if report.get("validation", {}).get("passed") is False
                else report.get("promote_skipped") or ("promoted" if auto_promoted else "staging_only")
            ),
        }
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["monthly_report"] = report
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(manifest_path)


def run_training_pipeline(
    *,
    days: float = 180.0,
    holdout_days: int = 10,
    bootstrap_only: bool = False,
    skip_collect: bool = False,
    skip_archive: bool = False,
    skip_fetch_klines: bool = False,
    skip_validate: bool = False,
    skip_promote: bool = False,
    skip_gate_tune: bool = False,
    tag: str = "",
    symbols_file: Optional[Path] = None,
    auto: Optional[bool] = None,
) -> Dict[str, Any]:
    """月度重训；staging 验收通过后自动 promote，失败则保留 production。"""
    t0 = time.time()
    ensure_model_dirs()
    auto_cfg = MlAutoConfig.from_env()
    use_auto = auto if auto is not None else True

    report: Dict[str, Any] = {
        "started_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "auto": use_auto,
        "auto_config": {
            "fetch_klines": auto_cfg.auto_fetch_klines,
            "validate": auto_cfg.auto_validate,
            "promote": auto_cfg.auto_promote,
            "gate_suggest": auto_cfg.auto_gate_suggest,
            "gate_apply": auto_cfg.auto_gate_apply,
        },
    }
    boot = bootstrap_from_legacy()
    if boot.get("bootstrapped"):
        report["bootstrap"] = boot

    if bootstrap_only:
        report["manifest"] = str(write_manifest(extra={"bootstrap_only": True}))
        report["elapsed_sec"] = round(time.time() - t0, 1)
        return report

    py = sys.executable
    sym_file = symbols_file or resolve_train_symbols_path()
    archive_tag = tag or datetime.now(timezone.utc).strftime("%Y%m")
    sample_days = float(days)
    kline_days = max(sample_days, float(auto_cfg.kline_days))

    if use_auto and auto_cfg.auto_fetch_klines and not skip_fetch_klines:
        kline_summary = fetch_universe_klines(
            symbols_file=sym_file,
            days=kline_days,
            skip_existing=auto_cfg.kline_skip_existing,
        )
        report["kline_fetch"] = kline_summary
        report["kline_days"] = kline_days
        if auto_cfg.fail_on_kline_errors and kline_summary.get("errors"):
            raise RuntimeError(f"kline fetch errors: {kline_summary['errors'][:3]}")

    ensure_staging_dirs()
    staging_samples = staging_samples_path()
    staging_gbm = staging_gbm_pkl_path()
    staging_meta = staging_gbm_meta_path()
    staging_report = staging_train_report_path()
    staging_profiles = staging_profiles_path()

    if not skip_collect:
        _run(
            [
                py,
                str(_TOOLS_ML / "collect_shared_breakout_samples.py"),
                "--days",
                str(sample_days),
                "--json-out",
                str(staging_samples),
                "--symbols-file",
                str(sym_file),
            ],
            label="collect samples",
        )
    elif SAMPLES_JSON.is_file() and not staging_samples.is_file():
        shutil.copy2(SAMPLES_JSON, staging_samples)

    if not staging_samples.is_file():
        raise FileNotFoundError(
            f"no staging samples at {staging_samples}; run collect or copy production samples"
        )

    _run(
        [
            py,
            str(_TOOLS_ML / "relabel_hold30_samples.py"),
            "--samples",
            str(staging_samples),
            "--json-out",
            str(staging_samples),
        ],
        label="relabel hold30",
    )
    _run(
        [
            py,
            str(_TOOLS_ML / "train_breakout_gbm.py"),
            "--samples",
            str(staging_samples),
            "--holdout-days",
            str(holdout_days),
            "--out",
            str(staging_gbm),
            "--meta-out",
            str(staging_meta),
            "--report-out",
            str(staging_report),
        ],
        label="train GBM",
    )
    _run(
        [
            py,
            str(_TOOLS_ML / "build_symbol_profiles.py"),
            "--samples",
            str(staging_samples),
            "--json-out",
            str(staging_profiles),
        ],
        label="build symbol profiles",
    )

    if staging_report.is_file():
        try:
            report["train"] = json.loads(staging_report.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    do_validate = use_auto and auto_cfg.auto_validate and not skip_validate
    if do_validate:
        passed, reasons, detail = validate_staging_artifacts(
            gbm_pkl=staging_gbm,
            gbm_meta=staging_meta,
            profiles=staging_profiles,
            samples=staging_samples,
            train_report=staging_report,
            cfg=auto_cfg,
        )
        report["validation"] = {"passed": passed, "reasons": reasons, "detail": detail}
        if not passed:
            report["promoted"] = False
            report["elapsed_sec"] = round(time.time() - t0, 1)
            _persist_monthly_report(report, archive_tag=archive_tag)
            raise TrainingValidationError(reasons, detail)

    skipped_validation = skip_validate or not (use_auto and auto_cfg.auto_validate)
    do_promote = use_auto and auto_cfg.auto_promote and not skip_promote
    if skipped_validation and do_promote and not auto_cfg.allow_promote_without_validate:
        report["promoted"] = False
        report["promote_skipped"] = "validation_skipped"
    elif do_promote or not use_auto:
        if do_promote:
            report["promoted"] = promote_staging_to_production(tag=archive_tag)
            clear_staging()
        elif not use_auto:
            from orb.ml.model.paths import GBM_META, GBM_PKL, PROFILES_JSON
            from orb.ml.live_bundle import sync_live_bundle_from_ml_models

            shutil.copy2(staging_gbm, GBM_PKL)
            shutil.copy2(staging_meta, GBM_META)
            shutil.copy2(staging_report, GBM_TRAIN_REPORT)
            shutil.copy2(staging_profiles, PROFILES_JSON)
            shutil.copy2(staging_samples, SAMPLES_JSON)
            live_synced = sync_live_bundle_from_ml_models(overwrite=True)
            report["promoted"] = "direct_overwrite"
            if live_synced:
                report["orb_live_sync"] = live_synced

    promoted_ok = isinstance(report.get("promoted"), dict) or report.get("promoted") == "direct_overwrite"
    if not skip_archive and promoted_ok:
        report["archive"] = str(archive_snapshot(archive_tag))

    if promoted_ok and use_auto and auto_cfg.auto_gate_suggest and not skip_gate_tune:
        try:
            from orb.ml.model.gate_tune import tune_gate_after_promote

            print("[orb_model] gate tune after promote ...", flush=True)
            report["gate_tune"] = tune_gate_after_promote(
                train_report=dict(report.get("train") or {}),
                archive_tag=archive_tag,
                cfg=auto_cfg,
            )
        except Exception as exc:
            report["gate_tune"] = {"action": "error", "error": str(exc)}
            print(f"[orb_model] gate tune failed: {exc}", flush=True)

    report["elapsed_sec"] = round(time.time() - t0, 1)
    report["manifest"] = _persist_monthly_report(report, archive_tag=archive_tag)
    return report
