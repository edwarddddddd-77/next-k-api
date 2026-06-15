"""ORB 实盘参数包：Gate + 大模型，统一在 orb_live/ 人工替换。"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Iterable, List

from orb.ml.paths import PROJECT_ROOT

# 实盘唯一参数目录（在 data/ 外，不受 DATA_DIR Volume 影响）
LIVE_BUNDLE_DIR = PROJECT_ROOT / "orb_live"

BUNDLE_FILENAMES = (
    "live_gate.json",
    "breakout_gbm.pkl",
    "breakout_gbm.json",
    "symbol_breakout_profiles.json",
    "breakout_gbm_train_report.json",
    "bundle_manifest.json",
)

REQUIRED_FILENAMES = (
    "live_gate.json",
    "breakout_gbm.pkl",
    "symbol_breakout_profiles.json",
)


def _env_override_issue() -> str:
    """ORB_LIVE_BUNDLE_ROOT 指向旧路径时给出明确修复提示。"""
    raw = (os.getenv("ORB_LIVE_BUNDLE_ROOT") or "").strip()
    if not raw:
        return ""
    norm = raw.replace("\\", "/").rstrip("/").lower()
    if norm.endswith("data/orb/live") or norm == "orb/live":
        return (
            f"ORB_LIVE_BUNDLE_ROOT={raw} 仍在使用旧目录 data/orb/live（Volume 会盖住模型）。"
            "请在 Railway 删除该环境变量，让程序默认读 orb_live/。"
        )
    return ""


def live_bundle_root() -> Path:
    """实盘参数根目录（可用 ORB_LIVE_BUNDLE_ROOT 覆盖）。"""
    raw = (os.getenv("ORB_LIVE_BUNDLE_ROOT") or "").strip()
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else PROJECT_ROOT / p
    return LIVE_BUNDLE_DIR


def _bundle_file(name: str) -> Path:
    return live_bundle_root() / name


def live_gate_json() -> Path:
    return _bundle_file("live_gate.json")


def live_gbm_pkl() -> Path:
    return _bundle_file("breakout_gbm.pkl")


def live_gbm_meta() -> Path:
    return _bundle_file("breakout_gbm.json")


def live_profiles_json() -> Path:
    return _bundle_file("symbol_breakout_profiles.json")


def live_train_report() -> Path:
    return _bundle_file("breakout_gbm_train_report.json")


def live_manifest() -> Path:
    return live_bundle_root() / "bundle_manifest.json"


BUNDLE_ARTIFACTS = (live_gate_json, live_gbm_meta, live_profiles_json)


def ensure_live_bundle_dir() -> Path:
    root = live_bundle_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.is_file():
            return p
    return paths[0]


def resolve_live_gate_path() -> Path:
    raw = (os.getenv("ORB_V2_GATE_CONFIG") or "").strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_file():
            return p
    return live_gate_json()


def resolve_live_gbm_path() -> Path:
    raw = (os.getenv("ORB_V2_GBM_PATH") or "").strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_file():
            return p
    return live_gbm_pkl()


def resolve_live_gbm_meta_path() -> Path:
    return live_gbm_meta()


def resolve_live_profiles_path() -> Path:
    raw = (os.getenv("ORB_V2_PROFILES_PATH") or "").strip()
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_file():
            return p
    return live_profiles_json()


def _rel_path(p: Path) -> str:
    try:
        return str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def _is_under_live_bundle(p: Path) -> bool:
    if not p.is_file():
        return False
    try:
        p.resolve().relative_to(live_bundle_root().resolve())
        return True
    except ValueError:
        return False


def bundle_status(*, relative_paths: bool = False) -> dict:
    ensure_live_bundle_dir()
    from orb.ml.model.paths import resolve_gbm_path, resolve_profiles_path

    root = live_bundle_root()
    gbm_p = resolve_gbm_path()
    prof_p = resolve_profiles_path()
    gate_p = resolve_live_gate_path()
    fmt = _rel_path if relative_paths else str
    return {
        "live_bundle_root": fmt(root),
        "gate": fmt(gate_p),
        "gate_exists": gate_p.is_file(),
        "gbm": fmt(gbm_p),
        "gbm_exists": gbm_p.is_file(),
        "profiles": fmt(prof_p),
        "profiles_exists": prof_p.is_file(),
        "using_live_bundle_gate": _is_under_live_bundle(gate_p),
        "using_live_bundle_gbm": _is_under_live_bundle(gbm_p),
        "using_live_bundle_profiles": _is_under_live_bundle(prof_p),
    }


def _artifact_rows(*, relative_paths: bool = True) -> list[dict]:
    """各产物路径与是否存在（供前端 / 运维诊断）。"""
    from orb.ml.model.paths import resolve_gbm_path, resolve_profiles_path

    fmt = _rel_path if relative_paths else str
    rows = [
        ("live_gate.json", resolve_live_gate_path()),
        ("breakout_gbm.pkl", resolve_gbm_path()),
        ("symbol_breakout_profiles.json", resolve_profiles_path()),
    ]
    return [
        {
            "name": name,
            "live_path": fmt(p),
            "live_exists": p.is_file(),
            "active_path": fmt(p),
            "active_exists": p.is_file(),
            "from_live": _is_under_live_bundle(p),
        }
        for name, p in rows
    ]


def ensure_live_bundle_on_startup() -> list[str]:
    """启动时若 orb_live/ 缺文件，从训练目录 bootstrap。"""
    if all(_bundle_file(name).is_file() for name in REQUIRED_FILENAMES):
        return []
    return bootstrap_from_legacy(overwrite=False)


def log_live_bundle_startup() -> None:
    """API 启动时打印 Live 包自检（Railway 日志可见）。"""
    import logging

    log = logging.getLogger(__name__)
    hint = live_bundle_hint()
    sev = hint.get("severity", "?")
    log.info(
        "ORB live bundle [%s] root=%s env=%s",
        sev,
        hint.get("root"),
        (os.getenv("ORB_LIVE_BUNDLE_ROOT") or "").strip() or "(default orb_live/)",
    )
    for row in hint.get("artifacts") or []:
        mark = "OK" if row.get("live_exists") else "MISSING"
        log.info("  %-36s %s", row["name"], mark)
    env_issue = _env_override_issue()
    if env_issue:
        log.warning("ORB live bundle: %s", env_issue)
    if sev != "ok":
        log.warning("ORB live bundle: %s", hint.get("message"))
        for step in hint.get("deploy_steps") or []:
            log.warning("  → %s", step)


def live_bundle_hint() -> dict:
    """前端 / 运维：Live 包是否就绪。"""
    st = bundle_status(relative_paths=True)
    gate_ok = bool(st["gate_exists"])
    gbm_ok = bool(st["gbm_exists"])
    prof_ok = bool(st["profiles_exists"])
    ready = gate_ok and gbm_ok and prof_ok
    using = (
        bool(st["using_live_bundle_gate"])
        and bool(st["using_live_bundle_gbm"])
        and bool(st["using_live_bundle_profiles"])
    )

    missing: list[str] = []
    if not gate_ok:
        missing.append("live_gate.json")
    if not gbm_ok:
        missing.append("breakout_gbm.pkl")
    if not prof_ok:
        missing.append("symbol_breakout_profiles.json")

    env_issue = _env_override_issue()
    deploy_steps = [
        "将 live_gate.json / breakout_gbm.pkl / symbol_breakout_profiles.json 放入 orb_live/",
        "确认 git 已提交 orb_live/breakout_gbm.pkl",
        "在 Railway 重新 Deploy next-k-api 服务",
    ]
    if env_issue:
        deploy_steps.insert(0, "删除 Railway 环境变量 ORB_LIVE_BUNDLE_ROOT（留空即读 orb_live/）")

    if not ready:
        message = (
            f"orb_live/ 不完整（缺 {', '.join(missing)}）。"
            "请将 Gate + 模型文件放入 orb_live/ 后 git 提交并部署。"
        )
        if env_issue:
            message += f" {env_issue}"
        severity = "block"
    elif not using:
        message = f"参数未从 orb_live/ 加载（当前 root={st['live_bundle_root']}）"
        if env_issue:
            message += f" {env_issue}"
        severity = "warn"
        deploy_steps = ["检查 ORB_LIVE_BUNDLE_ROOT 是否指向 orb_live/"]
        if env_issue:
            deploy_steps.insert(0, "删除 ORB_LIVE_BUNDLE_ROOT，使用默认 orb_live/")
    else:
        message = f"orb_live/ 就绪 · 直接覆盖文件即可更新，下次 scan 自动生效"
        severity = "ok"
        deploy_steps = []

    return {
        "ok": True,
        "ready": ready,
        "using_live_bundle": using,
        "bundle_dir": "orb_live",
        "root": st["live_bundle_root"],
        "env_override_issue": env_issue,
        "active_gate": st["gate"],
        "active_gbm": st["gbm"],
        "active_profiles": st["profiles"],
        "gate_exists": gate_ok,
        "gbm_exists": gbm_ok,
        "profiles_exists": prof_ok,
        "missing_in_live_bundle": missing,
        "using_live_bundle_gate": bool(st["using_live_bundle_gate"]),
        "using_live_bundle_gbm": bool(st["using_live_bundle_gbm"]),
        "using_live_bundle_profiles": bool(st["using_live_bundle_profiles"]),
        "artifacts": _artifact_rows(relative_paths=True),
        "orb_live_bundle_root_env": (os.getenv("ORB_LIVE_BUNDLE_ROOT") or "").strip(),
        "deploy_steps": deploy_steps,
        "message": message,
        "severity": severity,
    }


def write_bundle_manifest(*, note: str = "", target: Path | None = None) -> Path:
    root = target or live_bundle_root()
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "live_bundle_root": _rel_path(root),
        "note": note or "人工替换包：直接覆盖本目录下文件，Live 下次 scan 自动加载",
        "artifacts": {name: _rel_path(root / name) for name in BUNDLE_FILENAMES[:4]},
        "status": bundle_status(relative_paths=True),
    }
    out = root / "bundle_manifest.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def bootstrap_from_legacy(*, overwrite: bool = False) -> List[str]:
    """从 data/orb/ml/models/ 训练产物复制到 orb_live/（首次初始化用）。"""
    from orb.ml.model.paths import GBM_META, GBM_PKL, GBM_TRAIN_REPORT, PROFILES_JSON

    return sync_live_bundle_from_ml_models(overwrite=overwrite)


def sync_live_bundle_from_ml_models(*, overwrite: bool = True) -> List[str]:
    """将 data/orb/ml/models/ 同步到 orb_live/（训练 promote 后调用）。"""
    from orb.ml.model.paths import GBM_META, GBM_PKL, GBM_TRAIN_REPORT, PROFILES_JSON

    dst_root = live_bundle_root()
    dst_root.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    pairs: Iterable[tuple[Path, Path]] = (
        (GBM_PKL, dst_root / "breakout_gbm.pkl"),
        (GBM_META, dst_root / "breakout_gbm.json"),
        (PROFILES_JSON, dst_root / "symbol_breakout_profiles.json"),
        (GBM_TRAIN_REPORT, dst_root / "breakout_gbm_train_report.json"),
    )
    for src, dst in pairs:
        if src.is_file() and (overwrite or not dst.is_file()):
            shutil.copy2(src, dst)
            try:
                copied.append(str(dst.relative_to(PROJECT_ROOT)))
            except ValueError:
                copied.append(str(dst))
    if copied:
        write_bundle_manifest(note="synced from data/orb/ml/models", target=dst_root)
    return copied
