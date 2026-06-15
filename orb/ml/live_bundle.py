"""Live 人工替换包：随 git / 镜像发布的 data/orb/live/。"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Iterable, List

from orb.ml.paths import PROJECT_ROOT

LIVE_BUNDLE_DIR = PROJECT_ROOT / "data" / "orb" / "live"

BUNDLE_FILENAMES = (
    "live_gate.json",
    "breakout_gbm.pkl",
    "breakout_gbm.json",
    "symbol_breakout_profiles.json",
    "breakout_gbm_train_report.json",
    "bundle_manifest.json",
)


def live_bundle_root() -> Path:
    """运行时读取目录，默认 data/orb/live（可用 ORB_LIVE_BUNDLE_ROOT 覆盖）。"""
    raw = (os.getenv("ORB_LIVE_BUNDLE_ROOT") or "").strip()
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else PROJECT_ROOT / p
    return LIVE_BUNDLE_DIR


def live_gate_json() -> Path:
    return live_bundle_root() / "live_gate.json"


def live_gbm_pkl() -> Path:
    return live_bundle_root() / "breakout_gbm.pkl"


def live_gbm_meta() -> Path:
    return live_bundle_root() / "breakout_gbm.json"


def live_profiles_json() -> Path:
    return live_bundle_root() / "symbol_breakout_profiles.json"


def live_train_report() -> Path:
    return live_bundle_root() / "breakout_gbm_train_report.json"


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
    return _first_existing(
        live_gate_json(),
        PROJECT_ROOT / "config" / "orb" / "v2" / "live_gate.json",
    )


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
    return _first_existing(live_gbm_meta(), live_bundle_root() / "breakout_gbm.json")


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


def bundle_status(*, relative_paths: bool = False) -> dict:
    ensure_live_bundle_dir()
    from orb.ml.model.paths import resolve_gbm_path, resolve_profiles_path
    from orb.v2.paths import resolve_gate_config_path

    root = live_bundle_root()
    gbm_p = resolve_gbm_path()
    prof_p = resolve_profiles_path()
    gate_p = resolve_gate_config_path()
    fmt = _rel_path if relative_paths else str
    return {
        "live_bundle_root": fmt(root),
        "gate": fmt(gate_p),
        "gate_exists": gate_p.is_file(),
        "gbm": fmt(gbm_p),
        "gbm_exists": gbm_p.is_file(),
        "profiles": fmt(prof_p),
        "profiles_exists": prof_p.is_file(),
        "using_live_bundle_gate": gate_p.resolve() == live_gate_json().resolve()
        if gate_p.is_file() and live_gate_json().is_file()
        else False,
        "using_live_bundle_gbm": gbm_p.resolve() == live_gbm_pkl().resolve()
        if gbm_p.is_file() and live_gbm_pkl().is_file()
        else False,
    }


def live_bundle_hint() -> dict:
    """前端 / 运维：Live 包是否就绪、是否从 data/orb/live 加载。"""
    st = bundle_status(relative_paths=True)
    gate_ok = bool(st["gate_exists"])
    gbm_ok = bool(st["gbm_exists"])
    prof_ok = bool(st["profiles_exists"])
    ready = gate_ok and gbm_ok and prof_ok
    using = bool(st["using_live_bundle_gate"]) and bool(st["using_live_bundle_gbm"])

    if not ready:
        missing = []
        if not gate_ok:
            missing.append("Gate")
        if not gbm_ok:
            missing.append("GBM")
        if not prof_ok:
            missing.append("Profiles")
        message = (
            f"Live 包不完整（缺 {' / '.join(missing)}）。"
            "请将文件放入 data/orb/live/ 并提交部署。"
        )
        severity = "block"
    elif not using:
        message = "Gate 或 GBM 未从 data/orb/live 加载，当前使用回退路径。"
        severity = "warn"
    else:
        message = f"Live 包就绪 · {st['live_bundle_root']} · 随 git 部署更新"
        severity = "ok"

    return {
        "ok": True,
        "ready": ready,
        "using_live_bundle": using,
        "root": st["live_bundle_root"],
        "gate_exists": gate_ok,
        "gbm_exists": gbm_ok,
        "profiles_exists": prof_ok,
        "using_live_bundle_gate": bool(st["using_live_bundle_gate"]),
        "using_live_bundle_gbm": bool(st["using_live_bundle_gbm"]),
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
    """从 legacy 路径复制到 data/orb/live/（提交 git 后随部署发布）。"""
    from orb.ml.model.paths import GBM_META, GBM_PKL, GBM_TRAIN_REPORT, PROFILES_JSON
    from orb.ml.paths import CONFIG_V2

    dst_root = live_bundle_root()
    dst_root.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    pairs: Iterable[tuple[Path, Path]] = (
        (CONFIG_V2 / "live_gate.json", dst_root / "live_gate.json"),
        (GBM_PKL, dst_root / "breakout_gbm.pkl"),
        (GBM_META, dst_root / "breakout_gbm.json"),
        (PROFILES_JSON, dst_root / "symbol_breakout_profiles.json"),
        (GBM_TRAIN_REPORT, dst_root / "breakout_gbm_train_report.json"),
    )
    for src, dst in pairs:
        if src.is_file() and (overwrite or not dst.is_file()):
            shutil.copy2(src, dst)
            copied.append(str(dst.relative_to(PROJECT_ROOT)))
    write_bundle_manifest(note="bootstrapped from legacy paths", target=dst_root)
    return copied
