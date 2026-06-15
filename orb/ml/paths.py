"""ORB ML 路径：运行时配置/模型在 data/ 与 config/；output/ 仅本地回测报告写入。"""

from __future__ import annotations

from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PKG_ROOT.parent

CONFIG_V2 = PROJECT_ROOT / "config" / "orb" / "v2"

# 回测 / 调参报告（只写不读作生产配置）
V2_OUTPUT = PROJECT_ROOT / "output" / "orb" / "v2"
V2_EVAL = V2_OUTPUT / "eval"
V2_LIVE_GATE_EVAL = V2_EVAL / "live_gate_eval.json"
V2_LIVE_GATE_LAST30D = V2_EVAL / "live_gate_last30d.json"
V2_LIVE_GATE_SWEEP = V2_EVAL / "live_gate_sweep.json"
V2_GBM_SWEEP = V2_EVAL / "gbm_sweep.json"

# 生产环境变量若指向 data/ Volume，应忽略并告警（见 is_risky_production_data_path）
RISKY_PRODUCTION_ENV_VARS = (
    "ORB_LIVE_BUNDLE_ROOT",
    "ORB_V2_SYMBOLS_FILE",
    "ORB_V2_GATE_CONFIG",
    "ORB_V2_GBM_PATH",
    "ORB_V2_PROFILES_PATH",
)


def is_risky_production_data_path(raw: str) -> bool:
    """git/config/orb_live 安全；data/ 与旧 data/orb/live 会在 Volume 下被盖住。"""
    norm = raw.replace("\\", "/").strip().rstrip("/").lower()
    if not norm:
        return False
    if "orb_live" in norm or norm.startswith("config/"):
        return False
    if norm.endswith("data/orb/live") or norm == "orb/live":
        return True
    return norm.startswith("data/") or "/data/orb/" in f"/{norm}/"


def production_env_warnings() -> list[str]:
    import os

    fixes = {
        "ORB_LIVE_BUNDLE_ROOT": "删除该变量，默认 orb_live/",
        "ORB_V2_SYMBOLS_FILE": "删除该变量，默认 config/orb/v2/symbols.txt",
        "ORB_V2_GATE_CONFIG": "删除该变量，默认 orb_live/live_gate.json",
        "ORB_V2_GBM_PATH": "删除该变量，默认 orb_live/breakout_gbm.pkl",
        "ORB_V2_PROFILES_PATH": "删除该变量，默认 orb_live/symbol_breakout_profiles.json",
    }
    out: list[str] = []
    for name in RISKY_PRODUCTION_ENV_VARS:
        raw = (os.getenv(name) or "").strip()
        if raw and is_risky_production_data_path(raw):
            out.append(f"{name}={raw} 指向 data/（Volume 会盖住）。{fixes[name]}")
    return out


def resolve_production_env_path(env_var: str, default: Path) -> Path:
    """解析生产参数路径：忽略指向 data/ Volume 的 env 覆盖。"""
    import os

    raw = (os.getenv(env_var) or "").strip()
    if raw:
        if is_risky_production_data_path(raw):
            return default
        p = Path(raw)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.is_file():
            return p
    return default


def _first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.is_file():
            return p
    return paths[0]


def ensure_v2_eval_dirs() -> None:
    V2_EVAL.mkdir(parents=True, exist_ok=True)


def ensure_v2_dirs() -> None:
    """兼容旧工具名。"""
    ensure_v2_eval_dirs()


def default_shared_samples_path() -> Path:
    from orb.ml.model.paths import resolve_samples_path

    return resolve_samples_path()


def default_shared_true_model_path() -> Path:
    from orb.ml.model.paths import resolve_logistic_true_path

    return resolve_logistic_true_path()


def default_shared_fake_model_path() -> Path:
    from orb.ml.model.paths import LOGISTIC_FAKE_JSON

    return LOGISTIC_FAKE_JSON


def default_gbm_path() -> Path:
    from orb.ml.model.paths import resolve_gbm_path

    return resolve_gbm_path()


def default_profiles_path() -> Path:
    from orb.ml.model.paths import resolve_profiles_path

    return resolve_profiles_path()


def default_gbm_train_report_path() -> Path:
    from orb.ml.model.paths import resolve_train_report_path

    return resolve_train_report_path()


def default_live_gate_eval_path() -> Path:
    return V2_LIVE_GATE_EVAL


def default_live_gate_last30d_path() -> Path:
    return V2_LIVE_GATE_LAST30D


# 训练脚本默认写入 data/orb/ml（非 output）
def default_gbm_write_paths() -> tuple[Path, Path, Path]:
    from orb.ml.model.paths import GBM_META, GBM_PKL, GBM_TRAIN_REPORT

    return GBM_PKL, GBM_META, GBM_TRAIN_REPORT
