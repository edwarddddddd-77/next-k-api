"""ORB ML 默认产物路径（V1 研究 + V2 生产 + 旧路径回退）。"""

from __future__ import annotations

from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PKG_ROOT.parent

V1_OUTPUT = PROJECT_ROOT / "output" / "orb" / "v1"
V1_RESEARCH = V1_OUTPUT / "research"
V2_OUTPUT = PROJECT_ROOT / "output" / "orb" / "v2"
V2_EVAL = V2_OUTPUT / "eval"

LEGACY_OUTPUT = PROJECT_ROOT / "output"
LEGACY_V2_OUTPUT = PROJECT_ROOT / "output" / "orb_v2"

CONFIG_V1 = PROJECT_ROOT / "config" / "orb" / "v1"
CONFIG_V2 = PROJECT_ROOT / "config" / "orb" / "v2"

# V1 研究训练默认写入
DEFAULT_SAMPLES = V1_OUTPUT / "breakout_samples.json"
DEFAULT_GBM = V1_OUTPUT / "breakout_gbm.pkl"
DEFAULT_GBM_META = V1_OUTPUT / "breakout_gbm.json"
DEFAULT_GBM_TRAIN_REPORT = V1_OUTPUT / "breakout_gbm_train_report.json"
DEFAULT_PROFILES = V1_OUTPUT / "symbol_breakout_profiles.json"
DEFAULT_TRUE_MODEL = V1_OUTPUT / "true_breakout_model.json"
DEFAULT_FAKE_MODEL = V1_OUTPUT / "fake_breakout_model.json"

# V2 回测评估（ ephemeral，仍在 output）
V2_LIVE_GATE_EVAL = V2_EVAL / "live_gate_eval.json"
V2_LIVE_GATE_LAST30D = V2_EVAL / "live_gate_last30d.json"
V2_LIVE_GATE_SWEEP = V2_EVAL / "live_gate_sweep.json"

# 生产模型已迁至 data/orb/ml/ — 见 orb.ml.model.paths


def _first_existing(*paths: Path) -> Path:
    for p in paths:
        if p.is_file():
            return p
    return paths[0]


def ensure_v1_dirs() -> None:
    V1_OUTPUT.mkdir(parents=True, exist_ok=True)
    V1_RESEARCH.mkdir(parents=True, exist_ok=True)


def ensure_v2_dirs() -> None:
    V2_OUTPUT.mkdir(parents=True, exist_ok=True)
    V2_EVAL.mkdir(parents=True, exist_ok=True)


def default_shared_samples_path() -> Path:
    from orb.ml.model.paths import resolve_samples_path

    return resolve_samples_path()


def default_shared_true_model_path() -> Path:
    from orb.ml.model.paths import resolve_logistic_true_path

    return resolve_logistic_true_path()


def default_shared_fake_model_path() -> Path:
    return _first_existing(
        DEFAULT_FAKE_MODEL,
        LEGACY_OUTPUT / "orb_shared_fake_breakout_model.json",
        LEGACY_OUTPUT / "orb_fake_breakout_model.json",
    )


def default_gbm_path() -> Path:
    """V2 实盘加载优先，回退 V1 研究模型。"""
    from orb.ml.model.paths import resolve_gbm_path

    return resolve_gbm_path()


def default_profiles_path() -> Path:
    from orb.ml.model.paths import resolve_profiles_path

    return resolve_profiles_path()


def default_live_gate_eval_path() -> Path:
    return _first_existing(
        V2_LIVE_GATE_EVAL,
        LEGACY_OUTPUT / "live_gate_eval.json",
    )


def default_live_gate_last30d_path() -> Path:
    return _first_existing(
        V2_LIVE_GATE_LAST30D,
        LEGACY_OUTPUT / "live_gate_last30d.json",
    )
