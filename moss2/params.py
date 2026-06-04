"""DecisionParams：默认值来自 factory params_schema（HL / EN 各一份）。"""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from moss2.config import FactoryVariant

_SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"
DEFAULT_TEMPLATE = "balanced"

# 模板仅覆盖五维信号权重（与 factory SKILL 四风格一致）
_TEMPLATE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "balanced": {
        "trend_weight": 0.30,
        "momentum_weight": 0.25,
        "mean_revert_weight": 0.15,
        "volume_weight": 0.15,
        "volatility_weight": 0.15,
    },
    "momentum": {
        "trend_weight": 0.20,
        "momentum_weight": 0.50,
        "mean_revert_weight": 0.05,
        "volume_weight": 0.15,
        "volatility_weight": 0.10,
    },
    "trend": {
        "trend_weight": 0.50,
        "momentum_weight": 0.20,
        "mean_revert_weight": 0.05,
        "volume_weight": 0.15,
        "volatility_weight": 0.10,
    },
    "mean_revert": {
        "trend_weight": 0.15,
        "momentum_weight": 0.15,
        "mean_revert_weight": 0.45,
        "volume_weight": 0.15,
        "volatility_weight": 0.10,
    },
}


def _decision_params_cls(variant: FactoryVariant):
    if variant == "en":
        from moss2.variants.en.core.decision import DecisionParams

        return DecisionParams
    from moss2.variants.hl.core.decision import DecisionParams

    return DecisionParams


@lru_cache(maxsize=2)
def load_params_schema(variant: str) -> dict:
    v: FactoryVariant = "en" if variant == "en" else "hl"
    name = "params_schema_hl.json" if v == "hl" else "params_schema_en.json"
    path = _SCHEMA_DIR / name
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"params": {}}


@lru_cache(maxsize=2)
def schema_default_params(variant: str) -> Dict[str, Any]:
    """params_schema 中每个字段的 default。"""
    schema = load_params_schema(variant)
    out: Dict[str, Any] = {}
    for key, spec in (schema.get("params") or {}).items():
        if "default" in spec:
            out[key] = spec["default"]
    return out


@lru_cache(maxsize=2)
def personality_field_names(variant: str) -> Tuple[str, ...]:
    schema = load_params_schema(variant)
    return tuple(
        k
        for k, spec in (schema.get("params") or {}).items()
        if spec.get("category") == "personality"
    )


@lru_cache(maxsize=2)
def tactical_field_names(variant: str) -> Tuple[str, ...]:
    schema = load_params_schema(variant)
    return tuple(
        k
        for k, spec in (schema.get("params") or {}).items()
        if spec.get("category") == "tactical"
    )


def list_templates() -> List[str]:
    return sorted(_TEMPLATE_WEIGHTS.keys())


def resolve_params_dict(raw: Optional[dict], *, variant: FactoryVariant = "hl") -> dict:
    """合并 + DecisionParams 归一化（权重、钳制）。"""
    DecisionParams = _decision_params_cls(variant)
    clean = {k: v for k, v in (raw or {}).items() if v is not None}
    p = DecisionParams.from_dict(clean)
    p.normalize_weights()
    return p.to_dict()


def build_initial_params(
    template: str = DEFAULT_TEMPLATE,
    overrides: Optional[dict] = None,
    *,
    variant: FactoryVariant = "hl",
) -> dict:
    """完整默认参数字典（schema 默认 + 模板权重 + 可选覆盖）。"""
    merged = copy.deepcopy(schema_default_params(variant))
    tkey = (template or DEFAULT_TEMPLATE).strip().lower()
    merged.update(_TEMPLATE_WEIGHTS.get(tkey, _TEMPLATE_WEIGHTS[DEFAULT_TEMPLATE]))
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})
    return resolve_params_dict(merged, variant=variant)


def split_profile_params(
    full_params: dict, *, variant: FactoryVariant = "hl"
) -> Tuple[dict, dict]:
    """拆成 initial（性格）与 tactical（战术），创建 Profile 时用。"""
    resolved = resolve_params_dict(full_params, variant=variant)
    pers = {k: resolved[k] for k in personality_field_names(variant) if k in resolved}
    tact = {k: resolved[k] for k in tactical_field_names(variant) if k in resolved}
    return pers, tact


def extract_tactical_params(params: dict, *, variant: FactoryVariant = "hl") -> dict:
    tact = tactical_field_names(variant)
    return {k: params[k] for k in tact if k in params}


def extract_personality_params(params: dict, *, variant: FactoryVariant = "hl") -> dict:
    pers = personality_field_names(variant)
    return {k: params[k] for k in pers if k in params}


def merge_profile_params(profile: dict) -> dict:
    base = json.loads(profile["initial_params_json"])
    tactical = json.loads(profile.get("tactical_params_json") or "{}")
    from moss2.config import profile_variant

    variant: FactoryVariant = profile_variant(profile)
    merged = copy.deepcopy(base)
    merged.update(tactical)
    resolved = resolve_params_dict(merged, variant=variant)
    if variant == "hl":
        from moss2.variants.hl.core.leverage_caps import cap_params_for_symbol

        resolved = cap_params_for_symbol(resolved, str(profile.get("symbol") or ""))
    return resolved


def default_params_bundle(
    *,
    template: str = DEFAULT_TEMPLATE,
    variant: FactoryVariant = "hl",
    overrides: Optional[dict] = None,
) -> Dict[str, Any]:
    """API / 看板：一次返回 schema 默认、合并后全量、性格/战术拆分。"""
    merged = build_initial_params(template, overrides, variant=variant)
    initial, tactical = split_profile_params(merged, variant=variant)
    return {
        "template": (template or DEFAULT_TEMPLATE).strip().lower(),
        "variant": variant,
        "schema_defaults": schema_default_params(variant),
        "merged_params": merged,
        "initial_params": initial,
        "tactical_params": tactical,
        "personality_fields": list(personality_field_names(variant)),
        "tactical_fields": list(tactical_field_names(variant)),
    }
