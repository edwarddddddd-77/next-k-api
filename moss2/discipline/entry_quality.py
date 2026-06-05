"""扫描开仓质量：余量 + 多根 composite 确认（对齐 Moss Quant 纸面逻辑）。"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Tuple

import pandas as pd

from moss2 import config as cfg


def entry_confirm_bars() -> int:
    return max(0, int(cfg.MOSS2_ENTRY_CONFIRM_BARS or 0))


def effective_entry_margin() -> float:
    if not cfg.MOSS2_ENTRY_QUALITY_ENABLED:
        return 0.0
    return max(0.0, float(cfg.MOSS2_ENTRY_MARGIN or 0))


def entry_confirm_relax() -> float:
    """2K 时第 2 根相对第 1 根允许的最大回撤（composite 绝对值）。"""
    if not cfg.MOSS2_ENTRY_QUALITY_ENABLED:
        return 0.0
    return max(0.0, float(getattr(cfg, "MOSS2_ENTRY_CONFIRM_RELAX", 0) or 0))


def effective_entry_threshold(entry_threshold: float) -> float:
    """质量模式下钳制阈值下限，避免 DB 里旧战术参数（如 0.26）绕过。"""
    th = float(entry_threshold)
    if cfg.MOSS2_ENTRY_QUALITY_ENABLED:
        th = max(th, float(cfg.MOSS2_ENTRY_THRESHOLD_FLOOR or 0.40))
    return th


def params_for_quality_backtest(params: dict) -> dict:
    """回测用更高有效阈值（阈值+余量），与扫描门槛大致对齐（不含多 K 确认）。"""
    if not cfg.MOSS2_ENTRY_QUALITY_ENABLED:
        return params
    out = copy.deepcopy(params)
    th = effective_entry_threshold(float(out.get("entry_threshold") or 0.40))
    out["entry_threshold"] = round(th + effective_entry_margin(), 4)
    return out


def _variant_modules(variant: str):
    v = str(variant or "").lower()
    if v == "en":
        from moss2.variants.en.core.decision import DecisionParams
        from moss2.variants.en.core.regime import classify_regime

        cap = lambda p, s: p
    else:
        from moss2.variants.hl.core.decision import DecisionParams
        from moss2.variants.hl.core.regime import classify_regime
        from moss2.variants.hl.core.leverage_caps import cap_params_for_symbol

        cap = cap_params_for_symbol
    return DecisionParams, classify_regime, cap


def _composite_at_bar_fn(variant: str):
    v = str(variant or "").lower()
    if v == "hl":
        from moss2.variants.hl.core.decision import _composite_at_bar

        return _composite_at_bar
    from moss2.variants.en.core.decision import _composite_at_bar

    return _composite_at_bar


def _trailing_composites(
    df: pd.DataFrame, params_dict: dict, variant: str, *, bars: int
) -> Tuple[List[float], str]:
    DecisionParams, classify_regime, cap = _variant_modules(variant)
    sym = str(params_dict.get("_symbol") or "")
    clean = {k: v for k, v in params_dict.items() if not str(k).startswith("_")}
    clean = cap(dict(clean), sym)
    params = DecisionParams.from_dict(clean)
    params.normalize_weights()
    regime = classify_regime(df, version=cfg.MOSS2_REGIME_VERSION)
    regime_label = str(regime.iloc[-1]) if len(regime) else "SIDEWAYS"

    composite_at = _composite_at_bar_fn(variant)
    n = len(df)
    start_min = max(int(params.slow_ma_period), 50)
    if n <= start_min:
        return [], regime_label

    need = max(1, int(bars))
    cache: dict = {}
    out: List[float] = []
    for i in range(max(start_min, n - need), n):
        out.append(float(composite_at(df, params, regime, i, cache)))
    return out, regime_label


def _passes_entry_at_composites(
    composites: List[float],
    *,
    entry_threshold: float,
    entry_margin: float,
    confirm_bars: int,
    confirm_relax: float = 0.0,
) -> Tuple[int, str]:
    if not composites:
        return 0, "composite_unavailable"
    th = float(entry_threshold)
    long_eff = th + float(entry_margin)
    short_eff = th + float(entry_margin)
    c_last = float(composites[-1])
    raw_confirm = int(confirm_bars if confirm_bars is not None else 1)
    k = 1 if raw_confirm <= 0 else raw_confirm
    relax = max(0.0, float(confirm_relax or 0))
    if len(composites) < k:
        return 0, "confirm_bars_insufficient"

    def long_ok() -> bool:
        tail = [float(x) for x in composites[-k:]]
        if not all(x > long_eff for x in tail):
            return False
        if k >= 2 and tail[-1] < tail[-2] - relax:
            return False
        return True

    def short_ok() -> bool:
        tail = [float(x) for x in composites[-k:]]
        if not all(x < -short_eff for x in tail):
            return False
        if k >= 2 and tail[-1] > tail[-2] + relax:
            return False
        return True

    if long_ok():
        return 1, "signal_long"
    if short_ok():
        return -1, "signal_short"
    if abs(c_last) <= min(long_eff, short_eff):
        return 0, "composite_below_threshold"
    if c_last > 0:
        return 0, "long_margin_or_confirm_failed"
    return 0, "short_margin_or_confirm_failed"


def evaluate_open_signal(
    df: pd.DataFrame,
    params_dict: dict,
    variant: str,
    *,
    entry_threshold: float,
) -> Dict[str, Any]:
    """返回 signal、composite、regime、reason 及调试字段。"""
    th = effective_entry_threshold(entry_threshold)
    confirm = entry_confirm_bars() if cfg.MOSS2_ENTRY_QUALITY_ENABLED else 1
    margin = effective_entry_margin()
    relax = entry_confirm_relax()
    need = max(1, confirm)
    composites, regime_label = _trailing_composites(
        df, params_dict, variant, bars=need
    )
    sig, reason = _passes_entry_at_composites(
        composites,
        entry_threshold=th,
        entry_margin=margin,
        confirm_bars=confirm,
        confirm_relax=relax,
    )
    comp = float(composites[-1]) if composites else 0.0
    eff = th + margin
    raw_th = float(entry_threshold)
    return {
        "signal": sig,
        "composite": comp,
        "regime": regime_label,
        "reason": reason,
        "entry_threshold": round(th, 4),
        "entry_threshold_raw": round(raw_th, 4),
        "entry_threshold_eff": round(eff, 4),
        "entry_margin": margin,
        "confirm_bars": confirm,
        "confirm_relax": relax,
        "entry_quality_enabled": bool(cfg.MOSS2_ENTRY_QUALITY_ENABLED),
        "threshold_clamped": th > raw_th + 1e-9,
    }
