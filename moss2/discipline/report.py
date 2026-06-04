"""组装 discipline 报告。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from moss2.discipline.metrics import kelly_fraction, summary_from_backtest, trade_stats_from_rows


def regime_distribution(regime: pd.Series) -> Dict[str, float]:
    if regime is None or len(regime) == 0:
        return {}
    vc = regime.value_counts(normalize=True)
    return {str(k): round(float(v), 4) for k, v in vc.items()}


def build_discipline_report(
    *,
    summary: dict,
    trades: List[dict],
    regime: Optional[pd.Series] = None,
    template: str = "balanced",
    signal_contrib: Optional[dict] = None,
) -> Dict[str, Any]:
    base = summary_from_backtest(summary, trades)
    base["regime"] = regime_distribution(regime) if regime is not None else {}
    base["template"] = template
    if signal_contrib:
        base["signal_contrib"] = signal_contrib
    ev = base.get("ev") or {}
    kf = kelly_fraction(
        float(ev.get("win_rate") or 0),
        float(ev.get("avg_win_pct") or 0),
        float(ev.get("avg_loss_pct") or 0),
    )
    base["kelly"]["half_kelly_fraction"] = kf
    return base


def signal_contrib_from_df(
    df: pd.DataFrame, params_dict: dict, variant: str
) -> Dict[str, Any]:
    """五维权重占比（配置层贡献，非逐 bar 分解）。"""
    from moss2.paper_scanner import _variant_modules

    DecisionParams, _, _, classify_regime, cap, compute_last_composite = (
        _variant_modules(variant)
    )
    clean = {k: v for k, v in params_dict.items() if not str(k).startswith("_")}
    sym = str(params_dict.get("_symbol") or "")
    clean = cap(dict(clean), sym)
    params = DecisionParams.from_dict(clean)
    from moss2 import config as cfg

    regime = classify_regime(df, version=cfg.MOSS2_REGIME_VERSION)
    comp = float(compute_last_composite(df, params, regime))
    names = ("trend", "momentum", "mean_revert", "volume", "volatility")
    weights = {
        "trend": params.trend_weight,
        "momentum": params.momentum_weight,
        "mean_revert": params.mean_revert_weight,
        "volume": params.volume_weight,
        "volatility": params.volatility_weight,
    }
    total_w = sum(weights.values()) or 1.0
    contrib_pct = {k: round(weights[k] / total_w, 4) for k in names}
    dom = max(contrib_pct, key=contrib_pct.get)
    return {
        "contrib_pct": contrib_pct,
        "dominant_dimension": dom,
        "template": params_dict.get("_template", ""),
        "composite": round(comp, 4),
    }
