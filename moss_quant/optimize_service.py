"""网格搜索：模板 + 战术参数，按回测收益排序。"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional

import pandas as pd

from moss_quant import config as cfg
from moss_quant.core.backtest import run_backtest
from moss_quant.core.decision import DecisionParams
from moss_quant.core.regime import classify_regime
from moss_quant.kline_cache import load_cached
from moss_quant.params import (
    TACTICAL_FLOAT_FIELDS,
    build_initial_params,
    cap_leverage_for_symbol,
    resolve_params_dict,
)

TEMPLATES = ("balanced", "momentum", "trend", "mean_revert")

# 默认搜索空间（约 4×4×2×3 = 96 组）
DEFAULT_ENTRY_THRESHOLDS = (0.40, 0.44, 0.48, 0.52)
DEFAULT_SL_ATR_MULTS = (2.0, 2.5)
DEFAULT_TP_RR_RATIOS = (2.0, 2.5, 3.0)

TACTICAL_GRID_KEYS = (
    "entry_threshold",
    "sl_atr_mult",
    "tp_rr_ratio",
    "exit_threshold",
    "regime_sensitivity",
)


def _optimize_score(summary: Dict[str, Any]) -> float:
    """主排序：total_return；笔数过少略惩罚。"""
    ret = float(summary.get("total_return") or 0)
    n = int(summary.get("total_trades") or 0)
    if n < 3:
        return ret - 0.03
    if n < 8:
        return ret - 0.005
    return ret


def _run_one(
    df: pd.DataFrame,
    regime: pd.Series,
    *,
    symbol: str,
    template: str,
    tactical: Dict[str, Any],
    capital: float,
) -> Dict[str, Any]:
    params = build_initial_params(template=template)
    params.update(tactical)
    params = cap_leverage_for_symbol(resolve_params_dict(params), symbol)
    p = DecisionParams.from_dict(params)
    result = run_backtest(
        df,
        p,
        regime,
        initial_capital=capital,
        symbol=symbol,
    )
    summary = {
        "total_return": round(result.total_return, 4),
        "sharpe": round(result.sharpe_ratio, 4),
        "max_drawdown": round(result.max_drawdown, 4),
        "total_trades": int(result.total_trades),
        "win_rate": round(result.win_rate, 4),
        "blowup_count": int(result.blowup_count),
    }
    tact_out = {k: params[k] for k in TACTICAL_GRID_KEYS if k in params}
    for k in TACTICAL_FLOAT_FIELDS:
        if k in params and k not in tact_out:
            tact_out[k] = params[k]
    return {
        "template": template,
        "tactical_params": tact_out,
        "params": params,
        "summary": summary,
        "score": round(_optimize_score(summary), 6),
    }


def run_strategy_optimize(
    *,
    symbol: str,
    capital: Optional[float] = None,
    refresh_klines: bool = False,
    regime_version: Optional[str] = None,
    top_n: int = 15,
    max_combinations: int = 96,
    entry_thresholds: Optional[List[float]] = None,
    sl_atr_mults: Optional[List[float]] = None,
    tp_rr_ratios: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    遍历模板 × 战术网格，返回按 score（近似 total_return）排序的结果。
    K 线只加载一次；regime 只算一次。
    """
    capital = float(capital or cfg.MOSS_QUANT_DEFAULT_CAPITAL)
    regime_version = regime_version or cfg.MOSS_QUANT_REGIME_VERSION
    sym = str(symbol).strip().upper()

    entries = tuple(entry_thresholds or DEFAULT_ENTRY_THRESHOLDS)
    sls = tuple(sl_atr_mults or DEFAULT_SL_ATR_MULTS)
    tps = tuple(tp_rr_ratios or DEFAULT_TP_RR_RATIOS)

    grid = list(
        itertools.product(TEMPLATES, entries, sls, tps)
    )
    if len(grid) > max_combinations:
        grid = grid[:max_combinations]

    df = load_cached(sym, refresh=refresh_klines)
    regime = classify_regime(df, version=regime_version)

    results: List[Dict[str, Any]] = []
    for template, entry, sl, tp in grid:
        tactical = {
            "entry_threshold": float(entry),
            "sl_atr_mult": float(sl),
            "tp_rr_ratio": float(tp),
            "exit_threshold": 0.12,
            "regime_sensitivity": 0.55,
        }
        try:
            row = _run_one(
                df,
                regime,
                symbol=sym,
                template=template,
                tactical=tactical,
                capital=capital,
            )
            results.append(row)
        except Exception as e:
            results.append(
                {
                    "template": template,
                    "tactical_params": tactical,
                    "error": str(e),
                    "score": -999.0,
                    "summary": None,
                }
            )

    valid = [r for r in results if r.get("summary")]
    valid.sort(
        key=lambda r: (
            -float(r.get("score") or -999),
            -float((r.get("summary") or {}).get("sharpe") or -999),
        )
    )
    top_n = max(1, min(int(top_n), 50))
    best = valid[0] if valid else None

    return {
        "ok": True,
        "symbol": sym,
        "capital": capital,
        "combinations_tested": len(grid),
        "combinations_ok": len(valid),
        "bars": int(len(df)),
        "best": best,
        "ranking": valid[:top_n],
        "search_space": {
            "templates": list(TEMPLATES),
            "entry_threshold": list(entries),
            "sl_atr_mult": list(sls),
            "tp_rr_ratio": list(tps),
            "fixed": {"exit_threshold": 0.12, "regime_sensitivity": 0.55},
        },
    }
