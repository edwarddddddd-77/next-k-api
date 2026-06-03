"""SUI：6720 vs 1500 根寻优+精修对比。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi

load_env_oi()

import itertools

from moss_quant import config as cfg
from moss_quant.core.regime import classify_regime
from moss_quant.kline_cache import load_cached
from moss_quant.optimize_policy import regime_tactical_adjustments, templates_for_regime
from moss_quant.optimize_service import (
    _apply_post_grid_refinement,
    _attach_best_metadata,
    _run_one,
    _validate_candidate_walk_forward,
    apply_regime_to_tactical,
    pick_best_validated,
    split_train_validation_df,
    split_walk_forward_folds,
)

SYM = "SUIUSDT"
CAPITAL = 10000.0


def run_on_df(df, label: str) -> dict:
    df = df.reset_index(drop=True)
    rv = cfg.MOSS_QUANT_REGIME_VERSION
    df_train, df_val = split_train_validation_df(df)
    regime_train = classify_regime(df_train, version=rv)
    regime_adj = regime_tactical_adjustments(regime_train)
    templates = templates_for_regime(regime_adj)
    entries = tuple(cfg.MOSS_QUANT_OPTIMIZE_ENTRY_THRESHOLDS)
    sls = tuple(cfg.MOSS_QUANT_OPTIMIZE_SL_ATR_MULTS)
    tps = tuple(cfg.MOSS_QUANT_OPTIMIZE_TP_RR_RATIOS)
    grid = list(itertools.product(templates, entries, sls, tps))[
        : int(cfg.MOSS_QUANT_OPTIMIZE_MAX_COMBINATIONS)
    ]
    results = []
    for template, entry, sl, tp in grid:
        tactical = apply_regime_to_tactical(
            {
                "entry_threshold": float(entry),
                "sl_atr_mult": float(sl),
                "tp_rr_ratio": float(tp),
                "exit_threshold": 0.12,
                "regime_sensitivity": 0.55,
            },
            regime_adj,
        )
        try:
            results.append(
                _run_one(
                    df_train,
                    regime_train,
                    symbol=SYM,
                    template=template,
                    tactical=tactical,
                    capital=CAPITAL,
                )
            )
        except Exception as e:
            results.append({"error": str(e), "score": -999})
    valid = [
        r for r in results if r.get("summary") and float(r.get("score") or -999) > -900
    ]
    valid.sort(key=lambda r: -float(r.get("score") or -999))
    wf_folds = split_walk_forward_folds(df)
    validated = []
    for cand in valid[:5]:
        c2 = dict(cand)
        c2["validation"] = _validate_candidate_walk_forward(
            cand, wf_folds, symbol=SYM, capital=CAPITAL, regime_version=rv
        )
        validated.append(c2)
    best_raw = pick_best_validated(validated) or (valid[0] if valid else None)
    best = None
    if best_raw:
        best = _attach_best_metadata(
            best_raw,
            train_bars=len(df_train),
            val_bars=len(df_val),
            regime_adj=regime_adj,
        )
        best = _apply_post_grid_refinement(
            best,
            df_full=df,
            df_train=df_train,
            regime_train=regime_train,
            symbol=SYM,
            capital=CAPITAL,
            regime_version=rv,
            regime_adj=regime_adj,
        )
    sm = (best or {}).get("summary") or {}
    tact = (best or {}).get("tactical_params") or {}
    return {
        "label": label,
        "bars": len(df),
        "days_15m": round(len(df) * 15 / 60 / 24, 1),
        "train_bars": len(df_train),
        "val_bars": len(df_val),
        "wf_folds": len(wf_folds),
        "grid_ok": len(valid),
        "template": (best or {}).get("template"),
        "entry": tact.get("entry_threshold"),
        "sl": tact.get("sl_atr_mult"),
        "tp": tact.get("tp_rr_ratio"),
        "train_return_pct": round(
            float(sm.get("train_return", sm.get("total_return")) or 0) * 100, 2
        ),
        "val_return_pct": round(float(sm.get("val_return") or 0) * 100, 2),
        "val_sharpe": round(float(sm.get("val_sharpe") or 0), 3),
        "wf": f"{sm.get('wf_passed_folds')}/{sm.get('wf_folds')}",
        "trades_train": sm.get("total_trades"),
        "win_rate_train_pct": round(float(sm.get("win_rate") or 0) * 100, 1),
        "param_source": sm.get("param_source"),
        "refine_improved": sm.get("refine_improved"),
        "pool": sm.get("pool_tier"),
    }


def main() -> None:
    df_full = load_cached(SYM, refresh=False, research=True)
    df_1500 = df_full.iloc[-1500:].copy()
    out = {"6720": run_on_df(df_full, "6720"), "1500": run_on_df(df_1500, "1500")}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
