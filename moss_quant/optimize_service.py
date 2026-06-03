"""网格搜索：模板 + 战术参数 + 训练/验证窗 + 复合评分。"""

from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional

import pandas as pd

from moss_quant import config as cfg
from moss_quant.core.backtest import run_backtest
from moss_quant.core.decision import DecisionParams
from moss_quant.core.regime import classify_regime
from moss_quant.kline_cache import load_cached
from moss_quant.optimize_policy import (
    aggregate_walk_forward_validation,
    apply_regime_to_tactical,
    composite_optimize_score,
    enrich_summary,
    hard_reject_reason,
    pick_best_validated,
    regime_tactical_adjustments,
    split_train_validation_df,
    split_walk_forward_folds,
    stability_adjusted_val_score,
    templates_for_regime,
    validation_fail_reason,
)
from moss_quant.params import (
    TACTICAL_FLOAT_FIELDS,
    build_initial_params,
    cap_leverage_for_symbol,
    resolve_params_dict,
)

TEMPLATES = ("balanced", "momentum", "trend", "mean_revert")

TACTICAL_GRID_KEYS = (
    "entry_threshold",
    "sl_atr_mult",
    "tp_rr_ratio",
    "exit_threshold",
    "regime_sensitivity",
)

TRAILING_TEMPLATES = frozenset({"momentum", "trend"})


def _trailing_for_template(template: str) -> bool:
    return (
        cfg.MOSS_QUANT_OPTIMIZE_TRAILING_FOR_TREND
        and str(template).lower() in TRAILING_TEMPLATES
    )


def _build_run_params(
    template: str, tactical: Dict[str, Any], *, symbol: str
) -> Dict[str, Any]:
    params = build_initial_params(template=template)
    params.update(tactical)
    if _trailing_for_template(template):
        params["trailing_enabled"] = True
    return cap_leverage_for_symbol(resolve_params_dict(params), symbol)


def _optimize_score(summary: Dict[str, Any]) -> float:
    return composite_optimize_score(summary)


def _build_validation_context(
    df_prefix: pd.DataFrame,
    df_val: pd.DataFrame,
) -> pd.DataFrame:
    """验证段前截取 warmup，供 regime/增量状态预热。"""
    warmup = int(cfg.MOSS_QUANT_OPTIMIZE_VAL_WARMUP_BARS)
    if df_prefix is not None and len(df_prefix) > 0:
        prefix = df_prefix.iloc[-warmup:].copy()
    else:
        prefix = df_prefix.iloc[0:0].copy() if df_prefix is not None else pd.DataFrame()
    if len(df_val) == 0:
        return prefix.reset_index(drop=True)
    return pd.concat([prefix, df_val], ignore_index=True).reset_index(drop=True)


def _validation_window_bounds(df_val: pd.DataFrame, df_ctx: pd.DataFrame) -> tuple:
    """与 evolve 分段一致：仅在验证段时间窗内计收益/成交。"""
    n_val = len(df_val)
    n_ctx = len(df_ctx)
    if n_val <= 0 or n_ctx < n_val:
        raise ValueError("validation window bounds invalid")
    val_start_idx = n_ctx - n_val
    start_iloc = val_start_idx if val_start_idx > 0 else (1 if n_ctx > 1 else 0)
    ts = pd.to_datetime(df_ctx["timestamp"], utc=True)
    window_start = pd.Timestamp(ts.iloc[start_iloc])
    window_end = pd.Timestamp(ts.iloc[-1])
    return window_start, window_end


def _run_backtest_summary(
    df: pd.DataFrame,
    regime: pd.Series,
    *,
    symbol: str,
    template: str,
    tactical: Dict[str, Any],
    capital: float,
    window_start: Optional[pd.Timestamp] = None,
    window_end: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    params = _build_run_params(template, tactical, symbol=symbol)
    p = DecisionParams.from_dict(params)
    result = run_backtest(
        df,
        p,
        regime,
        initial_capital=capital,
        symbol=symbol,
        window_start=window_start,
        window_end=window_end,
    )
    return {
        "total_return": round(result.total_return, 4),
        "sharpe": round(result.sharpe_ratio, 4),
        "max_drawdown": round(result.max_drawdown, 4),
        "total_trades": int(result.total_trades),
        "win_rate": round(result.win_rate, 4),
        "profit_factor": round(float(result.profit_factor or 0), 4),
        "blowup_count": int(result.blowup_count),
    }


def _run_one(
    df: pd.DataFrame,
    regime: pd.Series,
    *,
    symbol: str,
    template: str,
    tactical: Dict[str, Any],
    capital: float,
) -> Dict[str, Any]:
    summary = _run_backtest_summary(
        df,
        regime,
        symbol=symbol,
        template=template,
        tactical=tactical,
        capital=capital,
    )
    reject = hard_reject_reason(summary)
    params_full = _build_run_params(template, tactical, symbol=symbol)
    tact_out = {k: tactical[k] for k in TACTICAL_GRID_KEYS if k in tactical}
    for k in TACTICAL_FLOAT_FIELDS:
        if k in params_full and k not in tact_out:
            tact_out[k] = params_full[k]
    if params_full.get("trailing_enabled"):
        tact_out["trailing_enabled"] = True
    return {
        "template": template,
        "tactical_params": tact_out,
        "params": params_full,
        "summary": summary,
        "train_reject": reject,
        "score": _optimize_score(summary),
    }


def _validate_candidate(
    candidate: Dict[str, Any],
    df_val: pd.DataFrame,
    *,
    symbol: str,
    capital: float,
    regime_version: str,
    df_prefix: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    if len(df_val) < 48:
        return {
            "validation_passed": False,
            "validation_reason": "验证窗K线不足",
            "validation_summary": None,
            "val_sharpe": None,
            "val_return": None,
        }
    tactical = dict(candidate.get("tactical_params") or {})
    template = str(candidate.get("template") or "balanced")
    prefix = df_prefix if df_prefix is not None else pd.DataFrame()
    df_ctx = _build_validation_context(prefix, df_val)
    if len(df_ctx) < len(df_val) + 1:
        return {
            "validation_passed": False,
            "validation_reason": "验证上下文K线不足",
            "validation_summary": None,
            "val_sharpe": None,
            "val_return": None,
        }
    regime_ctx = classify_regime(df_ctx, version=regime_version)
    try:
        ws, we = _validation_window_bounds(df_val, df_ctx)
        val_summary = _run_backtest_summary(
            df_ctx,
            regime_ctx,
            symbol=symbol,
            template=template,
            tactical=tactical,
            capital=capital,
            window_start=ws,
            window_end=we,
        )
    except Exception as e:
        return {
            "validation_passed": False,
            "validation_reason": str(e),
            "validation_summary": None,
            "val_sharpe": None,
            "val_return": None,
        }
    from moss_quant.core.decision import DecisionParams
    from moss_quant.gate_proxy import (
        funding_extreme_stats,
        gate_fail_reason,
        reachable_fail_reason,
        validation_gate_penalty,
        validation_reachable_penalty,
    )
    from moss_quant.signal_entry import validation_reachable_stats
    from moss_quant.trade_gates import train_regime_note_from_summary

    gate_stats = funding_extreme_stats(df_val, symbol)
    gate_penalty = validation_gate_penalty(gate_stats)
    val_summary["gate_extreme_ratio"] = gate_stats.get("extreme_ratio")
    val_summary["gate_penalty"] = gate_penalty

    n_ctx = len(df_ctx)
    val_start_idx = n_ctx - len(df_val) if len(df_val) > 0 else 0
    train_note = train_regime_note_from_summary(candidate.get("summary") or {})
    dec_params = DecisionParams.from_dict(
        _build_run_params(template, tactical, symbol=symbol)
    )
    base_th = float(tactical.get("entry_threshold") or dec_params.entry_threshold or 0.44)
    reach_stats = validation_reachable_stats(
        df_ctx,
        regime_ctx,
        symbol,
        dec_params,
        base_threshold=base_th,
        val_start_idx=val_start_idx,
        train_regime_note=train_note,
        template=template,
    )
    reach_penalty = validation_reachable_penalty(reach_stats)
    val_summary.update(reach_stats)
    val_summary["reachable_penalty"] = reach_penalty

    reason = (
        validation_fail_reason(val_summary)
        or gate_fail_reason(gate_stats)
        or reachable_fail_reason(reach_stats)
    )
    tr_ret = float((candidate.get("summary") or {}).get("total_return") or 0)
    val_ret_adj = float(val_summary.get("total_return") or 0) - gate_penalty - reach_penalty
    stability = stability_adjusted_val_score(
        float(val_summary.get("sharpe") or 0),
        train_return=tr_ret,
        val_return=val_ret_adj,
    )
    return {
        "validation_passed": reason is None,
        "validation_reason": reason or "验证通过",
        "validation_summary": val_summary,
        "val_sharpe": float(val_summary.get("sharpe") or 0),
        "val_return": float(val_summary.get("total_return") or 0),
        "gate_extreme_ratio": gate_stats.get("extreme_ratio"),
        "gate_penalty": gate_penalty,
        "reachable_ratio": reach_stats.get("reachable_ratio"),
        "reachable_penalty": reach_penalty,
        "reachable_sub_pf": reach_stats.get("reachable_sub_pf"),
        "stability_score": stability,
        "val_warmup_bars": int(min(len(prefix), cfg.MOSS_QUANT_OPTIMIZE_VAL_WARMUP_BARS)),
    }


def _validate_candidate_walk_forward(
    candidate: Dict[str, Any],
    folds: List[tuple],
    *,
    symbol: str,
    capital: float,
    regime_version: str,
) -> Dict[str, Any]:
    """对 Top-K 候选在每折验证窗上回测并汇总。"""
    if len(folds) <= 1:
        df_train, df_val = folds[0]
        single = _validate_candidate(
            candidate,
            df_val,
            symbol=symbol,
            capital=capital,
            regime_version=regime_version,
            df_prefix=df_train,
        )
        single["wf_folds"] = 1
        single["wf_passed_folds"] = 1 if single.get("validation_passed") else 0
        single["wf_validation_passed"] = single.get("validation_passed")
        single["wf_reason"] = single.get("validation_reason")
        return single

    fold_vals: List[Dict[str, Any]] = []
    for df_train, df_val in folds:
        fold_vals.append(
            _validate_candidate(
                candidate,
                df_val,
                symbol=symbol,
                capital=capital,
                regime_version=regime_version,
                df_prefix=df_train,
            )
        )
    agg = aggregate_walk_forward_validation(fold_vals)
    last_ok = next((v for v in reversed(fold_vals) if v.get("validation_summary")), None)
    val_summary = (last_ok or {}).get("validation_summary")
    out = {
        "validation_passed": agg["validation_passed"],
        "validation_reason": agg["validation_reason"],
        "validation_summary": val_summary,
        "val_sharpe": agg["val_sharpe"],
        "val_return": agg["val_return"],
        "wf_folds": agg["wf_folds"],
        "wf_passed_folds": agg["wf_passed_folds"],
        "wf_min_pass_folds": agg["wf_min_pass_folds"],
        "wf_validation_passed": agg["wf_validation_passed"],
        "wf_reason": agg["wf_reason"],
        "gate_penalty": agg.get("gate_penalty"),
        "gate_extreme_ratio": agg.get("gate_extreme_ratio"),
        "fold_validations": fold_vals,
    }
    tr_ret = float((candidate.get("summary") or {}).get("total_return") or 0)
    gate_pen = float(out.get("gate_penalty") or 0)
    reach_pen = float(agg.get("reachable_penalty") or 0)
    out["reachable_penalty"] = reach_pen
    out["reachable_ratio"] = agg.get("reachable_ratio")
    out["stability_score"] = stability_adjusted_val_score(
        float(out.get("val_sharpe") or 0),
        train_return=tr_ret,
        val_return=float(out.get("val_return") or 0) - gate_pen - reach_pen,
    )
    return out


def _apply_post_grid_refinement(
    best: Dict[str, Any],
    *,
    df_full: pd.DataFrame,
    df_train: pd.DataFrame,
    regime_train: pd.Series,
    symbol: str,
    capital: float,
    regime_version: str,
    regime_adj: Dict[str, Any],
) -> Dict[str, Any]:
    """70% 总结 + WF 邻域精修（与看板验证口径一致）；改善则覆盖 tactical_params。"""
    if not (
        cfg.MOSS_QUANT_OPTIMIZE_TUNING_DIAG_ENABLED
        or cfg.MOSS_QUANT_OPTIMIZE_LOCAL_REFINE_ENABLED
    ):
        return best
    tactical = dict(best.get("tactical_params") or {})
    if not tactical:
        return best
    try:
        from moss_quant.backtest_diagnosis import run_post_grid_pipeline

        wf_folds = split_walk_forward_folds(df_full)
        grid_wf = dict(best.get("validation") or {})
        train_ret = float((best.get("summary") or {}).get("train_return") or 0)
        template = str(best.get("template") or "balanced")

        def validate_wf_fn(tact: Dict[str, Any], tpl: str) -> Dict[str, Any]:
            cand = {
                "tactical_params": tact,
                "template": tpl,
                "summary": {"total_return": train_ret},
            }
            return _validate_candidate_walk_forward(
                cand,
                wf_folds,
                symbol=symbol,
                capital=capital,
                regime_version=regime_version,
            )

        pipe = run_post_grid_pipeline(
            df_train=df_train,
            regime_train=regime_train,
            symbol=symbol,
            template=template,
            tactical=tactical,
            capital=capital,
            regime_note=str(regime_adj.get("regime_note") or ""),
            build_params_fn=_build_run_params,
            validate_wf_fn=validate_wf_fn,
            grid_wf_validation=grid_wf,
        )
        best = dict(best)
        summary = dict(best.get("summary") or {})
        summary["post_grid_pipeline"] = pipe
        summary["grid_val_return"] = float(summary.get("val_return") or 0)
        summary["grid_val_sharpe"] = float(summary.get("val_sharpe") or 0)

        final_tact = pipe.get("final_tactical_params") or tactical
        if pipe.get("param_source") == "local_refine":
            refine = pipe.get("local_refine") or {}
            rwf = refine.get("refined_wf_validation") or {}
            summary["val_return"] = refine.get("refined_val_return")
            summary["val_sharpe"] = refine.get("refined_val_sharpe")
            summary["wf_passed_folds"] = rwf.get("wf_passed_folds")
            summary["wf_folds"] = rwf.get("wf_folds")
            summary["wf_validation_passed"] = rwf.get("wf_validation_passed")
            summary["wf_reason"] = rwf.get("wf_reason")
            summary["validation_passed"] = bool(rwf.get("validation_passed"))
            summary["validation_reason"] = rwf.get("validation_reason")
            summary["param_source"] = "local_refine"
            summary["refine_improved"] = True
            best["tactical_params"] = final_tact
            best["params"] = _build_run_params(template, final_tact, symbol=symbol)
            best["validation"] = rwf
        else:
            summary["param_source"] = "grid"
            summary["refine_improved"] = False

        best["summary"] = enrich_summary(summary)
    except Exception as e:
        best = dict(best)
        summary = dict(best.get("summary") or {})
        summary["post_grid_pipeline"] = {"error": str(e), "skipped": True}
        best["summary"] = summary
    return best


def _attach_best_metadata(
    best: Dict[str, Any],
    *,
    train_bars: int,
    val_bars: int,
    regime_adj: Dict[str, Any],
) -> Dict[str, Any]:
    train_summary = dict(best.get("summary") or {})
    val_block = dict(best.get("validation") or {})
    val_summary = val_block.get("validation_summary") or {}
    merged = {
        **train_summary,
        "train_return": float(train_summary.get("total_return") or 0),
        "train_score": float(best.get("score") or 0),
        "train_bars": train_bars,
        "val_bars": val_bars,
        "validation_passed": bool(val_block.get("validation_passed")),
        "validation_reason": val_block.get("validation_reason"),
        "validation_summary": val_summary,
        "val_sharpe": val_block.get("val_sharpe"),
        "val_return": val_block.get("val_return"),
        "wf_folds": val_block.get("wf_folds"),
        "wf_passed_folds": val_block.get("wf_passed_folds"),
        "wf_validation_passed": val_block.get("wf_validation_passed"),
        "wf_reason": val_block.get("wf_reason"),
        "stability_score": val_block.get("stability_score"),
        "gate_extreme_ratio": val_block.get("gate_extreme_ratio"),
        "gate_penalty": val_block.get("gate_penalty"),
        "regime_adjustment": regime_adj or {},
    }
    if val_summary:
        merged["val_total_trades"] = val_summary.get("total_trades")
        merged["val_max_drawdown"] = val_summary.get("max_drawdown")
    best = dict(best)
    best["summary"] = enrich_summary(merged)
    return best


def run_strategy_optimize(
    *,
    symbol: str,
    capital: Optional[float] = None,
    refresh_klines: bool = False,
    regime_version: Optional[str] = None,
    top_n: Optional[int] = None,
) -> Dict[str, Any]:
    """训练窗网格寻优 → Top-K → 验证窗样本外 → 复合评分入选。"""
    capital = float(capital or cfg.MOSS_QUANT_DEFAULT_CAPITAL)
    regime_version = regime_version or cfg.MOSS_QUANT_REGIME_VERSION
    sym = str(symbol).strip().upper()

    entries = tuple(cfg.MOSS_QUANT_OPTIMIZE_ENTRY_THRESHOLDS)
    sls = tuple(cfg.MOSS_QUANT_OPTIMIZE_SL_ATR_MULTS)
    tps = tuple(cfg.MOSS_QUANT_OPTIMIZE_TP_RR_RATIOS)
    max_combinations = int(cfg.MOSS_QUANT_OPTIMIZE_MAX_COMBINATIONS)
    top_n = max(1, min(int(top_n or cfg.MOSS_QUANT_OPTIMIZE_API_TOP_N), 50))

    df_full = load_cached(sym, refresh=refresh_klines, research=True)
    df_train, df_val = split_train_validation_df(df_full)
    regime_train = classify_regime(df_train, version=regime_version)
    regime_adj = regime_tactical_adjustments(regime_train)
    active_templates = templates_for_regime(regime_adj)
    grid = list(itertools.product(active_templates, entries, sls, tps))
    if len(grid) > max_combinations:
        grid = grid[:max_combinations]

    wf_folds = split_walk_forward_folds(df_full)

    results: List[Dict[str, Any]] = []
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
            row = _run_one(
                df_train,
                regime_train,
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

    valid = [r for r in results if r.get("summary") and float(r.get("score") or -999) > -900]
    valid.sort(
        key=lambda r: (
            -float(r.get("score") or -999),
            -float((r.get("summary") or {}).get("sharpe") or -999),
        )
    )

    top_k = max(1, min(int(cfg.MOSS_QUANT_OPTIMIZE_VALIDATION_TOP_K), len(valid)))
    candidates = valid[:top_k]
    validated: List[Dict[str, Any]] = []
    for cand in candidates:
        c2 = dict(cand)
        if len(wf_folds) > 1:
            c2["validation"] = _validate_candidate_walk_forward(
                cand,
                wf_folds,
                symbol=sym,
                capital=capital,
                regime_version=regime_version,
            )
        else:
            c2["validation"] = _validate_candidate(
                cand,
                df_val,
                symbol=sym,
                capital=capital,
                regime_version=regime_version,
                df_prefix=df_train,
            )
        validated.append(c2)

    best_raw = pick_best_validated(validated)
    if best_raw is None and valid:
        best_raw = dict(valid[0])
        if "validation" not in best_raw:
            if len(wf_folds) > 1:
                best_raw["validation"] = _validate_candidate_walk_forward(
                    best_raw,
                    wf_folds,
                    symbol=sym,
                    capital=capital,
                    regime_version=regime_version,
                )
            else:
                best_raw["validation"] = _validate_candidate(
                    best_raw,
                    df_val,
                    symbol=sym,
                    capital=capital,
                    regime_version=regime_version,
                    df_prefix=df_train,
                )

    best: Optional[Dict[str, Any]] = None
    if best_raw:
        best = _attach_best_metadata(
            best_raw,
            train_bars=int(len(df_train)),
            val_bars=int(len(df_val)),
            regime_adj=regime_adj,
        )
        if best:
            best = _apply_post_grid_refinement(
                best,
                df_full=df_full,
                df_train=df_train,
                regime_train=regime_train,
                symbol=sym,
                capital=capital,
                regime_version=regime_version,
                regime_adj=regime_adj,
            )

    ranking: List[Dict[str, Any]] = []
    for c in validated[:top_n]:
        if not c.get("summary"):
            continue
        ranking.append(
            _attach_best_metadata(
                c,
                train_bars=int(len(df_train)),
                val_bars=int(len(df_val)),
                regime_adj=regime_adj,
            )
        )

    best_ret = float((best or {}).get("summary", {}).get("total_return", 0) or 0)
    all_non_positive = bool(valid) and best_ret <= 0
    val_passed = bool((best or {}).get("summary", {}).get("validation_passed"))

    return {
        "ok": True,
        "symbol": sym,
        "capital": capital,
        "data_source": cfg.MOSS_QUANT_DATA_SOURCE,
        "data_source_label": cfg.data_source_label(),
        "combinations_tested": len(grid),
        "combinations_ok": len(valid),
        "bars": int(len(df_full)),
        "train_bars": int(len(df_train)),
        "val_bars": int(len(df_val)),
        "kline_start": str(df_full["timestamp"].iloc[0]) if len(df_full) else None,
        "kline_end": str(df_full["timestamp"].iloc[-1]) if len(df_full) else None,
        "best": best,
        "validation_passed": val_passed,
        "all_non_positive": all_non_positive,
        "warning": (
            "本次窗口内所有组合收益均≤0，最优仅为相对亏损最小；不宜直接应用实盘。"
            if all_non_positive
            else (
                "最优组合未通过样本外验证，不会自动同步纸面 Profile。"
                if best and not val_passed and cfg.MOSS_QUANT_OPTIMIZE_REQUIRE_VALIDATION
                else None
            )
        ),
        "ranking": ranking,
        "regime_adjustment": regime_adj,
        "search_space": {
            "templates": list(active_templates),
            "entry_threshold": list(entries),
            "sl_atr_mult": list(sls),
            "tp_rr_ratio": list(tps),
            "fixed": {"exit_threshold": 0.12, "regime_sensitivity": 0.55},
            "train_ratio": float(cfg.MOSS_QUANT_OPTIMIZE_TRAIN_RATIO),
            "wf_folds": len(wf_folds),
            "wf_min_pass_folds": int(cfg.MOSS_QUANT_OPTIMIZE_WF_MIN_PASS_FOLDS),
            "val_warmup_bars": int(cfg.MOSS_QUANT_OPTIMIZE_VAL_WARMUP_BARS),
            "gate_proxy_enabled": bool(cfg.MOSS_QUANT_OPTIMIZE_GATE_PROXY_ENABLED),
            "tuning_diag_enabled": bool(cfg.MOSS_QUANT_OPTIMIZE_TUNING_DIAG_ENABLED),
            "local_refine_enabled": bool(cfg.MOSS_QUANT_OPTIMIZE_LOCAL_REFINE_ENABLED),
            "local_refine_max_rounds": int(
                cfg.MOSS_QUANT_OPTIMIZE_LOCAL_REFINE_MAX_ROUNDS
            ),
        },
    }
