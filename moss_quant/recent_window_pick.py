"""L3：最近 1500 根（与纸面扫描同长）归因 + 短窗收益最高模板/战术参数。"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from moss_quant import config as cfg
from moss_quant.backtest_diagnosis import (
    analyze_trades,
    build_neighbor_candidates,
    suggest_tactical_adjustments,
)
from moss_quant.core.backtest import run_backtest
from moss_quant.core.decision import DecisionParams
from moss_quant.core.regime import classify_regime
from moss_quant.kline_cache import load_cached
from moss_quant.optimize_policy import (
    apply_regime_to_tactical,
    regime_tactical_adjustments,
    templates_for_regime,
)
from moss_quant.optimize_service import (
    _build_run_params,
    _build_validation_context,
    _run_backtest_summary,
    _validation_window_bounds,
)
from moss_quant.params import TACTICAL_FLOAT_FIELDS

logger = logging.getLogger(__name__)

_ALL_TEMPLATES = ("balanced", "momentum", "trend", "mean_revert")


def _recent_bars() -> int:
    return int(cfg.MOSS_QUANT_RECENT_PICK_BARS or cfg.MOSS_QUANT_KLINE_LIMIT)


def _train_cut(n: int) -> int:
    ratio = float(cfg.MOSS_QUANT_OPTIMIZE_TRAIN_RATIO)
    return max(48, min(n - 48, int(n * ratio)))


def _rank_recent_score(summary: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """收益 × 笔数饱和系数 × PF 加成（未达标笔数强降权，达标后略奖多笔）。"""
    ret = float(summary.get("total_return") or 0)
    trades = int(summary.get("total_trades") or 0)
    min_t = max(1, int(cfg.MOSS_QUANT_RECENT_PICK_MIN_TRADES))
    if trades >= min_t:
        extra = min(0.25, 0.08 * (trades - min_t) / min_t)
        trade_f = 1.0 + extra
    else:
        trade_f = (trades / min_t) ** 1.25
    pf = float(summary.get("profit_factor") or 0)
    if cfg.MOSS_QUANT_RECENT_PICK_SCORE_USE_PF and pf > 0:
        pf_f = min(2.0, pf) / 2.0
        score = ret * trade_f * (0.5 + 0.5 * pf_f)
    else:
        score = ret * trade_f
    return (score, ret, float(summary.get("sharpe") or 0), -abs(float(summary.get("max_drawdown") or 0)))


def _tail_validation_summary(
    df: pd.DataFrame,
    regime: pd.Series,
    *,
    symbol: str,
    template: str,
    tactical: Dict[str, Any],
    capital: float,
    train_cut: int,
) -> Dict[str, Any]:
    df_tail = df.iloc[train_cut:].reset_index(drop=True)
    if len(df_tail) < 24:
        return {"total_return": 0.0, "total_trades": 0, "error": "tail_too_short"}
    df_prefix = df.iloc[:train_cut]
    df_ctx = _build_validation_context(df_prefix, df_tail)
    regime_ctx = classify_regime(df_ctx, version=cfg.MOSS_QUANT_REGIME_VERSION)
    ws, we = _validation_window_bounds(df_tail, df_ctx)
    return _run_backtest_summary(
        df_ctx,
        regime_ctx,
        symbol=symbol,
        template=template,
        tactical=tactical,
        capital=capital,
        window_start=ws,
        window_end=we,
    )


def _passes_guards(
    full_s: Dict[str, Any],
    tail_s: Dict[str, Any],
) -> Tuple[bool, str]:
    trades = int(full_s.get("total_trades") or 0)
    if trades < int(cfg.MOSS_QUANT_RECENT_PICK_MIN_TRADES):
        return False, f"1500窗笔数不足({trades}<{cfg.MOSS_QUANT_RECENT_PICK_MIN_TRADES})"
    full_ret = float(full_s.get("total_return") or 0)
    if full_ret <= 0:
        return False, "1500窗收益≤0"
    if int(full_s.get("blowup_count") or 0) > 0:
        return False, "1500窗爆仓"
    min_pf = float(cfg.MOSS_QUANT_RECENT_PICK_MIN_PROFIT_FACTOR or 0)
    pf = float(full_s.get("profit_factor") or 0)
    if min_pf > 0 and pf < min_pf:
        return False, f"1500窗盈亏比{pf:.2f}<{min_pf:.2f}"
    tail_ret = float(tail_s.get("total_return") or 0)
    floor = float(cfg.MOSS_QUANT_RECENT_PICK_TAIL_MIN_RETURN)
    if tail_ret < floor:
        return False, f"后{int(cfg.MOSS_QUANT_RECENT_PICK_TAIL_RATIO * 100)}%窗收益{tail_ret * 100:.2f}%<{floor * 100:.0f}%"
    return True, "短窗+尾段通过"


def _grid_combos(
    regime_adj: Dict[str, Any],
    *,
    prefer_template: Optional[str] = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    templates = (
        templates_for_regime(regime_adj)
        if cfg.MOSS_QUANT_RECENT_PICK_REGIME_TEMPLATES
        else _ALL_TEMPLATES
    )
    entries = tuple(cfg.MOSS_QUANT_RECENT_PICK_ENTRY_THRESHOLDS)
    sls = tuple(cfg.MOSS_QUANT_RECENT_PICK_SL_ATR_MULTS)
    tps = tuple(cfg.MOSS_QUANT_RECENT_PICK_TP_RR_RATIOS)
    exits = tuple(cfg.MOSS_QUANT_RECENT_PICK_EXIT_THRESHOLDS)
    out: List[Tuple[str, Dict[str, Any]]] = []
    for template, entry, sl, tp, ex in itertools.product(
        templates, entries, sls, tps, exits
    ):
        tactical = apply_regime_to_tactical(
            {
                "entry_threshold": float(entry),
                "sl_atr_mult": float(sl),
                "tp_rr_ratio": float(tp),
                "exit_threshold": float(ex),
                "regime_sensitivity": 0.55,
            },
            regime_adj,
        )
        out.append((str(template), tactical))
    pref = str(prefer_template or "").strip().lower()
    if pref:
        out.sort(key=lambda row: 0 if str(row[0]).lower() == pref else 1)
    cap = int(cfg.MOSS_QUANT_RECENT_PICK_MAX_COMBINATIONS)
    return out[:cap] if len(out) > cap else out


def pick_best_on_recent_window(
    symbol: str,
    *,
    capital: Optional[float] = None,
    l1_summary: Optional[Dict[str, Any]] = None,
    refresh_klines: bool = False,
) -> Dict[str, Any]:
    """
    在最近 N 根（默认 1500）上选复合分最高的模板+战术参数。
    70% 归因指导邻域精修；后 30% 尾段不得明显崩坏。
    """
    sym = str(symbol or "").strip().upper()
    cap = float(capital or cfg.MOSS_QUANT_DEFAULT_CAPITAL)
    if not cfg.MOSS_QUANT_RECENT_PICK_ENABLED:
        return {"adopted": False, "skipped": True, "reason": "recent_pick_disabled"}

    l1 = dict(l1_summary or {})
    if cfg.MOSS_QUANT_RECENT_PICK_REQUIRE_L1_OK:
        tier = str(l1.get("pool_tier") or "C").upper()
        if tier != "A":
            return {
                "adopted": False,
                "skipped": True,
                "reason": f"L1池级非A({tier})",
                "l1_pool_tier": tier,
            }
        if not l1.get("sync_allowed") and cfg.MOSS_QUANT_RECENT_PICK_REQUIRE_SYNC:
            return {
                "adopted": False,
                "skipped": True,
                "reason": "L1不可同步",
            }

    try:
        df = load_cached(sym, refresh=refresh_klines, research=False)
    except Exception as e:
        return {"adopted": False, "skipped": True, "reason": str(e)}

    need = max(96, int(cfg.MOSS_QUANT_RECENT_PICK_MIN_BARS))
    bars = _recent_bars()
    if len(df) < need:
        return {
            "adopted": False,
            "skipped": True,
            "reason": f"K线不足({len(df)}<{need})",
        }
    if len(df) > bars:
        df = df.iloc[-bars:].copy().reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    regime = classify_regime(df, version=cfg.MOSS_QUANT_REGIME_VERSION)
    train_cut = _train_cut(len(df))
    df_train = df.iloc[:train_cut].reset_index(drop=True)
    regime_train = regime.iloc[:train_cut].reset_index(drop=True)
    regime_adj = regime_tactical_adjustments(regime_train)
    note = str(regime_adj.get("regime_note") or "")
    prefer_tpl = str(l1.get("l1_template") or l1.get("template") or "").strip().lower()

    best_tpl = ""
    best_tact: Dict[str, Any] = {}
    best_full: Dict[str, Any] = {}
    best_tail: Dict[str, Any] = {}
    best_rank: Tuple[float, float, float, float] = (-999.0, -999.0, -999.0, -999.0)
    tried = 0

    for template, tactical in _grid_combos(regime_adj, prefer_template=prefer_tpl or None):
        tried += 1
        full_s = _run_backtest_summary(
            df, regime, symbol=sym, template=template, tactical=tactical, capital=cap
        )
        tail_s = _tail_validation_summary(
            df,
            regime,
            symbol=sym,
            template=template,
            tactical=tactical,
            capital=cap,
            train_cut=train_cut,
        )
        ok, _ = _passes_guards(full_s, tail_s)
        if not ok:
            continue
        rk = _rank_recent_score(full_s)
        if rk > best_rank:
            best_rank = rk
            best_tpl = template
            best_tact = dict(tactical)
            best_full = full_s
            best_tail = tail_s

    refine_log: List[Dict[str, Any]] = []
    last_train_result = None
    if best_tpl and int(cfg.MOSS_QUANT_RECENT_PICK_REFINE_ROUNDS) > 0:
        params = _build_run_params(best_tpl, best_tact, symbol=sym)
        train_result = run_backtest(
            df_train,
            DecisionParams.from_dict(params),
            regime_train,
            initial_capital=cap,
            symbol=sym,
        )
        last_train_result = train_result
        for rnd in range(1, int(cfg.MOSS_QUANT_RECENT_PICK_REFINE_ROUNDS) + 1):
            analysis = analyze_trades(train_result.trades, regime_train)
            sug = suggest_tactical_adjustments(
                analysis, best_tact, template=best_tpl, regime_note=note
            )
            round_best_tpl = best_tpl
            round_best_tact = best_tact
            round_rank = best_rank
            for cand in build_neighbor_candidates(best_tact, sug.get("tuned_tactical")):
                full_s = _run_backtest_summary(
                    df, regime, symbol=sym, template=best_tpl, tactical=cand, capital=cap
                )
                tail_s = _tail_validation_summary(
                    df,
                    regime,
                    symbol=sym,
                    template=best_tpl,
                    tactical=cand,
                    capital=cap,
                    train_cut=train_cut,
                )
                ok, _ = _passes_guards(full_s, tail_s)
                if not ok:
                    continue
                rk = _rank_recent_score(full_s)
                if rk > round_rank:
                    round_rank = rk
                    round_best_tact = cand
            improved = round_rank > best_rank
            refine_log.append(
                {
                    "round": rnd,
                    "improved": improved,
                    "score": round(float(round_rank[0]), 4),
                    "return_pct": round(float(round_rank[1]) * 100, 2),
                }
            )
            if improved:
                best_rank = round_rank
                best_tact = round_best_tact
                best_tpl = round_best_tpl
                best_full = _run_backtest_summary(
                    df, regime, symbol=sym, template=best_tpl, tactical=best_tact, capital=cap
                )
                best_tail = _tail_validation_summary(
                    df,
                    regime,
                    symbol=sym,
                    template=best_tpl,
                    tactical=best_tact,
                    capital=cap,
                    train_cut=train_cut,
                )
                params = _build_run_params(best_tpl, best_tact, symbol=sym)
                train_result = run_backtest(
                    df_train,
                    DecisionParams.from_dict(params),
                    regime_train,
                    initial_capital=cap,
                    symbol=sym,
                )
                last_train_result = train_result
            else:
                break

    if not best_tpl:
        return {
            "adopted": False,
            "skipped": False,
            "reason": "1500窗无满足门槛的组合",
            "bars": len(df),
            "combinations_tried": tried,
            "train_cut": train_cut,
            "regime_note": note,
        }

    ok, reason = _passes_guards(best_full, best_tail)
    reach_stats: Dict[str, Any] = {}
    if best_tpl and ok and cfg.MOSS_QUANT_OPTIMIZE_REACHABLE_ENABLED:
        from moss_quant.core.decision import DecisionParams
        from moss_quant.signal_entry import validation_reachable_stats

        dec = DecisionParams.from_dict(_build_run_params(best_tpl, best_tact, symbol=sym))
        base_th = float(best_tact.get("entry_threshold") or dec.entry_threshold or 0.44)
        reach_stats = validation_reachable_stats(
            df,
            regime,
            sym,
            dec,
            base_threshold=base_th,
            val_start_idx=train_cut,
            train_regime_note=note,
            template=best_tpl,
        )
        min_reach = float(cfg.MOSS_QUANT_RECENT_PICK_MIN_REACHABLE_RATIO)
        if float(reach_stats.get("reachable_ratio") or 0) < min_reach:
            ok = False
            reason = (
                f"1500窗信号可达性{float(reach_stats.get('reachable_ratio') or 0) * 100:.2f}%"
                f"（门槛 {min_reach * 100:.2f}%）"
            )
    side_stats: Dict[str, Any] = {}
    if best_tpl:
        if last_train_result is None:
            params = _build_run_params(best_tpl, best_tact, symbol=sym)
            last_train_result = run_backtest(
                df_train,
                DecisionParams.from_dict(params),
                regime_train,
                initial_capital=cap,
                symbol=sym,
            )
        side_stats = dict(
            (
                analyze_trades(last_train_result.trades, regime_train).get(
                    "side_stats"
                )
                or {}
            )
        )
    narrative = (
        f"最近{len(df)}根：{best_tpl} entry={best_tact.get('entry_threshold')} "
        f"exit={best_tact.get('exit_threshold')} "
        f"收益{float(best_full.get('total_return') or 0) * 100:+.2f}% "
        f"PF={float(best_full.get('profit_factor') or 0):.2f} "
        f"尾段{float(best_tail.get('total_return') or 0) * 100:+.2f}% "
        f"({reason})"
    )

    return {
        "adopted": ok,
        "skipped": False,
        "reason": reason,
        "bars": len(df),
        "train_cut": train_cut,
        "validation_mode": "recent_window",
        "regime_note": note,
        "combinations_tried": tried,
        "template": best_tpl,
        "tactical_params": {k: best_tact[k] for k in TACTICAL_FLOAT_FIELDS if k in best_tact},
        "full_window": best_full,
        "tail_window": best_tail,
        "recent_return_pct": round(float(best_full.get("total_return") or 0) * 100, 2),
        "tail_return_pct": round(float(best_tail.get("total_return") or 0) * 100, 2),
        "win_rate_pct": round(float(best_full.get("win_rate") or 0) * 100, 1),
        "total_trades": best_full.get("total_trades"),
        "profit_factor": best_full.get("profit_factor"),
        "pick_score": round(float(best_rank[0]), 4),
        "reachable_ratio": reach_stats.get("reachable_ratio"),
        "reachable_sub_pf": reach_stats.get("reachable_sub_pf"),
        "side_stats": side_stats,
        "refine_rounds": refine_log,
        "narrative": narrative,
        "param_source": "recent_1500" if ok else "grid",
    }


def apply_recent_pick_to_best(
    best: Dict[str, Any],
    symbol: str,
    *,
    capital: Optional[float] = None,
    refresh_klines: bool = False,
) -> Dict[str, Any]:
    """在 L1 best 上叠加 L3；采纳则覆盖 template / tactical_params / summary。"""
    best = dict(best)
    summary = dict(best.get("summary") or {})
    l1_tpl = best.get("template")
    summary["l1_template"] = l1_tpl
    summary["l1_tactical_params"] = dict(best.get("tactical_params") or {})
    summary["l1_param_source"] = summary.get("param_source") or "grid"
    summary["l1_val_return"] = summary.get("val_return")
    summary["l1_train_return"] = summary.get(
        "train_return", summary.get("total_return")
    )
    summary["l1_wf_passed_folds"] = summary.get("wf_passed_folds")

    pick = pick_best_on_recent_window(
        symbol,
        capital=capital,
        l1_summary=summary,
        refresh_klines=refresh_klines,
    )
    summary["recent_pick"] = pick
    pick = summary.get("recent_pick") or pick

    if pick.get("adopted"):
        tpl = str(pick.get("template") or l1_tpl or "balanced")
        tact = dict(pick.get("tactical_params") or {})
        best["template"] = tpl
        best["tactical_params"] = tact
        best["params"] = _build_run_params(tpl, tact, symbol=symbol)
        summary["param_source"] = "recent_1500"
        summary["recent_applied"] = True
        summary["recent_return_pct"] = pick.get("recent_return_pct")
        summary["tail_return_pct"] = pick.get("tail_return_pct")
    else:
        summary["recent_applied"] = False
        summary["recent_return_pct"] = pick.get("recent_return_pct")
        if not summary.get("param_source"):
            summary["param_source"] = summary.get("l1_param_source") or "grid"

    from moss_quant.optimize_policy import enrich_summary

    summary = enrich_summary(summary)
    best["summary"] = summary
    if summary.get("sync_block_reason") and not summary.get("sync_allowed"):
        summary["l1_sync_block_reason"] = summary.get("sync_block_reason")
    return best
