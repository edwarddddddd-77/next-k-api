"""70% 训练窗归因总结 + 30% 验证；网格赢家后的局部精修（最多 N 轮早停）。"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from moss_quant import config as cfg
from moss_quant.core.backtest import run_backtest
from moss_quant.core.decision import DecisionParams
from moss_quant.core.engine import Trade

_SIGNAL_EXITS = frozenset(
    {
        "close_long",
        "close_short",
        "flip_close_long",
        "flip_close_short",
    }
)

TACTICAL_FIELDS = (
    "entry_threshold",
    "sl_atr_mult",
    "tp_rr_ratio",
    "exit_threshold",
    "regime_sensitivity",
)


def _clamp_entry(v: float) -> float:
    return round(max(0.36, min(0.56, float(v))), 4)


def _clamp_sl(v: float) -> float:
    return round(max(1.5, min(4.0, float(v))), 2)


def _clamp_tp(v: float) -> float:
    return round(max(1.5, min(4.0, float(v))), 2)


def _clamp_exit(v: float) -> float:
    return round(max(0.08, min(0.25, float(v))), 4)


def _tactical_key(t: Dict[str, Any]) -> Tuple:
    return tuple(round(float(t.get(k) or 0), 4) for k in TACTICAL_FIELDS if k in t)


def _trade_regime(regime: Optional[pd.Series], entry_idx: int) -> str:
    if regime is None or len(regime) == 0:
        return "UNKNOWN"
    i = max(0, min(int(entry_idx), len(regime) - 1))
    return str(regime.iloc[i]).upper()


def _side_stats(trades: Sequence[Trade]) -> Dict[str, Any]:
    longs = [t for t in trades if int(t.direction) == 1]
    shorts = [t for t in trades if int(t.direction) == -1]

    def _wr(ts: List[Trade]) -> Optional[float]:
        if not ts:
            return None
        return round(sum(1 for t in ts if float(t.gross_pnl) > 0) / len(ts), 4)

    return {
        "long_count": len(longs),
        "short_count": len(shorts),
        "long_win_rate": _wr(longs),
        "short_win_rate": _wr(shorts),
    }


def analyze_trades(
    trades: Sequence[Trade],
    regime: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    ts = list(trades)
    n = len(ts)
    if n == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "insufficient_sample": True,
        }

    wins = [t for t in ts if float(t.gross_pnl) > 0]
    losses = [t for t in ts if float(t.gross_pnl) <= 0]
    win_rate = len(wins) / n
    gross_win = sum(float(t.gross_pnl) for t in wins)
    gross_loss = abs(
        sum(float(t.gross_pnl) for t in losses if float(t.gross_pnl) < 0)
    )
    pf = gross_win / gross_loss if gross_loss > 0 else (2.0 if gross_win > 0 else 0.0)

    exit_counts: Dict[str, int] = {}
    for t in ts:
        r = str(t.exit_reason or "unknown")
        exit_counts[r] = exit_counts.get(r, 0) + 1
    signal_exits = sum(exit_counts.get(k, 0) for k in _SIGNAL_EXITS)

    regime_buckets: Dict[str, Dict[str, Any]] = {}
    for t in ts:
        rg = _trade_regime(regime, int(t.entry_idx))
        b = regime_buckets.setdefault(rg, {"trades": 0, "wins": 0, "pnl_sum": 0.0})
        b["trades"] += 1
        if float(t.gross_pnl) > 0:
            b["wins"] += 1
        b["pnl_sum"] += float(t.gross_pnl)
    for b in regime_buckets.values():
        b["win_rate"] = round(b["wins"] / b["trades"], 4) if b["trades"] else 0.0
        b["pnl_sum"] = round(b["pnl_sum"], 4)

    return {
        "total_trades": n,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(pf, 4),
        "exit_reasons": exit_counts,
        "signal_exit_ratio": round(signal_exits / n, 4),
        "liquidation_count": sum(1 for t in ts if str(t.exit_reason) == "liquidation"),
        "avg_win_pct": round(
            sum(float(t.pnl_pct) for t in wins) / len(wins) if wins else 0.0, 4
        ),
        "avg_loss_pct": round(
            sum(float(t.pnl_pct) for t in losses) / len(losses) if losses else 0.0, 4
        ),
        "regime_at_entry": regime_buckets,
        "side_stats": _side_stats(ts),
        "insufficient_sample": n < int(cfg.MOSS_QUANT_OPTIMIZE_MIN_TRAIN_TRADES),
    }


def suggest_tactical_adjustments(
    analysis: Dict[str, Any],
    tactical: Dict[str, Any],
    *,
    template: str,
    regime_note: str = "",
) -> Dict[str, Any]:
    tact = {k: tactical[k] for k in TACTICAL_FIELDS if k in tactical}
    causes: List[str] = []
    adjustments: List[Dict[str, Any]] = []

    if analysis.get("insufficient_sample"):
        return {
            "adjustments": [],
            "tuned_tactical": tact,
            "causes": ["训练窗成交笔数不足"],
            "narrative": "训练窗样本过少，保持网格参数。",
            "skipped": True,
        }

    wr = float(analysis.get("win_rate") or 0)
    pf = float(analysis.get("profit_factor") or 0)
    flip_ratio = float(analysis.get("signal_exit_ratio") or 0)
    liq = int(analysis.get("liquidation_count") or 0)
    avg_win = float(analysis.get("avg_win_pct") or 0)
    avg_loss = float(analysis.get("avg_loss_pct") or 0)
    note = str(regime_note or "").strip()
    tpl = str(template or "balanced").lower()
    entry = float(tact.get("entry_threshold") or 0.44)
    sl = float(tact.get("sl_atr_mult") or 2.0)
    tp = float(tact.get("tp_rr_ratio") or 2.5)
    exit_th = float(tact.get("exit_threshold") or 0.12)

    def _propose(field: str, new_val: float, reason: str) -> None:
        if field not in tact:
            return
        old_f = float(tact[field])
        nv = new_val
        if field == "entry_threshold":
            nv = _clamp_entry(nv)
        elif field == "sl_atr_mult":
            nv = _clamp_sl(nv)
        elif field == "tp_rr_ratio":
            nv = _clamp_tp(nv)
        elif field == "exit_threshold":
            nv = _clamp_exit(nv)
        if abs(old_f - nv) < 1e-6:
            return
        adjustments.append(
            {"field": field, "from": old_f, "to": nv, "reason": reason}
        )
        tact[field] = nv
        causes.append(reason)

    if wr < 0.42 and flip_ratio >= 0.40:
        _propose(
            "entry_threshold",
            entry + 0.04,
            "胜率偏低且反手平仓占比高",
        )
    if pf < 1.0 and avg_loss < 0 and abs(avg_loss) > abs(avg_win) * 1.15:
        _propose("sl_atr_mult", sl + 0.5, "亏损幅度大于盈利，放宽止损")
        if tp > 2.0:
            _propose("tp_rr_ratio", tp - 0.5, "降低止盈倍数提高兑现")
    if note == "sideways_heavy" and tpl in ("momentum", "trend") and wr < 0.45:
        _propose(
            "entry_threshold",
            float(tact.get("entry_threshold") or entry) + 0.04,
            "震荡训练窗+趋势模板需更高门槛",
        )
        _propose("exit_threshold", exit_th - 0.02, "震荡市宜更早离场")
    if liq > 0:
        _propose(
            "entry_threshold",
            float(tact.get("entry_threshold") or entry) + 0.04,
            "训练窗爆仓，收紧开仓",
        )
        _propose(
            "sl_atr_mult",
            float(tact.get("sl_atr_mult") or sl) + 0.5,
            "训练窗爆仓，放宽止损距离",
        )
    if wr >= 0.42 and pf < 0.9 and int(analysis.get("total_trades") or 0) >= 20:
        _propose(
            "entry_threshold",
            float(tact.get("entry_threshold") or entry) + 0.04,
            "笔数多但盈亏比差",
        )

    if adjustments:
        narrative = (
            f"训练窗胜率 {wr * 100:.0f}%。建议："
            + "；".join(f"{a['field']} {a['from']}→{a['to']}" for a in adjustments)
            + "。"
        )
    else:
        narrative = f"训练窗胜率 {wr * 100:.0f}%，保持当前战术参数。"

    return {
        "adjustments": adjustments,
        "tuned_tactical": tact,
        "causes": causes,
        "narrative": narrative,
        "skipped": False,
    }


def compare_holdout_validation(
    base_summary: Dict[str, Any],
    tuned_summary: Dict[str, Any],
) -> Dict[str, Any]:
    br = float(base_summary.get("total_return") or 0)
    tr = float(tuned_summary.get("total_return") or 0)
    bs = float(base_summary.get("sharpe") or 0)
    ts = float(tuned_summary.get("sharpe") or 0)
    bm = abs(float(base_summary.get("max_drawdown") or 0))
    tm = abs(float(tuned_summary.get("max_drawdown") or 0))

    return_improved = tr > br + 0.002
    sharpe_improved = ts > bs + 0.05
    mdd_ok = tm <= bm + 0.03
    adopted = mdd_ok and (return_improved or sharpe_improved)
    if not int(base_summary.get("total_trades") or 0):
        adopted = False
        reason = "验证窗无成交"
    elif adopted:
        reason = "验证收益或 Sharpe 改善且回撤可控"
    elif tr >= br - 0.005 and ts >= bs - 0.02 and mdd_ok:
        adopted = True
        reason = "验证不劣于基线"
    else:
        reason = "验证未改善"

    return {
        "adopted": adopted,
        "validation_reason": reason,
        "base_val_return": round(br, 4),
        "tuned_val_return": round(tr, 4),
        "base_val_sharpe": round(bs, 4),
        "tuned_val_sharpe": round(ts, 4),
        "delta_return": round(tr - br, 4),
        "delta_sharpe": round(ts - bs, 4),
    }


def _val_rank_key(summary: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        float(summary.get("total_return") or 0),
        float(summary.get("sharpe") or 0),
        -abs(float(summary.get("max_drawdown") or 0)),
    )


def _wf_rank_key(val_block: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """与看板一致：WF 验证收益 / Sharpe / 过关折数。"""
    vs = val_block.get("validation_summary") or {}
    return (
        float(val_block.get("val_return") or 0),
        float(val_block.get("val_sharpe") or 0),
        float(val_block.get("wf_passed_folds") or 0),
        -abs(float(vs.get("max_drawdown") or 0)),
    )


def wf_validation_to_summary(val_block: Dict[str, Any]) -> Dict[str, Any]:
    """将 WF 验证块转为 compare_holdout_validation 可用的 summary。"""
    vs = val_block.get("validation_summary") or {}
    return {
        "total_return": float(val_block.get("val_return") or 0),
        "sharpe": float(val_block.get("val_sharpe") or 0),
        "max_drawdown": float(vs.get("max_drawdown") or 0),
        "total_trades": int(vs.get("total_trades") or 0),
    }


def build_neighbor_candidates(
    base: Dict[str, Any],
    suggestion: Optional[Dict[str, Any]] = None,
    *,
    max_candidates: int = 14,
) -> List[Dict[str, Any]]:
    """围绕当前战术参数 ±1 档邻域 + 归因建议点。"""
    seen: set = set()
    out: List[Dict[str, Any]] = []

    def _add(t: Dict[str, Any]) -> None:
        tt = {k: base[k] for k in TACTICAL_FIELDS if k in base}
        for k, v in t.items():
            if k in TACTICAL_FIELDS:
                tt[k] = v
        key = _tactical_key(tt)
        if key in seen:
            return
        seen.add(key)
        out.append(tt)

    _add(base)
    if suggestion:
        _add(suggestion)

    entry = float(base.get("entry_threshold") or 0.44)
    sl = float(base.get("sl_atr_mult") or 2.0)
    tp = float(base.get("tp_rr_ratio") or 2.5)
    ex = float(base.get("exit_threshold") or 0.12)

    for de in (-0.04, 0.04):
        _add({"entry_threshold": _clamp_entry(entry + de)})
    for ds in (-0.5, 0.5):
        _add({"sl_atr_mult": _clamp_sl(sl + ds)})
    for dt in (-0.5, 0.5):
        _add({"tp_rr_ratio": _clamp_tp(tp + dt)})
    for dx in (-0.02, 0.02):
        _add({"exit_threshold": _clamp_exit(ex + dx)})

    return out[:max_candidates]


def refine_best_locally(
    *,
    df_train: pd.DataFrame,
    regime_train: pd.Series,
    symbol: str,
    template: str,
    tactical: Dict[str, Any],
    capital: float,
    regime_note: str,
    build_params_fn: Callable[..., Dict[str, Any]],
    validate_wf_fn: Callable[[Dict[str, Any], str], Dict[str, Any]],
    grid_wf_validation: Dict[str, Any],
    max_rounds: Optional[int] = None,
) -> Dict[str, Any]:
    """
    多轮：70% 归因 → 邻域候选 → WF 滚动验证（与看板同口径）选最优；无改善早停。
    """
    max_r = int(
        max_rounds
        if max_rounds is not None
        else cfg.MOSS_QUANT_OPTIMIZE_LOCAL_REFINE_MAX_ROUNDS
    )
    current = {k: tactical[k] for k in TACTICAL_FIELDS if k in tactical}
    params = build_params_fn(template, current, symbol=symbol)
    p = DecisionParams.from_dict(params)
    train_result = run_backtest(
        df_train, p, regime_train, initial_capital=capital, symbol=symbol
    )

    baseline_wf = dict(grid_wf_validation or {})
    baseline_summary = wf_validation_to_summary(baseline_wf)
    best_wf = dict(baseline_wf)
    best_tactical = dict(current)
    grid_val_return = float(baseline_wf.get("val_return") or 0)
    grid_val_sharpe = float(baseline_wf.get("val_sharpe") or 0)
    grid_wf_passed = int(baseline_wf.get("wf_passed_folds") or 0)
    grid_wf_ok = bool(baseline_wf.get("wf_validation_passed"))

    rounds_log: List[Dict[str, Any]] = []
    improved = False

    for rnd in range(1, max_r + 1):
        analysis = analyze_trades(train_result.trades, regime_train)
        suggestion = suggest_tactical_adjustments(
            analysis,
            best_tactical,
            template=template,
            regime_note=regime_note,
        )
        candidates = build_neighbor_candidates(
            best_tactical,
            suggestion.get("tuned_tactical"),
        )

        round_best_tact = best_tactical
        round_best_wf = best_wf
        tried: List[Dict[str, Any]] = []

        for cand in candidates:
            wf_block = validate_wf_fn(cand, template)
            tried.append(
                {
                    "tactical": cand,
                    "val_return": wf_block.get("val_return"),
                    "val_sharpe": wf_block.get("val_sharpe"),
                    "wf_passed": wf_block.get("wf_passed_folds"),
                    "wf_ok": wf_block.get("wf_validation_passed"),
                }
            )
            if _wf_rank_key(wf_block) > _wf_rank_key(round_best_wf):
                round_best_wf = wf_block
                round_best_tact = cand

        round_improved = _wf_rank_key(round_best_wf) > _wf_rank_key(best_wf)
        rounds_log.append(
            {
                "round": rnd,
                "train_analysis": analysis,
                "suggestion_narrative": suggestion.get("narrative"),
                "candidates_tested": len(candidates),
                "improved": round_improved,
                "best_tactical": round_best_tact,
                "val_return": round_best_wf.get("val_return"),
                "val_sharpe": round_best_wf.get("val_sharpe"),
                "wf_passed_folds": round_best_wf.get("wf_passed_folds"),
                "tried": tried[:8],
            }
        )

        if round_improved:
            improved = True
            best_wf = round_best_wf
            best_tactical = round_best_tact
            params = build_params_fn(template, best_tactical, symbol=symbol)
            train_result = run_backtest(
                df_train,
                DecisionParams.from_dict(params),
                regime_train,
                initial_capital=capital,
                symbol=symbol,
            )
        else:
            break

    best_summary = wf_validation_to_summary(best_wf)
    final_cmp = compare_holdout_validation(baseline_summary, best_summary)
    refined_wf_ok = bool(best_wf.get("wf_validation_passed"))
    adopt_ok = bool(final_cmp.get("adopted")) and (
        refined_wf_ok or not grid_wf_ok
    )

    return {
        "enabled": True,
        "validation_mode": "walk_forward",
        "max_rounds": max_r,
        "rounds_run": len(rounds_log),
        "rounds": rounds_log,
        "improved_vs_grid": improved and adopt_ok,
        "grid_val_return": round(grid_val_return, 4),
        "grid_val_sharpe": round(grid_val_sharpe, 4),
        "grid_wf_passed_folds": grid_wf_passed,
        "refined_val_return": round(float(best_wf.get("val_return") or 0), 4),
        "refined_val_sharpe": round(float(best_wf.get("val_sharpe") or 0), 4),
        "refined_wf_passed_folds": int(best_wf.get("wf_passed_folds") or 0),
        "refined_wf_validation": best_wf,
        "refined_tactical_params": best_tactical,
        "holdout_comparison": final_cmp,
        "narrative": (
            f"局部精修 {len(rounds_log)} 轮（WF）：验证收益 "
            f"{grid_val_return * 100:+.2f}% → {float(best_wf.get('val_return') or 0) * 100:+.2f}%"
            f"，{int(best_wf.get('wf_passed_folds') or 0)}/{int(best_wf.get('wf_folds') or 3)} 折过关；"
            f"{final_cmp.get('validation_reason')}。"
        ),
    }


def run_post_grid_pipeline(
    *,
    df_train: pd.DataFrame,
    regime_train: pd.Series,
    symbol: str,
    template: str,
    tactical: Dict[str, Any],
    capital: float,
    regime_note: str,
    build_params_fn: Callable[..., Dict[str, Any]],
    validate_wf_fn: Callable[[Dict[str, Any], str], Dict[str, Any]],
    grid_wf_validation: Dict[str, Any],
) -> Dict[str, Any]:
    """70% 训练窗归因 + WF 局部精修（验证口径与看板一致）。"""
    out: Dict[str, Any] = {
        "split": {"train_ratio": 0.7, "val_ratio": 0.3},
        "validation_mode": "walk_forward",
    }

    if cfg.MOSS_QUANT_OPTIMIZE_TUNING_DIAG_ENABLED:
        params = build_params_fn(template, tactical, symbol=symbol)
        train_result = run_backtest(
            df_train,
            DecisionParams.from_dict(params),
            regime_train,
            initial_capital=capital,
            symbol=symbol,
        )
        analysis = analyze_trades(train_result.trades, regime_train)
        suggestion = suggest_tactical_adjustments(
            analysis, tactical, template=template, regime_note=regime_note
        )
        base_wf = dict(grid_wf_validation or {})
        tuned_wf = validate_wf_fn(
            suggestion.get("tuned_tactical") or tactical, template
        )
        hold = compare_holdout_validation(
            wf_validation_to_summary(base_wf),
            wf_validation_to_summary(tuned_wf),
        )
        out["tuning_diagnosis"] = {
            "train_analysis": analysis,
            "suggestion": suggestion,
            "holdout_validation": hold,
            "suggestion_wf": {
                "val_return": tuned_wf.get("val_return"),
                "val_sharpe": tuned_wf.get("val_sharpe"),
                "wf_passed_folds": tuned_wf.get("wf_passed_folds"),
            },
            "narrative_full": str(suggestion.get("narrative") or "")
            + f" WF建议验证：{hold.get('validation_reason', '')}",
        }

    working_tactical = dict(tactical)
    if cfg.MOSS_QUANT_OPTIMIZE_LOCAL_REFINE_ENABLED:
        refine = refine_best_locally(
            df_train=df_train,
            regime_train=regime_train,
            symbol=symbol,
            template=template,
            tactical=working_tactical,
            capital=capital,
            regime_note=regime_note,
            build_params_fn=build_params_fn,
            validate_wf_fn=validate_wf_fn,
            grid_wf_validation=grid_wf_validation,
        )
        out["local_refine"] = refine
        if refine.get("improved_vs_grid"):
            working_tactical = dict(refine.get("refined_tactical_params") or tactical)
            out["final_tactical_params"] = working_tactical
            out["param_source"] = "local_refine"
        else:
            out["final_tactical_params"] = tactical
            out["param_source"] = "grid"
    else:
        out["final_tactical_params"] = tactical
        out["param_source"] = "grid"

    return out
