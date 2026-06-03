"""寻优评分、样本外验证、币池分层（每日寻优）。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import sqlite3

from moss_quant import config as cfg
from moss_quant.daily_auto_enable import evaluate_profile_auto_enable


def hard_reject_reason(summary: Optional[Dict[str, Any]]) -> Optional[str]:
    """训练窗硬淘汰（用于网格候选）。"""
    if not summary or summary.get("error"):
        return "invalid_summary"
    if int(summary.get("blowup_count") or 0) > 0:
        return "回测爆仓"
    trades = int(summary.get("total_trades") or 0)
    if trades < int(cfg.MOSS_QUANT_OPTIMIZE_MIN_TRAIN_TRADES):
        return "训练回合不足"
    if float(summary.get("total_return") or 0) <= 0:
        return "训练收益≤0"
    mdd = abs(float(summary.get("max_drawdown") or 0))
    if mdd > float(cfg.MOSS_QUANT_OPTIMIZE_MAX_TRAIN_DRAWDOWN):
        return "训练回撤过大"
    return None


def composite_optimize_score(summary: Dict[str, Any]) -> float:
    """复合评分：收益 + Sharpe + 回撤 + 笔数；硬淘汰返回 -999。"""
    reason = hard_reject_reason(summary)
    if reason:
        return -999.0
    ret = float(summary.get("total_return") or 0)
    sharpe = max(0.0, min(float(summary.get("sharpe") or 0), 2.0)) / 2.0
    mdd = abs(float(summary.get("max_drawdown") or 0))
    trades = int(summary.get("total_trades") or 0)
    trade_factor = min(trades / 12.0, 1.0)
    return round(
        0.45 * ret + 0.25 * sharpe + 0.20 * (1.0 - mdd) + 0.10 * trade_factor,
        6,
    )


def stability_gap(train_return: float, val_return: float) -> float:
    return abs(float(train_return or 0) - float(val_return or 0))


def stability_penalty(train_return: float, val_return: float) -> float:
    return float(cfg.MOSS_QUANT_OPTIMIZE_STABILITY_PENALTY) * stability_gap(
        train_return, val_return
    )


def stability_adjusted_val_score(
    val_sharpe: float,
    *,
    train_return: float,
    val_return: float,
) -> float:
    """验证 Sharpe 为主，惩罚训练/验证收益差距过大。"""
    return round(
        float(val_sharpe or 0) - stability_penalty(train_return, val_return),
        6,
    )


def train_val_ratio_ok(train_return: float, val_return: float) -> Optional[str]:
    """训练/验证收益比异常 → 过拟合嫌疑。"""
    tr = float(train_return or 0)
    vr = float(val_return or 0)
    if tr <= 0 or vr <= 0:
        return None
    ratio = tr / vr if abs(vr) > 1e-9 else 999.0
    if ratio > float(cfg.MOSS_QUANT_OPTIMIZE_MAX_TRAIN_VAL_RATIO):
        return "训练验证收益比过高"
    if ratio < float(cfg.MOSS_QUANT_OPTIMIZE_MIN_TRAIN_VAL_RATIO):
        return "训练验证收益比过低"
    return None


_ALL_TEMPLATES = ("balanced", "momentum", "trend", "mean_revert")


def templates_for_regime(regime_adj: Dict[str, Any]) -> Tuple[str, ...]:
    """按训练窗 regime 占比缩小模板搜索空间。"""
    if not cfg.MOSS_QUANT_OPTIMIZE_REGIME_FILTER_TEMPLATES:
        return _ALL_TEMPLATES
    note = str(regime_adj.get("regime_note") or "")
    if note == "sideways_heavy":
        return ("mean_revert", "balanced")
    if note == "trend_heavy":
        return ("trend", "momentum", "balanced")
    return _ALL_TEMPLATES


def split_train_validation_df(
    df: pd.DataFrame,
    train_ratio: Optional[float] = None,
    *,
    min_bars: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ratio = float(train_ratio if train_ratio is not None else cfg.MOSS_QUANT_OPTIMIZE_TRAIN_RATIO)
    ratio = max(0.5, min(0.85, ratio))
    n = int(len(df))
    min_bars = int(min_bars if min_bars is not None else cfg.MOSS_QUANT_OPTIMIZE_MIN_BARS)
    if n < min_bars:
        return df.copy(), df.iloc[0:0].copy()
    cut = max(int(n * ratio), int(n * 0.5))
    cut = min(cut, n - max(48, int(n * 0.15)))
    if cut <= 0 or cut >= n:
        return df.copy(), df.iloc[0:0].copy()
    return df.iloc[:cut].copy().reset_index(drop=True), df.iloc[cut:].copy().reset_index(
        drop=True
    )


def split_walk_forward_folds(
    df: pd.DataFrame,
    *,
    n_folds: Optional[int] = None,
    train_ratio: Optional[float] = None,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """把时间轴切成多段，每段内 70/30 训练/验证（滚动多考几次试）。"""
    k = max(1, int(n_folds if n_folds is not None else cfg.MOSS_QUANT_OPTIMIZE_WF_FOLDS))
    if k <= 1:
        tr, va = split_train_validation_df(df, train_ratio)
        return [(tr, va)]

    n = int(len(df))
    seg = max(96, n // k)
    ratio = train_ratio
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    for i in range(k):
        start = i * seg
        end = n if i >= k - 1 else min(n, (i + 1) * seg)
        if end - start < 96:
            continue
        chunk = df.iloc[start:end].copy().reset_index(drop=True)
        fold_min = max(96, int(cfg.MOSS_QUANT_OPTIMIZE_MIN_BARS) // max(k * 2, 2))
        tr, va = split_train_validation_df(chunk, ratio, min_bars=fold_min)
        if len(tr) >= 48 and len(va) >= 48:
            folds.append((tr, va))
    if not folds:
        tr, va = split_train_validation_df(df, train_ratio)
        return [(tr, va)]
    return folds


def aggregate_walk_forward_validation(
    fold_validations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """多折验证汇总：达标折数、中位 Sharpe/收益、稳定性分。"""
    passed = [v for v in fold_validations if v.get("validation_passed")]
    n_pass = len(passed)
    min_pass = int(cfg.MOSS_QUANT_OPTIMIZE_WF_MIN_PASS_FOLDS)
    n_folds = len(fold_validations)
    sharpes = [
        float(v.get("val_sharpe") or 0)
        for v in passed
        if v.get("val_sharpe") is not None
    ]
    returns = [
        float(v.get("val_return") or 0)
        for v in passed
        if v.get("val_return") is not None
    ]
    med_sharpe = float(median(sharpes)) if sharpes else 0.0
    med_return = float(median(returns)) if returns else 0.0
    penalties = [float(v.get("gate_penalty") or 0) for v in fold_validations]
    ratios = [float(v.get("gate_extreme_ratio") or 0) for v in fold_validations]
    med_penalty = float(median(penalties)) if penalties else 0.0
    med_ratio = float(median(ratios)) if ratios else 0.0
    wf_ok = n_pass >= min_pass if n_folds > 1 else bool(passed)
    reason = "滚动验证通过" if wf_ok else f"滚动验证仅 {n_pass}/{n_folds} 折达标(需>={min_pass})"
    return {
        "wf_folds": n_folds,
        "wf_passed_folds": n_pass,
        "wf_min_pass_folds": min_pass,
        "wf_validation_passed": wf_ok,
        "wf_reason": reason,
        "val_sharpe": med_sharpe,
        "val_return": med_return,
        "gate_penalty": med_penalty,
        "gate_extreme_ratio": med_ratio,
        "validation_passed": wf_ok,
        "validation_reason": reason if wf_ok else reason,
        "fold_validations": fold_validations,
    }


def regime_tactical_adjustments(regime: pd.Series) -> Dict[str, Any]:
    """训练窗 regime 占比 → 战术微调（不替换模板）。"""
    if regime is None or len(regime) == 0:
        return {}
    vc = regime.astype(str).value_counts(normalize=True)
    sideways = float(
        vc.get("SIDEWAYS", 0)
        + vc.get("CHOP", 0)
        + vc.get("RANGE", 0)
    )
    trend = float(
        vc.get("TREND_UP", 0)
        + vc.get("TREND_DOWN", 0)
        + vc.get("BULL", 0)
        + vc.get("BEAR", 0)
        + vc.get("UPTREND", 0)
        + vc.get("DOWNTREND", 0)
    )
    out: Dict[str, Any] = {}
    if sideways >= 0.55:
        out["entry_threshold_bump"] = 0.04
        out["tp_rr_mult"] = 0.9
        out["regime_note"] = "sideways_heavy"
    elif trend >= 0.45:
        out["entry_threshold_bump"] = -0.02
        out["regime_note"] = "trend_heavy"
    return out


def apply_regime_to_tactical(tactical: Dict[str, Any], regime_adj: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(tactical)
    bump = float(regime_adj.get("entry_threshold_bump") or 0)
    if bump and "entry_threshold" in out:
        out["entry_threshold"] = round(
            max(0.05, min(0.60, float(out["entry_threshold"]) + bump)), 4
        )
    mult = float(regime_adj.get("tp_rr_mult") or 1.0)
    if mult != 1.0 and "tp_rr_ratio" in out:
        out["tp_rr_ratio"] = round(max(1.0, min(10.0, float(out["tp_rr_ratio"]) * mult)), 4)
    return out


def validation_fail_reason(summary: Optional[Dict[str, Any]]) -> Optional[str]:
    if not summary or summary.get("error"):
        return "验证窗无结果"
    if float(summary.get("total_return") or 0) <= 0:
        return "验证收益≤0"
    mdd = abs(float(summary.get("max_drawdown") or 0))
    if mdd > float(cfg.MOSS_QUANT_OPTIMIZE_MAX_VAL_DRAWDOWN):
        return "验证回撤过大"
    trades = int(summary.get("total_trades") or 0)
    if trades < int(cfg.MOSS_QUANT_OPTIMIZE_MIN_VAL_TRADES):
        return "验证回合不足"
    return None


def evaluate_validation(summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    reason = validation_fail_reason(summary)
    return {
        "validation_passed": reason is None,
        "validation_reason": reason or "验证通过",
    }


def classify_pool_tier(summary: Dict[str, Any]) -> Dict[str, Any]:
    """A=可交易 B=观察 C=剔除（用于看板与同步）。"""
    if summary.get("error"):
        return {"pool_tier": "C", "pool_label": "剔除", "pool_reason": "寻优失败"}

    gate = evaluate_profile_auto_enable(summary)
    val_passed = bool(summary.get("validation_passed"))
    val_required = bool(cfg.MOSS_QUANT_OPTIMIZE_REQUIRE_VALIDATION)

    if not gate.get("auto_enabled"):
        return {
            "pool_tier": "C",
            "pool_label": "剔除",
            "pool_reason": str(gate.get("auto_enable_reason") or "不达标"),
        }
    if val_required and not val_passed:
        reason = str(summary.get("validation_reason") or "验证未通过")
        return {"pool_tier": "B", "pool_label": "观察", "pool_reason": reason}
    return {
        "pool_tier": "A",
        "pool_label": "可交易",
        "pool_reason": "训练+验证+门槛通过",
    }


def sync_deny_reason(summary: Dict[str, Any]) -> Optional[str]:
    """本批次寻优结果不可同步的原因（不含纸面近期亏损）。"""
    if summary.get("error"):
        return "寻优失败"
    tier = classify_pool_tier(summary)
    if tier["pool_tier"] != "A":
        return str(tier.get("pool_reason") or "非 A 池")
    if bool(cfg.MOSS_QUANT_OPTIMIZE_REQUIRE_VALIDATION) and not summary.get(
        "validation_passed"
    ):
        return str(summary.get("validation_reason") or "验证未通过")
    if summary.get("wf_validation_passed") is False:
        return str(summary.get("wf_reason") or "滚动验证未达标")
    tr = float(summary.get("total_return") or summary.get("train_return") or 0)
    vr = float(summary.get("val_return") or 0)
    ratio_reason = train_val_ratio_ok(tr, vr)
    if ratio_reason:
        return ratio_reason
    auto_gate = evaluate_profile_auto_enable(summary)
    if not auto_gate.get("auto_enabled"):
        return str(
            summary.get("auto_enable_reason")
            or auto_gate.get("auto_enable_reason")
            or "达标门禁未通过"
        )
    tail_reason = _l3_sync_tail_deny_reason(summary)
    if tail_reason:
        return tail_reason
    return None


def _l3_sync_tail_deny_reason(summary: Dict[str, Any]) -> Optional[str]:
    """
    L3 仅在有「过关组合」但尾段太差时拦同步；无合格组合不否决 L1 网格同步。
    """
    if not cfg.MOSS_QUANT_SYNC_REQUIRE_RECENT_TAIL_OK or not cfg.MOSS_QUANT_RECENT_PICK_ENABLED:
        return None
    rp = summary.get("recent_pick")
    if not isinstance(rp, dict) or rp.get("skipped"):
        return None
    reason = str(rp.get("reason") or "")
    if "无满足门槛" in reason:
        return None
    tail_pct = rp.get("tail_return_pct")
    if tail_pct is None:
        return None
    floor_pct = float(cfg.MOSS_QUANT_RECENT_PICK_TAIL_MIN_RETURN) * 100
    if float(tail_pct) < floor_pct:
        return (
            f"1500尾段收益{float(tail_pct):.2f}%"
            f"（门槛 {floor_pct:.0f}%）"
        )
    if rp.get("adopted") is False and rp.get("recent_return_pct") is not None:
        return reason or "L3未采纳"
    return None


def can_sync_profile_params(summary: Dict[str, Any]) -> bool:
    """是否允许将本批次寻优结果写入纸面 Profile（仅看寻优 summary）。"""
    return sync_deny_reason(summary) is None


def paper_recent_pnl_block_reason(
    conn: sqlite3.Connection,
    profile_id: int,
    *,
    profile_capital: Optional[float] = None,
) -> Optional[str]:
    """启用 Profile 近 N 日纸面收益过差 → 本批不同步覆盖参数。"""
    if not cfg.MOSS_QUANT_SYNC_BLOCK_RECENT_LOSS_ENABLED:
        return None
    cap = float(profile_capital or cfg.MOSS_QUANT_PROFILE_CAPITAL)
    if cap <= 0:
        return None
    days = int(cfg.MOSS_QUANT_SYNC_BLOCK_LOSS_DAYS)
    floor_pct = float(cfg.MOSS_QUANT_SYNC_BLOCK_LOSS_PCT)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """SELECT
               COALESCE(SUM(
                   CASE WHEN outcome IS NOT NULL AND outcome_at_utc >= ?
                        THEN pnl_usdt ELSE 0 END
               ), 0) AS realized,
               COALESCE(SUM(
                   CASE WHEN outcome IS NULL THEN unrealized_pnl_usdt ELSE 0 END
               ), 0) AS unrealized
           FROM moss_signals WHERE profile_id=?""",
        (cutoff, int(profile_id)),
    ).fetchone()
    if not row:
        return None
    pnl = float(row["realized"] or 0) + float(row["unrealized"] or 0)
    pct = pnl / cap
    if pct <= floor_pct:
        return f"近{days}日纸面收益 {pct * 100:.1f}%（阈值 {floor_pct * 100:.0f}%）"
    return None


def enrich_summary(
    summary: Dict[str, Any],
    *,
    conn: Optional[sqlite3.Connection] = None,
    profile_id: Optional[int] = None,
    profile_capital: Optional[float] = None,
) -> Dict[str, Any]:
    """合并门禁、验证、币池标签到 summary（写入 DB / API）。"""
    out = dict(summary)
    out.update(evaluate_profile_auto_enable(out))
    if "validation_passed" not in out:
        val_summary = out.get("validation_summary")
        if isinstance(val_summary, dict):
            out.update(evaluate_validation(val_summary))
    out.update(classify_pool_tier(out))
    rp = out.get("recent_pick")
    if isinstance(rp, dict) and rp.get("side_stats"):
        out["side_bias_stats"] = rp["side_stats"]
    out["sync_allowed"] = can_sync_profile_params(out)
    deny = sync_deny_reason(out)
    if deny:
        out["sync_block_reason"] = deny
    if conn is not None and profile_id is not None:
        paper_block = paper_recent_pnl_block_reason(
            conn, int(profile_id), profile_capital=profile_capital
        )
        if paper_block:
            out["sync_allowed"] = False
            out["sync_block_reason"] = paper_block
    elif not out.get("sync_allowed") and not out.get("sync_block_reason") and deny:
        out["sync_block_reason"] = deny
    return out


def _candidate_sort_key(c: Dict[str, Any]) -> Tuple[float, float, float]:
    val = c.get("validation") or {}
    tr_ret = float((c.get("summary") or {}).get("total_return") or 0)
    vr_ret = float(val.get("val_return") or 0)
    gate_pen = float(val.get("gate_penalty") or 0)
    adj = stability_adjusted_val_score(
        float(val.get("val_sharpe") or -999),
        train_return=tr_ret,
        val_return=vr_ret - gate_pen,
    )
    return (-adj, -(vr_ret - gate_pen), -float(c.get("score") or -999))


def pick_best_validated(
    candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """在通过验证的候选中选稳定性调整后的验证分最高者。"""
    ok = [
        c
        for c in candidates
        if c.get("summary")
        and float(c.get("score") or -999) > -900
        and (
            c.get("validation", {}).get("validation_passed")
            or not cfg.MOSS_QUANT_OPTIMIZE_REQUIRE_VALIDATION
        )
    ]
    if not ok:
        return None
    if not cfg.MOSS_QUANT_OPTIMIZE_REQUIRE_VALIDATION:
        ok.sort(
            key=lambda r: (
                -float(r.get("score") or -999),
                -float((r.get("summary") or {}).get("sharpe") or -999),
            )
        )
        return ok[0]
    ok.sort(key=_candidate_sort_key)
    return ok[0]


def risk_scale_for_rank(
    rank_index: int,
    *,
    val_sharpe: Optional[float] = None,
    pool_max_val_sharpe: Optional[float] = None,
) -> float:
    """A 池内排序：前 N 基础满仓；其余半仓；可按验证 Sharpe 相对缩放。"""
    full = max(1, int(cfg.MOSS_QUANT_OPTIMIZE_FULL_RISK_SLOTS))
    base = 1.0 if rank_index < full else float(cfg.MOSS_QUANT_OPTIMIZE_REDUCED_RISK_SCALE)
    vs = float(val_sharpe or 0)
    mx = float(pool_max_val_sharpe or 0)
    if vs > 0 and mx > 0:
        rel = max(0.5, min(1.0, vs / mx))
        return round(base * rel, 4)
    return base
