"""Moss2 选优与淘汰闸门：创建 / evolve 共用回测标准 + 赢家战术窄搜。"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, List, Optional, Tuple

from moss2 import config as cfg
from moss2.backtest_service import run_factory_backtest
from moss2.discipline.report import build_discipline_report
from moss2.params import build_initial_params, list_templates

logger = logging.getLogger(__name__)

# 在选定模板内窄搜的战术字段（小网格，控制算力）
_TACTICAL_GRID: Tuple[Tuple[str, Tuple[Any, ...]], ...] = (
    ("entry_threshold", (0.18, 0.20, 0.22, 0.25)),
    ("exit_threshold", (0.08, 0.10, 0.12)),
    ("fast_ma_period", (8, 10, 12)),
)


def passes_backtest_gates(
    summary: Dict[str, Any],
    discipline: Dict[str, Any],
    *,
    min_trades: Optional[int] = None,
) -> bool:
    """回测候选过关：EV、成交数、Sharpe、最大回撤。"""
    min_trades = int(min_trades if min_trades is not None else cfg.MOSS2_SELECTION_MIN_TRADES)
    trades = int(summary.get("total_trades") or 0)
    if trades < min_trades:
        return False
    ev = float((discipline.get("ev") or {}).get("ev_per_trade_pct") or -1.0)
    if ev < float(cfg.MOSS2_SELECTION_MIN_EV_PCT):
        return False
    sharpe = float(summary.get("sharpe") or 0)
    if sharpe < float(cfg.MOSS2_SELECTION_MIN_SHARPE):
        return False
    mdd = abs(float(summary.get("max_drawdown") or 0))
    if mdd > float(cfg.MOSS2_SELECTION_MAX_MDD):
        return False
    return True


def composite_score(summary: Dict[str, Any], discipline: Dict[str, Any]) -> float:
    """越大越好；回撤超阈大幅扣分。"""
    sharpe = float(summary.get("sharpe") or 0)
    ret = float(summary.get("total_return") or 0)
    ev = float((discipline.get("ev") or {}).get("ev_per_trade_pct") or 0)
    mdd = abs(float(summary.get("max_drawdown") or 0))
    score = sharpe + ret * 0.1 + ev * 0.02
    if mdd > float(cfg.MOSS2_SELECTION_MAX_MDD):
        score -= 2.0
    return score


def _backtest_row(
    *,
    symbol: str,
    variant: str,
    template: str,
    params: dict,
    capital: float,
    limit_bars: int,
) -> Dict[str, Any]:
    out = run_factory_backtest(
        symbol=symbol,
        params=params,
        variant=variant,  # type: ignore[arg-type]
        capital=capital,
        limit_bars=limit_bars,
    )
    summ = out.get("summary") or {}
    trades = out.get("trades") or []
    disc = build_discipline_report(summary=summ, trades=trades, template=template)
    ok = passes_backtest_gates(summ, disc, min_trades=min_trades)
    return {
        "template": template,
        "params": params,
        "summary": summ,
        "discipline": disc,
        "score": round(composite_score(summ, disc), 4),
        "passes_gates": ok,
        "total_trades": int(summ.get("total_trades") or 0),
        "sharpe": float(summ.get("sharpe") or 0),
        "max_drawdown": float(summ.get("max_drawdown") or 0),
        "ev_per_trade_pct": float((disc.get("ev") or {}).get("ev_per_trade_pct") or 0),
    }


def _iter_tactical_overrides() -> List[Dict[str, Any]]:
    if not cfg.MOSS2_SELECTION_TACTICAL_NARROW:
        return [{}]
    keys = [k for k, _ in _TACTICAL_GRID]
    value_lists = [vals for _, vals in _TACTICAL_GRID]
    combos: List[Dict[str, Any]] = []
    for values in itertools.product(*value_lists):
        ov = dict(zip(keys, values))
        if int(ov.get("fast_ma_period") or 10) >= 45:
            continue
        combos.append(ov)
    return combos or [{}]


def optimize_template_params(
    symbol: str,
    template: str,
    *,
    variant: Optional[str] = None,
    capital: Optional[float] = None,
    limit_bars: Optional[int] = None,
    base_overrides: Optional[dict] = None,
    min_trades: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """在单一模板默认参数上做战术窄搜，返回最优且过关的一行。"""
    variant = variant or cfg.MOSS2_OPS_VARIANT
    capital = float(capital or cfg.MOSS2_PROFILE_CAPITAL)
    limit_bars = int(limit_bars or cfg.MOSS2_EVOLVE_LIMIT_BARS)
    best: Optional[Dict[str, Any]] = None
    for tactical_ov in _iter_tactical_overrides():
        merged = dict(base_overrides or {})
        merged.update(tactical_ov)
        try:
            params = build_initial_params(
                template, merged, variant=variant  # type: ignore[arg-type]
            )
            row = _backtest_row(
                symbol=symbol,
                variant=variant,
                template=template,
                params=params,
                capital=capital,
                limit_bars=limit_bars,
                min_trades=min_trades,
            )
        except Exception as e:
            logger.debug("[moss2] tactical search %s %s %s: %s", symbol, template, tactical_ov, e)
            continue
        if not row.get("passes_gates"):
            continue
        if best is None or float(row["score"]) > float(best["score"]):
            best = row
    return best


def compete_templates(
    symbol: str,
    *,
    variant: Optional[str] = None,
    capital: Optional[float] = None,
    limit_bars: Optional[int] = None,
    optimize_tactical: Optional[bool] = None,
    min_trades: Optional[int] = None,
) -> Dict[str, Any]:
    """
    两阶段选优：
    1) 四模板默认参数粗赛 → 过关者 score 最高
    2) 仅对胜出模板做战术窄搜（可选）
    """
    variant = variant or cfg.MOSS2_OPS_VARIANT
    capital = float(capital or cfg.MOSS2_PROFILE_CAPITAL)
    limit_bars = int(limit_bars or cfg.MOSS2_EVOLVE_LIMIT_BARS)
    do_narrow = (
        cfg.MOSS2_SELECTION_TACTICAL_NARROW
        if optimize_tactical is None
        else optimize_tactical
    )

    coarse_rows: List[Dict[str, Any]] = []
    for template in list_templates():
        try:
            params = build_initial_params(template, variant=variant)  # type: ignore[arg-type]
            row = _backtest_row(
                symbol=symbol,
                variant=variant,
                template=template,
                params=params,
                capital=capital,
                limit_bars=limit_bars,
                min_trades=min_trades,
            )
            if row.get("passes_gates"):
                coarse_rows.append(row)
        except Exception as e:
            logger.warning("[moss2] template coarse %s %s: %s", symbol, template, e)

    if not coarse_rows:
        return {
            "symbol": symbol,
            "variant": variant,
            "limit_bars": limit_bars,
            "rows": [],
            "best": None,
            "has_winner": False,
        }

    coarse_best = max(coarse_rows, key=lambda x: float(x["score"]))
    winner_tpl = str(coarse_best["template"])
    best = coarse_best
    tactical_refined = False
    if do_narrow:
        refined = optimize_template_params(
            symbol,
            winner_tpl,
            variant=variant,
            capital=capital,
            limit_bars=limit_bars,
            min_trades=min_trades,
        )
        if refined and float(refined["score"]) > float(coarse_best["score"]):
            best = refined
            tactical_refined = True

    return {
        "symbol": symbol,
        "variant": variant,
        "limit_bars": limit_bars,
        "rows": coarse_rows,
        "best": best,
        "has_winner": True,
        "coarse_winner": winner_tpl,
        "tactical_refined": tactical_refined,
    }
