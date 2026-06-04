"""Factory 同款回测：HL replay-aligned / EN cross-margin。"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from moss2 import config as cfg
from moss2.config import FactoryVariant, MOSS2_DEFAULT_CAPITAL, MOSS2_REGIME_VERSION
from moss2.dataset import load_ohlcv, resolve_csv_path
from moss2.params import merge_profile_params


def _trades_payload(result) -> list:
    rows = []
    for t in (result.trades or [])[:500]:
        rows.append(
            {
                "entry_time": t.entry_time or "",
                "exit_time": t.exit_time or "",
                "direction": "LONG" if t.direction == 1 else "SHORT",
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2) if t.exit_price else None,
                "pnl_pct": round(t.pnl_pct, 4),
                "leverage": t.leverage,
                "exit_reason": t.exit_reason,
            }
        )
    return rows


def run_factory_backtest(
    *,
    symbol: str,
    params: dict,
    variant: FactoryVariant,
    capital: Optional[float] = None,
    regime_version: Optional[str] = None,
    limit_bars: Optional[int] = None,
) -> Dict[str, Any]:
    from moss2.config import MOSS2_HL_ENABLED, MOSS2_OPS_VARIANT, effective_variant

    variant = effective_variant(variant)
    if variant != MOSS2_OPS_VARIANT and not MOSS2_HL_ENABLED:
        raise ValueError(f"variant_{variant}_disabled")
    capital = float(capital or MOSS2_DEFAULT_CAPITAL)
    regime_version = regime_version or MOSS2_REGIME_VERSION
    csv_path = resolve_csv_path(symbol, variant)
    df = load_ohlcv(symbol, variant, limit=limit_bars)

    if variant == "hl":
        from moss2.variants.hl.core.decision import DecisionParams
        from moss2.variants.hl.core.leverage_caps import cap_params_for_symbol
        from moss2.variants.hl.core.regime import classify_regime
        from moss2.variants.hl.core.backtest import run_backtest
        from moss2.variants.hl.core.replay_baseline import infer_replay_symbol_from_path

        replay_sym = infer_replay_symbol_from_path(str(csv_path or symbol))
        params = cap_params_for_symbol(params, replay_sym)
        regime = classify_regime(df, version=regime_version)
        p = DecisionParams.from_dict(params)
        result = run_backtest(
            df, p, regime, initial_capital=capital, symbol=replay_sym
        )
        engine = "hl_replay_aligned"
    else:
        from moss2.variants.en.core.decision import DecisionParams
        from moss2.variants.en.core.regime import classify_regime
        from moss2.variants.en.core.backtest import run_backtest

        regime = classify_regime(df, version=regime_version)
        p = DecisionParams.from_dict(params)
        result = run_backtest(df, p, regime, initial_capital=capital)
        engine = "en_cross_margin"

    trades = _trades_payload(result)
    summary = {
        "total_return": round(float(result.total_return), 4),
        "sharpe": round(float(result.sharpe_ratio), 4),
        "max_drawdown": round(float(result.max_drawdown), 4),
        "total_trades": int(result.total_trades),
        "win_rate": round(float(result.win_rate), 4),
        "profit_factor": round(float(getattr(result, "profit_factor", 0) or 0), 4),
        "blowup_count": int(getattr(result, "blowup_count", 0) or 0),
    }
    discipline = None
    if cfg.MOSS2_DISCIPLINE_ENABLED:
        from moss2.discipline.report import build_discipline_report, signal_contrib_from_df

        params["_template"] = str(params.get("_template") or "balanced")
        contrib = signal_contrib_from_df(df, params, variant)
        discipline = build_discipline_report(
            summary=summary,
            trades=trades,
            regime=regime,
            template=params["_template"],
            signal_contrib=contrib,
        )
    return {
        "lane": "moss2",
        "variant": variant,
        "engine": engine,
        "symbol": symbol,
        "data_csv": str(csv_path) if csv_path else None,
        "initial_params": params,
        "backtest_result": result.to_dict(),
        "equity_curve": result.equity_curve.tolist()
        if hasattr(result.equity_curve, "tolist")
        else list(result.equity_curve),
        "trades": trades,
        "summary": summary,
        "discipline": discipline,
    }


def run_profile_backtest(
    profile: dict,
    *,
    capital: Optional[float] = None,
    limit_bars: Optional[int] = None,
) -> Dict[str, Any]:
    from moss2.config import profile_variant

    variant: FactoryVariant = profile_variant(profile)
    params = merge_profile_params(profile)
    params["_template"] = str(profile.get("template") or "balanced")
    return run_factory_backtest(
        symbol=str(profile["symbol"]),
        params=params,
        variant=variant,
        capital=capital,
        limit_bars=limit_bars,
    )
