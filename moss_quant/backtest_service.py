"""全量回测。"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from moss_quant import config as cfg
from moss_quant.core.backtest import run_backtest
from moss_quant.core.decision import DecisionParams
from moss_quant.core.regime import classify_regime
from moss_quant.kline_cache import load_cached
from moss_quant.params import cap_leverage_for_symbol, resolve_params_dict


def run_full_backtest(
    *,
    symbol: str,
    params: dict,
    capital: Optional[float] = None,
    regime_version: Optional[str] = None,
    refresh_klines: bool = False,
) -> Dict[str, Any]:
    capital = float(capital or cfg.MOSS_QUANT_DEFAULT_CAPITAL)
    regime_version = regime_version or cfg.MOSS_QUANT_REGIME_VERSION
    params = cap_leverage_for_symbol(resolve_params_dict(params), symbol)
    df = load_cached(symbol, refresh=refresh_klines)
    regime = classify_regime(df, version=regime_version)
    p = DecisionParams.from_dict(params)
    result = run_backtest(
        df,
        p,
        regime,
        initial_capital=capital,
        symbol=symbol,
    )
    trades_list = []
    for t in result.trades[:500]:
        trades_list.append(
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
    return {
        "symbol": symbol,
        "initial_params": params,
        "backtest_result": result.to_dict(),
        "equity_curve": result.equity_curve.tolist()
        if hasattr(result.equity_curve, "tolist")
        else list(result.equity_curve),
        "trades": trades_list,
        "summary": {
            "total_return": round(result.total_return, 4),
            "sharpe": round(result.sharpe_ratio, 4),
            "max_drawdown": round(result.max_drawdown, 4),
            "total_trades": result.total_trades,
            "win_rate": round(result.win_rate, 4),
            "blowup_count": result.blowup_count,
        },
    }
