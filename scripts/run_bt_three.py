"""One-off: balanced replay backtest for HYPE / ARB / APT."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi

load_env_oi()

from moss_quant import config as cfg
from moss_quant.backtest_service import run_full_backtest
from moss_quant.kline_cache import load_cached
from moss_quant.params import build_initial_params

SYMS = ("HYPEUSDT", "ARBUSDT", "APTUSDT", "ADAUSDT", "DOGEUSDT")


def main() -> None:
    capital = float(cfg.MOSS_QUANT_DEFAULT_CAPITAL)
    print(
        f"data_source={cfg.MOSS_QUANT_DATA_SOURCE} "
        f"research_bars={cfg.MOSS_QUANT_RESEARCH_KLINE_BARS} "
        f"capital={capital} template=balanced"
    )
    print("-" * 78)
    for sym in SYMS:
        params = build_initial_params(template="balanced")
        try:
            df = load_cached(sym, refresh=True, research=True)
            out = run_full_backtest(
                symbol=sym,
                params=params,
                capital=capital,
                refresh_klines=False,
            )
            s = out.get("summary") or {}
            ret = float(s.get("total_return") or 0) * 100
            mdd = abs(float(s.get("max_drawdown") or 0)) * 100
            print(
                f"{sym:10} bars={len(df):5}  "
                f"return={ret:+7.2f}%  sharpe={float(s.get('sharpe') or 0):6.3f}  "
                f"mdd={mdd:5.2f}%  trades={int(s.get('total_trades') or 0):3}  "
                f"win={float(s.get('win_rate') or 0)*100:5.1f}%  "
                f"blowup={int(s.get('blowup_count') or 0)}"
            )
            if df is not None and len(df):
                print(
                    f"           kline {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}"
                )
        except Exception as e:
            print(f"{sym:10} ERROR: {e}")
    print("-" * 78)


if __name__ == "__main__":
    main()
