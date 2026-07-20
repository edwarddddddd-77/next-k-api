#!/usr/bin/env python3
"""Compare baseline vs filtered Fractal ICT backtest."""
from __future__ import annotations

from datetime import datetime, timezone

from quant.common.kline_cache import load_klines, norm_symbol
from quant.fractal_ict.backtest import BacktestParams, simulate_bars
from quant.fractal_ict.config import FractalIctConfig


def _load(days: int = 30):
    import time

    sym = norm_symbol("BTCUSDT")
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000
    return load_klines(sym, "5m", start_ms=start_ms, end_ms=end_ms).sort_values("open_time")


def _fmt_trade(t) -> str:
    ts = datetime.fromtimestamp(t.entry_ms / 1000, tz=timezone.utc).strftime("%m-%d %H:%M")
    return (
        f"  {t.side:5} {t.pattern:9} {ts} UTC  "
        f"in={t.entry_price:.1f} out={t.exit_price:.1f}  "
        f"net={t.pnl_net_usdt:+.2f}  {t.exit_reason}"
    )


def _run(name: str, cfg: FractalIctConfig, df):
    r = simulate_bars(df, symbol="BTCUSDT", params=BacktestParams(symbols=["BTCUSDT"], days=30, cfg=cfg))
    print(f"\n=== {name} ===")
    print(f"trades={len(r.trades)}  win%={r.win_rate}  pnl={r.total_pnl_net:+.2f}  maxDD={r.max_drawdown_usdt:.2f}")
    if r.trades:
        for t in r.trades:
            print(_fmt_trade(t))
    return r


def main() -> None:
    df = _load(30)
    print(f"data bars={len(df)}")

    baseline = FractalIctConfig(ltf_interval="5m")
    filtered = FractalIctConfig(
        ltf_interval="5m",
        bias="bullish",
        require_fractal_touch=True,
        allowed_patterns=("c2_sweep",),
        range_only=True,
        range_max_pct=0.015,
        allowed_sessions=("asia", "london", "late"),
    )

    r0 = _run("Baseline (default cisd_c2)", baseline, df)
    r1 = _run(
        "Filtered (long + touch + c2_sweep + range + no NY)",
        filtered,
        df,
    )

    print("\n=== Comparison ===")
    print(f"{'':40} {'trades':>6} {'win%':>7} {'pnl':>10} {'maxDD':>8}")
    print(f"{'Baseline':40} {len(r0.trades):6d} {r0.win_rate:6.1f}% {r0.total_pnl_net:+10.2f} {r0.max_drawdown_usdt:8.2f}")
    print(f"{'Filtered':40} {len(r1.trades):6d} {r1.win_rate:6.1f}% {r1.total_pnl_net:+10.2f} {r1.max_drawdown_usdt:8.2f}")


if __name__ == "__main__":
    main()
