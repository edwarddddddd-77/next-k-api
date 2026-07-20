#!/usr/bin/env python3
"""Print full Fractal ICT backtest trade details."""
from __future__ import annotations

import sys
from datetime import datetime, timezone

from quant.fractal_ict.backtest import BacktestParams, run_backtest
from quant.fractal_ict.config import FractalIctConfig


def fmt_ms(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def main() -> int:
    cfg = FractalIctConfig(ltf_interval="5m")
    params = BacktestParams(symbols=["BTCUSDT"], days=30, interval="5m", cfg=cfg)
    results = run_backtest(params)
    r = results[0]
    trades = r.trades

    print(
        f"Symbol: {r.symbol}  LTF=5m  HTF={r.htf_interval}  bars={r.bars}"
    )
    print(
        f"Trades: {len(trades)}  PnL: {r.total_pnl_net}  "
        f"Win%: {r.win_rate}  MaxDD: {r.max_drawdown_usdt}"
    )
    print()
    header = (
        "No | Side  | Pattern   | Entry Time          | Exit Time           | "
        "Entry    | Exit     | Stop     | TP       | Qty      | "
        "Gross   | Fee    | Net      | Exit"
    )
    print(header)
    print("-" * len(header))

    for i, t in enumerate(trades, 1):
        print(
            f"{i:2d} | {t.side:5} | {t.pattern:9} | "
            f"{fmt_ms(t.entry_ms)} | {fmt_ms(t.exit_ms)} | "
            f"{t.entry_price:8.2f} | {t.exit_price:8.2f} | "
            f"{t.stop_price:8.2f} | {t.tp_price:8.2f} | {t.qty:8.4f} | "
            f"{t.pnl_gross_usdt:7.2f} | {t.fee_usdt:6.2f} | "
            f"{t.pnl_net_usdt:8.2f} | {t.exit_reason}"
        )

    wins = [t for t in trades if t.pnl_net_usdt > 0]
    losses = [t for t in trades if t.pnl_net_usdt <= 0]
    longs = [t for t in trades if t.side == "LONG"]
    shorts = [t for t in trades if t.side == "SHORT"]

    print()
    print("Summary:")
    print(f"  LONG  {len(longs):2d} trades  net {sum(t.pnl_net_usdt for t in longs):.2f}")
    print(f"  SHORT {len(shorts):2d} trades  net {sum(t.pnl_net_usdt for t in shorts):.2f}")
    if wins:
        print(
            f"  WIN   {len(wins):2d} trades  net {sum(t.pnl_net_usdt for t in wins):.2f}  "
            f"avg {sum(t.pnl_net_usdt for t in wins) / len(wins):.2f}"
        )
    if losses:
        print(
            f"  LOSS  {len(losses):2d} trades  net {sum(t.pnl_net_usdt for t in losses):.2f}  "
            f"avg {sum(t.pnl_net_usdt for t in losses) / len(losses):.2f}"
        )
    print(
        f"  TP {sum(1 for t in trades if t.exit_reason == 'tp')}  "
        f"STOP {sum(1 for t in trades if t.exit_reason == 'stop')}"
    )
    print(
        f"  c2_sweep {sum(1 for t in trades if t.pattern == 'c2_sweep')}  "
        f"c4 {sum(1 for t in trades if t.pattern == 'c4')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
