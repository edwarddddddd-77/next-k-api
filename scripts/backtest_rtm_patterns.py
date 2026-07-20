#!/usr/bin/env python3
"""RTM pattern win-rate backtest CLI."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.rtm_patterns import RTMConfig, backtest_rtm_patterns, BacktestParams, summary_to_dataframe


def main() -> int:
    parser = argparse.ArgumentParser(description="RTM pattern win-rate backtest")
    parser.add_argument("csv", type=Path, help="OHLC CSV")
    parser.add_argument("--resample", default="", help="Pandas resample rule, e.g. 4h, 1h")
    parser.add_argument("--target-r", type=float, default=2.0, help="Default target R if no target_level")
    parser.add_argument("--max-hold", type=int, default=30, help="Max bars to hold")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Not found: {args.csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.csv)
    if "timestamp" in {c.lower() for c in df.columns}:
        ts_col = next(c for c in df.columns if c.lower() == "timestamp")
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.set_index(ts_col)

    if args.resample:
        df = (
            df.resample(args.resample)
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna()
            .reset_index()
        )

    summary = backtest_rtm_patterns(
        df,
        config=RTMConfig(),
        params=BacktestParams(target_r=args.target_r, max_hold_bars=args.max_hold),
    )

    print("=== RTM Backtest Summary ===")
    print(f"Total trades: {summary.total_trades}")
    print(f"Win rate:     {summary.win_rate * 100:.1f}%")
    print(f"Avg R:        {summary.avg_r:.2f}")
    print(f"Profit factor:{summary.profit_factor:.2f}")
    print()

    table = summary_to_dataframe(summary)
    if table.empty:
        print("No trades to backtest.")
        return 0

    pd.set_option("display.width", 200)
    print("--- By pattern ---")
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
