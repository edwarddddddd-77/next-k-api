#!/usr/bin/env python3
"""IBS aggressive 日线回测 — 对齐 CazSyd Colab（TQQQ / 0.19 / 0.95 / 全仓）。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import StringIO

import pandas as pd
import requests

from quant.ibs.profile import PROFILE_AGGRESSIVE, PROFILE_DEFAULTS


@dataclass
class Trade:
    entry_day: str
    exit_day: str
    entry_price: float
    exit_price: float
    pnl: float


def fetch_bitget_spot_daily(symbol: str, *, days: int = 120) -> pd.DataFrame:
    import time

    from quant.market import klines_to_df
    from quant.market.bitget_spot import fetch_klines_forward

    end_ms = int(time.time() * 1000)
    start_ms = end_ms - max(30, int(days)) * 86_400_000
    rows = fetch_klines_forward(symbol, "1d", start_ms, end_ms)
    df = klines_to_df(rows)
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    return df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})


def fetch_stooq_daily(ticker: str) -> pd.DataFrame:
    sym = ticker.lower().strip()
    if not sym.endswith(".us"):
        sym = f"{sym}.us"
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    return df.sort_values("Date").reset_index(drop=True)


def fetch_daily(ticker: str, *, source: str = "auto") -> pd.DataFrame:
    sym = ticker.upper().strip()
    if source == "bitget" or (source == "auto" and sym in ("RTQQQUSDT", "TQQQ")):
        pair = "RTQQQUSDT" if sym == "TQQQ" else sym
        return fetch_bitget_spot_daily(pair)
    return fetch_stooq_daily(ticker)


def run_backtest(
    df: pd.DataFrame,
    *,
    entry: float,
    exit_: float,
    equity0: float,
    start: str,
    end: str,
) -> tuple[list[Trade], float]:
    work = df.copy()
    work["IBS"] = (work["Close"] - work["Low"]) / (work["High"] - work["Low"]).replace(0, pd.NA)

    cash = float(equity0)
    shares = 0.0
    in_pos = False
    entry_px = 0.0
    entry_day = ""
    trades: list[Trade] = []

    for i in range(1, len(work)):
        day = str(work["Date"].iloc[i])
        prev_ibs = float(work["IBS"].iloc[i - 1])
        op = float(work["Open"].iloc[i])

        if not in_pos and prev_ibs < entry:
            shares = cash / op
            cash = 0.0
            in_pos = True
            entry_px = op
            entry_day = day
        elif in_pos and prev_ibs > exit_:
            cash = shares * op
            trades.append(
                Trade(entry_day, day, entry_px, op, (op - entry_px) * shares)
            )
            shares = 0.0
            in_pos = False

    last_day = str(work["Date"].iloc[-1])
    if in_pos:
        cl = float(work["Close"].iloc[-1])
        cash = shares * cl
        trades.append(Trade(entry_day, f"{last_day}*", entry_px, cl, (cl - entry_px) * shares))

    window = [t for t in trades if t.entry_day >= start and t.entry_day <= end]
    equity = cash
    return window, equity


def main() -> None:
    defaults = PROFILE_DEFAULTS[PROFILE_AGGRESSIVE]
    parser = argparse.ArgumentParser(description="IBS aggressive daily backtest (CazSyd)")
    parser.add_argument("--source", default="auto", choices=("auto", "stooq", "bitget"))
    parser.add_argument("--ticker", default="TQQQ", help="TQQQ or RTQQQUSDT (Bitget spot)")
    parser.add_argument("--start", default="2026-06-10")
    parser.add_argument("--end", default="2026-07-10")
    parser.add_argument("--equity", type=float, default=10_000.0)
    parser.add_argument("--entry", type=float, default=defaults.entry_threshold)
    parser.add_argument("--exit", type=float, default=defaults.exit_threshold)
    args = parser.parse_args()

    df = fetch_daily(args.ticker, source=str(args.source))
    trades, equity = run_backtest(
        df,
        entry=float(args.entry),
        exit_=float(args.exit),
        equity0=float(args.equity),
        start=str(args.start),
        end=str(args.end),
    )
    net = equity - float(args.equity)
    sub = df[(df["Date"] >= args.start) & (df["Date"] <= args.end)]
    bh = 0.0
    if len(sub) >= 2:
        bh = (float(sub["Close"].iloc[-1]) / float(sub["Open"].iloc[0]) - 1.0) * 100.0

    print(f"\n=== ibs_aggressive | {args.ticker} daily | {args.start} .. {args.end} ===")
    print(f"entry={args.entry} exit={args.exit} | 100% equity | long_only | no EMA | no fee")
    print(f"trades={len(trades)}")
    for t in trades:
        print(
            f"  {t.entry_day} -> {t.exit_day}  "
            f"{t.entry_price:.2f} -> {t.exit_price:.2f}  pnl={t.pnl:+.2f}"
        )
    print(
        f"equity {args.equity:.0f} -> {equity:.2f}  "
        f"net={net:+.2f} ({net / args.equity * 100:+.2f}%)"
    )
    print(f"buy&hold same window: {bh:+.2f}%")
    print(f"data through: {df['Date'].iloc[-1]}")


if __name__ == "__main__":
    main()
