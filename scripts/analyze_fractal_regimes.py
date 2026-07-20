#!/usr/bin/env python3
"""Analyze Fractal ICT win rate by market regime slices."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd

from quant.common.kline_cache import load_klines, norm_symbol
from quant.fractal_ict.backtest import BacktestParams, simulate_bars
from quant.fractal_ict.config import FractalIctConfig


def _fetch(symbol: str, days: int = 30) -> pd.DataFrame:
    import time

    sym = norm_symbol(symbol)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000
    return load_klines(sym, "5m", start_ms=start_ms, end_ms=end_ms).sort_values("open_time")


def _trend_at_entry(df: pd.DataFrame, entry_ms: int, lookback_bars: int = 288) -> str:
    """288 x 5m = 24h return regime."""
    sub = df[df["open_time"] <= entry_ms].tail(lookback_bars + 1)
    if len(sub) < 20:
        return "unknown"
    ret = (sub["close"].iloc[-1] - sub["close"].iloc[0]) / sub["close"].iloc[0]
    if ret > 0.015:
        return "uptrend"
    if ret < -0.015:
        return "downtrend"
    return "range"


def _vol_at_entry(df: pd.DataFrame, entry_ms: int, lookback_bars: int = 288) -> str:
    sub = df[df["open_time"] <= entry_ms].tail(lookback_bars + 1)
    if len(sub) < 20:
        return "unknown"
    rets = sub["close"].pct_change().dropna()
    vol = float(rets.std())
    if vol > 0.0035:
        return "high_vol"
    if vol < 0.0020:
        return "low_vol"
    return "mid_vol"


def _stats(items, key_fn):
    buckets: dict[str, list[float]] = defaultdict(list)
    for item in items:
        t = item[0] if isinstance(item, tuple) else item
        buckets[key_fn(item)].append(t.pnl_net_usdt)
    rows = []
    for k, pnls in sorted(buckets.items()):
        wins = sum(1 for p in pnls if p > 0)
        rows.append(
            {
                "bucket": k,
                "trades": len(pnls),
                "wins": wins,
                "win_rate": round(wins / len(pnls) * 100, 1) if pnls else 0,
                "net_pnl": round(sum(pnls), 2),
                "avg_pnl": round(sum(pnls) / len(pnls), 2) if pnls else 0,
            }
        )
    return rows


def main() -> None:
    cfg = FractalIctConfig(ltf_interval="5m")
    params = BacktestParams(symbols=["BTCUSDT"], days=30, interval="5m", cfg=cfg)
    df = _fetch("BTCUSDT", 30)
    r = simulate_bars(df, symbol="BTCUSDT", params=params)
    trades = r.trades

    enriched = []
    for t in trades:
        trend = _trend_at_entry(df, t.entry_ms)
        vol = _vol_at_entry(df, t.entry_ms)
        aligned = (
            "with_trend"
            if (t.side == "LONG" and trend == "uptrend")
            or (t.side == "SHORT" and trend == "downtrend")
            else "counter_trend"
            if trend in ("uptrend", "downtrend")
            else "neutral"
        )
        hour_utc = datetime.fromtimestamp(t.entry_ms / 1000, tz=timezone.utc).hour
        session = (
            "asia"
            if 0 <= hour_utc < 8
            else "london"
            if 8 <= hour_utc < 13
            else "ny"
            if 13 <= hour_utc < 21
            else "late"
        )
        enriched.append((t, trend, vol, aligned, session))

    print("=== Overall ===")
    print(f"trades={len(trades)} win_rate={r.win_rate}% net={r.total_pnl_net}")
    print()

    for title, fn in [
        ("By Side", lambda x: x[0].side),
        ("By Pattern", lambda x: x[0].pattern),
        ("By 24h Trend at Entry", lambda x: x[1]),
        ("By Volatility at Entry", lambda x: x[2]),
        ("By Trend Alignment", lambda x: x[3]),
        ("By Session (UTC)", lambda x: x[4]),
        ("Side + Trend", lambda x: f"{x[0].side}/{x[1]}"),
        ("Side + Alignment", lambda x: f"{x[0].side}/{x[3]}"),
        ("Pattern + Trend", lambda x: f"{x[0].pattern}/{x[1]}"),
    ]:
        print(f"--- {title} ---")
        rows = _stats(enriched, fn)
        for row in rows:
            print(
                f"  {row['bucket']:22} n={row['trades']:2}  "
                f"win%={row['win_rate']:5.1f}  net={row['net_pnl']:8.2f}  "
                f"avg={row['avg_pnl']:7.2f}"
            )
        print()


if __name__ == "__main__":
    main()
