#!/usr/bin/env python3
"""Estimate ORB stop-loss rate from equity CSV (wallet_usdt series)."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

# Full backtest reference (outcome == loss)
KNOWN = {"COIN": (44, 68), "MSTR": (54, 70)}


def analyze(path: Path) -> dict:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    wallets = [float(r["wallet_usdt"]) for r in rows]
    symbol = path.stem.replace("orb_", "").replace("_equity_premarket", "").upper()
    if len(wallets) < 2:
        return {"symbol": symbol, "trades": 0, "sl": 0, "sl_rate_pct": 0.0}

    init = wallets[0]
    trade_rets: list[float] = []
    for i in range(1, len(wallets)):
        prev, cur = wallets[i - 1], wallets[i]
        if prev <= 0:
            continue
        trade_rets.append((cur - prev) / prev * 100.0)

    n = len(trade_rets)
    # ~1R stop at 1% risk: typically -0.5% .. -1.8% per trade on compounding wallet
    sl = sum(1 for r in trade_rets if -1.8 < r < -0.3)
    neg = sum(1 for r in trade_rets if r < -0.01)
    pos = sum(1 for r in trade_rets if r > 0.01)
    return {
        "symbol": symbol,
        "init": init,
        "final": wallets[-1],
        "return_pct": (wallets[-1] / init - 1) * 100.0,
        "trades": n,
        "sl": sl,
        "sl_rate_pct": sl / n * 100.0 if n else 0.0,
        "neg_trades": neg,
        "pos_trades": pos,
    }


def main() -> None:
    files = sorted(OUT.glob("orb_*_equity_premarket.csv"))
    results = [analyze(p) for p in files]
    results.sort(key=lambda r: r["sl_rate_pct"])

    print("ORB stop-loss rate from equity CSV")
    print("Method: count trades where wallet change is in (-1.8%, -0.3%) (~1R SL band)")
    print()
    print(f"{'symbol':<6} {'trades':>6} {'SL':>4} {'SL%':>7} {'loss':>5} {'win':>5} {'ret%':>8}")
    print("-" * 54)
    for r in results:
        print(
            f"{r['symbol']:<6} {r['trades']:>6} {r['sl']:>4} {r['sl_rate_pct']:>6.1f}% "
            f"{r['neg_trades']:>5} {r['pos_trades']:>5} {r['return_pct']:>+7.1f}%"
        )

    print()
    print("Validation vs full backtest (where available):")
    for sym, (act_sl, act_n) in KNOWN.items():
        row = next((x for x in results if x["symbol"] == sym), None)
        if not row:
            continue
        act_pct = act_sl / act_n * 100.0
        print(
            f"  {sym}: backtest {act_sl}/{act_n} ({act_pct:.1f}%) | "
            f"CSV est {row['sl']}/{row['trades']} ({row['sl_rate_pct']:.1f}%)"
        )


if __name__ == "__main__":
    main()
