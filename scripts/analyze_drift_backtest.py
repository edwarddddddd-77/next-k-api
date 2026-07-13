#!/usr/bin/env python3
"""Quick anchor drift backtest post-mortem."""
from __future__ import annotations

import json
import subprocess
import sys
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    raw = subprocess.check_output(
        [
            sys.executable,
            "-m",
            "quant.anchor_drift.backtest",
            "--symbols",
            "MSTR,COIN,HOOD",
            "--days",
            "60",
            "--interval",
            "5m",
            "--json",
        ],
        cwd=str(ROOT),
        stderr=subprocess.DEVNULL,
        text=True,
    )
    out = json.loads(raw)
    trades = [t for sym in out["symbols"] for t in sym["trades"]]
    if not trades:
        print("no trades")
        return 1

    wins = [t for t in trades if t["pnl_net_usdt"] > 0]
    losses = [t for t in trades if t["pnl_net_usdt"] <= 0]
    avg_win = sum(t["pnl_net_usdt"] for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t["pnl_net_usdt"] for t in losses) / len(losses) if losses else 0.0
    avg_fee = sum(t["fee_usdt"] for t in trades) / len(trades)
    avg_gross = sum(t["pnl_gross_usdt"] for t in trades) / len(trades)
    total_gross = sum(t["pnl_gross_usdt"] for t in trades)
    total_fee = sum(t["fee_usdt"] for t in trades)

    print("=== 总体 ===")
    print(f"trades={len(trades)} win_rate={len(wins)/len(trades)*100:.1f}%")
    payoff = abs(avg_win / avg_loss) if avg_loss else 0.0
    print(f"avg_win={avg_win:+.4f}  avg_loss={avg_loss:+.4f}  payoff={payoff:.2f}")
    print(f"avg_gross/trade={avg_gross:+.4f}  avg_fee/trade={avg_fee:.4f}")
    print(f"total_gross={total_gross:+.4f}  total_fee={total_fee:.4f}  total_net={total_gross-total_fee:+.4f}")

    by_reason: dict[str, list[float]] = defaultdict(list)
    for t in trades:
        by_reason[t["exit_reason"]].append(t["pnl_net_usdt"])

    print("\n=== 平仓原因 ===")
    for k, v in sorted(by_reason.items(), key=lambda x: -len(x[1])):
        w = sum(1 for x in v if x > 0)
        print(f"  {k}: n={len(v)} win={w/len(v)*100:.0f}% pnl={sum(v):+.4f} avg={sum(v)/len(v):+.4f}")

    by_period: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        by_period[t["period"]].append(t)

    print("\n=== 时段 ===")
    for k, ts in by_period.items():
        w = sum(1 for t in ts if t["pnl_net_usdt"] > 0)
        gross = sum(t["pnl_gross_usdt"] for t in ts)
        fee = sum(t["fee_usdt"] for t in ts)
        print(f"  {k}: n={len(ts)} win={w/len(ts)*100:.0f}% gross={gross:+.4f} fee={fee:.4f} net={gross-fee:+.4f}")

    win_drift = [abs(t["drift_at_entry_pct"]) for t in wins]
    loss_drift = [abs(t["drift_at_entry_pct"]) for t in losses]
    print("\n=== 入场 |drift| ===")
    print(f"  winners median={st.median(win_drift):.2f}%  losers median={st.median(loss_drift):.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
