#!/usr/bin/env python3
"""Print QQQ backtest day-by-day ledger with macro filter days marked."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orb.macro_calendar import is_macro_skip_day, macro_events_for_day  # noqa: E402
from orb.us_equity_calendar import is_us_equity_trading_day  # noqa: E402


def main() -> None:
    csv_path = ROOT / "orb_qqq_bot_10k_risk.csv"
    trades: dict[str, dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            trades[row["session_date"]] = row

    dates = sorted(trades)
    start, end = dates[0], dates[-1]
    days = pd.date_range(start, end, freq="D")

    print("=" * 105)
    print(f"QQQ 50d backtest daily ledger ({start} ~ {end})")
    print("=" * 105)
    hdr = (
        f"{'date':<12} {'status':<16} {'macro':<18} "
        f"{'side':<6} {'result':<12} {'pnl_u':>10} {'entry':>8} {'sl':>8}"
    )
    print(hdr)
    print("-" * 105)

    macro_days: list[tuple[str, tuple[str, ...]]] = []
    no_signal_days: list[str] = []

    for d in days:
        ds = d.strftime("%Y-%m-%d")
        if not is_us_equity_trading_day(ds):
            continue
        ev = macro_events_for_day(ds)
        macro = is_macro_skip_day(ds)
        if ds in trades:
            t = trades[ds]
            pnl = float(t["pnl_usdt"])
            oc = t["outcome"]
            if oc == "session_close":
                oc_disp = "eod_profit" if pnl > 0 else "eod"
            elif oc == "loss":
                oc_disp = "stop_loss"
            elif oc == "win":
                oc_disp = "take_profit"
            else:
                oc_disp = oc or "-"
            ev_str = ",".join(ev) if ev else "-"
            print(
                f"{ds:<12} {'TRADE':<16} {ev_str:<18} "
                f"{t['side']:<6} {oc_disp:<12} {pnl:>10.2f} "
                f"{float(t['entry']):>8.2f} {float(t['sl']):>8.2f}"
            )
        elif macro:
            macro_days.append((ds, ev))
            ev_str = ",".join(ev) if ev else "FOMC/CPI"
            print(
                f"{ds:<12} {'MACRO_SKIP':<16} {ev_str:<18} "
                f"{'—':<6} {'no_entry':<12} {'—':>10} {'—':>8} {'—':>8}"
            )
        else:
            no_signal_days.append(ds)
            print(
                f"{ds:<12} {'NO_SIGNAL':<16} {'-':<18} "
                f"{'—':<6} {'no_breakout':<12} {'—':>10} {'—':>8} {'—':>8}"
            )

    print("-" * 105)
    print(
        f"trades={len(trades)} | macro_skip={len(macro_days)} | "
        f"no_signal={len(no_signal_days)}"
    )
    print()
    print("Macro filter days:")
    for ds, ev in macro_days:
        print(f"  {ds}  ->  {', '.join(ev)}")
    if no_signal_days:
        print()
        print("No-signal days (regular session, no OR breakout):")
        for ds in no_signal_days:
            print(f"  {ds}")


if __name__ == "__main__":
    main()
