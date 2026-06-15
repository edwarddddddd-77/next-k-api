#!/usr/bin/env python3
"""Print daily true breakout counts from universe backtest JSON."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json",
        default=str(ROOT / "output" / "orb" / "v2" / "eval" / "universe_60d_trading_backtest.json"),
    )
    args = ap.parse_args()
    d = json.loads(Path(args.json).read_text(encoding="utf-8"))
    rows = []
    for day in d.get("days") or []:
        sd = day["session_date"]
        true_n = int(day.get("true_opens") or 0)
        opens = int(day.get("opens") or 0)
        pnl = round(sum(float(r.get("pnl_usdt") or 0) for r in day.get("opened") or []), 1)
        rows.append((sd, true_n, opens, pnl))

    dr = d.get("date_range") or {}
    n = len(rows)
    total_true = sum(r[1] for r in rows)
    print(f"Range: {dr.get('from')} .. {dr.get('to')} ({n} NYSE days)")
    print(f"Total true: {total_true} | Total opens: {sum(r[2] for r in rows)}")
    print(f"Avg true/day: {total_true / n:.2f}" if n else "")
    print()
    print(f"{'Date':<12} {'True':>4} {'Opens':>5} {'PnL':>10}")
    print("-" * 34)
    for sd, t, o, pnl in rows:
        print(f"{sd:<12} {t:>4} {o:>5} {pnl:>+9.1f}U")
    print()
    print("Distribution:")
    for k in sorted(Counter(r[1] for r in rows)):
        cnt = Counter(r[1] for r in rows)[k]
        print(f"  {k} true: {cnt} days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
