#!/usr/bin/env python3
"""Export ORB backtest trade ledger (incl. notional) to CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from env_loader import load_env_oi
from orb.backtest import run_backtest
from orb.config import OrbConfig
from tools.print_pltr_backtest_detail import days_since_onboard, symbol_onboard_ms


def outcome_label(oc: str) -> str:
    if oc == "session_close":
        return "eod"
    if oc == "win":
        return "tp"
    if oc == "loss":
        return "sl"
    return oc or "-"


def main() -> None:
    load_env_oi()
    ap = argparse.ArgumentParser(description="Export ORB trade detail CSV")
    ap.add_argument("--symbol", default="COINUSDT")
    ap.add_argument("--days", type=float, default=None)
    ap.add_argument("--since-listing", action="store_true")
    ap.add_argument("--out", default=None, help="output CSV path")
    args = ap.parse_args()

    cfg = OrbConfig.for_backtest()
    sym = str(args.symbol).strip().upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    base = sym[:-4]

    if args.days is not None:
        days = float(args.days)
    else:
        days = days_since_onboard(sym)

    init = cfg.per_symbol_bot_equity()
    raw = run_backtest(days=days, symbols=[sym], cfg=cfg, json_path=None, csv_path=None)
    trades = raw.get("trades") or []

    rows: list[dict] = []
    wallet = init
    for t in trades:
        oc = t.get("outcome")
        if oc is None or oc == "supersede":
            continue
        wallet_after = float(t.get("wallet_after") or wallet)
        notion = float(t.get("notional_usdt") or 0)
        pu = float(t.get("pnl_usdt") or 0)
        wallet_before = round(wallet_after - pu, 4)
        entry = float(t.get("entry") or 0)
        sl = float(t.get("sl") or 0)
        risk_frac_pct = abs(entry - sl) / entry * 100 if entry > 0 and sl > 0 else 0.0
        rows.append(
            {
                "session_date": t.get("session_date", ""),
                "symbol": base,
                "side": str(t.get("side") or ""),
                "outcome": outcome_label(str(oc)),
                "entry": round(entry, 4),
                "sl": round(sl, 4),
                "exit": round(float(t.get("exit_price") or 0), 4),
                "risk_frac_pct": round(risk_frac_pct, 4),
                "notional_usdt": round(notion, 2),
                "margin_usdt": round(notion / max(float(cfg.leverage or 10), 1), 2),
                "pnl_usdt": round(pu, 2),
                "pnl_r": round(float(t.get("pnl_r") or 0), 2),
                "wallet_before": wallet_before,
                "wallet_after": wallet_after,
            }
        )
        wallet = wallet_after

    out_path = Path(args.out) if args.out else ROOT / "output" / f"orb_{base.lower()}_trades_detail.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    onboard = symbol_onboard_ms(sym)
    listed = pd.Timestamp(onboard, unit="ms", tz=cfg.session_tz).strftime("%Y-%m-%d")
    wins = sum(1 for r in rows if r["pnl_usdt"] > 0)
    final = rows[-1]["wallet_after"] if rows else init
    ret = (final / init - 1) * 100 if init > 0 else 0

    print(f"{base} trades={len(rows)} | init={init:,.0f} U | final={final:,.2f} U | ret={ret:+.1f}%")
    print(f"since {listed} ({days:.1f}d) | win/loss={wins}/{len(rows)-wins}")
    print(f"Wrote {out_path.resolve()}")
    print()
    print(f"{'#':>3}  {'date':<12} {'side':<6} {'out':<4} {'notional':>10} {'pnl':>9} {'wallet':>10}")
    print("-" * 58)
    for i, r in enumerate(rows, 1):
        print(
            f"{i:>3}  {r['session_date']:<12} {r['side']:<6} {r['outcome']:<4} "
            f"{r['notional_usdt']:>10.0f} {r['pnl_usdt']:>+9.2f} {r['wallet_after']:>10.0f}"
        )


if __name__ == "__main__":
    main()
