#!/usr/bin/env python3
"""ORB backtest with live-style compounding wallet (per-symbol detail)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from binance_fapi import api_get_raw  # noqa: E402
from orb.backtest import run_backtest  # noqa: E402
from orb.config import OrbConfig  # noqa: E402


def outcome_label(oc: str) -> str:
    if oc == "session_close":
        return "eod"
    if oc == "win":
        return "tp"
    if oc == "loss":
        return "sl"
    return oc or "-"


def symbol_onboard_ms(symbol: str) -> int:
    sym = str(symbol).strip().upper()
    data, _ = api_get_raw("/fapi/v1/exchangeInfo", {})
    for row in data.get("symbols") or []:
        if str(row.get("symbol", "")).upper() == sym:
            od = int(row.get("onboardDate") or 0)
            if od > 0:
                return od
            break
    raise SystemExit(f"onboardDate not found for {sym}")


def days_since_onboard(symbol: str, *, end_ms: int | None = None) -> float:
    end = int(end_ms if end_ms is not None else time.time() * 1000)
    onboard = symbol_onboard_ms(symbol)
    return max(1.0, (end - onboard) / 86_400_000.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="ORB compound backtest detail")
    ap.add_argument("--symbol", default="PLTRUSDT", help="e.g. INTCUSDT")
    ap.add_argument("--days", type=float, default=None, help="lookback calendar days")
    ap.add_argument(
        "--since-listing",
        action="store_true",
        help="from Binance onboardDate to now (default when --days omitted)",
    )
    args = ap.parse_args()

    cfg = OrbConfig.for_backtest()
    sym = str(args.symbol).strip().upper()
    if not sym.endswith("USDT"):
        sym = sym + "USDT"
    base = sym[:-4] if sym.endswith("USDT") else sym

    if args.days is not None:
        days = float(args.days)
        window = f"{int(days)}d"
    else:
        days = days_since_onboard(sym)
        onboard_ms = symbol_onboard_ms(sym)
        listed = pd.Timestamp(onboard_ms, unit="ms", tz=cfg.session_tz).strftime("%Y-%m-%d")
        window = f"since_listing ({listed} ~ {days:.1f}d)"

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
        rows.append(
            {
                "session": t.get("session_date", ""),
                "side": str(t["side"]),
                "outcome": outcome_label(str(oc)),
                "entry": float(t["entry"]),
                "sl": float(t["sl"]),
                "exit": float(t["exit_price"]),
                "pnl_r": float(t.get("pnl_r") or 0),
                "notional": round(notion, 2),
                "pnl_u": round(pu, 2),
                "wallet_before": round(wallet_before, 2),
                "wallet_after": wallet_after,
            }
        )
        wallet = wallet_after

    wins = sum(1 for r in rows if r["pnl_u"] > 0)
    losses = sum(1 for r in rows if r["pnl_u"] < 0)
    sum_pnl = round(wallet - init, 2)

    print("=" * 120)
    print(
        f"{base} ORB backtest | compound wallet | {window} | init {init:,.0f} U | "
        f"risk {cfg.risk_pct*100:.0f}% | EoD | 5%ATR SL"
    )
    print(
        f"config: 15m OR | 5m breakout | macro_filter={cfg.macro_filter} | vol_mult={cfg.vol_mult}"
    )
    print("=" * 120)
    hdr = (
        f"{'#':>3}  {'date':<12} {'side':<6} {'result':<6} "
        f"{'entry':>9} {'sl':>9} {'exit':>9} {'R':>6} "
        f"{'notional':>9} {'pnl_u':>9} {'wallet':>9} {'after':>9}"
    )
    print(hdr)
    print("-" * 120)
    for i, r in enumerate(rows, 1):
        print(
            f"{i:>3}  {r['session']:<12} {r['side']:<6} {r['outcome']:<6} "
            f"{r['entry']:>9.4f} {r['sl']:>9.4f} {r['exit']:>9.4f} {r['pnl_r']:>6.2f} "
            f"{r['notional']:>9.0f} {r['pnl_u']:>+9.2f} {r['wallet_before']:>9.0f} {r['wallet_after']:>9.0f}"
        )
    print("-" * 120)
    print(
        f"trades={len(rows)} | win={wins} loss={losses} | "
        f"compound_pnl={sum_pnl:+.2f} U | final_wallet={wallet:,.2f} U ({(wallet/init-1)*100:+.2f}%)"
    )
    unresolved = [t for t in trades if t.get("outcome") is None]
    if unresolved:
        print(f"open/unresolved at data end: {len(unresolved)} (often 1m history gap — check resolve_note)")


if __name__ == "__main__":
    main()
