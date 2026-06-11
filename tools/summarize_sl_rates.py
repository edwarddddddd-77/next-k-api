#!/usr/bin/env python3
"""Summarize ORB stop-loss rates for all output symbols (since listing)."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from env_loader import load_env_oi
from orb.backtest import run_backtest
from orb.config import OrbConfig
from tools.print_pltr_backtest_detail import days_since_onboard, symbol_onboard_ms

SYMS = [
    "COINUSDT",
    "PAYPUSDT",
    "INTCUSDT",
    "MSTRUSDT",
    "EWYUSDT",
    "PLTRUSDT",
]


def summarize_symbol(sym: str, cfg: OrbConfig) -> dict:
    base = sym[:-4]
    days = days_since_onboard(sym)
    onboard = symbol_onboard_ms(sym)
    listed = pd.Timestamp(onboard, unit="ms", tz=cfg.session_tz).strftime("%Y-%m-%d")
    raw = run_backtest(days=days, symbols=[sym], cfg=cfg, json_path=None, csv_path=None)
    trades = [
        t for t in (raw.get("trades") or []) if t.get("outcome") not in (None, "supersede")
    ]
    sl = sum(1 for t in trades if t.get("outcome") == "loss")
    eod = sum(1 for t in trades if t.get("outcome") == "session_close")
    tp = sum(1 for t in trades if t.get("outcome") == "win")
    wins = sum(1 for t in trades if float(t.get("pnl_usdt") or 0) > 0)
    total = len(trades)
    init = cfg.per_symbol_bot_equity()
    final = float(trades[-1].get("wallet_after") or init) if trades else init
    return {
        "symbol": base,
        "sym": sym,
        "listed": listed,
        "days": round(days, 1),
        "trades": total,
        "sl": sl,
        "eod": eod,
        "tp": tp,
        "sl_rate_pct": round(sl / total * 100, 1) if total else 0.0,
        "win_pnl": wins,
        "loss_pnl": total - wins,
        "return_pct": round((final / init - 1) * 100, 1) if init > 0 else 0.0,
    }


def main() -> None:
    load_env_oi()
    cfg = OrbConfig.for_backtest()
    out_dir = ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "orb_sl_summary.json"
    csv_path = out_dir / "orb_sl_summary.csv"

    rows: list[dict] = []
    for sym in SYMS:
        t0 = time.time()
        row = summarize_symbol(sym, cfg)
        row["elapsed_sec"] = round(time.time() - t0, 1)
        rows.append(row)
        json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(
            f"DONE {row['symbol']}: SL {row['sl']}/{row['trades']} ({row['sl_rate_pct']}%) "
            f"ret={row['return_pct']:+.1f}% ({row['elapsed_sec']:.0f}s)",
            flush=True,
        )

    rows.sort(key=lambda r: r["sl_rate_pct"])
    fieldnames = [
        "symbol",
        "listed",
        "days",
        "trades",
        "sl",
        "eod",
        "tp",
        "sl_rate_pct",
        "win_pnl",
        "loss_pnl",
        "return_pct",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print()
    print(f"{'symbol':<6} {'trades':>6} {'SL':>4} {'SL%':>6} {'EoD':>4} {'ret%':>8}")
    print("-" * 44)
    for r in rows:
        print(
            f"{r['symbol']:<6} {r['trades']:>6} {r['sl']:>4} {r['sl_rate_pct']:>5.1f}% "
            f"{r['eod']:>4} {r['return_pct']:>+7.1f}%"
        )
    print(f"\nWrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
