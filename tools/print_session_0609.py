#!/usr/bin/env python3
"""Compare 2026-06-09 session: live screenshot vs fixed backtest."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from orb.backtest import run_backtest  # noqa: E402
from orb.config import DEFAULT_SYMBOLS, OrbConfig  # noqa: E402

DAY = "2026-06-09"

# From live dashboard screenshot (old engine / old ATR)
LIVE_0609 = {
    "COIN": {"side": "LONG", "outcome": "loss", "entry": 163.64, "sl": 163.089, "pnl": -85.0},
    "INTC": {"side": "SHORT", "outcome": "loss", "entry": 112.04, "sl": 112.4469, "pnl": -85.0},
    "PAYP": None,
    "GOOGL": {"side": "SHORT", "outcome": "loss", "entry": 366.72, "sl": 367.185, "pnl": -85.0},
    "PLTR": {"side": "SHORT", "outcome": "session_close", "entry": 134.8, "sl": 135.1861, "pnl": 746.89},
    "EWY": {"side": "SHORT", "outcome": "session_close", "entry": 189.24, "sl": 189.7994, "pnl": 806.87},
    "QQQ": {"side": "LONG", "outcome": "loss", "entry": 724.35, "sl": 723.6955, "pnl": -85.0},
}


def main() -> None:
    cfg = OrbConfig.for_backtest()
    init = cfg.per_symbol_bot_equity()
    syms = []
    for s in DEFAULT_SYMBOLS.split(","):
        s = s.strip().upper()
        if not s:
            continue
        syms.append(s if s.endswith("USDT") else s + "USDT")

    raw = run_backtest(days=7, symbols=syms, cfg=cfg, json_path=None, csv_path=None)
    trades = [t for t in raw.get("trades") or [] if t.get("session_date") == DAY and t.get("outcome") not in (None, "supersede")]

    by_sym: dict[str, dict] = {}
    for t in trades:
        base = str(t["symbol"]).replace("USDT", "")
        by_sym[base] = t

    oc_label = {"session_close": "收盘平仓", "loss": "止损", "win": "止盈"}

    print("=" * 100)
    print(f"2026-06-09  |  init {init:,.0f}U/symbol  |  engine=live_scan_sim + ATR fix + 1m chunk")
    print("=" * 100)
    print(f"{'标的':<6} {'实盘(昨)':<28} {'改后回测':<28} {'差异'}")
    print("-" * 100)

    for base in ["COIN", "INTC", "PAYP", "GOOGL", "PLTR", "EWY", "QQQ"]:
        live = LIVE_0609.get(base)
        sim = by_sym.get(base)
        if live is None and sim is None:
            live_s = "无交易 0U"
            sim_s = "无交易"
            diff = "—"
        elif live is None and sim:
            live_s = "无交易 0U"
            oc = oc_label.get(str(sim["outcome"]), sim["outcome"])
            sim_s = f"{sim['side']} {oc} {sim['pnl_usdt']:+.0f}U"
            diff = "新增"
        elif live and sim is None:
            live_oc = "止损" if live["outcome"] == "loss" else "收盘"
            live_s = f"{live['side']} {live_oc} {live['pnl']:+.0f}U"
            sim_s = "无交易"
            diff = "消失"
        else:
            live_oc = "止损" if live["outcome"] == "loss" else "收盘平仓"
            live_s = f"{live['side']} {live_oc} {live['pnl']:+.0f}U"
            oc = oc_label.get(str(sim["outcome"]), sim["outcome"])
            sim_s = f"{sim['side']} {oc} {sim['pnl_usdt']:+.0f}U"
            if live["outcome"] == sim["outcome"]:
                diff = f"PnL {sim['pnl_usdt'] - live['pnl']:+.0f}U"
            else:
                diff = f"{live_oc}→{oc}"
        print(f"{base:<6} {live_s:<28} {sim_s:<28} {diff}")

    print("-" * 100)
    print("\n改后明细 (SL / exit):")
    for base in ["COIN", "INTC", "PAYP", "GOOGL", "PLTR", "EWY", "QQQ"]:
        sim = by_sym.get(base)
        if not sim:
            print(f"  {base}: —")
            continue
        live = LIVE_0609.get(base)
        oc = oc_label.get(str(sim["outcome"]), sim["outcome"])
        sl_note = ""
        if live and live.get("sl"):
            sl_note = f"  (SL {live['sl']:.4f}→{sim['sl']:.4f})"
        print(
            f"  {base}: {sim['side']} {oc}  entry={sim['entry']:.4f}  "
            f"sl={sim['sl']:.4f}  exit={sim['exit_price']:.4f}  pnl={sim['pnl_usdt']:+.2f}U{sl_note}"
        )

    print("\n改后各标钱包 (复合, 含 6/9 前历史):")
    wallets = {s: init for s in syms}
    for t in raw.get("trades") or []:
        if t.get("outcome") in (None, "supersede"):
            continue
        if str(t.get("session_date", "")) > DAY:
            continue
        wallets[t["symbol"]] = float(t.get("wallet_after") or wallets[t["symbol"]])
    for s in syms:
        base = s.replace("USDT", "")
        w = wallets[s]
        print(f"  {base}: {w:,.0f}U  ({w - init:+.0f}U vs 初始)")


if __name__ == "__main__":
    main()
