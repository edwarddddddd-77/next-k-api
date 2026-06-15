#!/usr/bin/env python3
"""ORB walk-forward 回测 CLI。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace

from env_loader import load_env_oi
from orb.core.backtest import run_backtest
from orb.core.config import OrbConfig, DEFAULT_SYMBOLS


def main() -> int:
    load_env_oi()
    ap = argparse.ArgumentParser(description="ORB 量价策略 — walk-forward 回测")
    ap.add_argument("--days", type=float, default=14.0)
    ap.add_argument("--symbols", type=str, default="")
    ap.add_argument("--or-minutes", type=int, default=None)
    ap.add_argument("--interval", type=str, default=None)
    ap.add_argument("--entry-mode", choices=["breakout", "retest"], default=None)
    ap.add_argument("--confirm-bars", type=int, default=None)
    ap.add_argument("--exit-mode", choices=["eod", "fixed_r"], default=None)
    ap.add_argument("--sl-mode", choices=["atr_pct", "or_range"], default=None)
    ap.add_argument("--risk-pct", type=float, default=None)
    ap.add_argument("--account-equity", type=float, default=None)
    ap.add_argument("--symbol-bot-equity", type=float, default=None, help="单标机器人虚拟本金 U")
    ap.add_argument("--fixed-notional", type=float, default=None, help="固定名义 U/笔；>0 时覆盖风险仓位")
    ap.add_argument("--vol-mult", type=float, default=None, help="成交量过滤：当前K线 vol >= vol_ma×mult；0=关闭")
    ap.add_argument("--vol-ma-period", type=int, default=None, help="成交量均线周期（默认20）")
    ap.add_argument("--csv", type=str, default="")
    ap.add_argument("--json-out", type=str, default="")
    args = ap.parse_args()
    cfg = OrbConfig.for_backtest()
    if args.or_minutes is not None:
        cfg = replace(cfg, or_minutes=max(1, args.or_minutes))
    if args.interval:
        cfg = replace(cfg, signal_interval=args.interval.strip().lower())
    if args.entry_mode:
        cfg = replace(cfg, entry_mode=args.entry_mode)
    if args.confirm_bars is not None:
        cfg = replace(cfg, confirm_bars=max(1, args.confirm_bars))
    if args.sl_mode:
        cfg = replace(cfg, sl_mode=args.sl_mode)
    if args.exit_mode:
        cfg = replace(cfg, exit_mode=args.exit_mode)
        if args.exit_mode == "eod":
            cfg = replace(cfg, tp_r_multiple=0.0)
    if args.risk_pct is not None:
        cfg = replace(cfg, risk_pct=max(0.0, args.risk_pct))
    if args.account_equity is not None:
        cfg = replace(cfg, account_equity_usdt=max(0.0, args.account_equity))
    if args.symbol_bot_equity is not None:
        cfg = replace(cfg, symbol_bot_equity_usdt=max(0.0, args.symbol_bot_equity))
    if args.fixed_notional is not None:
        cfg = replace(cfg, fixed_notional_usdt=max(0.0, args.fixed_notional))
    if args.vol_mult is not None:
        cfg = replace(cfg, vol_mult=max(0.0, args.vol_mult))
    if args.vol_ma_period is not None:
        cfg = replace(cfg, vol_ma_period=max(2, args.vol_ma_period))
    syms = (
        [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
        if args.symbols.strip()
        else cfg.symbol_list()
    )
    summary = run_backtest(
        days=float(args.days),
        symbols=syms,
        cfg=cfg,
        json_path=args.json_out or None,
        csv_path=args.csv or None,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "trades"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
