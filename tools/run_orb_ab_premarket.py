#!/usr/bin/env python3
"""ORB 盘前过滤 A/B 回测（同一标的、同一窗口）。"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402
from orb.backtest import run_backtest  # noqa: E402
from orb.config import OrbConfig  # noqa: E402
from tools.print_pltr_backtest_detail import days_since_onboard, symbol_onboard_ms  # noqa: E402


def _summarize(raw: dict, init: float) -> dict:
    trades = [t for t in (raw.get("trades") or []) if t.get("outcome") and t["outcome"] != "supersede"]
    wins = sum(1 for t in trades if float(t.get("pnl_usdt") or 0) > 0)
    losses = sum(1 for t in trades if float(t.get("pnl_usdt") or 0) < 0)
    pnl = sum(float(t.get("pnl_usdt") or 0) for t in trades)
    wallet = init + pnl
    return {
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "wr": wins / (wins + losses) if wins + losses else 0.0,
        "pnl": pnl,
        "ret_pct": (wallet / init - 1) * 100 if init > 0 else 0.0,
        "longs": sum(1 for t in trades if t["side"] == "LONG"),
        "shorts": sum(1 for t in trades if t["side"] == "SHORT"),
        "trades_list": trades,
    }


def _strict_cfg(cfg: OrbConfig) -> OrbConfig:
    """加强盘前：Gap-Go/Fade + RVOL + 最低量 + PMH/PML/VWAP。"""
    return replace(
        cfg,
        premarket_filter=True,
        premarket_source="alpaca",
        premarket_mode="gap_go_fade",
        premarket_rvol_min=2.0,
        premarket_min_volume=10_000.0,
        premarket_min_gap_pct=0.5,
        premarket_require_pmh_long=True,
        premarket_require_pml_short=True,
        premarket_require_vwap_long=True,
        premarket_require_vwap_short=True,
    )


def _enhanced_cfg(cfg: OrbConfig) -> OrbConfig:
    return replace(
        cfg,
        premarket_filter=True,
        premarket_source="alpaca",
    )


def _medium_cfg(cfg: OrbConfig) -> OrbConfig:
    """折中：enhanced + RVOL>=1.5 + min_vol 5k + PMH/PML/VWAP（无 gap_go_fade）。"""
    return replace(
        cfg,
        premarket_filter=True,
        premarket_source="alpaca",
        premarket_mode="enhanced",
        premarket_rvol_min=1.5,
        premarket_min_volume=5_000.0,
        premarket_require_pmh_long=True,
        premarket_require_pml_short=True,
        premarket_require_vwap_long=True,
        premarket_require_vwap_short=True,
    )


def _medium_soft_cfg(cfg: OrbConfig) -> OrbConfig:
    """更松折中：enhanced + RVOL>=1.0，无 min_vol / gap_go_fade。"""
    return replace(
        cfg,
        premarket_filter=True,
        premarket_source="alpaca",
        premarket_mode="enhanced",
        premarket_rvol_min=1.0,
        premarket_min_volume=0.0,
        premarket_require_pmh_long=True,
        premarket_require_pml_short=True,
        premarket_require_vwap_long=True,
        premarket_require_vwap_short=True,
    )


def main() -> None:
    load_env_oi()
    ap = argparse.ArgumentParser(description="ORB premarket A/B backtest")
    ap.add_argument("--symbol", default="COINUSDT")
    ap.add_argument("--days", type=float, default=None)
    ap.add_argument("--since-listing", action="store_true", default=True)
    ap.add_argument(
        "--strict",
        action="store_true",
        help="B 使用 gap_go_fade + RVOL>=2 + min_vol 10k + PMH/VWAP",
    )
    ap.add_argument(
        "--medium",
        action="store_true",
        help="B 折中：enhanced + RVOL>=1.5 + min_vol 5k + PMH/PML/VWAP",
    )
    ap.add_argument(
        "--medium-soft",
        action="store_true",
        help="B 更松：enhanced + RVOL>=1.0，无 min_vol",
    )
    ap.add_argument(
        "--skip-baseline",
        action="store_true",
        help="跳过 A（已知 baseline 时可省时间）",
    )
    args = ap.parse_args()

    sym = str(args.symbol).strip().upper()
    if not sym.endswith("USDT"):
        sym += "USDT"
    base = sym[:-4]

    cfg_base = OrbConfig.for_backtest()
    onboard_ms = symbol_onboard_ms(sym)
    listed = pd.Timestamp(onboard_ms, unit="ms", tz=cfg_base.session_tz).strftime("%Y-%m-%d")
    days = float(args.days) if args.days is not None else days_since_onboard(sym)
    init = cfg_base.per_symbol_bot_equity()

    print("=" * 88)
    print(f"{base} ORB A/B | since Binance listing {listed} | {days:.1f} calendar days")
    print(
        f"init {init:,.0f} U | risk {cfg_base.risk_pct * 100:.0f}% | "
        f"EoD | 5%ATR | macro_filter={cfg_base.macro_filter}"
    )
    if args.strict and args.medium:
        raise SystemExit("use only one of --strict / --medium / --medium-soft")
    if args.strict and args.medium_soft:
        raise SystemExit("use only one of --strict / --medium / --medium-soft")
    if args.medium and args.medium_soft:
        raise SystemExit("use only one of --strict / --medium / --medium-soft")
    if args.strict:
        b_label = "B strict PM"
        b_desc = "B config: gap_go_fade | rvol>=2 | min_vol>=10k | PMH/PML/VWAP | min_gap>=0.5%"
        cfg_b_fn = _strict_cfg
    elif args.medium:
        b_label = "B medium PM"
        b_desc = "B config: enhanced | rvol>=1.5 | min_vol>=5k | PMH/PML/VWAP"
        cfg_b_fn = _medium_cfg
    elif args.medium_soft:
        b_label = "B medium-soft PM"
        b_desc = "B config: enhanced | rvol>=1.0 | PMH/PML/VWAP"
        cfg_b_fn = _medium_soft_cfg
    else:
        b_label = "B alpaca PM"
        b_desc = ""
        cfg_b_fn = _enhanced_cfg
    if b_desc:
        print(b_desc)
    print("=" * 88)

    if args.skip_baseline:
        print("\n[A] Baseline — skipped (--skip-baseline)")
        sa = None
    else:
        print("\n[A] Baseline — no premarket filter ...")
        t0 = time.time()
        raw_a = run_backtest(
            days=days,
            symbols=[sym],
            cfg=replace(cfg_base, premarket_filter=False),
            json_path=None,
        )
        sa = _summarize(raw_a, init)
        print(
            f"    {sa['trades']} trades | pnl {sa['pnl']:+.2f} U ({sa['ret_pct']:+.1f}%) | {time.time()-t0:.0f}s"
        )

    print(f"\n[B] {b_label} ...")
    t0 = time.time()
    cfg_b = cfg_b_fn(cfg_base)
    raw_b = run_backtest(
        days=days,
        symbols=[sym],
        cfg=cfg_b,
        json_path=None,
    )
    sb = _summarize(raw_b, init)
    feed = cfg_base.alpaca_data_feed
    print(
        f"    {sb['trades']} trades | pnl {sb['pnl']:+.2f} U ({sb['ret_pct']:+.1f}%) | "
        f"feed={feed} | {time.time()-t0:.0f}s"
    )

    if sa is None:
        print("\n" + "=" * 88)
        print(f"{'metric':<24} {'B only':>18}")
        print("-" * 88)
        for name, vb in [
            ("trades", sb["trades"]),
            ("wins", sb["wins"]),
            ("losses", sb["losses"]),
            ("win_rate", sb["wr"]),
            ("longs", sb["longs"]),
            ("shorts", sb["shorts"]),
            ("pnl_u", sb["pnl"]),
            ("return_pct", sb["ret_pct"]),
        ]:
            if name == "win_rate":
                print(f"{name:<24} {vb:>17.1%}")
            elif name == "return_pct":
                print(f"{name:<24} {vb:>+17.2f}%")
            elif name == "pnl_u":
                print(f"{name:<24} {vb:>+17.2f}")
            else:
                print(f"{name:<24} {vb:>18.0f}")
        only_b = sb["trades_list"]
        print("\n--- All B trades ---")
        for t in only_b:
            pu = float(t.get("pnl_usdt") or 0)
            print(f"  {t['session_date']} {t['side']:<5} {t['outcome']:<14} pnl {pu:+.2f}")
        return

    print("\n" + "=" * 88)
    print(f"{'metric':<24} {'A baseline':>18} {b_label:>18} {'delta':>14}")
    print("-" * 88)
    rows = [
        ("trades", sa["trades"], sb["trades"], sb["trades"] - sa["trades"], "d"),
        ("wins", sa["wins"], sb["wins"], sb["wins"] - sa["wins"], "d"),
        ("losses", sa["losses"], sb["losses"], sb["losses"] - sa["losses"], "d"),
        ("win_rate", sa["wr"], sb["wr"], sb["wr"] - sa["wr"], "pct"),
        ("longs", sa["longs"], sb["longs"], sb["longs"] - sa["longs"], "d"),
        ("shorts", sa["shorts"], sb["shorts"], sb["shorts"] - sa["shorts"], "d"),
        ("pnl_u", sa["pnl"], sb["pnl"], sb["pnl"] - sa["pnl"], "money"),
        ("return_pct", sa["ret_pct"], sb["ret_pct"], sb["ret_pct"] - sa["ret_pct"], "pct2"),
    ]
    for name, va, vb, d, kind in rows:
        if kind == "pct":
            print(f"{name:<24} {va:>17.1%} {vb:>17.1%} {d*100:>+13.1f}pp")
        elif kind == "money":
            print(f"{name:<24} {va:>+17.2f} {vb:>+17.2f} {d:>+14.2f}")
        elif kind == "pct2":
            print(f"{name:<24} {va:>+17.2f}% {vb:>+17.2f}% {d:>+13.2f}pp")
        else:
            print(f"{name:<24} {va:>18.0f} {vb:>18.0f} {d:>+14.0f}")

    keys_a = {t["session_date"] + t["side"] for t in sa["trades_list"]}
    keys_b = {t["session_date"] + t["side"] for t in sb["trades_list"]}
    only_a = [t for t in sa["trades_list"] if t["session_date"] + t["side"] not in keys_b]
    only_b = [t for t in sb["trades_list"] if t["session_date"] + t["side"] not in keys_a]

    print("\n--- Filtered OUT by B (trades only in A) ---")
    print(f"count={len(only_a)} | sum_pnl={sum(float(t.get('pnl_usdt') or 0) for t in only_a):+.2f} U")
    for t in only_a:
        pu = float(t.get("pnl_usdt") or 0)
        print(f"  {t['session_date']} {t['side']:<5} {t['outcome']:<14} pnl {pu:+.2f}")

    print("\n--- Only in B ---")
    print(f"count={len(only_b)} | sum_pnl={sum(float(t.get('pnl_usdt') or 0) for t in only_b):+.2f} U")
    for t in only_b:
        pu = float(t.get("pnl_usdt") or 0)
        print(f"  {t['session_date']} {t['side']:<5} {t['outcome']:<14} pnl {pu:+.2f}")

    both = []
    for ta in sa["trades_list"]:
        k = ta["session_date"] + ta["side"]
        if k in keys_b:
            tb = next(t for t in sb["trades_list"] if t["session_date"] + t["side"] == k)
            if ta.get("outcome") != tb.get("outcome") or abs(float(ta.get("pnl_usdt") or 0) - float(tb.get("pnl_usdt") or 0)) > 0.01:
                both.append((ta, tb))
    if both:
        print("\n--- Same session+side but different outcome (wallet path diverged) ---")
        for ta, tb in both:
            print(
                f"  {ta['session_date']} {ta['side']} A:{ta['outcome']} {float(ta.get('pnl_usdt') or 0):+.2f} "
                f"B:{tb['outcome']} {float(tb.get('pnl_usdt') or 0):+.2f}"
            )


if __name__ == "__main__":
    main()
