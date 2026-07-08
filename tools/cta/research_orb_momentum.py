#!/usr/bin/env python3
"""ORB baseline vs 5-day momentum filter on pool7 (honest 1m fills)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402

load_env_oi()

import pandas as pd  # noqa: E402

from orb.core.config import OrbConfig  # noqa: E402
from orb.core.kline_cache import norm_symbol, session_dates_from_cache  # noqa: E402
from orb.core.symbols import parse_symbol_list  # noqa: E402
from orb.gtl.engine import compute_gtl_dataframe  # noqa: E402
from orb.gtl.resample import resample_ohlcv  # noqa: E402
from orb.core.symbols_path import resolve_symbols_path  # noqa: E402
from tools.cta.compare_kk_orb_pool7 import (  # noqa: E402
    POOL7,
    backtest_orb_honest,
    _load_range,
)

DEFAULT_SYMBOLS_FILE = ROOT / "config" / "trading_orb" / "symbols.txt"

EMPTY = {
    "trades": 0,
    "win_rate": 0.0,
    "sum_usd": 0.0,
    "fees": 0.0,
    "filtered": 0,
    "conflict": 0,
    "ret_pct": 0.0,
}


def _summarize(out: Dict[str, Any], equity: float) -> Dict[str, Any]:
    if not out or out.get("error"):
        return dict(EMPTY, error=out.get("error") if out else "no_data")
    net = float(out.get("sum_usd", 0) or 0)
    n = int(out.get("trades", 0) or 0)
    return {
        "trades": n,
        "win_rate": float(out.get("win_rate", 0) or 0),
        "sum_usd": round(net, 2),
        "fees": 0.0,
        "filtered": int(out.get("filtered", 0)),
        "no_fill": int(out.get("no_fill", 0)),
        "ret_pct": round(100.0 * net / equity, 1) if equity > 0 else 0.0,
    }


def run_symbol(
    sym: str,
    lo: str,
    hi: str,
    *,
    equity: float,
    slip_bps: float,
    momentum: bool,
) -> Dict[str, Any]:
    cfg = OrbConfig.from_env()
    cfg.macro_filter = True
    cfg.resolve_at_session_close = True
    cfg.momentum_filter = momentum
    cfg.momentum_days = max(1, int(cfg.momentum_days))

    fetch_lo = (pd.Timestamp(lo) - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    df_1m = _load_range(sym, fetch_lo, hi, cfg)
    if df_1m.empty:
        return {"symbol": sym, "error": "no_data"}

    lo_ms = int(pd.Timestamp(lo, tz=cfg.session_tz).value // 1_000_000)
    hi_ms = int(
        (pd.Timestamp(hi, tz=cfg.session_tz) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).value
        // 1_000_000
    )
    df_1m = df_1m[(df_1m["open_time"] >= lo_ms - 30 * 86400 * 1000) & (df_1m["open_time"] <= hi_ms)].copy()
    df_5m = resample_ohlcv(df_1m, "5m")
    df_30m = resample_ohlcv(df_1m, "30m")
    gtl = compute_gtl_dataframe(df_30m, lookback=23, vol_window=500)

    raw = backtest_orb_honest(
        df_1m,
        df_5m,
        gtl,
        df_30m,
        cfg,
        equity=equity,
        slip_bps=slip_bps,
        gtl_mode="none",
    )
    summary = _summarize(raw, equity)
    summary["symbol"] = sym.replace("USDT", "")
    summary["sessions"] = len([d for d in session_dates_from_cache(sym, cfg) if lo <= d <= hi])
    return summary


def _resolve_symbols(args: argparse.Namespace) -> List[str]:
    if args.symbols:
        raw = parse_symbol_list(args.symbols)
    elif args.symbols_file:
        raw = parse_symbol_list(Path(args.symbols_file).read_text(encoding="utf-8"))
    elif DEFAULT_SYMBOLS_FILE.is_file():
        raw = parse_symbol_list(DEFAULT_SYMBOLS_FILE.read_text(encoding="utf-8"))
    else:
        raw = parse_symbol_list(Path(resolve_symbols_path()).read_text(encoding="utf-8"))
    if not raw:
        raw = list(POOL7)
    return [norm_symbol(s) for s in raw]


def main() -> int:
    ap = argparse.ArgumentParser(description="ORB vs ORB+5d momentum filter (pool7)")
    ap.add_argument("--from-date", default="2026-02-01")
    ap.add_argument("--to-date", default="2026-06-30")
    ap.add_argument("--equity", type=float, default=1000.0)
    ap.add_argument("--slip-bps", type=float, default=5.0)
    ap.add_argument("--momentum-days", type=int, default=5)
    ap.add_argument("--symbols", default="", help="comma-separated symbols, e.g. INTC,COIN,TSLA")
    ap.add_argument(
        "--symbols-file",
        default="",
        help=f"symbol list file (default: {DEFAULT_SYMBOLS_FILE.name})",
    )
    args = ap.parse_args()

    syms = _resolve_symbols(args)

    os.environ["ORB_MOMENTUM_DAYS"] = str(max(1, int(args.momentum_days)))

    lo, hi = args.from_date, args.to_date
    eq, slip = float(args.equity), float(args.slip_bps)

    print(f"=== ORB momentum filter A/B | {len(syms)} symbols | {lo}..{hi} ===")
    print(f"symbols: {', '.join(s.replace('USDT','') for s in syms)}")
    print(f"equity={eq}U/symbol | slip={slip}bps | mom={args.momentum_days}d | 15m OR + 5m breakout + EoD")
    print()

    rows: List[Dict[str, Any]] = []
    t0 = time.time()
    for sym in syms:
        try:
            base = run_symbol(sym, lo, hi, equity=eq, slip_bps=slip, momentum=False)
            mom = run_symbol(sym, lo, hi, equity=eq, slip_bps=slip, momentum=True)
        except Exception as exc:
            print(f"  {sym.replace('USDT',''):5s}  ERROR: {exc}", flush=True)
            continue
        row = {
            "symbol": base.get("symbol", sym.replace("USDT", "")),
            "base_trades": base.get("trades", 0),
            "base_usd": base.get("sum_usd", 0),
            "base_wr": base.get("win_rate", 0),
            "mom_trades": mom.get("trades", 0),
            "mom_usd": mom.get("sum_usd", 0),
            "mom_wr": mom.get("win_rate", 0),
            "mom_filtered": mom.get("filtered", 0),
            "delta_usd": round(float(mom.get("sum_usd", 0)) - float(base.get("sum_usd", 0)), 2),
        }
        if base.get("error") or mom.get("error"):
            row["error"] = base.get("error") or mom.get("error")
            print(f"  {row['symbol']:5s}  skip ({row.get('error')})", flush=True)
            continue
        rows.append(row)
        print(
            f"  {row['symbol']:5s}  base {row['base_usd']:+7.2f}U ({row['base_trades']:2d}t)  "
            f"mom {row['mom_usd']:+7.2f}U ({row['mom_trades']:2d}t, filt={row['mom_filtered']})  "
            f"Δ {row['delta_usd']:+.2f}U",
            flush=True,
        )

    tot_base = sum(r["base_usd"] for r in rows)
    tot_mom = sum(r["mom_usd"] for r in rows)
    print()
    print(f"{'TOTAL':5s}  base {tot_base:+7.2f}U          mom {tot_mom:+7.2f}U          Δ {tot_mom - tot_base:+.2f}U")
    print(f"elapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
