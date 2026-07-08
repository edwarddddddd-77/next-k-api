#!/usr/bin/env python3
"""GTL vnpy 回测 — 接入 BacktestingEngine + GtlBreakoutStrategy。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402

load_env_oi()

import pandas as pd  # noqa: E402

from binance_fapi import fetch_klines_forward, klines_to_df  # noqa: E402
from orb.core.config import OrbConfig  # noqa: E402
from orb.core.kline_cache import load_klines, norm_symbol  # noqa: E402
from orb.core.symbols import parse_symbol_list  # noqa: E402
from orb.cta.vnpy.registry import VNPY_CTA_STRATEGIES  # noqa: E402
from orb.gtl.resample import resample_ohlcv  # noqa: E402
from orb.gtl.vnpy.backtest import run_gtl_vnpy_backtest  # noqa: E402
from orb.core.symbols_path import resolve_symbols_path  # noqa: E402

GTL_KEYS = ["gtl_birth_break", "gtl_break", "gtl_signal", "gtl_signal_break"]


def _fetch_range(sym: str, from_date: str, to_date: str, cfg: OrbConfig) -> pd.DataFrame:
    tz = cfg.session_tz
    lo = pd.Timestamp(from_date.strip(), tz=tz)
    hi = pd.Timestamp(to_date.strip(), tz=tz) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    rows = fetch_klines_forward(sym, "1m", int(lo.value // 1_000_000), int(hi.value // 1_000_000))
    df = klines_to_df(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time").reset_index(drop=True)


def _load_symbol_df(sym: str, from_date: str, to_date: str, cfg: OrbConfig) -> pd.DataFrame:
    tz = cfg.session_tz
    lo = pd.Timestamp(from_date.strip(), tz=tz)
    hi = pd.Timestamp(to_date.strip(), tz=tz) + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
    lo_ms, hi_ms = int(lo.value // 1_000_000), int(hi.value // 1_000_000)
    df = load_klines(sym, "1m")
    if df is not None and not df.empty:
        sl = df[(df["open_time"] >= lo_ms) & (df["open_time"] <= hi_ms)].copy()
        if not sl.empty:
            return sl.sort_values("open_time").reset_index(drop=True)
    return _fetch_range(sym, from_date, to_date, cfg)


def main() -> int:
    ap = argparse.ArgumentParser(description="GTL vnpy backtest")
    ap.add_argument("--symbol", default="")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--symbols-file", default=str(resolve_symbols_path()))
    ap.add_argument("--strategy", default="all", help="gtl_break | gtl_signal | gtl_signal_break | all")
    ap.add_argument("--from-date", default="2026-02-01")
    ap.add_argument("--to-date", default="2026-06-30")
    ap.add_argument("--equity", type=float, default=1000.0)
    ap.add_argument("--resample", default="", help="optional e.g. 5m, 15m, 30m")
    ap.add_argument("--max-hold-bars", type=int, default=0, help="0=disabled; research exit cap")
    ap.add_argument("--exit-opposite-break", action="store_true", help="flat on opposite aligned break")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    cfg = OrbConfig.from_env()
    lo, hi = args.from_date.strip(), args.to_date.strip()
    start = pd.Timestamp(lo).to_pydatetime()
    end = (pd.Timestamp(hi) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).to_pydatetime()
    # Warmup fetch starts earlier for vol_window
    fetch_lo = (pd.Timestamp(lo) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")

    if (args.symbols or "").strip():
        symbols = [norm_symbol(s.strip()) for s in args.symbols.split(",") if s.strip()]
    elif (args.symbol or "").strip():
        symbols = [norm_symbol(args.symbol.strip())]
    else:
        symbols = parse_symbol_list(Path(args.symbols_file).read_text(encoding="utf-8"))

    keys = GTL_KEYS if args.strategy == "all" else [args.strategy.strip()]
    for k in keys:
        if k not in VNPY_CTA_STRATEGIES:
            print(f"Unknown strategy: {k}")
            return 1

    overrides: Dict[str, Any] = {}
    if args.max_hold_bars > 0:
        overrides["max_hold_bars"] = int(args.max_hold_bars)
    if args.exit_opposite_break:
        overrides["exit_on_opposite_break"] = True

    print(f"[GTL vnpy] {lo}..{hi} | {len(symbols)} sym | equity={args.equity}", flush=True)
    t0 = time.time()
    results: List[Dict[str, Any]] = []
    data_cache: Dict[str, pd.DataFrame] = {}

    for k in keys:
        total_net = 0.0
        total_trades = 0
        per_sym = []
        for sym in symbols:
            if sym not in data_cache:
                raw = _load_symbol_df(sym, fetch_lo, hi, cfg)
                if args.resample.strip():
                    raw = resample_ohlcv(raw, args.resample.strip())
                data_cache[sym] = raw
            df = data_cache[sym]
            if df.empty:
                per_sym.append({"symbol": sym, "error": "no_data"})
                continue
            r = run_gtl_vnpy_backtest(
                k,
                sym,
                df=df,
                start=start,
                end=end,
                capital=float(args.equity),
                quiet=True,
                strategy_overrides=overrides or None,
            )
            s = r.get("summary") or {}
            net = float(s.get("net_pnl") or 0)
            realized = float(s.get("realized_pnl") or 0)
            trades = int(s.get("total_trade_count") or 0)
            total_net += net
            total_trades += trades
            per_sym.append(
                {
                    "symbol": r.get("symbol", sym),
                    "net_pnl": s.get("net_pnl"),
                    "realized_pnl": s.get("realized_pnl"),
                    "realized_round_trips": s.get("realized_round_trips"),
                    "trades": trades,
                    "opens": s.get("opens"),
                    "error": r.get("error"),
                }
            )
        results.append(
            {
                "strategy": k,
                "title": VNPY_CTA_STRATEGIES[k]["title"],
                "net_pnl": round(total_net, 2),
                "realized_pnl": round(
                    sum(float(p.get("realized_pnl") or 0) for p in per_sym if isinstance(p, dict)), 2
                ),
                "trades": total_trades,
                "per_symbol": per_sym,
            }
        )
        realized_total = float(results[-1]["realized_pnl"])
        print(
            f"  {k:18s} net={total_net:+.2f} realized={realized_total:+.2f} trades={total_trades}",
            flush=True,
        )

    out_dir = ROOT / "output" / "orb" / "cta"
    out_dir.mkdir(parents=True, exist_ok=True)
    sym_tag = symbols[0].replace("USDT", "") if len(symbols) == 1 else f"pool{len(symbols)}"
    out_path = Path(args.json_out) if args.json_out else out_dir / f"gtl_vnpy_{sym_tag}_{lo}_{hi}.json"
    payload = {
        "date_range": {"from": lo, "to": hi},
        "equity": float(args.equity),
        "resample": args.resample or "1m",
        "strategies": keys,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\njson -> {out_path} ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
