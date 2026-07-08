#!/usr/bin/env python3
"""Print daily equity detail for GTL flip sim variants."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi

load_env_oi()

import pandas as pd
from orb.core.config import OrbConfig
from orb.core.kline_cache import norm_symbol
from orb.core.symbols import parse_symbol_list
from orb.core.symbols_path import resolve_symbols_path
from tools.cta._gtl_flip_sim import (
    POOL7,
    _load_symbol_cache,
    _run_symbol,
    last_trading_days,
)
from tools.cta._snap_gtl_pool7_day import POOL7 as _P7


def _print_variant(name: str, rows: list[dict], days: list[str], syms: list[str]) -> None:
    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"{'='*72}")
    labels = [s.replace("USDT", "") for s in syms]
    for sym in labels:
        sub = [r for r in rows if r["symbol"] == sym]
        if not sub:
            continue
        fe = sub[-1]["equity_end"]
        fs = sub[0]["equity_start"]
        print(f"\n--- {sym}  ({fs:.0f} -> {fe:.2f}U) ---")
        for day in days:
            day_rows = [r for r in sub if r["day"] == day]
            if not day_rows:
                continue
            r = day_rows[0]
            if r.get("skipped"):
                print(f"  {day}  SKIP  ({r.get('reason')})")
                continue
            et = r.get("entry_time") or "-"
            legs = int(r.get("n_legs") or 0)
            print(
                f"  {day}  {r['open_dir']:>4s} @{et}  "
                f"{r['equity_start']:7.2f} -> {r['equity_end']:7.2f}U  "
                f"({r['day_return_pct']:+.2f}%)  legs={legs}  "
                f"pnl={r['total_pnl_usdt']:+.2f}U"
            )


def main() -> int:
    cfg = OrbConfig.from_env()
    days = last_trading_days(7, "2026-07-05", cfg)
    cap = 1000.0
    exclude = {"SNDK"}
    syms = [norm_symbol(s) for s in parse_symbol_list(Path(resolve_symbols_path()).read_text(encoding="utf-8"))]
    if not syms:
        syms = [norm_symbol(s) for s in POOL7]

    fetch_lo = (pd.Timestamp(days[0]) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    fetch_hi = (pd.Timestamp(days[-1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    opts = dict(sl_flat=True, allow_flip=True, capital=cap)
    variants = [
        ("open_0930 | break+forecast @09:30 | SL-flat+flip", dict(entry="open", open_mode="break_forecast")),
        ("first_break | 等首个 aligned break | SL-flat+flip", dict(entry="first_break", open_mode="break")),
    ]

    print(f"GTL daily detail | {days[0]} .. {days[-1]} | 5m RTH | {cap:.0f}U/sym compound | ex-SNDK")
    print("B-rule: SL后flat, 允许flip, 否则15:55收盘平")

    cache: dict = {}
    for sym in syms:
        label = sym.replace("USDT", "")
        if label in exclude:
            continue
        loaded = _load_symbol_cache(sym, fetch_lo, fetch_hi, cfg, "5m")
        if loaded:
            cache[sym] = loaded

    all_rows: dict[str, list] = {}
    for name, vopts in variants:
        summary: list[dict] = []
        for sym, (df, gtl) in cache.items():
            rows, _, _ = _run_symbol(days, sym, df, gtl, cfg, **opts, **vopts)
            summary.extend(rows)
        all_rows[name] = summary
        _print_variant(name, summary, days, list(cache.keys()))

    # pool daily
    print(f"\n{'='*72}")
    print("  POOL 每日合计 (USDT)")
    print(f"{'='*72}")
    hdr = f"{'day':12s}" + "".join(f"{n.split('|')[0].strip():>14s}" for n, _ in variants)
    print(hdr)
    print("-" * len(hdr))
    for day in days:
        cells = []
        for name, _ in variants:
            rows = all_rows[name]
            v = sum(float(r["total_pnl_usdt"]) for r in rows if r["day"] == day and not r.get("skipped"))
            cells.append(f"{v:+12.2f}U")
        print(f"{day:12s}" + "".join(f"{c:>14s}" for c in cells))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
