#!/usr/bin/env python3
"""Batch GTL honest research for 7 US stock tokens (30m)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402

load_env_oi()

import pandas as pd  # noqa: E402

from orb.core.config import OrbConfig  # noqa: E402
from orb.core.kline_cache import norm_symbol  # noqa: E402
from orb.core.symbols import parse_symbol_list  # noqa: E402
from orb.gtl.engine import compute_gtl_dataframe  # noqa: E402
from orb.gtl.resample import resample_ohlcv  # noqa: E402
from orb.gtl.vnpy.backtest import run_gtl_vnpy_backtest  # noqa: E402
from orb.core.symbols_path import resolve_symbols_path  # noqa: E402
from tools.cta.research_gtl_vnpy import _load_symbol_df  # noqa: E402
from tools.cta.validate_gtl import _honest_trading_sim, _load  # noqa: E402

POOL7 = ["INTC", "SOXL", "HOOD", "CRCL", "COIN", "SNDK", "MSTR"]


def _analyze(sym: str, lo: str, hi: str, rs: str, equity: float, cfg: OrbConfig) -> dict:
    fetch_lo = (pd.Timestamp(lo) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    raw = _load_symbol_df(sym, fetch_lo, hi, cfg)
    if raw.empty:
        raw = _load(sym, lo, hi)
    if raw.empty:
        return {"symbol": sym.replace("USDT", ""), "error": "no_data"}
    if rs != "1m":
        raw = resample_ohlcv(raw, rs)

    gtl = compute_gtl_dataframe(raw, lookback=23, vol_window=500)
    breaks = gtl[gtl["break_dir"] != 0]
    aligned_n = int(gtl["break_aligns_birth"].sum())
    birth_mask = breaks["birth_hit"] >= 0
    birth_hit = float(breaks.loc[birth_mask, "birth_hit"].mean()) if birth_mask.any() else float("nan")
    sim = _honest_trading_sim(raw, gtl)

    start = pd.Timestamp(lo).to_pydatetime()
    end = (pd.Timestamp(hi) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).to_pydatetime()
    honest = run_gtl_vnpy_backtest(
        "gtl_birth_break_honest", sym, df=raw, start=start, end=end, capital=equity, quiet=True
    )
    hs = honest.get("summary") or {}

    last = gtl.iloc[-1]
    return {
        "symbol": sym.replace("USDT", ""),
        "bars": len(raw),
        "structures": len(breaks),
        "birth_hit_rate": round(birth_hit, 3) if birth_hit == birth_hit else None,
        "aligned_setups": aligned_n,
        "buy_hold_move": sim.get("buy_hold_move"),
        "sim_hold_20": sim.get("hold_20_sum"),
        "honest_realized": hs.get("realized_pnl"),
        "honest_opens": hs.get("opens"),
        "honest_win_rate": hs.get("realized_win_rate"),
        "display_up_pct": round(float(last.get("display_prob_up", last["prob_up"])) * 100, 1),
        "forecast": "up" if last.get("forecast_up") else "down" if last.get("forecast_down") else "-",
        "forecast_conf": last.get("forecast_confidence", "?"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="GTL pool7 batch honest research")
    ap.add_argument("--from-date", default="2026-02-01")
    ap.add_argument("--to-date", default="2026-06-30")
    ap.add_argument("--resample", default="30m")
    ap.add_argument("--equity", type=float, default=1000.0)
    ap.add_argument("--symbols-file", default=str(resolve_symbols_path()))
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    syms = [norm_symbol(s) for s in parse_symbol_list(Path(args.symbols_file).read_text(encoding="utf-8"))]
    if not syms:
        syms = [norm_symbol(s) for s in POOL7]

    lo, hi, rs = args.from_date, args.to_date, args.resample.strip() or "1m"
    cfg = OrbConfig.from_env()
    print(f"[GTL pool7] {lo}..{hi} resample={rs} n={len(syms)}", flush=True)
    t0 = time.time()
    rows = []
    for sym in syms:
        t1 = time.time()
        print(f"  {sym.replace('USDT',''):5s} ...", end=" ", flush=True)
        row = _analyze(sym, lo, hi, rs, float(args.equity), cfg)
        rows.append(row)
        if row.get("error"):
            print(row["error"], flush=True)
        else:
            print(
                f"birth_hit={row.get('birth_hit_rate')} aligned={row.get('aligned_setups')} "
                f"realized={row.get('honest_realized'):+.2f} bh={row.get('buy_hold_move'):+.2f} "
                f"display={row.get('display_up_pct')}% ({row.get('forecast')}) [{time.time()-t1:.0f}s]",
                flush=True,
            )

    print("\n=== SUMMARY TABLE ===")
    hdr = f"{'sym':6s} {'birth%':>6s} {'align':>5s} {'bh':>8s} {'hold20':>8s} {'realized':>9s} {'opens':>5s} {'disp%':>5s} {'fc':>4s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r.get("error"):
            print(f"{r['symbol']:6s} ERROR: {r['error']}")
            continue
        print(
            f"{r['symbol']:6s} {r.get('birth_hit_rate') or 0:6.3f} {r.get('aligned_setups') or 0:5d} "
            f"{float(r.get('buy_hold_move') or 0):+8.2f} {float(r.get('sim_hold_20') or 0):+8.2f} "
            f"{float(r.get('honest_realized') or 0):+9.2f} {int(r.get('honest_opens') or 0):5d} "
            f"{float(r.get('display_up_pct') or 0):5.1f} {str(r.get('forecast') or '-'):>4s}"
        )

    out_dir = ROOT / "output" / "orb" / "cta"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.json_out) if args.json_out else out_dir / f"gtl_pool7_{lo}_{hi}.json"
    payload = {
        "date_range": {"from": lo, "to": hi},
        "resample": rs,
        "symbols": [r.get("symbol") for r in rows],
        "results": rows,
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o)),
        encoding="utf-8",
    )
    print(f"\njson -> {out_path} ({time.time()-t0:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
