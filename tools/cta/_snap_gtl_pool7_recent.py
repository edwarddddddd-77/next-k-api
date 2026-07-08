#!/usr/bin/env python3
"""GTL pool7 batch snapshot for last N NYSE trading days (5m RTH default)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi

load_env_oi()

import pandas as pd
from orb.core.config import OrbConfig
from orb.core.kline_cache import norm_symbol
from orb.core.us_equity_calendar import is_us_equity_trading_day
from orb.core.symbols import parse_symbol_list
from orb.core.symbols_path import resolve_symbols_path
from tools.cta._snap_gtl_pool7_day import POOL7, _load_range, _session_mask, snap_symbol

OUTPUT = ROOT / "output" / "orb" / "cta"


def last_trading_days(n: int, asof: str, cfg: OrbConfig) -> list[str]:
    """Last n NYSE trading days strictly before asof calendar date."""
    d = pd.Timestamp(asof.strip(), tz=cfg.session_tz)
    out: list[str] = []
    while len(out) < n:
        d -= pd.Timedelta(days=1)
        day = d.strftime("%Y-%m-%d")
        if is_us_equity_trading_day(day):
            out.append(day)
    return sorted(out)


def snap_symbol_cached(
    sym: str,
    day: str,
    df,
    gtl,
    cfg: OrbConfig,
    *,
    rth_only: bool,
) -> dict:
    """Day slice from precomputed resampled df + gtl."""
    from tools.cta.validate_gtl import _honest_trading_sim

    mask = _session_mask(df, day, cfg, rth_only=rth_only)
    day_gtl = gtl[mask].reset_index(drop=True)
    day_df = df[mask].reset_index(drop=True)
    if day_gtl.empty:
        return {"symbol": sym.replace("USDT", ""), "error": "no_bars_on_day"}

    breaks = day_gtl[day_gtl["break_dir"] != 0]
    births = day_gtl[day_gtl["is_birth_bar"]]
    aligned = day_gtl[day_gtl["break_aligns_birth"]]
    fc_up = len(day_gtl[day_gtl["forecast_up"]])
    fc_dn = len(day_gtl[day_gtl["forecast_down"]])

    birth_hit = float("nan")
    if not breaks.empty:
        bm = breaks["birth_hit"] >= 0
        if bm.any():
            birth_hit = float(breaks.loc[bm, "birth_hit"].mean())

    first = day_gtl.iloc[0]
    last = day_gtl.iloc[-1]
    last_px = float(day_df.iloc[-1]["close"])
    sim = _honest_trading_sim(day_df, day_gtl)

    open_break = ""
    if bool(first.get("break_aligns_birth")):
        open_break = "up" if int(first["break_dir"]) > 0 else "down"

    return {
        "symbol": sym.replace("USDT", ""),
        "day": day,
        "bars": len(day_gtl),
        "structures_born": len(births),
        "breaks": len(breaks),
        "aligned": len(aligned),
        "birth_hit_rate": round(birth_hit, 3) if birth_hit == birth_hit else None,
        "forecast_up_bars": fc_up,
        "forecast_down_bars": fc_dn,
        "open_break": open_break,
        "open_forecast": "up" if first["forecast_up"] else "down" if first["forecast_down"] else "-",
        "open_disp_pct": round(float(first["display_prob_up"]) * 100, 1),
        "close": round(last_px, 2),
        "hh": round(float(last["frozen_hh"]), 2),
        "ll": round(float(last["frozen_ll"]), 2),
        "display_up_pct": round(float(last["display_prob_up"]) * 100, 1),
        "forecast": "up" if last["forecast_up"] else "down" if last["forecast_down"] else "-",
        "conf": last["forecast_confidence"],
        "n_eff": round(float(last["n_eff"]), 1),
        "hold_1bar": sim.get("hold_1_sum"),
        "hold_4bar": sim.get("hold_4_sum"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7, help="number of NYSE trading days")
    ap.add_argument("--asof", default="", help="count back from this date (default: today ET)")
    ap.add_argument("--resample", default="5m")
    ap.add_argument("--no-rth", action="store_true", help="include extended hours")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--symbols-file", default=str(resolve_symbols_path()))
    args = ap.parse_args()

    cfg = OrbConfig.from_env()
    rth = not bool(args.no_rth)
    rs = args.resample.strip() or "5m"
    asof = args.asof.strip() or pd.Timestamp.now(tz=cfg.session_tz).strftime("%Y-%m-%d")
    days = last_trading_days(int(args.days), asof, cfg)

    syms = [norm_symbol(s) for s in parse_symbol_list(Path(args.symbols_file).read_text(encoding="utf-8"))]
    if not syms:
        syms = [norm_symbol(s) for s in POOL7]

    sess = f"RTH {cfg.session_open_time}-{cfg.session_close_time} ET" if rth else "full day ET"
    print(f"=== GTL pool7 | last {len(days)} sessions | {rs} | {sess} ===")
    print(f"asof={asof}  days={days[0]} .. {days[-1]}\n")

    from orb.gtl.engine import compute_gtl_dataframe
    from orb.gtl.resample import resample_ohlcv

    t0 = time.time()
    all_rows: list[dict] = []
    fetch_lo = (pd.Timestamp(days[0]) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    fetch_hi = (pd.Timestamp(days[-1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    for sym in syms:
        label = sym.replace("USDT", "")
        print(f"--- {label} ---", flush=True)
        raw = _load_range(sym, fetch_lo, fetch_hi, cfg)
        if raw.empty:
            for day in days:
                all_rows.append({"symbol": label, "day": day, "error": "no_data"})
            print("  ERROR no_data")
            continue
        df = resample_ohlcv(raw, rs) if rs != "1m" else raw
        gtl = compute_gtl_dataframe(df, lookback=23, vol_window=500)
        for day in days:
            r = snap_symbol_cached(sym, day, df, gtl, cfg, rth_only=rth)
            all_rows.append(r)
            if r.get("error"):
                print(f"  {day} ERROR {r['error']}")
                continue
            hit = r.get("birth_hit_rate")
            hit_s = f"{hit:.3f}" if hit is not None else "n/a"
            ob = r.get("open_break") or "-"
            print(
                f"  {day} brk={r['breaks']:2d} aln={r['aligned']:2d} hit={hit_s} "
                f"open_brk={ob:4s} close_fc={r['forecast']:4s} disp={r['display_up_pct']:5.1f}%"
            )

    print(f"\n{'day':10s} {'sym':6s} {'brk':>3s} {'aln':>3s} {'hit%':>5s} "
          f"{'op_brk':>6s} {'op_fc':>5s} {'cl_fc':>5s} {'disp%':>5s} {'close':>8s}")
    print("-" * 70)
    for r in all_rows:
        if r.get("error"):
            print(f"{r.get('day','?'):10s} {r['symbol']:6s} ERROR {r['error']}")
            continue
        hit = f"{r['birth_hit_rate']:.3f}" if r.get("birth_hit_rate") is not None else "  n/a"
        print(
            f"{r['day']:10s} {r['symbol']:6s} {r['breaks']:3d} {r['aligned']:3d} {hit:>5s} "
            f"{(r.get('open_break') or '-'):>6s} {r.get('open_forecast','-'):>5s} "
            f"{r['forecast']:>5s} {r['display_up_pct']:5.1f} {r['close']:8.2f}"
        )

    # daily pool totals
    print("\n--- daily pool totals (aligned breaks) ---")
    print(f"{'day':10s} {'aligned':>7s} {'avg hit%':>8s} {'symbols':>7s}")
    for day in days:
        sub = [r for r in all_rows if r.get("day") == day and not r.get("error")]
        if not sub:
            continue
        aln = sum(int(r["aligned"]) for r in sub)
        hits = [r["birth_hit_rate"] for r in sub if r.get("birth_hit_rate") is not None]
        avg_hit = sum(hits) / len(hits) if hits else float("nan")
        hit_s = f"{avg_hit:.3f}" if avg_hit == avg_hit else "n/a"
        print(f"{day:10s} {aln:7d} {hit_s:>8s} {len(sub):7d}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.json_out) if args.json_out else OUTPUT / f"gtl_pool7_recent_{days[0]}_{days[-1]}_{rs}.json"
    payload = {
        "asof": asof,
        "days": days,
        "resample": rs,
        "rth": rth,
        "results": all_rows,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\njson -> {out_path} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
