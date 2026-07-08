#!/usr/bin/env python3
"""GTL pool7 aligned-break detail for recent NYSE sessions (5m RTH)."""

from __future__ import annotations

import argparse
import csv
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
from orb.core.symbols import parse_symbol_list
from orb.core.us_equity_calendar import is_us_equity_trading_day
from orb.gtl.engine import compute_gtl_dataframe
from orb.gtl.resample import resample_ohlcv
from orb.core.symbols_path import resolve_symbols_path
from tools.cta._snap_gtl_pool7_day import POOL7, _load_range, _session_mask

HOLDS = (1, 4, 20)
OUTPUT = ROOT / "output" / "orb" / "cta"


def last_trading_days(n: int, asof: str, cfg: OrbConfig) -> list[str]:
    d = pd.Timestamp(asof.strip(), tz=cfg.session_tz)
    out: list[str] = []
    while len(out) < n:
        d -= pd.Timedelta(days=1)
        day = d.strftime("%Y-%m-%d")
        if is_us_equity_trading_day(day):
            out.append(day)
    return sorted(out)


def aligned_details(
    sym: str,
    day: str,
    df: pd.DataFrame,
    gtl: pd.DataFrame,
    cfg: OrbConfig,
) -> list[dict]:
    mask = _session_mask(df, day, cfg, rth_only=True)
    sub_df = df[mask].reset_index(drop=True)
    sub_gtl = gtl[mask].reset_index(drop=True)
    if sub_gtl.empty:
        return []

    px = sub_df["close"].astype(float).values
    hi = sub_df["high"].astype(float).values
    lo = sub_df["low"].astype(float).values
    n = len(px)
    label = sym.replace("USDT", "")
    rows: list[dict] = []

    for i in sub_gtl.index[sub_gtl["break_aligns_birth"]]:
        r = sub_gtl.loc[i]
        d = int(r["break_dir"])
        ep = float(px[i])
        bhh = float(r["broken_hh"] or r["frozen_hh"])
        bll = float(r["broken_ll"] or r["frozen_ll"])
        stop = bll if d > 0 else bhh
        span = max(bhh - bll, 1e-9)
        ts = pd.Timestamp(int(sub_df.iloc[i]["open_time"]), unit="ms", tz="UTC").tz_convert(cfg.session_tz)

        rec: dict = {
            "day": day,
            "symbol": label,
            "time_et": ts.strftime("%H:%M"),
            "dir": "up" if d > 0 else "down",
            "entry": round(ep, 2),
            "stop": round(stop, 2),
            "box_hh": round(bhh, 2),
            "box_ll": round(bll, 2),
            "box_pct": round(span / ep * 100, 2),
            "birth_disp_pct": round(float(r.get("birth_display_prob_up", r.get("display_prob_up", 0.5))) * 100, 1),
            "disp_pct": round(float(r["display_prob_up"]) * 100, 1),
        }

        for h in HOLDS:
            j = min(i + h, n - 1)
            signed = (px[j] - ep) if d > 0 else (ep - px[j])
            rec[f"pct_{h}"] = round(signed / ep * 100, 2)
            rec[f"ok_{h}"] = "+" if signed > 0 else "-"

        # structure stop hit before 20 bars?
        stopped = False
        for j in range(i + 1, min(i + 21, n)):
            if d > 0 and lo[j] <= stop:
                rec["stop_hit_20"] = "Y"
                stopped = True
                break
            if d < 0 and hi[j] >= stop:
                rec["stop_hit_20"] = "Y"
                stopped = True
                break
        if not stopped:
            rec["stop_hit_20"] = "N"

        rows.append(rec)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--asof", default="")
    ap.add_argument("--from-date", default="")
    ap.add_argument("--to-date", default="")
    ap.add_argument("--resample", default="5m")
    ap.add_argument("--csv-out", default="")
    ap.add_argument("--symbols-file", default=str(resolve_symbols_path()))
    args = ap.parse_args()

    cfg = OrbConfig.from_env()
    rs = args.resample.strip() or "5m"
    if args.from_date.strip() and args.to_date.strip():
        d0 = pd.Timestamp(args.from_date.strip())
        d1 = pd.Timestamp(args.to_date.strip())
        days = []
        d = d0
        while d <= d1:
            ds = d.strftime("%Y-%m-%d")
            if is_us_equity_trading_day(ds):
                days.append(ds)
            d += pd.Timedelta(days=1)
    else:
        asof = args.asof.strip() or pd.Timestamp.now(tz=cfg.session_tz).strftime("%Y-%m-%d")
        days = last_trading_days(int(args.days), asof, cfg)

    syms = [norm_symbol(s) for s in parse_symbol_list(Path(args.symbols_file).read_text(encoding="utf-8"))]
    if not syms:
        syms = [norm_symbol(s) for s in POOL7]

    fetch_lo = (pd.Timestamp(days[0]) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    fetch_hi = (pd.Timestamp(days[-1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    all_rows: list[dict] = []
    t0 = time.time()
    print(f"=== GTL aligned-break detail | {days[0]}..{days[-1]} | {rs} RTH ===\n")

    for sym in syms:
        label = sym.replace("USDT", "")
        raw = _load_range(sym, fetch_lo, fetch_hi, cfg)
        if raw.empty:
            print(f"{label}: no data")
            continue
        df = resample_ohlcv(raw, rs) if rs != "1m" else raw
        gtl = compute_gtl_dataframe(df, lookback=23, vol_window=500)
        for day in days:
            rows = aligned_details(sym, day, df, gtl, cfg)
            all_rows.extend(rows)

    fieldnames = [
        "day", "symbol", "time_et", "dir", "entry", "stop", "box_hh", "box_ll", "box_pct",
        "birth_disp_pct", "disp_pct", "pct_1", "ok_1", "pct_4", "ok_4", "pct_20", "ok_20", "stop_hit_20",
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.csv_out) if args.csv_out else OUTPUT / f"gtl_pool7_detail_{days[0]}_{days[-1]}_{rs}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    # print grouped detail
    cur_day = ""
    cur_sym = ""
    idx = 0
    for day in days:
        print(f"\n######## {day} ########")
        for sym in syms:
            label = sym.replace("USDT", "")
            sub = [r for r in all_rows if r["day"] == day and r["symbol"] == label]
            if not sub:
                print(f"\n--- {label} (0) ---")
                continue
            print(f"\n--- {label} ({len(sub)}) ---")
            print(f"{'#':>3s} {'time':>5s} {'dir':>4s} {'entry':>8s} {'stop':>8s} {'box%':>5s} "
                  f"{'5m%':>6s} {'20m%':>6s} {'100m%':>7s} {'SL20':>4s}")
            for r in sub:
                idx += 1
                print(
                    f"{idx:3d} {r['time_et']:>5s} {r['dir']:>4s} {r['entry']:8.2f} {r['stop']:8.2f} "
                    f"{r['box_pct']:5.2f} {r['pct_1']:+5.2f}{r['ok_1']} {r['pct_4']:+5.2f}{r['ok_4']} "
                    f"{r['pct_20']:+6.2f}{r['ok_20']} {r['stop_hit_20']:>4s}"
                )

    print(f"\n--- summary ---")
    print(f"total aligned breaks: {len(all_rows)}")
    print(f"csv -> {csv_path}")
    print(f"({time.time()-t0:.0f}s)")
    print("  5m/20m/100m = move in break direction; SL20 = structure stop hit within 20 bars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
