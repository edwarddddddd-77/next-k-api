#!/usr/bin/env python3
"""GTL snapshot for pool7 on a single day (5m default)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi

load_env_oi()

import pandas as pd
from orb.core.config import OrbConfig
from orb.core.kline_cache import norm_symbol
from orb.core.session import is_trading_session, session_day_str
from orb.core.symbols import parse_symbol_list
from orb.gtl.engine import compute_gtl_dataframe
from orb.gtl.resample import resample_ohlcv
from orb.core.symbols_path import resolve_symbols_path
from tools.cta.research_gtl_vnpy import _load_symbol_df
from tools.cta.validate_gtl import _honest_trading_sim, _load

POOL7 = ["INTC", "SOXL", "HOOD", "CRCL", "COIN", "SNDK", "MSTR"]


from binance_fapi import fetch_klines_forward, klines_to_df  # noqa: E402


def _load_range(sym: str, fetch_lo: str, fetch_hi: str, cfg: OrbConfig) -> pd.DataFrame:
    lo_ms = int(pd.Timestamp(fetch_lo, tz="UTC").value // 1_000_000)
    hi_ms = int(pd.Timestamp(fetch_hi, tz="UTC").value // 1_000_000)
    raw = _load_symbol_df(sym, fetch_lo, fetch_hi, cfg)
    last_ms = int(raw.iloc[-1]["open_time"]) if not raw.empty else 0
    if last_ms < hi_ms - 86400_000:
        fetched = klines_to_df(fetch_klines_forward(sym, "1m", lo_ms, hi_ms))
        if not fetched.empty:
            raw = (
                pd.concat([raw, fetched], ignore_index=True)
                .drop_duplicates(subset=["open_time"], keep="last")
                .sort_values("open_time")
                .reset_index(drop=True)
            )
    if raw.empty:
        raw = klines_to_df(fetch_klines_forward(sym, "1m", lo_ms, hi_ms))
    return raw


def _session_mask(df: pd.DataFrame, day: str, cfg: OrbConfig, *, rth_only: bool) -> pd.Series:
    """Session day uses 09:30 ET anchor; optional RTH 09:30-16:00 filter."""
    ms = df["open_time"].astype(int)

    def _ok(m: int) -> bool:
        if session_day_str(m, tz=cfg.session_tz, session_open_time=cfg.session_open_time) != day:
            return False
        if not rth_only:
            return True
        return bool(
            is_trading_session(
                m,
                tz=cfg.session_tz,
                session_open_time=cfg.session_open_time,
                session_close_time=cfg.session_close_time,
                market=cfg.market,
            )
        )

    return ms.map(_ok)


def snap_symbol(sym: str, day: str, rs: str, cfg: OrbConfig, *, rth_only: bool) -> dict:
    fetch_lo = (pd.Timestamp(day) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    fetch_hi = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = _load_range(sym, fetch_lo, fetch_hi, cfg)
    if raw.empty:
        return {"symbol": sym.replace("USDT", ""), "error": "no_data"}

    df = resample_ohlcv(raw, rs) if rs != "1m" else raw
    gtl = compute_gtl_dataframe(df, lookback=23, vol_window=500)
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

    last = day_gtl.iloc[-1]
    last_px = float(day_df.iloc[-1]["close"])
    sim = _honest_trading_sim(day_df, day_gtl)

    return {
        "symbol": sym.replace("USDT", ""),
        "bars": len(day_gtl),
        "structures_born": len(births),
        "breaks": len(breaks),
        "aligned": len(aligned),
        "birth_hit_rate": round(birth_hit, 3) if birth_hit == birth_hit else None,
        "forecast_up_bars": fc_up,
        "forecast_down_bars": fc_dn,
        "close": round(last_px, 2),
        "hh": round(float(last["frozen_hh"]), 2),
        "ll": round(float(last["frozen_ll"]), 2),
        "display_up_pct": round(float(last["display_prob_up"]) * 100, 1),
        "raw_up_pct": round(float(last["prob_up"]) * 100, 1),
        "forecast": "up" if last["forecast_up"] else "down" if last["forecast_down"] else "-",
        "conf": last["forecast_confidence"],
        "n_eff": round(float(last["n_eff"]), 1),
        "hold_1bar": sim.get("hold_1_sum"),
        "hold_4bar": sim.get("hold_4_sum"),
        "aligned_setups": sim.get("aligned_setups"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-07-02")
    ap.add_argument("--resample", default="5m")
    ap.add_argument("--rth", action="store_true", help="only regular session 09:30-16:00 ET")
    ap.add_argument("--symbols-file", default=str(resolve_symbols_path()))
    args = ap.parse_args()

    syms = [norm_symbol(s) for s in parse_symbol_list(Path(args.symbols_file).read_text(encoding="utf-8"))]
    if not syms:
        syms = [norm_symbol(s) for s in POOL7]

    cfg = OrbConfig.from_env()
    day, rs = args.date, args.resample.strip() or "1m"
    rth = bool(args.rth)
    sess = f"RTH {cfg.session_open_time}-{cfg.session_close_time} {cfg.session_tz}" if rth else "full day ET"
    print(f"=== GTL pool7 | {day} | {rs} | {sess} ===\n")
    print(
        f"{'sym':6s} {'bars':>4s} {'brk':>4s} {'aln':>4s} {'hit%':>5s} "
        f"{'fc↑':>4s} {'fc↓':>4s} {'disp%':>5s} {'fc':>4s} {'close':>8s} {'hh/ll':>18s}"
    )
    print("-" * 78)

    rows = []
    for sym in syms:
        r = snap_symbol(sym, day, rs, cfg, rth_only=rth)
        rows.append(r)
        if r.get("error"):
            print(f"{r['symbol']:6s} ERROR: {r['error']}")
            continue
        hhll = f"{r['hh']:.0f}/{r['ll']:.0f}"
        hit = f"{r['birth_hit_rate']:.3f}" if r.get("birth_hit_rate") is not None else " n/a"
        print(
            f"{r['symbol']:6s} {r['bars']:4d} {r['breaks']:4d} {r['aligned']:4d} {hit:>5s} "
            f"{r['forecast_up_bars']:4d} {r['forecast_down_bars']:4d} {r['display_up_pct']:5.1f} "
            f"{r['forecast']:>4s} {r['close']:8.2f} {hhll:>18s}"
        )

    print(f"\n--- {'正盘末' if rth else '日末'}状态 (最后一根 bar) ---")
    for r in rows:
        if r.get("error"):
            continue
        print(
            f"{r['symbol']:6s} display={r['display_up_pct']:.1f}% raw={r['raw_up_pct']:.1f}% "
            f"n_eff={r['n_eff']} conf={r['conf']} | aligned breaks={r['aligned']} "
            f"sim1bar={r.get('hold_1bar')} sim4bar={r.get('hold_4bar')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
