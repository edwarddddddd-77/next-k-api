#!/usr/bin/env python3
"""GTL open-direction hold: enter at RTH open, exit on opposite aligned break or EOD."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi

load_env_oi()

import pandas as pd
from binance_fapi import fetch_klines_forward, klines_to_df
from orb.core.config import OrbConfig
from orb.core.kline_cache import norm_symbol
from orb.core.session import is_trading_session, session_day_str
from orb.core.symbols import parse_symbol_list
from orb.gtl.engine import compute_gtl_dataframe
from orb.gtl.resample import resample_ohlcv
from orb.core.symbols_path import resolve_symbols_path
from tools.cta.research_gtl_vnpy import _load_symbol_df

POOL7 = ["INTC", "SOXL", "HOOD", "CRCL", "COIN", "SNDK", "MSTR"]


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


def _rth_day_slice(df: pd.DataFrame, gtl: pd.DataFrame, day: str, cfg: OrbConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    ms = df["open_time"].astype(int)

    def _ok(m: int) -> bool:
        if session_day_str(m, tz=cfg.session_tz, session_open_time=cfg.session_open_time) != day:
            return False
        return bool(
            is_trading_session(
                m,
                tz=cfg.session_tz,
                session_open_time=cfg.session_open_time,
                session_close_time=cfg.session_close_time,
                market=cfg.market,
            )
        )

    m = ms.map(_ok)
    return df[m].reset_index(drop=True), gtl[m].reset_index(drop=True)


def _open_direction(row: pd.Series, *, mode: str) -> tuple[int, str]:
    """1=long, -1=short, 0=no trade. Returns (dir, source)."""
    if mode in ("break", "break_first"):
        if bool(row.get("break_aligns_birth")):
            d = int(row.get("break_dir") or 0)
            if d > 0:
                return 1, "aligned_break_up"
            if d < 0:
                return -1, "aligned_break_down"
        if mode == "break":
            return 0, "no_aligned_break_at_open"

    up = bool(row.get("forecast_up"))
    dn = bool(row.get("forecast_down"))
    if up and not dn:
        return 1, "forecast_up"
    if dn and not up:
        return -1, "forecast_down"
    disp = float(row.get("display_prob_up", 0.5))
    if disp >= 0.55:
        return 1, f"display_{disp:.0%}_up"
    if disp <= 0.45:
        return -1, f"display_{disp:.0%}_down"
    return 0, "neutral"


def sim_day(day_df: pd.DataFrame, day_gtl: pd.DataFrame, cfg: OrbConfig, *, mode: str) -> dict | None:
    if day_gtl.empty:
        return None
    first_g = day_gtl.iloc[0]
    pos, src = _open_direction(first_g, mode=mode)
    if pos == 0:
        return {"skipped": True, "reason": src}

    entry_px = float(day_df.iloc[0]["close"])
    ts0 = pd.Timestamp(int(day_df.iloc[0]["open_time"]), unit="ms", tz="UTC").tz_convert(cfg.session_tz)
    exit_px = float(day_df.iloc[-1]["close"])
    exit_reason = "eod"
    exit_time = pd.Timestamp(int(day_df.iloc[-1]["open_time"]), unit="ms", tz="UTC").tz_convert(cfg.session_tz)
    exit_i = len(day_df) - 1

    for i in range(1, len(day_gtl)):
        r = day_gtl.iloc[i]
        if not bool(r.get("break_aligns_birth")):
            continue
        d = int(r["break_dir"])
        if pos == 1 and d < 0:
            exit_px = float(day_df.iloc[i]["close"])
            exit_reason = "opposite_break"
            exit_time = pd.Timestamp(int(day_df.iloc[i]["open_time"]), unit="ms", tz="UTC").tz_convert(cfg.session_tz)
            exit_i = i
            break
        if pos == -1 and d > 0:
            exit_px = float(day_df.iloc[i]["close"])
            exit_reason = "opposite_break"
            exit_time = pd.Timestamp(int(day_df.iloc[i]["open_time"]), unit="ms", tz="UTC").tz_convert(cfg.session_tz)
            exit_i = i
            break

    pnl = (exit_px - entry_px) if pos > 0 else (entry_px - exit_px)
    pct = pnl / entry_px * 100
    hh = float(first_g.get("frozen_hh") or 0)
    ll = float(first_g.get("frozen_ll") or 0)
    return {
        "skipped": False,
        "side": "long" if pos > 0 else "short",
        "signal": src,
        "open_disp_pct": round(float(first_g.get("display_prob_up", 0.5)) * 100, 1),
        "entry": round(entry_px, 2),
        "exit": round(exit_px, 2),
        "entry_time": ts0.strftime("%H:%M"),
        "exit_time": exit_time.strftime("%H:%M"),
        "exit_reason": exit_reason,
        "pnl": round(pnl, 2),
        "pct": round(pct, 2),
        "bars_held": exit_i,
        "box": f"{hh:.2f}/{ll:.2f}",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", action="append", default=[], help="e.g. 2026-07-01 (repeatable)")
    ap.add_argument("--resample", default="5m")
    ap.add_argument(
        "--mode",
        default="break",
        choices=["break", "forecast"],
        help="break=aligned break at open bar; forecast=soft forecast/display",
    )
    args = ap.parse_args()

    days = args.day or ["2026-07-01", "2026-07-02"]
    rs = args.resample.strip() or "5m"
    cfg = OrbConfig.from_env()
    syms = [norm_symbol(s) for s in parse_symbol_list(Path(resolve_symbols_path()).read_text(encoding="utf-8"))]
    if not syms:
        syms = [norm_symbol(s) for s in POOL7]

    print(f"=== GTL open-hold sim | {rs} RTH | mode={args.mode} | days={days} ===")
    print("Rule: first RTH bar -> enter @ close; exit opposite aligned break else EOD\n")

    for day in days:
        print(f"--- {day} ---")
        print(
            f"{'sym':6s} {'side':>5s} {'signal':>18s} {'entry':>8s} {'exit':>8s} "
            f"{'pnl':>8s} {'pct':>6s} {'exit':>8s} {'t_out':>5s}"
        )
        print("-" * 72)
        rows = []
        for sym in syms:
            fetch_lo = (pd.Timestamp(day) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
            fetch_hi = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            raw = _load_range(sym, fetch_lo, fetch_hi, cfg)
            if raw.empty:
                continue
            df = resample_ohlcv(raw, rs) if rs != "1m" else raw
            gtl = compute_gtl_dataframe(df, lookback=23, vol_window=500)
            day_df, day_gtl = _rth_day_slice(df, gtl, day, cfg)
            r = sim_day(day_df, day_gtl, cfg, mode=args.mode)
            if r is None:
                continue
            label = sym.replace("USDT", "")
            if r.get("skipped"):
                print(f"{label:6s}  SKIP  ({r.get('reason')})")
                continue
            rows.append({**r, "symbol": label})
            print(
                f"{label:6s} {r['side']:>5s} {r['signal']:>18s} {r['entry']:8.2f} {r['exit']:8.2f} "
                f"{r['pnl']:+8.2f} {r['pct']:+5.2f}% {r['exit_reason']:>8s} {r['exit_time']:>5s}"
            )

        if rows:
            tot = sum(x["pnl"] for x in rows)
            wins = sum(1 for x in rows if x["pnl"] > 0)
            opp = sum(1 for x in rows if x["exit_reason"] == "opposite_break")
            print("-" * 72)
            print(
                f"{'TOTAL':6s} {len(rows)} trades | win {wins}/{len(rows)} | "
                f"opposite_exit {opp} | sum pnl {tot:+.2f} USD | avg {tot/len(rows):+.2f}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
