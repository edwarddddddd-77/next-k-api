#!/usr/bin/env python3
"""Research downstream strategies: ORB+GTL filter vs GTL retest entry."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402

load_env_oi()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from orb.core.config import OrbConfig  # noqa: E402
from orb.core.indicators import daily_atr_asof  # noqa: E402
from orb.core.kline_cache import norm_symbol  # noqa: E402
from orb.core.resolve import pnl_r, resolve_forward  # noqa: E402
from orb.core.session import (  # noqa: E402
    is_trading_session,
    session_anchor_ms,
    session_close_ms,
    session_day_str,
)
from orb.core.signals import classify_signal  # noqa: E402
from orb.core.symbols import parse_symbol_list  # noqa: E402
from orb.gtl.engine import compute_gtl_dataframe  # noqa: E402
from orb.gtl.resample import resample_ohlcv  # noqa: E402
from orb.core.symbols_path import resolve_symbols_path  # noqa: E402
from tools.cta.research_gtl_vnpy import _load_symbol_df  # noqa: E402
from tools.cta.validate_gtl import _load  # noqa: E402

POOL7 = ["INTC", "SOXL", "HOOD", "CRCL", "COIN", "SNDK", "MSTR"]


def _daily_bars(df_1m: pd.DataFrame, cfg: OrbConfig) -> pd.DataFrame:
    if df_1m.empty:
        return df_1m
    tz = cfg.session_tz
    tmp = df_1m.copy()
    tmp["_day"] = pd.to_datetime(tmp["open_time"], unit="ms", utc=True).dt.tz_convert(tz).dt.normalize()
    rows = []
    for _, g in tmp.groupby("_day"):
        rows.append(
            {
                "open_time": int(g["open_time"].iloc[0]),
                "open": float(g["open"].iloc[0]),
                "high": float(g["high"].max()),
                "low": float(g["low"].min()),
                "close": float(g["close"].iloc[-1]),
                "volume": float(g["volume"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("open_time").reset_index(drop=True)


def _gtl_bias(gtl: pd.DataFrame, df: pd.DataFrame, asof_ms: int) -> str:
    """long / short / neutral from last 30m bar at or before asof_ms."""
    sub = df[df["open_time"] <= asof_ms]
    if sub.empty:
        return "neutral"
    idx = sub.index[-1]
    r = gtl.loc[idx]
    if bool(r.get("birth_forecast_up")) or bool(r.get("forecast_up")):
        if float(r.get("display_prob_up", 0.5)) >= 0.55:
            return "long"
    if bool(r.get("birth_forecast_down")) or bool(r.get("forecast_down")):
        if float(r.get("display_prob_down", 0.5)) >= 0.55:
            return "short"
    return "neutral"


def _session_anchors(df_1m: pd.DataFrame, cfg: OrbConfig) -> List[int]:
    tz = cfg.session_tz
    days = sorted(
        {
            session_day_str(int(ms), tz=tz, session_open_time=cfg.session_open_time)
            for ms in df_1m["open_time"]
        }
    )
    anchors: List[int] = []
    for day in days:
        ts = pd.Timestamp(day, tz=tz)
        if cfg.market == "us_equity":
            from orb.core.us_equity_calendar import is_us_equity_trading_day

            if not is_us_equity_trading_day(day):
                continue
        anchor = session_anchor_ms(
            int(pd.Timestamp(f"{day} 12:00", tz=tz).value // 1_000_000),
            tz=tz,
            session_open_time=cfg.session_open_time,
        )
        anchors.append(anchor)
    return anchors


def backtest_orb(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    gtl: pd.DataFrame,
    df_30m: pd.DataFrame,
    cfg: OrbConfig,
    *,
    gtl_filter: bool,
) -> Dict[str, Any]:
    daily = _daily_bars(df_1m, cfg)
    bar_step = cfg.bar_step_ms()
    trades: List[Dict[str, Any]] = []
    filtered = 0

    for anchor in _session_anchors(df_1m, cfg):
        close_ms = session_close_ms(anchor, tz=cfg.session_tz, session_close_time=cfg.session_close_time)
        if close_ms is None:
            continue
        bias = _gtl_bias(gtl, df_30m, anchor) if gtl_filter else "any"
        session_traded = False
        sess_bars = df_5m[(df_5m["open_time"] >= anchor) & (df_5m["open_time"] < close_ms)]
        for _, bar in sess_bars.iterrows():
            ms = int(bar["open_time"])
            if not is_trading_session(
                ms,
                tz=cfg.session_tz,
                session_open_time=cfg.session_open_time,
                session_close_time=cfg.session_close_time,
                market=cfg.market,
            ):
                continue
            atr = daily_atr_asof(daily, ms, period=cfg.atr_period, tz=cfg.session_tz)
            sig = classify_signal(
                "SYM",
                df_1m,
                asof_open_ms=ms,
                cfg=cfg,
                session_traded=session_traded,
                daily_atr=atr,
                daily_df=daily,
            )
            if sig.side not in ("LONG", "SHORT"):
                continue
            if gtl_filter:
                if sig.side == "LONG" and bias not in ("long", "any"):
                    filtered += 1
                    continue
                if sig.side == "SHORT" and bias not in ("short", "any"):
                    filtered += 1
                    continue
            if sig.sl_price is None:
                continue
            outcome, exit_px, note, bars_seen, exit_bo = resolve_forward(
                df_1m,
                entry=float(sig.price),
                entry_bar_open_ms=ms,
                side=sig.side,
                sl=float(sig.sl_price),
                tp=sig.tp_price,
                hist_end_ms=int(df_1m["open_time"].max()),
                bar_step_ms=bar_step,
                cfg=cfg,
            )
            if outcome is None:
                continue
            session_traded = True
            r_pnl = pnl_r(sig.side, float(sig.price), float(exit_px), float(sig.sl_price))
            trades.append(
                {
                    "session": sig.session_date,
                    "side": sig.side,
                    "entry": float(sig.price),
                    "exit": float(exit_px),
                    "r_pnl": r_pnl,
                    "outcome": outcome,
                    "note": note,
                    "bias": bias,
                }
            )
            break  # one trade per session

    r_vals = [t["r_pnl"] for t in trades]
    wins = sum(1 for x in r_vals if x > 0)
    return {
        "trades": len(trades),
        "filtered_by_gtl": filtered,
        "win_rate": round(wins / len(r_vals), 3) if r_vals else 0.0,
        "sum_r": round(float(sum(r_vals)), 3),
        "avg_r": round(float(np.mean(r_vals)), 3) if r_vals else 0.0,
    }


def _exit_trade(
    px: np.ndarray,
    gtl: pd.DataFrame,
    entry_i: int,
    direction: int,
    max_hold: int,
    aligned_idx: List[int],
) -> Tuple[int, float]:
    n = len(px)
    for j in range(entry_i + 1, min(entry_i + max_hold + 1, n)):
        if int(gtl.loc[j, "break_dir"]) != 0 and int(gtl.loc[j, "break_dir"]) != direction:
            if bool(gtl.loc[j, "break_aligns_birth"]):
                return j, float(px[j])
    exit_i = min(entry_i + max_hold, n - 1)
    return exit_i, float(px[exit_i])


def backtest_gtl_entries(
    df: pd.DataFrame,
    gtl: pd.DataFrame,
    *,
    mode: str,
    retest_bars: int = 20,
    retest_tol: float = 0.003,
    max_hold: int = 40,
) -> Dict[str, Any]:
    px = df["close"].astype(float).values
    lo = df["low"].astype(float).values
    hi = df["high"].astype(float).values
    aligned_idx = list(gtl.index[gtl["break_aligns_birth"]])
    trades: List[float] = []

    for i in aligned_idx:
        d = int(gtl.loc[i, "break_dir"])
        box_h = float(gtl.loc[i, "broken_hh"] or gtl.loc[i, "frozen_hh"])
        box_l = float(gtl.loc[i, "broken_ll"] or gtl.loc[i, "frozen_ll"])
        entry_i = -1
        if mode == "immediate":
            entry_i = i
        elif mode == "retest":
            for j in range(i + 1, min(i + retest_bars + 1, len(px))):
                if d > 0:
                    touched = float(lo[j]) <= box_h * (1.0 + retest_tol)
                    hold = float(px[j]) >= box_h * (1.0 - retest_tol)
                    if touched and hold:
                        entry_i = j
                        break
                else:
                    touched = float(hi[j]) >= box_l * (1.0 - retest_tol)
                    hold = float(px[j]) <= box_l * (1.0 + retest_tol)
                    if touched and hold:
                        entry_i = j
                        break
        if entry_i < 0:
            continue
        ep = float(px[entry_i])
        stop = box_l if d > 0 else box_h
        # structure stop hit intrabar before normal exit
        exit_i, xp = _exit_trade(px, gtl, entry_i, d, max_hold, aligned_idx)
        for j in range(entry_i + 1, exit_i + 1):
            if d > 0 and float(lo[j]) <= stop:
                xp = stop
                exit_i = j
                break
            if d < 0 and float(hi[j]) >= stop:
                xp = stop
                exit_i = j
                break
        pnl = (xp - ep) if d > 0 else (ep - xp)
        trades.append(float(pnl))

    wins = sum(1 for p in trades if p > 0)
    return {
        "mode": mode,
        "trades": len(trades),
        "win_rate": round(wins / len(trades), 3) if trades else 0.0,
        "sum_pnl": round(float(sum(trades)), 2) if trades else 0.0,
        "avg_pnl": round(float(np.mean(trades)), 3) if trades else 0.0,
    }


def analyze_symbol(sym: str, lo: str, hi: str, cfg: OrbConfig) -> Dict[str, Any]:
    fetch_lo = (pd.Timestamp(lo) - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    df_1m = _load_symbol_df(sym, fetch_lo, hi, cfg)
    if df_1m.empty:
        df_1m = _load(sym, lo, hi)
    if df_1m.empty:
        return {"symbol": sym.replace("USDT", ""), "error": "no_data"}

    # clip to study window (keep warmup before lo for GTL vol)
    lo_ms = int(pd.Timestamp(lo, tz=cfg.session_tz).value // 1_000_000)
    hi_ms = int(
        (pd.Timestamp(hi, tz=cfg.session_tz) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).value // 1_000_000
    )
    df_1m = df_1m[(df_1m["open_time"] >= lo_ms - 30 * 86400 * 1000) & (df_1m["open_time"] <= hi_ms)].copy()
    df_5m = resample_ohlcv(df_1m, "5m")
    df_30m = resample_ohlcv(df_1m, "30m")
    gtl = compute_gtl_dataframe(df_30m, lookback=23, vol_window=500)

    orb_cfg = OrbConfig.from_env()
    orb_cfg.macro_filter = False  # research: don't skip macro days

    orb_pure = backtest_orb(df_1m, df_5m, gtl, df_30m, orb_cfg, gtl_filter=False)
    orb_gtl = backtest_orb(df_1m, df_5m, gtl, df_30m, orb_cfg, gtl_filter=True)
    gtl_imm = backtest_gtl_entries(df_30m, gtl, mode="immediate")
    gtl_ret = backtest_gtl_entries(df_30m, gtl, mode="retest")

    p0 = float(df_30m[df_30m["open_time"] >= lo_ms]["close"].iloc[0]) if (df_30m["open_time"] >= lo_ms).any() else 0
    p1 = float(df_30m["close"].iloc[-1])
    bh = round(p1 - p0, 2) if p0 > 0 else 0.0

    breaks = gtl[gtl["break_dir"] != 0]
    birth_mask = breaks["birth_hit"] >= 0
    birth_hit = float(breaks.loc[birth_mask, "birth_hit"].mean()) if birth_mask.any() else float("nan")

    return {
        "symbol": sym.replace("USDT", ""),
        "birth_hit_rate": round(birth_hit, 3) if birth_hit == birth_hit else None,
        "buy_hold_30m": bh,
        "orb_pure": orb_pure,
        "orb_gtl_filter": orb_gtl,
        "gtl_immediate": gtl_imm,
        "gtl_retest": gtl_ret,
        "orb_gtl_delta_r": round(float(orb_gtl["sum_r"]) - float(orb_pure["sum_r"]), 3),
        "retest_vs_immediate": round(float(gtl_ret["sum_pnl"]) - float(gtl_imm["sum_pnl"]), 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="GTL downstream strategy research")
    ap.add_argument("--from-date", default="2026-02-01")
    ap.add_argument("--to-date", default="2026-06-30")
    ap.add_argument("--symbols-file", default=str(resolve_symbols_path()))
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    syms = [norm_symbol(s) for s in parse_symbol_list(Path(args.symbols_file).read_text(encoding="utf-8"))]
    if not syms:
        syms = [norm_symbol(s) for s in POOL7]

    cfg = OrbConfig.from_env()
    lo, hi = args.from_date, args.to_date
    print(f"[downstream] pool7 {lo}..{hi} ORB(5m RTH) + GTL(30m)", flush=True)
    t0 = time.time()
    rows = []
    for sym in syms:
        print(f"  {sym.replace('USDT',''):5s} ...", end=" ", flush=True)
        row = analyze_symbol(sym, lo, hi, cfg)
        rows.append(row)
        if row.get("error"):
            print(row["error"], flush=True)
        else:
            print(
                f"ORB R {row['orb_pure']['sum_r']:+.2f}->{row['orb_gtl_filter']['sum_r']:+.2f} | "
                f"GTL $ {row['gtl_immediate']['sum_pnl']:+.2f}->{row['gtl_retest']['sum_pnl']:+.2f}",
                flush=True,
            )

    print("\n=== A. ORB pure vs ORB+GTL filter (PnL in R units) ===")
    print(f"{'sym':6s} {'pure_R':>8s} {'+GTL_R':>8s} {'delta':>8s} {'pure_n':>6s} {'filt_n':>6s} {'filt':>5s}")
    for r in rows:
        if r.get("error"):
            continue
        op, og = r["orb_pure"], r["orb_gtl_filter"]
        print(
            f"{r['symbol']:6s} {op['sum_r']:+8.2f} {og['sum_r']:+8.2f} {r['orb_gtl_delta_r']:+8.2f} "
            f"{op['trades']:6d} {og['trades']:6d} {og['filtered_by_gtl']:5d}"
        )

    print("\n=== B. GTL aligned break: immediate vs retest (price PnL, 30m) ===")
    print(f"{'sym':6s} {'imm$':>8s} {'ret$':>8s} {'delta':>8s} {'imm_n':>6s} {'ret_n':>6s} {'bh30m':>8s}")
    for r in rows:
        if r.get("error"):
            continue
        gi, gr = r["gtl_immediate"], r["gtl_retest"]
        print(
            f"{r['symbol']:6s} {gi['sum_pnl']:+8.2f} {gr['sum_pnl']:+8.2f} {r['retest_vs_immediate']:+8.2f} "
            f"{gi['trades']:6d} {gr['trades']:6d} {r['buy_hold_30m']:+8.2f}"
        )

    # pool aggregates
    def _agg(key_path: str) -> float:
        vals = []
        for r in rows:
            if r.get("error"):
                continue
            obj = r
            for part in key_path.split("."):
                obj = obj[part]
            vals.append(float(obj))
        return round(sum(vals), 2)

    print("\n=== POOL TOTALS ===")
    print(f"  ORB pure sum_R      = {_agg('orb_pure.sum_r'):+.2f}")
    print(f"  ORB + GTL sum_R     = {_agg('orb_gtl_filter.sum_r'):+.2f}")
    print(f"  GTL immediate sum$  = {_agg('gtl_immediate.sum_pnl'):+.2f}")
    print(f"  GTL retest sum$     = {_agg('gtl_retest.sum_pnl'):+.2f}")

    out_dir = ROOT / "output" / "orb" / "cta"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.json_out) if args.json_out else out_dir / f"gtl_downstream_pool7_{lo}_{hi}.json"
    out_path.write_text(
        json.dumps({"date_range": {"from": lo, "to": hi}, "results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\njson -> {out_path} ({time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
