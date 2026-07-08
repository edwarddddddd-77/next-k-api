#!/usr/bin/env python3
"""GTL flip strategy: 1 long + 1 short max/day/symbol, structure SL, opposite-break flip."""

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
from orb.core.us_equity_calendar import is_us_equity_trading_day, us_equity_session_close_time
from orb.core.session import session_day_str
from orb.core.fees import trade_fee_usdt
from orb.core.indicators import daily_atr_asof
from orb.core.signals import compute_position_notional, compute_sl_tp
from orb.gtl.engine import compute_gtl_dataframe
from orb.gtl.resample import resample_ohlcv
from orb.core.symbols_path import resolve_symbols_path
from tools.cta._snap_gtl_pool7_day import POOL7, _load_range, _session_mask

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


def trading_days_in_range(from_date: str, to_date: str) -> list[str]:
    d0 = pd.Timestamp(from_date.strip())
    d1 = pd.Timestamp(to_date.strip())
    out: list[str] = []
    d = d0
    while d <= d1:
        day = d.strftime("%Y-%m-%d")
        if is_us_equity_trading_day(day):
            out.append(day)
        d += pd.Timedelta(days=1)
    return out


def resolve_days(args, cfg: OrbConfig) -> list[str]:
    if args.from_date.strip() and args.to_date.strip():
        return trading_days_in_range(args.from_date.strip(), args.to_date.strip())
    return last_trading_days(int(args.days), args.asof, cfg)


def _open_dir(row: pd.Series, *, open_mode: str = "break_forecast") -> int:
    if bool(row.get("break_aligns_birth")):
        d = int(row.get("break_dir") or 0)
        if d > 0:
            return 1
        if d < 0:
            return -1
    if open_mode == "break":
        return 0
    if bool(row.get("forecast_up")) and not bool(row.get("forecast_down")):
        return 1
    if bool(row.get("forecast_down")) and not bool(row.get("forecast_up")):
        return -1
    disp = float(row.get("display_prob_up", 0.5))
    if disp >= 0.55:
        return 1
    if disp <= 0.45:
        return -1
    return 0


def _stop_for(row: pd.Series, side: int) -> float:
    if side > 0:
        return float(row["broken_ll"] or row["frozen_ll"])
    return float(row["broken_hh"] or row["frozen_hh"])


def _hhmm_to_mins(hhmm: str) -> int:
    h, m = map(int, hhmm.strip().split(":"))
    return h * 60 + m


def _session_date(day_df: pd.DataFrame, cfg: OrbConfig) -> str:
    ms = int(day_df.iloc[0]["open_time"])
    return session_day_str(ms, tz=cfg.session_tz, session_open_time=cfg.session_open_time)


def _eod_bar_index(
    day_df: pd.DataFrame,
    cfg: OrbConfig,
    *,
    eod_early_min: int,
    ts_fn,
) -> int:
    """Last bar with open time <= session_close - eod_early_min."""
    n = len(day_df)
    if eod_early_min <= 0:
        return n - 1
    day = _session_date(day_df, cfg)
    close_s = us_equity_session_close_time(day, cfg.session_close_time or "16:00")
    eod_m = _hhmm_to_mins(close_s) - eod_early_min
    last = 0
    for i in range(n):
        if _hhmm_to_mins(ts_fn(i)) <= eod_m:
            last = i
        else:
            break
    return last


def _daily_bars(raw: pd.DataFrame, cfg: OrbConfig) -> pd.DataFrame:
    if raw.empty:
        return raw
    tz = cfg.session_tz
    tmp = raw.copy()
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


def _orb_stop(side: int, entry: float, ms: int, orb_cfg: OrbConfig, daily_df: pd.DataFrame) -> float | None:
    atr = daily_atr_asof(daily_df, ms, period=orb_cfg.atr_period, tz=orb_cfg.session_tz)
    side_u = "LONG" if side > 0 else "SHORT"
    sl, _, _ = compute_sl_tp(
        side=side_u,
        entry=entry,
        or_high=entry + 1.0,
        or_low=entry - 1.0,
        cfg=orb_cfg,
        daily_atr=atr,
    )
    return float(sl) if sl is not None else None


def _first_aligned_break(day_gtl: pd.DataFrame) -> tuple[int, int] | None:
    """Return (bar_index, side) for first aligned break, else None."""
    for i in range(len(day_gtl)):
        r = day_gtl.iloc[i]
        if not bool(r.get("break_aligns_birth")):
            continue
        d = int(r.get("break_dir") or 0)
        if d > 0:
            return i, 1
        if d < 0:
            return i, -1
    return None


def simulate_day(
    day_df: pd.DataFrame,
    day_gtl: pd.DataFrame,
    cfg: OrbConfig,
    *,
    entry: str = "open",
    open_mode: str = "break_forecast",
    sl_flat: bool = False,
    allow_flip: bool = True,
    equity_start: float = 0.0,
    eod_early_min: int = 0,
    entry_cutoff: str = "",
    orb_risk: bool = False,
    stop_mode: str = "structure",
    daily_df: pd.DataFrame | None = None,
) -> dict:
    if day_gtl.empty:
        return {"skipped": True, "reason": "no_bars"}

    px = day_df["close"].astype(float).values
    hi = day_df["high"].astype(float).values
    lo = day_df["low"].astype(float).values
    n = len(px)
    use_usdt = equity_start > 0
    equity = float(equity_start)

    start_i = 0
    if entry == "first_break":
        fb = _first_aligned_break(day_gtl)
        if fb is None:
            return {"skipped": True, "reason": "no_first_break"}
        start_i, od = fb
    else:
        od = _open_dir(day_gtl.iloc[0], open_mode=open_mode)
        if od == 0:
            return {"skipped": True, "reason": "no_open_dir"}

    legs: list[dict] = []
    pos = 0
    entry_i = 0
    entry_px = 0.0
    stop = 0.0
    pos_notional = 0.0
    long_used = short_used = False
    done = False
    cutoff_m = _hhmm_to_mins(entry_cutoff) if entry_cutoff.strip() else -1

    def _ts(i: int) -> str:
        return (
            pd.Timestamp(int(day_df.iloc[i]["open_time"]), unit="ms", tz="UTC")
            .tz_convert(cfg.session_tz)
            .strftime("%H:%M")
        )

    def _can_open(i: int) -> bool:
        if cutoff_m < 0:
            return True
        return _hhmm_to_mins(_ts(i)) < cutoff_m

    eod_i = _eod_bar_index(day_df, cfg, eod_early_min=eod_early_min, ts_fn=_ts)
    eod_tag = "eod_early" if eod_early_min > 0 else "eod"

    def _close(i: int, xp: float, reason: str) -> None:
        nonlocal pos, entry_px, entry_i, equity, pos_notional
        side = "long" if pos > 0 else "short"
        pnl = (xp - entry_px) if pos > 0 else (entry_px - xp)
        pct = pnl / entry_px * 100 if entry_px else 0.0
        eq_before = equity
        if use_usdt and orb_risk and pos_notional > 0:
            gross = (xp - entry_px) / entry_px * pos_notional if pos > 0 else (entry_px - xp) / entry_px * pos_notional
            fee = trade_fee_usdt(
                pos_notional,
                entry_mode="signal",
                maker_bps=cfg.fee_maker_bps,
                taker_bps=cfg.fee_taker_bps,
            )
            pnl_usdt = round(float(gross) - float(fee), 2)
        elif use_usdt:
            pnl_usdt = equity * pct / 100.0
        else:
            pnl_usdt = 0.0
        if use_usdt:
            equity += pnl_usdt
        leg = {
            "side": side,
            "entry_time": _ts(entry_i),
            "exit_time": _ts(i),
            "entry": round(entry_px, 2),
            "exit": round(xp, 2),
            "stop": round(stop, 2),
            "pnl": round(pnl, 2),
            "pct": round(pct, 2),
            "reason": reason,
        }
        if use_usdt:
            leg["equity_before"] = round(eq_before, 2)
            leg["notional_usdt"] = round(pos_notional, 2)
            leg["pnl_usdt"] = round(pnl_usdt, 2)
            leg["equity_after"] = round(equity, 2)
        legs.append(leg)
        pos = 0
        pos_notional = 0.0

    def _open(i: int, side: int) -> bool:
        nonlocal pos, entry_i, entry_px, stop, long_used, short_used, pos_notional
        if not _can_open(i):
            return False
        r = day_gtl.iloc[i]
        pos = side
        entry_i = i
        entry_px = float(px[i])
        if stop_mode == "atr" and daily_df is not None and not daily_df.empty:
            ms = int(day_df.iloc[i]["open_time"])
            orb_sl = _orb_stop(side, entry_px, ms, cfg, daily_df)
            stop = orb_sl if orb_sl is not None else _stop_for(r, side)
        else:
            stop = _stop_for(r, side)
        if use_usdt and orb_risk:
            pos_notional = compute_position_notional(
                entry=entry_px, sl=stop, cfg=cfg, bot_equity_usdt=equity,
            )
            if pos_notional <= 0:
                pos = 0
                return False
        if side > 0:
            long_used = True
        else:
            short_used = True
        return True

    if not _can_open(start_i):
        return {"skipped": True, "reason": "entry_after_cutoff"}

    if od > 0:
        _open(start_i, 1)
    else:
        _open(start_i, -1)

    for i in range(start_i + 1, eod_i + 1):
        if done:
            break

        if pos == 0:
            if sl_flat:
                continue
            r = day_gtl.iloc[i]
            if not bool(r.get("break_aligns_birth")):
                continue
            d = int(r["break_dir"])
            if d > 0 and not long_used and _can_open(i):
                _open(i, 1)
            elif d < 0 and not short_used and _can_open(i):
                _open(i, -1)
            continue

        if pos > 0:
            if lo[i] <= stop:
                _close(i, stop, "sl")
                if sl_flat:
                    done = True
                continue
            r = day_gtl.iloc[i]
            if bool(r.get("break_aligns_birth")) and int(r["break_dir"]) < 0:
                _close(i, float(px[i]), "flip_down" if allow_flip else "break_exit")
                if allow_flip and not short_used and _can_open(i):
                    _open(i, -1)
                elif sl_flat:
                    done = True
        else:
            if hi[i] >= stop:
                _close(i, stop, "sl")
                if sl_flat:
                    done = True
                continue
            r = day_gtl.iloc[i]
            if bool(r.get("break_aligns_birth")) and int(r["break_dir"]) > 0:
                _close(i, float(px[i]), "flip_up" if allow_flip else "break_exit")
                if allow_flip and not long_used and _can_open(i):
                    _open(i, 1)
                elif sl_flat:
                    done = True

    if pos != 0:
        _close(eod_i, float(px[eod_i]), eod_tag)

    total = sum(x["pnl"] for x in legs)
    total_usdt = round(equity - equity_start, 2) if use_usdt else 0.0
    day_ret = round((equity / equity_start - 1) * 100, 2) if use_usdt and equity_start else 0.0
    return {
        "skipped": False,
        "entry": entry,
        "entry_time": legs[0]["entry_time"] if legs else "",
        "open_dir": "up" if od > 0 else "down",
        "legs": legs,
        "n_legs": len(legs),
        "total_pnl": round(total, 2),
        "total_pnl_usdt": total_usdt,
        "equity_start": round(equity_start, 2) if use_usdt else 0.0,
        "equity_end": round(equity, 2) if use_usdt else 0.0,
        "day_return_pct": day_ret,
        "long_pnl": round(sum(x["pnl"] for x in legs if x["side"] == "long"), 2),
        "short_pnl": round(sum(x["pnl"] for x in legs if x["side"] == "short"), 2),
    }


def _load_symbol_cache(
    sym: str, fetch_lo: str, fetch_hi: str, cfg: OrbConfig, resample: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    raw = _load_range(sym, fetch_lo, fetch_hi, cfg)
    if raw.empty:
        return None
    df = resample_ohlcv(raw, resample)
    gtl = compute_gtl_dataframe(df, lookback=23, vol_window=500)
    daily = _daily_bars(raw, cfg)
    return raw, df, gtl, daily


def _run_symbol(
    days: list[str],
    sym: str,
    df: pd.DataFrame,
    gtl: pd.DataFrame,
    cfg: OrbConfig,
    *,
    entry: str,
    open_mode: str,
    sl_flat: bool,
    allow_flip: bool,
    capital: float,
    eod_early_min: int = 0,
    entry_cutoff: str = "",
    orb_risk: bool = False,
    stop_mode: str = "structure",
    daily_df: pd.DataFrame | None = None,
) -> tuple[list[dict], list[dict], float]:
    label = sym.replace("USDT", "")
    equity = float(capital)
    summary: list[dict] = []
    leg_rows: list[dict] = []
    for day in days:
        m = _session_mask(df, day, cfg, rth_only=True)
        day_df = df[m].reset_index(drop=True)
        day_gtl = gtl[m].reset_index(drop=True)
        eq_in = equity if capital > 0 else 0.0
        r = simulate_day(
            day_df,
            day_gtl,
            cfg,
            entry=entry,
            open_mode=open_mode,
            sl_flat=sl_flat,
            allow_flip=allow_flip,
            equity_start=eq_in,
            eod_early_min=eod_early_min,
            entry_cutoff=entry_cutoff,
            orb_risk=orb_risk,
            stop_mode=stop_mode,
            daily_df=daily_df,
        )
        if r.get("skipped"):
            summary.append({
                "day": day, "symbol": label, "skipped": True, "reason": r.get("reason"),
                "total_pnl": 0, "total_pnl_usdt": 0.0,
                "equity_start": round(eq_in, 2), "equity_end": round(equity, 2),
                "day_return_pct": 0.0,
            })
            continue
        if capital > 0:
            equity = float(r["equity_end"])
        summary.append({
            "day": day, "symbol": label, "skipped": False, "reason": "",
            "entry": r.get("entry", entry), "entry_time": r.get("entry_time", ""),
            "open_dir": r["open_dir"], "n_legs": r["n_legs"],
            "long_pnl": r["long_pnl"], "short_pnl": r["short_pnl"],
            "total_pnl": r["total_pnl"], "total_pnl_usdt": r["total_pnl_usdt"],
            "equity_start": r["equity_start"], "equity_end": r["equity_end"],
            "day_return_pct": r["day_return_pct"],
        })
        for lg in r["legs"]:
            leg_rows.append({"day": day, "symbol": label, "open_dir": r["open_dir"], **lg})
    return summary, leg_rows, equity


def _pnl_field(capital: float) -> str:
    return "total_pnl_usdt" if capital > 0 else "total_pnl"


def _print_matrix(
    days: list[str],
    syms: list[str],
    summary_rows: list[dict],
    *,
    exclude: set[str],
    capital: float,
    unit: str,
) -> float:
    labels = [s.replace("USDT", "") for s in syms if s.replace("USDT", "") not in exclude]
    pf = _pnl_field(capital)
    hdr = f"{'sym':6s}" + "".join(f"{d[5:]:>8s}" for d in days) + f"{' SUM':>9s}"
    print(hdr)
    print("-" * len(hdr))
    pool_total = 0.0
    for sym in labels:
        row_sum = 0.0
        cells = []
        for day in days:
            sub = [x for x in summary_rows if x["symbol"] == sym and x["day"] == day and not x.get("skipped")]
            v = float(sub[0][pf]) if sub else 0.0
            row_sum += v
            cells.append(f"{v:+7.2f}")
        print(f"{sym:6s}" + "".join(f"{c:>8s}" for c in cells) + f"{row_sum:+8.2f}{unit}")
        pool_total += row_sum
    pool_by_day = {
        d: sum(float(x[pf]) for x in summary_rows if x["day"] == d and not x.get("skipped"))
        for d in days
    }
    print(f"{'POOL':6s}" + "".join(f"{pool_by_day[d]:+7.2f}" for d in days) + f"{pool_total:+8.2f}{unit}")
    return pool_total


def _print_equity_summary(
    syms: list[str],
    finals: dict[str, float],
    *,
    exclude: set[str],
    capital: float,
) -> None:
    labels = [s.replace("USDT", "") for s in syms if s.replace("USDT", "") not in exclude]
    print(f"\n=== Equity ({capital:.0f}U/sym compound) ===")
    start = capital * len(labels)
    end = 0.0
    for sym in labels:
        fe = finals.get(sym, capital)
        end += fe
        ret = (fe / capital - 1) * 100 if capital else 0
        print(f"  {sym:5s} {capital:7.0f} -> {fe:8.2f}U  ({ret:+.1f}%)")
    pool_ret = (end / start - 1) * 100 if start else 0
    print(f"  POOL  {start:7.0f} -> {end:8.2f}U  ({pool_ret:+.1f}%)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--asof", default="2026-07-05")
    ap.add_argument("--from-date", default="", help="range start YYYY-MM-DD (with --to-date)")
    ap.add_argument("--to-date", default="", help="range end YYYY-MM-DD")
    ap.add_argument("--resample", default="5m")
    ap.add_argument("--csv-out", default="")
    ap.add_argument("--entry", choices=["open", "first_break"], default="open",
                    help="open=09:30 entry; first_break=wait for first aligned break")
    ap.add_argument("--open-mode", choices=["break", "break_forecast"], default="break_forecast")
    ap.add_argument("--sl-flat", action="store_true", help="after structure SL, flat for rest of day")
    ap.add_argument("--no-flip", action="store_true", help="exit on opposite break, no reverse entry")
    ap.add_argument("--exclude", default="", help="comma symbols to skip, e.g. SNDK")
    ap.add_argument("--capital", type=float, default=0.0, help="initial USDT per symbol; compound daily")
    ap.add_argument("--eod-early-min", type=int, default=0,
                    help="close all positions this many minutes before session close (60=1h early)")
    ap.add_argument("--entry-cutoff", default="",
                    help="no new entries at/after this ET time, e.g. 12:00")
    ap.add_argument("--compare", action="store_true", help="run original vs recommended vs open-hold")
    ap.add_argument("--compare-entry", action="store_true",
                    help="open@09:30 vs first_break; uses --capital, sl-flat+flip")
    ap.add_argument("--compare-entry-orb", action="store_true",
                    help="GTL pct-compound vs ORB risk+ATR SL+fees for both entry modes")
    ap.add_argument("--orb-risk", action="store_true",
                    help="ORB risk_pct sizing + fees (needs --capital)")
    ap.add_argument("--stop-mode", choices=["structure", "atr"], default="structure",
                    help="SL: structure=GTL box; atr=ORB ATR (with --orb-risk)")
    ap.add_argument("--compare-open-sizing", action="store_true",
                    help="open@09:30: GTL_pct vs ORB risk+structure SL vs ORB risk+ATR SL")
    ap.add_argument("--compare-session", action="store_true",
                    help="full EOD vs 1h early EOD + 12:00 entry cutoff; open_0930, 1000U")
    args = ap.parse_args()

    cfg = OrbConfig.from_env()
    days = resolve_days(args, cfg)
    if not days:
        print("no trading days in range")
        return 1
    syms = [norm_symbol(s) for s in parse_symbol_list(Path(resolve_symbols_path()).read_text(encoding="utf-8"))]
    if not syms:
        syms = [norm_symbol(s) for s in POOL7]
    exclude = {s.strip().upper() for s in args.exclude.split(",") if s.strip()}

    fetch_lo = (pd.Timestamp(days[0]) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    fetch_hi = (pd.Timestamp(days[-1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    t0 = time.time()
    unit = "U" if args.capital > 0 else ""
    cap = float(args.capital)

    if args.compare_session:
        cap = cap if cap > 0 else 1000.0
        base = dict(
            entry="open", open_mode="break_forecast",
            sl_flat=True, allow_flip=True,
            eod_early_min=0, entry_cutoff="",
        )
        early = dict(
            entry="open", open_mode="break_forecast",
            sl_flat=True, allow_flip=True,
            eod_early_min=60, entry_cutoff="12:00",
        )
        variants = [
            ("full_EOD", base),
            ("EOD-1h_12cut", early),
        ]
        print(f"=== Session compare | {days[0]}..{days[-1]} | 5m RTH | open@09:30 | B-rule ===")
        print(f"capital={cap:.0f}U/symbol compound\n")
        if exclude:
            print(f"exclude: {', '.join(sorted(exclude))}\n")
        cache = {}
        for sym in syms:
            label = sym.replace("USDT", "")
            if label in exclude:
                continue
            loaded = _load_symbol_cache(sym, fetch_lo, fetch_hi, cfg, args.resample)
            if loaded:
                cache[sym] = loaded
        for name, vopts in variants:
            desc = f"eod_early={vopts['eod_early_min']}min entry_cutoff={vopts['entry_cutoff'] or 'none'}"
            print(f"\n--- {name} ({desc}) ---")
            summary_rows: list[dict] = []
            finals: dict[str, float] = {}
            for sym, (_raw, df, gtl, daily) in cache.items():
                label = sym.replace("USDT", "")
                rows, _, fe = _run_symbol(days, sym, df, gtl, cfg, capital=cap, **vopts)
                summary_rows.extend(rows)
                finals[label] = fe
            _print_matrix(days, syms, summary_rows, exclude=exclude, capital=cap, unit=unit or "U")
            _print_equity_summary(syms, finals, exclude=exclude, capital=cap)
            traded = [r for r in summary_rows if not r.get("skipped")]
            wins = sum(1 for r in traded if float(r[_pnl_field(cap)]) > 0)
            print(f"  day win rate: {wins}/{len(traded)} = {100*wins/len(traded):.0f}%" if traded else "")
        print(f"\n({time.time()-t0:.0f}s)")
        return 0

    if args.compare_open_sizing:
        cap = cap if cap > 0 else 1000.0
        variants = [
            ("GTL_pct", dict(entry="open", open_mode="break_forecast", orb_risk=False, stop_mode="structure")),
            ("ORB_risk+GTL_SL", dict(entry="open", open_mode="break_forecast", orb_risk=True, stop_mode="structure")),
            ("ORB_risk+ATR_SL", dict(entry="open", open_mode="break_forecast", orb_risk=True, stop_mode="atr")),
        ]
        opts = dict(sl_flat=True, allow_flip=True)
        print(f"=== open@09:30 sizing compare | {days[0]}..{days[-1]} | {len(days)} days ===")
        print("GTL_pct: structure SL + equity pct compound")
        print("ORB_risk+GTL_SL: GTL structure SL + ORB risk sizing + fees")
        print("ORB_risk+ATR_SL: ORB ATR SL + ORB risk sizing + fees")
        print(f"capital={cap:.0f}U/symbol compound\n")
        if exclude:
            print(f"exclude: {', '.join(sorted(exclude))}\n")
        cache = {}
        for sym in syms:
            if sym.replace("USDT", "") in exclude:
                continue
            loaded = _load_symbol_cache(sym, fetch_lo, fetch_hi, cfg, args.resample)
            if loaded:
                cache[sym] = loaded
        totals: dict[str, float] = {}
        for name, vopts in variants:
            print(f"\n--- {name} ---")
            summary_rows: list[dict] = []
            finals: dict[str, float] = {}
            for sym, (_raw, df, gtl, daily) in cache.items():
                label = sym.replace("USDT", "")
                rows, _, fe = _run_symbol(
                    days, sym, df, gtl, cfg, capital=cap,
                    daily_df=daily, **opts, **vopts,
                )
                summary_rows.extend(rows)
                finals[label] = fe
            totals[name] = _print_matrix(
                days, syms, summary_rows, exclude=exclude, capital=cap, unit=unit or "U",
            )
            _print_equity_summary(syms, finals, exclude=exclude, capital=cap)
            traded = [r for r in summary_rows if not r.get("skipped")]
            wins = sum(1 for r in traded if float(r[_pnl_field(cap)]) > 0)
            print(f"  day win rate: {wins}/{len(traded)} = {100*wins/len(traded):.0f}%" if traded else "")
        print("\n=== POOL total ===")
        for name, tot in totals.items():
            print(f"  {name:20s} {tot:+8.2f}U")
        print(f"\n({time.time()-t0:.0f}s)")
        return 0

    if args.compare_entry_orb:
        cap = cap if cap > 0 else 1000.0
        entry_variants = [
            ("open_0930", dict(entry="open", open_mode="break_forecast")),
            ("first_break", dict(entry="first_break", open_mode="break")),
        ]
        sizing_variants = [
            ("GTL_pct", dict(orb_risk=False, stop_mode="structure")),
            ("ORB_risk", dict(orb_risk=True, stop_mode="atr")),
        ]
        opts = dict(sl_flat=True, allow_flip=True)
        print(f"=== Entry x sizing | {days[0]}..{days[-1]} | {len(days)} days | 5m RTH ===")
        print(f"GTL_pct: structure SL + full equity pct compound")
        print(f"ORB_risk+ATR: ATR SL risk={cfg.risk_pct*100:.1f}% + fees")
        print(f"capital={cap:.0f}U/symbol compound\n")
        if exclude:
            print(f"exclude: {', '.join(sorted(exclude))}\n")
        cache = {}
        for sym in syms:
            if sym.replace("USDT", "") in exclude:
                continue
            loaded = _load_symbol_cache(sym, fetch_lo, fetch_hi, cfg, args.resample)
            if loaded:
                cache[sym] = loaded
        totals: dict[str, float] = {}
        for ename, ev in entry_variants:
            for sname, sv in sizing_variants:
                tag = f"{ename}|{sname}"
                print(f"\n--- {tag} ---")
                summary_rows: list[dict] = []
                finals: dict[str, float] = {}
                for sym, (_raw, df, gtl, daily) in cache.items():
                    label = sym.replace("USDT", "")
                    rows, _, fe = _run_symbol(
                        days, sym, df, gtl, cfg, capital=cap,
                        daily_df=daily, **opts, **ev, **sv,
                    )
                    summary_rows.extend(rows)
                    finals[label] = fe
                totals[tag] = _print_matrix(
                    days, syms, summary_rows, exclude=exclude, capital=cap, unit=unit or "U",
                )
                _print_equity_summary(syms, finals, exclude=exclude, capital=cap)
        print("\n=== POOL total ===")
        for tag, tot in totals.items():
            print(f"  {tag:28s} {tot:+8.2f}U")
        print(f"\n({time.time()-t0:.0f}s)")
        return 0

    if args.compare_entry:
        cap = cap if cap > 0 else 1000.0
        variants = [
            ("open_0930", dict(entry="open", open_mode="break_forecast")),
            ("first_break", dict(entry="first_break", open_mode="break")),
        ]
        opts = dict(sl_flat=True, allow_flip=True)
        print(f"=== Entry compare | {days[0]}..{days[-1]} | 5m RTH | B-rule SL-flat+flip ===")
        print(f"capital={cap:.0f}U/symbol compound\n")
        if exclude:
            print(f"exclude: {', '.join(sorted(exclude))}\n")
        cache: dict[str, tuple] = {}
        for sym in syms:
            label = sym.replace("USDT", "")
            if label in exclude:
                continue
            loaded = _load_symbol_cache(sym, fetch_lo, fetch_hi, cfg, args.resample)
            if loaded:
                cache[sym] = loaded
        for name, vopts in variants:
            print(f"\n--- {name} ---")
            print(f"entry={vopts['entry']} open_mode={vopts['open_mode']}")
            summary_rows: list[dict] = []
            finals: dict[str, float] = {}
            for sym, (_raw, df, gtl, _daily) in cache.items():
                label = sym.replace("USDT", "")
                rows, _, fe = _run_symbol(
                    days, sym, df, gtl, cfg, capital=cap, **opts, **vopts,
                )
                summary_rows.extend(rows)
                finals[label] = fe
            _print_matrix(days, syms, summary_rows, exclude=exclude, capital=cap, unit=unit or "U")
            _print_equity_summary(syms, finals, exclude=exclude, capital=cap)
            traded = [r for r in summary_rows if not r.get("skipped")]
            wins = sum(1 for r in traded if float(r[_pnl_field(cap)]) > 0)
            print(f"  day win rate: {wins}/{len(traded)} = {100*wins/len(traded):.0f}%" if traded else "")
        print(f"\n({time.time()-t0:.0f}s)")
        return 0

    if args.compare:
        variants = [
            ("A_original", dict(entry="open", open_mode="break_forecast", sl_flat=False, allow_flip=True)),
            ("B_sl_flat_flip", dict(entry="open", open_mode="break_forecast", sl_flat=True, allow_flip=True)),
            ("C_open_hold", dict(entry="open", open_mode="break_forecast", sl_flat=True, allow_flip=False)),
        ]
        print(f"=== GTL compare | {days[0]}..{days[-1]} | 5m RTH ===")
        if exclude:
            print(f"exclude: {', '.join(sorted(exclude))}\n")
        cache = {}
        for sym in syms:
            label = sym.replace("USDT", "")
            if label in exclude:
                continue
            loaded = _load_symbol_cache(sym, fetch_lo, fetch_hi, cfg, args.resample)
            if loaded:
                cache[sym] = loaded
        totals: dict[str, float] = {}
        for name, vopts in variants:
            print(f"\n--- {name} ---")
            print(f"entry={vopts['entry']} open={vopts['open_mode']} sl_flat={vopts['sl_flat']} flip={vopts['allow_flip']}")
            summary_rows = []
            for sym, (_raw, df, gtl, _daily) in cache.items():
                rows, _, _ = _run_symbol(days, sym, df, gtl, cfg, capital=cap, **vopts)
                summary_rows.extend(rows)
            totals[name] = _print_matrix(days, syms, summary_rows, exclude=exclude, capital=cap, unit=unit)
        print("\n=== 7d POOL total ===")
        for name, tot in totals.items():
            sfx = unit or ""
            print(f"  {name:18s} {tot:+8.2f}{sfx}")
        print(f"\n({time.time()-t0:.0f}s)")
        return 0

    summary_rows: list[dict] = []
    leg_rows: list[dict] = []

    mode_bits = []
    if args.sl_flat:
        mode_bits.append("SL->flat")
    if args.no_flip:
        mode_bits.append("no-flip")
    mode_s = " | ".join(mode_bits) if mode_bits else "full flip+reentry"
    print(f"=== GTL flip sim | {days[0]}..{days[-1]} | 5m RTH ===")
    print(f"entry={args.entry} | open_mode={args.open_mode} | {mode_s}")
    if cap > 0:
        print(f"capital={cap:.0f}U/symbol compound")
    if exclude:
        print(f"exclude: {', '.join(sorted(exclude))}")
    print()

    finals: dict[str, float] = {}
    for sym in syms:
        label = sym.replace("USDT", "")
        if label in exclude:
            continue
        loaded = _load_symbol_cache(sym, fetch_lo, fetch_hi, cfg, args.resample)
        if not loaded:
            continue
        _raw, df, gtl, daily = loaded
        rows, legs, fe = _run_symbol(
            days, sym, df, gtl, cfg,
            entry=args.entry,
            open_mode=args.open_mode,
            sl_flat=args.sl_flat,
            allow_flip=not args.no_flip,
            capital=cap,
            eod_early_min=int(args.eod_early_min),
            entry_cutoff=args.entry_cutoff.strip(),
            orb_risk=bool(args.orb_risk),
            stop_mode=args.stop_mode,
            daily_df=daily,
        )
        sym_total = sum(float(r[_pnl_field(cap)]) for r in rows if not r.get("skipped"))
        finals[label] = fe
        print(f"--- {label} ---")
        for r in rows:
            if r.get("skipped"):
                print(f"  {r['day']} SKIP ({r.get('reason')})")
                summary_rows.append(r)
                continue
            et = f" @{r.get('entry_time','')}" if r.get("entry_time") else ""
            if cap > 0:
                print(
                    f"  {r['day']} {r['open_dir']}{et} "
                    f"{r['equity_start']:.0f}->{r['equity_end']:.0f}U ({r['day_return_pct']:+.1f}%)"
                )
            else:
                print(f"  {r['day']} open={r['open_dir']}{et} total={r['total_pnl']:+.2f}")
            summary_rows.append(r)
        leg_rows.extend(legs)
        sfx = "U" if cap > 0 else ""
        print(f"  >> 7d sum {sym_total:+.2f}{sfx}  end={fe:.2f}{sfx}\n")

    print(f"\n=== PnL matrix ({'USDT' if cap else 'price units'}) ===")
    _print_matrix(days, syms, summary_rows, exclude=exclude, capital=cap, unit=unit or "")
    if cap > 0:
        _print_equity_summary(syms, finals, exclude=exclude, capital=cap)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    tag = "slflat" if args.sl_flat else "full"
    if args.no_flip:
        tag = "hold"
    csv_path = Path(args.csv_out) if args.csv_out else OUTPUT / f"gtl_flip_{tag}_{days[0]}_{days[-1]}.csv"
    summary_fields = [
        "day", "symbol", "skipped", "reason", "entry", "entry_time", "open_dir", "n_legs",
        "long_pnl", "short_pnl", "total_pnl", "total_pnl_usdt",
        "equity_start", "equity_end", "day_return_pct",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary_rows)
    legs_path = csv_path.with_name(csv_path.stem + "_legs.csv")
    if leg_rows:
        with legs_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(leg_rows[0].keys()))
            w.writeheader()
            w.writerows(leg_rows)

    print(f"\nsummary -> {csv_path}")
    print(f"legs    -> {legs_path}")
    print(f"({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
