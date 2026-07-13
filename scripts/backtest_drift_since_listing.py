#!/usr/bin/env python3
"""Anchor drift backtest from perp listing to now."""
from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import pandas as pd

from quant.anchor_drift.backtest import BacktestParams, fetch_bars, run_backtest, simulate_bars
from quant.anchor_drift.config import AnchorDriftConfig
from quant.common.kline_cache import load_klines, norm_symbol
from quant.common.us_equity_calendar import is_us_equity_trading_day
from quant.common.session import session_day_str
from quant.common.config import OrbConfig

SYMBOLS = ["MSTR", "COIN", "HOOD", "CRCL", "SOXL", "SNDK"]
INTERVAL = "5m"
EQUITY = 1000.0
# Binance equity perps ~Feb 2026; fetch from Jan 1 2026 to be safe
LISTING_PROBE_MS = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp() * 1000)

SCHEMES = [
    ("A ALL 2.5% no-adverse", 0.025, dict(weekend_only=False, sat_sun_entry_only=False, disable_adverse_stop=True)),
    ("B ALL 2.5% + adverse", 0.025, dict(weekend_only=False, sat_sun_entry_only=False, disable_adverse_stop=False)),
    ("C ALL 3.5% + adverse", 0.035, dict(weekend_only=False, sat_sun_entry_only=False, disable_adverse_stop=False)),
]


def probe_listing(symbols: list[str]) -> dict[str, dict[str, Any]]:
    end_ms = int(time.time() * 1000)
    out: dict[str, dict[str, Any]] = {}
    for raw in symbols:
        sym = norm_symbol(raw)
        df = load_klines(sym, INTERVAL, start_ms=LISTING_PROBE_MS, end_ms=end_ms)
        if df.empty or len(df) < 10:
            print(f"[fetch] {sym} no cache, downloading full history...")
            df = fetch_bars(sym, days=9999, interval=INTERVAL, exchange_id="binance", refresh=True)
            if df.empty:
                # try with explicit early start via forward fetch
                from quant.market import fetch_klines_forward, klines_to_df
                from quant.common.kline_cache import save_klines

                rows = fetch_klines_forward(sym, INTERVAL, LISTING_PROBE_MS, end_ms, exchange_id="binance")
                df = klines_to_df(rows)
                if not df.empty:
                    save_klines(sym, INTERVAL, df)
        if df.empty:
            out[sym] = {"ok": False, "bars": 0}
            continue
        t0 = pd.Timestamp(int(df["open_time"].min()), unit="ms", tz="America/New_York")
        t1 = pd.Timestamp(int(df["open_time"].max()), unit="ms", tz="America/New_York")
        sess = OrbConfig.from_env()
        days_seen = {
            session_day_str(int(ms), tz=sess.session_tz, session_open_time=sess.session_open_time)
            for ms in df["open_time"]
        }
        trading_days = sorted(d for d in days_seen if is_us_equity_trading_day(d))
        out[sym] = {
            "ok": True,
            "bars": len(df),
            "first": t0,
            "last": t1,
            "cal_days": (t1 - t0).days,
            "trading_days": len(trading_days),
            "first_trading": trading_days[0] if trading_days else "?",
            "last_trading": trading_days[-1] if trading_days else "?",
        }
    return out


def run_since_listing(
    symbols: list[str],
    *,
    th: float,
    flags: dict,
) -> dict[str, Any]:
    end_ms = int(time.time() * 1000)
    cfg = replace(AnchorDriftConfig.from_env(), signal_threshold=th)
    params = BacktestParams(
        symbols=symbols,
        days=9999,
        interval=INTERVAL,
        equity_usdt=EQUITY,
        compound=True,
        cfg=cfg,
        **flags,
    )
    # Per-symbol simulate from its own first bar (true listing-to-now)
    results = []
    for raw in symbols:
        sym = norm_symbol(raw)
        df = load_klines(sym, INTERVAL, start_ms=LISTING_PROBE_MS, end_ms=end_ms)
        if df.empty:
            df = fetch_bars(sym, days=9999, interval=INTERVAL, exchange_id="binance", refresh=False)
        if df.empty:
            continue
        results.append(simulate_bars(df, symbol=sym, params=params))

    trades = [t for r in results for t in r.trades]
    total_pnl = sum(r.total_pnl_net for r in results)
    wins = sum(1 for t in trades if t.pnl_net_usdt > 0)
    by_sym = {
        r.symbol: {
            "trades": len(r.trades),
            "pnl": r.total_pnl_net,
            "win_rate": r.win_rate,
            "max_dd": r.max_drawdown_usdt,
            "final_equity": round(EQUITY + r.total_pnl_net, 2),
        }
        for r in results
    }
    by_reason: dict[str, dict] = {}
    for t in trades:
        b = by_reason.setdefault(t.exit_reason, {"n": 0, "pnl": 0.0})
        b["n"] += 1
        b["pnl"] += t.pnl_net_usdt
    start_total = EQUITY * len(results)
    final_total = start_total + total_pnl
    return {
        "symbols_ok": len(results),
        "trades": len(trades),
        "pnl": round(total_pnl, 2),
        "final": round(final_total, 2),
        "ret_pct": round((final_total / start_total - 1) * 100, 2) if start_total else 0,
        "win_rate": round(wins / len(trades), 4) if trades else 0,
        "by_sym": by_sym,
        "by_reason": by_reason,
    }


def main() -> None:
    print("=== Listing probe (5m klines) ===")
    listing = probe_listing(SYMBOLS)
    ok_syms = []
    for raw in SYMBOLS:
        sym = norm_symbol(raw)
        info = listing.get(sym, {})
        if not info.get("ok"):
            print(f"  {sym}: NO DATA")
            continue
        ok_syms.append(raw)
        print(
            f"  {sym}: {info['first'].date()} -> {info['last'].date()} | "
            f"{info['bars']} bars | {info['trading_days']} trading days"
        )

    if not ok_syms:
        print("No symbol data available.")
        return

    # Use symbols with data; common start = latest listing among pool (for fair comparison)
    first_dates = [listing[norm_symbol(s)]["first"] for s in ok_syms]
    common_start = max(first_dates)
    print(f"\nPool: {', '.join(ok_syms)}")
    print(f"Earliest listing in pool: {min(first_dates).date()}")
    print(f"Latest listing in pool:   {common_start.date()} (shortest history)")
    print(f"End: {max(listing[norm_symbol(s)]['last'] for s in ok_syms).date()}")
    print(f"\nEquity: {EQUITY} USDT/symbol, compound=True, interval={INTERVAL}")
    print("\n=== Since listing (each symbol from its own first bar) ===")
    print(f"{'方案':<28} {'标的':>4} {'笔数':>5} {'净PnL':>10} {'收益率':>8} {'胜率':>6}")
    print("-" * 70)

    for label, th, flags in SCHEMES:
        r = run_since_listing(ok_syms, th=th, flags=flags)
        print(
            f"{label:<28} {r['symbols_ok']:>4} {r['trades']:>5} {r['pnl']:>+10.2f} "
            f"{r['ret_pct']:>+7.2f}% {r['win_rate']*100:>5.1f}%"
        )
        for sym, st in sorted(r["by_sym"].items()):
            ret = (st["final_equity"] / EQUITY - 1) * 100
            print(f"    {sym:<12} {st['trades']:>3}t  pnl={st['pnl']:+.2f}  ret={ret:+.2f}%  dd={st['max_dd']:.2f}")
        parts = [f"{k} {v['n']}t {v['pnl']:+.1f}" for k, v in sorted(r["by_reason"].items(), key=lambda x: -x[1]["n"])]
        print(f"    exit: {' | '.join(parts)}")
        print()


if __name__ == "__main__":
    main()
