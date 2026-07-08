#!/usr/bin/env python3
"""Dump GTL vnpy trade fills for post-mortem."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi

load_env_oi()

import pandas as pd
from orb.core.config import OrbConfig
from orb.gtl.resample import resample_ohlcv
from orb.gtl.vnpy.backtest import (
    _as_utc,
    _configure_sqlite_db,
    _df_to_bars,
    _force_flat_at_close,
    _pricetick,
    _vt_symbol,
)
from orb.gtl.vnpy.strategy import GtlBreakoutStrategy
from orb.vnpy.bootstrap import ensure_vnpy_path
from tools.cta.research_gtl_vnpy import _load_symbol_df

ensure_vnpy_path()

from vnpy.trader.constant import Interval, Offset  # noqa: E402
from vnpy.trader.database import get_database  # noqa: E402
from vnpy_ctastrategy.backtesting import BacktestingEngine, load_bar_data  # noqa: E402


def main() -> None:
    sym = "INTCUSDT"
    lo, hi = "2026-02-01", "2026-06-30"
    mode = sys.argv[1] if len(sys.argv) > 1 else "birth_break"
    trade_mode = {
        "break": "break",
        "signal": "signal",
        "signal_break": "signal_break",
        "birth_break": "birth_break",
    }.get(mode, mode)

    cfg = OrbConfig.from_env()
    fetch_lo = (pd.Timestamp(lo) - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    df = _load_symbol_df(sym, fetch_lo, hi, cfg)
    df = resample_ohlcv(df, "30m")
    start = pd.Timestamp(lo).to_pydatetime()
    end = (pd.Timestamp(hi) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).to_pydatetime()
    vt = _vt_symbol(sym)
    px = float(df.iloc[-1]["close"])
    bars = _df_to_bars(df, vt)
    db = _configure_sqlite_db()
    load_bar_data.cache_clear()
    get_database().save_bar_data(bars)

    engine = BacktestingEngine()
    engine.output = lambda _m: None
    engine.set_parameters(
        vt_symbol=vt,
        interval=Interval.MINUTE,
        start=_as_utc(start),
        end=_as_utc(end),
        rate=0.3 / 10_000,
        slippage=0.2,
        size=1,
        pricetick=_pricetick(px),
        capital=1000,
    )
    engine.add_strategy(GtlBreakoutStrategy, {"trade_mode": trade_mode, "fixed_size": 1, "force_flat_on_stop": True})
    engine.load_data()
    engine.run_backtesting()
    _force_flat_at_close(engine)
    engine.calculate_result()
    stats = engine.calculate_statistics(output=False)

    trades = sorted(engine.trades.values(), key=lambda t: t.datetime or pd.Timestamp(0))
    print(f"=== INTC 30m gtl_{trade_mode} {lo}..{hi} ===")
    print(f"net_pnl={float(stats.get('total_net_pnl') or 0):.2f} end={float(stats.get('end_balance') or 0):.2f}")
    print(f"fills={len(trades)} win_rate={stats.get('winning_rate')}")
    print()
    print("--- raw fills ---")
    for i, t in enumerate(trades, 1):
        print(f"{i:2d} {t.datetime} {t.direction} {t.offset} px={float(t.price):.4f} vol={float(t.volume)}")
    print()

    round_trips = []
    entry = None
    for t in trades:
        px_fill = float(t.price)
        if t.offset == Offset.OPEN:
            entry = {"side": t.direction, "dt": t.datetime, "px": px_fill}
        elif t.offset == Offset.CLOSE and entry:
            from vnpy.trader.constant import Direction

            if entry["side"] == Direction.LONG and t.direction == Direction.SHORT:
                pnl = px_fill - entry["px"]
            elif entry["side"] == Direction.SHORT and t.direction == Direction.LONG:
                pnl = entry["px"] - px_fill
            else:
                pnl = 0.0
            round_trips.append(
                {
                    "open_dt": entry["dt"],
                    "close_dt": t.datetime,
                    "side": entry["side"].value,
                    "entry_px": entry["px"],
                    "exit_px": px_fill,
                    "pnl_per_unit": pnl,
                }
            )
            entry = None

    for i, rt in enumerate(round_trips, 1):
        win = "win" if rt["pnl_per_unit"] > 0 else "loss"
        print(
            f"#{i} {win} {rt['side']} "
            f"open {rt['open_dt']} @{rt['entry_px']:.2f} -> "
            f"close {rt['close_dt']} @{rt['exit_px']:.2f} "
            f"move={rt['pnl_per_unit']:+.2f}"
        )

    if round_trips:
        wins = sum(1 for r in round_trips if r["pnl_per_unit"] > 0)
        print()
        print(f"round_trips={len(round_trips)} wins={wins} losses={len(round_trips)-wins}")
        print(f"sum_price_pnl={sum(r['pnl_per_unit'] for r in round_trips):+.2f} (before fees/slip)")


if __name__ == "__main__":
    main()
