#!/usr/bin/env python3
"""Backtest anchor_drift (last night's strategy) on available history."""
from __future__ import annotations

import time
from dataclasses import replace

import pandas as pd

from quant.anchor_drift.backtest import BacktestParams, fetch_bars, run_backtest, simulate_bars
from quant.anchor_dift.config import AnchorDriftConfig
from quant.common.config import OrbConfig
from quant.common.kline_cache import load_klines, norm_symbol
from quant.common.session import session_day_str
from quant.common.us_equity_calendar import is_us_equity_trading_day

# fix typo
from quant.anchor_drift.config import AnchorDriftConfig as _ADC

EQUITY = 1000.0
INTERVAL = "5m"
PROBE_MS = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp() * 1000)


def data_span(symbols: list[str]) -> tuple[str, str, int]:
    sess = OrbConfig.from_env()
    end_ms = int(time.time() * 1000)
    all_days: set[str] = set()
    t0 = t1 = None
    for raw in symbols:
        sym = norm_symbol(raw)
        df = load_klines(sym, INTERVAL, start_ms=PROBE_MS, end_ms=end_ms)
        if df.empty:
            continue
        a = pd.Timestamp(int(df["open_time"].min()), unit="ms", tz="America/New_York")
        b = pd.Timestamp(int(df["open_time"].max()), unit="ms", tz="America/New_York")
        t0 = a if t0 is None or a < t0 else t0
        t1 = b if t1 is None or b > t1 else t1
        for ms in df["open_time"]:
            all_days.add(
                session_day_str(int(ms), tz=sess.session_tz, session_open_time=sess.session_open_time)
            )
    td = len([d for d in all_days if is_us_equity_trading_day(d)])
    return (str(t0.date()) if t0 else "?", str(t1.date()) if t1 else "?", td)


def run_label(
    label: str,
    symbols: list[str],
    *,
    cfg: AnchorDriftConfig,
    weekend_only: bool = False,
    disable_adverse: bool = False,
) -> None:
    end_ms = int(time.time() * 1000)
    params = BacktestParams(
        symbols=symbols,
        days=9999,
        interval=INTERVAL,
        equity_usdt=EQUITY,
        compound=True,
        cfg=cfg,
        weekend_only=weekend_only,
        disable_adverse_stop=disable_adverse,
    )
    results = []
    for raw in symbols:
        sym = norm_symbol(raw)
        df = load_klines(sym, INTERVAL, start_ms=PROBE_MS, end_ms=end_ms)
        if df.empty:
            continue
        results.append(simulate_bars(df, symbol=sym, params=params))

    trades = [t for r in results for t in r.trades]
    pnl = sum(r.total_pnl_net for r in results)
    wins = sum(1 for t in trades if t.pnl_net_usdt > 0)
    start = EQUITY * len(results)
    by_reason: dict = {}
    for t in trades:
        b = by_reason.setdefault(t.exit_reason, {"n": 0, "pnl": 0.0})
        b["n"] += 1
        b["pnl"] += t.pnl_net_usdt

    print(f"\n--- {label} ---")
    print(
        f"  threshold={cfg.signal_threshold*100:.1f}%  adverse={'off' if disable_adverse else f'on({cfg.max_adverse_extension*100:.1f}%)'}  "
        f"weekend_only={weekend_only}"
    )
    print(
        f"  trades={len(trades)}  pnl={pnl:+.2f} USDT  ret={(pnl/start*100):+.2f}%  "
        f"win={wins/len(trades)*100:.1f}%  start={start:.0f}U"
    )
    for r in results:
        ret = r.total_pnl_net / EQUITY * 100
        print(
            f"    {r.symbol}: {len(r.trades)}t  pnl={r.total_pnl_net:+.2f}  ret={ret:+.2f}%  dd={r.max_drawdown_usdt:.2f}"
        )
    parts = [f"{k} {v['n']}t {v['pnl']:+.1f}" for k, v in sorted(by_reason.items(), key=lambda x: -x[1]["n"])]
    print(f"  exit: {' | '.join(parts)}")


def main() -> None:
    cfg_default = AnchorDriftConfig.from_env()
    pool = cfg_default.symbol_list()
    # symbols with local kline cache
    ok = [s for s in pool if not load_klines(norm_symbol(s), INTERVAL, start_ms=PROBE_MS).empty]
    missing = [s for s in pool if s not in ok]

    t0, t1, td = data_span(ok)
    print("=" * 60)
    print("Anchor Drift 回测（昨晚新增策略）")
    print("=" * 60)
    print(f"标的池: {', '.join(pool)}")
    print(f"有数据: {', '.join(ok)}")
    if missing:
        print(f"无K线: {', '.join(missing)} (需联网拉取)")
    print(f"区间: {t0} -> {t1}  |  {td} 美股交易日")
    print(f"每标的 {EQUITY}U  复利  5m  one_trade_per_anchor")

    # 1) 昨晚默认参数（config 默认 1.5% + adverse）
    run_label("默认参数（昨晚实现）", ok, cfg=cfg_default)

    # 2) 讨论后推荐的实盘方案 B
    cfg_b = replace(cfg_default, signal_threshold=0.025)
    run_label("推荐方案 B（2.5% + adverse）", ok, cfg=cfg_b)

    # 3) 默认 + 仅周末
    run_label("默认 + weekend_only", ok, cfg=cfg_default, weekend_only=True)


if __name__ == "__main__":
    main()
