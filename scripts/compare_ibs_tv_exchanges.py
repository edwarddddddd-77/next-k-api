#!/usr/bin/env python3
"""同 31 天窗口对比 Bitget vs Binance（long_only）。"""

from __future__ import annotations

import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import List

os.environ["IBS_TV_VNPY_TRADE_TYPE"] = "long_only"

import pandas as pd

from quant.common.config import OrbConfig
from quant.common.kline_cache import norm_symbol
from quant.ibs.session_daily import aggregate_session_daily
from quant.ibs.symbols import resolve_ibs_trading_symbol
from quant.ibs_tv.config import IbsTvConfig
from quant.market import klines_to_df
from quant.market.binance import fetch_klines_forward as bn_fwd
from quant.market.bitget import _public_get
from scripts.backtest_ibs_tv import (
    IbsTvBacktestParams,
    IbsTrade,
    SymbolResult,
    _apply_action,
    _ctx_at_close,
    _ctx_at_open,
    _evaluate,
    _params_for_symbol,
    _Sim,
)

WINDOW_START = "2026-06-09"
WINDOW_END = "2026-07-10"


def _day_ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def bitget_5m(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    sym = norm_symbol(symbol)
    cur_end = end_ms
    out: list = []
    seen: set[int] = set()
    while cur_end > start_ms and len(out) < 200_000:
        batch = _public_get(
            "/api/v2/mix/market/candles",
            {
                "symbol": sym,
                "productType": "USDT-FUTURES",
                "granularity": "5m",
                "endTime": str(cur_end),
                "limit": "1000",
            },
        )
        if not isinstance(batch, list) or not batch:
            break
        rows = sorted(batch, key=lambda r: int(r[0]))
        first = int(rows[0][0])
        for row in rows:
            ot = int(row[0])
            if start_ms <= ot <= end_ms and ot not in seen:
                seen.add(ot)
                out.append([ot, row[1], row[2], row[3], row[4], row[5]])
        if first <= start_ms or len(rows) < 1000:
            break
        cur_end = first - 1
    out.sort(key=lambda r: int(r[0]))
    return klines_to_df(out)


def fetch_5m(symbol: str, exchange: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    if exchange == "bitget":
        return bitget_5m(symbol, start_ms, end_ms)
    return klines_to_df(bn_fwd(symbol, "5m", start_ms, end_ms))


def run_on_df(
    symbol: str,
    df: pd.DataFrame,
    params: IbsTvBacktestParams,
    *,
    sess: OrbConfig,
    count_start: str,
    count_end: str,
) -> SymbolResult:
    sym = resolve_ibs_trading_symbol(symbol, params.product_type)
    daily = aggregate_session_daily(df.sort_values("open_time").reset_index(drop=True), sess=sess)
    if len(daily) < 3:
        return SymbolResult(symbol=sym, equity_start=params.equity_usdt, equity_end=params.equity_usdt)

    year_prefix = f"{int(params.year):04d}-"
    sim = _Sim(equity=float(params.equity_usdt))
    trades: List[IbsTrade] = []

    for i, bar in enumerate(daily):
        day = bar.session_day
        if i == 0:
            continue
        if sim.pending_action and day > sim.pending_signal_day:
            pending = sim.pending_action
            sim.pending_action = ""
            sim.pending_signal_day = ""
            ctx_open = _ctx_at_open(daily[:i], bar.open)
            if ctx_open is not None and pending in ("BUY", "SHORT"):
                recheck = _evaluate(
                    ctx_open,
                    params=params,
                    position_side=0,
                    holding_days=0,
                    last_entry_price=sim.last_closed_entry,
                )
                if recheck != pending:
                    pending = ""
            if pending:
                _apply_action(pending, sim=sim, price=bar.open, day=day, params=params, trades=trades, symbol=sym)

        in_year = day.startswith(year_prefix)
        if sim.position_side != 0 and in_year:
            sim.holding_days += 1

        ctx_close = _ctx_at_close(daily[: i + 1], bar.close)
        if ctx_close is None:
            continue
        action = _evaluate(
            ctx_close,
            params=params,
            position_side=sim.position_side,
            holding_days=sim.holding_days,
            last_entry_price=sim.last_closed_entry,
        )
        if action == "HOLD":
            continue
        if params.execute_at_next_open:
            sim.pending_action = action
            sim.pending_signal_day = day
            continue
        if in_year or sim.position_side != 0:
            _apply_action(action, sim=sim, price=bar.close, day=day, params=params, trades=trades, symbol=sym)

    if sim.position_side != 0:
        last = daily[-1]
        if sim.position_side > 0:
            _apply_action("SELL", sim=sim, price=last.close, day=last.session_day, params=params, trades=trades, symbol=sym)
        else:
            _apply_action("COVER", sim=sim, price=last.close, day=last.session_day, params=params, trades=trades, symbol=sym)

    window_trades = [t for t in trades if t.entry_day >= count_start and t.exit_day <= count_end]
    net = sum(t.pnl_net for t in window_trades)
    return SymbolResult(
        symbol=sym,
        trades=window_trades,
        equity_start=params.equity_usdt,
        equity_end=params.equity_usdt + net,
    )


def compare_symbol(symbol: str, cfg: IbsTvConfig, sess: OrbConfig, equity: float) -> None:
    end_ms = int(time.time() * 1000)
    win_start = _day_ms(WINDOW_START)
    win_end = _day_ms(WINDOW_END) + 86_400_000

    # A) 严格只有 31 天 5m（EMA 不够，filter 自动放行）
    print(f"\n### {symbol} | 严格 31 天数据（{WINDOW_START}~{WINDOW_END}）###")
    for ex in ("bitget", "binance"):
        df = fetch_5m(resolve_ibs_trading_symbol(symbol, "perp"), ex, win_start, win_end)
        p = _params_for_symbol(cfg, symbol, equity=equity, year=2026, warmup=31, exchange=ex)
        res = run_on_df(symbol, df, p, sess=sess, count_start=WINDOW_START, count_end=WINDOW_END)
        net = sum(t.pnl_net for t in res.trades)
        print(f"  {ex:7s} bars={len(df):5d} daily={len(aggregate_session_daily(df, sess=sess)):2d} trades={len(res.trades)} net={net:+.2f}")
        for t in res.trades:
            print(f"    {t.entry_day}->{t.exit_day} {t.entry_price:.2f}->{t.exit_price:.2f} net={t.pnl_net:+.2f}")

    # B) 同窗口统计，但币安用更长 warmup 算 EMA（Bitget 做不到）
    print(f"\n### {symbol} | 同窗口统计 + 币安 EMA 预热（Bitget 无法复制）###")
    warm_start = _day_ms("2025-08-01")
    bn_df = fetch_5m(resolve_ibs_trading_symbol(symbol, "perp"), "binance", warm_start, end_ms)
    bg_df = fetch_5m(resolve_ibs_trading_symbol(symbol, "perp"), "bitget", win_start, end_ms)
    for ex, df in (("bitget", bg_df), ("binance", bn_df)):
        p = _params_for_symbol(cfg, symbol, equity=equity, year=2026, warmup=400, exchange=ex)
        res = run_on_df(symbol, df, p, sess=sess, count_start=WINDOW_START, count_end=WINDOW_END)
        net = sum(t.pnl_net for t in res.trades)
        print(f"  {ex:7s} bars={len(df):5d} trades={len(res.trades)} net={net:+.2f}")
        for t in res.trades:
            print(f"    {t.entry_day}->{t.exit_day} {t.entry_price:.2f}->{t.exit_price:.2f} net={t.pnl_net:+.2f}")


def main() -> None:
    cfg = IbsTvConfig.from_env()
    sess = OrbConfig.from_env()
    for sym in ("SPY", "QQQ"):
        compare_symbol(sym, cfg, sess, 10_000.0)


if __name__ == "__main__":
    main()
