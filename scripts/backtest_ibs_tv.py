#!/usr/bin/env python3
"""IBS TV lane 回测 — 对齐 ibs_tv（long_short + 永续 + 收盘信号/次日开盘执行）。"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from quant.common.config import OrbConfig
from quant.common.fees import trade_fee_usdt
from quant.common.kline_cache import norm_symbol, save_klines
from quant.common.symbols import parse_symbol_list
from quant.ibs.core import (
    IbsSignalContext,
    evaluate_signal_context,
    select_signal_context,
)
from quant.ibs.session_daily import aggregate_session_daily
from quant.ibs.sizing import size_for_ibs
from quant.ibs.symbols import resolve_ibs_trading_symbol
from quant.ibs_tv.config import IbsTvConfig
from quant.ibs_tv.paths import resolve_ibs_tv_symbols_path
from quant.market import fetch_klines_forward, klines_to_df


@dataclass
class IbsTvBacktestParams:
    symbols: List[str]
    exchange_id: str = "bitget"
    product_type: str = "perp"
    trade_type: str = "long_short"
    equity_usdt: float = 100.0
    compound: bool = True
    year: int = 2026
    warmup_days: int = 400
    entry_threshold: float = 0.09
    exit_threshold: float = 0.985
    position_pct: float = 1.0
    trend_ma_type: str = "ema"
    trend_ma_period: int = 220
    min_entry_distance_pct: float = 0.0
    max_trade_duration_days: int = 14
    execute_at_next_open: bool = True
    fee_bps_per_side: float = 4.0


@dataclass
class IbsTrade:
    symbol: str
    side: str
    entry_day: str
    exit_day: str
    entry_price: float
    exit_price: float
    qty: float
    pnl_gross: float
    fee_usdt: float
    pnl_net: float


@dataclass
class SymbolResult:
    symbol: str
    trades: List[IbsTrade] = field(default_factory=list)
    equity_start: float = 100.0
    equity_end: float = 100.0


def _fetch_bitget_5m_backward(symbol: str, *, start_ms: int, end_ms: int) -> pd.DataFrame:
    from quant.market.bitget import _public_get

    sym = norm_symbol(symbol)
    cur_end = int(end_ms)
    out: list = []
    seen: set[int] = set()
    while cur_end > int(start_ms) and len(out) < 200_000:
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
        rows_asc = sorted(batch, key=lambda r: int(r[0]))
        first_open = int(rows_asc[0][0])
        for row in rows_asc:
            ot = int(row[0])
            if ot < int(start_ms) or ot > int(end_ms) or ot in seen:
                continue
            seen.add(ot)
            out.append([ot, row[1], row[2], row[3], row[4], row[5]])
        if first_open <= int(start_ms) or len(rows_asc) < 1000:
            break
        cur_end = first_open - 1
    if not out:
        return pd.DataFrame()
    out.sort(key=lambda r: int(r[0]))
    return klines_to_df(out)


def _fetch_session_5m(symbol: str, *, days: int, exchange_id: str) -> pd.DataFrame:
    sym = norm_symbol(symbol)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - max(30, int(days)) * 86_400_000
    print(f"[fetch] {sym} 5m {days}d from {exchange_id} ...")
    if exchange_id == "bitget":
        df = _fetch_bitget_5m_backward(sym, start_ms=start_ms, end_ms=end_ms)
    else:
        rows = fetch_klines_forward(sym, "5m", start_ms, end_ms, exchange_id=exchange_id)
        df = klines_to_df(rows)
    if not df.empty:
        save_klines(sym, "5m", df)
        print(f"[fetch] {sym} got {len(df)} bars")
    return df.sort_values("open_time").reset_index(drop=True)


def _ctx_at_close(daily: list, close_px: float) -> Optional[IbsSignalContext]:
    if len(daily) < 2:
        return None
    return select_signal_context(daily, trend_price_mode="current", current_price=close_px)


def _ctx_at_open(daily_before_today: list, open_px: float) -> Optional[IbsSignalContext]:
    if len(daily_before_today) < 2:
        return None
    return select_signal_context(
        daily_before_today,
        trend_price_mode="current",
        current_price=open_px,
        ma_excludes_last_bar=True,
    )


def _evaluate(
    ctx: IbsSignalContext,
    *,
    params: IbsTvBacktestParams,
    position_side: int,
    holding_days: int,
    last_entry_price: float,
) -> str:
    return evaluate_signal_context(
        ctx,
        position_side=position_side,
        trade_type=params.trade_type,
        entry_threshold=params.entry_threshold,
        exit_threshold=params.exit_threshold,
        trend_ma_type=params.trend_ma_type,
        trend_ma_period=params.trend_ma_period,
        holding_days=holding_days,
        max_trade_duration_days=params.max_trade_duration_days,
        last_entry_price=last_entry_price,
        min_entry_distance_pct=params.min_entry_distance_pct,
    )


@dataclass
class _Sim:
    position_side: int = 0
    entry_price: float = 0.0
    qty: float = 0.0
    entry_day: str = ""
    holding_days: int = 0
    last_closed_entry: float = 0.0
    pending_action: str = ""
    pending_signal_day: str = ""
    equity: float = 100.0


class _SizingCfg:
    def __init__(self, equity: float, position_pct: float) -> None:
        self.equity_usdt = equity
        self.position_pct = position_pct
        self.max_notional_usdt = 0.0


def _open_position(
    sim: _Sim,
    *,
    side: int,
    price: float,
    day: str,
    params: IbsTvBacktestParams,
) -> None:
    cfg = _SizingCfg(sim.equity, params.position_pct)
    qty = size_for_ibs(cfg, price, equity_usdt=sim.equity)
    if qty <= 0:
        return
    sim.position_side = side
    sim.entry_price = float(price)
    sim.qty = float(qty)
    sim.entry_day = day
    sim.holding_days = 0


def _close_position(
    sim: _Sim,
    *,
    price: float,
    day: str,
    params: IbsTvBacktestParams,
    trades: List[IbsTrade],
    symbol: str,
    side_label: str,
) -> None:
    if sim.position_side == 0 or sim.qty <= 0:
        return
    entry = float(sim.entry_price)
    exit_px = float(price)
    qty = float(sim.qty)
    if sim.position_side > 0:
        gross = (exit_px - entry) * qty
    else:
        gross = (entry - exit_px) * qty
    notional = entry * qty
    fee = trade_fee_usdt(notional, fee_bps_per_side=params.fee_bps_per_side)
    net = gross - fee
    trades.append(
        IbsTrade(
            symbol=symbol,
            side=side_label,
            entry_day=sim.entry_day,
            exit_day=day,
            entry_price=entry,
            exit_price=exit_px,
            qty=qty,
            pnl_gross=round(gross, 4),
            fee_usdt=round(fee, 4),
            pnl_net=round(net, 4),
        )
    )
    if params.compound:
        sim.equity = max(0.0, sim.equity + net)
    sim.last_closed_entry = entry
    sim.position_side = 0
    sim.entry_price = 0.0
    sim.qty = 0.0
    sim.entry_day = ""
    sim.holding_days = 0


def _apply_action(
    action: str,
    *,
    sim: _Sim,
    price: float,
    day: str,
    params: IbsTvBacktestParams,
    trades: List[IbsTrade],
    symbol: str,
) -> None:
    if action == "SELL" and sim.position_side > 0:
        _close_position(sim, price=price, day=day, params=params, trades=trades, symbol=symbol, side_label="long")
        return
    if action == "COVER" and sim.position_side < 0:
        _close_position(sim, price=price, day=day, params=params, trades=trades, symbol=symbol, side_label="short")
        return
    if action == "BUY" and sim.position_side == 0:
        _open_position(sim, side=1, price=price, day=day, params=params)
        return
    if action == "SHORT" and sim.position_side == 0:
        _open_position(sim, side=-1, price=price, day=day, params=params)


def run_symbol_backtest(symbol: str, params: IbsTvBacktestParams, *, sess: OrbConfig) -> SymbolResult:
    sym = resolve_ibs_trading_symbol(symbol, params.product_type)
    df = _fetch_session_5m(sym, days=params.warmup_days, exchange_id=params.exchange_id)
    daily = aggregate_session_daily(df, sess=sess)
    if len(daily) < 3:
        return SymbolResult(symbol=sym, equity_start=params.equity_usdt, equity_end=params.equity_usdt)

    year_prefix = f"{int(params.year):04d}-"
    sim = _Sim(equity=float(params.equity_usdt))
    trades: List[IbsTrade] = []

    for i, bar in enumerate(daily):
        day = bar.session_day
        if i == 0:
            continue

        # 次日开盘执行 pending
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

        # 收盘评估（仅统计年内持仓天数）
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

    # 年末强平（便于统计）
    if sim.position_side != 0:
        last = daily[-1]
        if sim.position_side > 0:
            _apply_action("SELL", sim=sim, price=last.close, day=last.session_day, params=params, trades=trades, symbol=sym)
        else:
            _apply_action("COVER", sim=sim, price=last.close, day=last.session_day, params=params, trades=trades, symbol=sym)

    year_trades = [t for t in trades if t.entry_day.startswith(year_prefix) or t.exit_day.startswith(year_prefix)]
    return SymbolResult(
        symbol=sym,
        trades=year_trades,
        equity_start=params.equity_usdt,
        equity_end=sim.equity,
    )


def _summarize(results: List[SymbolResult], *, year: int) -> None:
    print(f"\n=== IBS TV backtest {year} | long_short | perp ===\n")
    total_net = 0.0
    total_trades = 0
    for res in results:
        net = sum(t.pnl_net for t in res.trades)
        wins = sum(1 for t in res.trades if t.pnl_net > 0)
        longs = [t for t in res.trades if t.side == "long"]
        shorts = [t for t in res.trades if t.side == "short"]
        total_net += net
        total_trades += len(res.trades)
        ret_pct = (res.equity_end / res.equity_start - 1.0) * 100.0 if res.equity_start > 0 else 0.0
        print(f"{res.symbol}")
        print(f"  trades={len(res.trades)} (long={len(longs)} short={len(shorts)}) win={wins}/{len(res.trades) or 1}")
        print(f"  net_pnl={net:+.2f} USDT  equity {res.equity_start:.2f} -> {res.equity_end:.2f} ({ret_pct:+.2f}%)")
        for t in res.trades:
            print(
                f"    {t.side:5s} {t.entry_day} @ {t.entry_price:.2f} -> {t.exit_day} @ {t.exit_price:.2f} "
                f"qty={t.qty:.4f} net={t.pnl_net:+.2f}"
            )
        print()
    print(f"POOL total net={total_net:+.2f} USDT  trades={total_trades}")


def _params_for_symbol(cfg: IbsTvConfig, symbol: str, *, equity: float, year: int, warmup: int, exchange: str) -> IbsTvBacktestParams:
    sym_cfg = cfg.lane_config_for_symbol(symbol)
    return IbsTvBacktestParams(
        symbols=[symbol],
        exchange_id=exchange,
        equity_usdt=equity,
        year=year,
        warmup_days=warmup,
        entry_threshold=sym_cfg.entry_threshold,
        exit_threshold=sym_cfg.exit_threshold,
        position_pct=sym_cfg.position_pct,
        trend_ma_type=sym_cfg.trend_ma_type,
        trend_ma_period=sym_cfg.trend_ma_period,
        min_entry_distance_pct=sym_cfg.min_entry_distance_pct,
        max_trade_duration_days=sym_cfg.max_trade_duration_days,
        trade_type=sym_cfg.trade_type,
        product_type=sym_cfg.product_type,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="IBS TV backtest")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--equity", type=float, default=100.0)
    parser.add_argument("--exchange", default="bitget")
    parser.add_argument("--warmup-days", type=int, default=400)
    parser.add_argument(
        "--symbols",
        default="",
        help="comma list; default config/ibs_tv/symbols.txt",
    )
    args = parser.parse_args()

    if args.symbols.strip():
        symbols = parse_symbol_list(args.symbols)
    else:
        symbols = parse_symbol_list(resolve_ibs_tv_symbols_path().read_text(encoding="utf-8"))

    cfg = IbsTvConfig.from_env()
    sess = OrbConfig.from_env()
    results = [
        run_symbol_backtest(
            sym,
            _params_for_symbol(
                cfg,
                sym,
                equity=float(args.equity),
                year=int(args.year),
                warmup=int(args.warmup_days),
                exchange=str(args.exchange).strip().lower(),
            ),
            sess=sess,
        )
        for sym in symbols
    ]
    _summarize(results, year=int(args.year))


if __name__ == "__main__":
    main()
