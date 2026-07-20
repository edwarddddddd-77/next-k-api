#!/usr/bin/env python3
"""Backtest MtfMomo + KAMA Trend — last N days (default 60)."""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any, List, Sequence

import pandas as pd

from quant.common.fees import fee_taker_bps_from_env, trade_fee_usdt
from quant.common.kline_cache import load_klines, norm_symbol, save_klines
from quant.kama_trend.config import KamaTrendConfig
from quant.kama_trend.core import (
    BarOhlc,
    bar_hits_stop_tp as kama_bar_hits,
    compute_snapshot,
    entry_signal as kama_entry,
    stop_tp_prices as kama_stop_tp,
)
from quant.kama_trend.sizing import size_for_kama
from quant.market import fetch_klines_forward, klines_to_df
from quant.mtfmomo.config import MtfMomoConfig
from quant.mtfmomo.core import (
    HourOhlc,
    bar_hits_stop_tp as momo_bar_hits,
    compute_levels,
    entry_signal as momo_entry,
    resample_utc,
    should_ema_exit,
    stop_tp_prices as momo_stop_tp,
)
from quant.mtfmomo.sizing import size_for_momo


@dataclass
class Trade:
    symbol: str
    side: str
    entry_ms: int
    exit_ms: int
    entry_price: float
    exit_price: float
    exit_reason: str
    qty: float
    pnl_net: float


def _apply_slippage(price: float, *, side: int, is_entry: bool, bps: float) -> float:
    slip = max(0.0, bps) / 10_000.0
    px = float(price)
    if is_entry:
        return px * (1.0 + slip) if side > 0 else px * (1.0 - slip)
    return px * (1.0 - slip) if side > 0 else px * (1.0 + slip)


def _close_pnl(*, side: int, qty: float, entry_px: float, exit_px: float, taker_bps: float) -> float:
    qty = abs(float(qty))
    if side > 0:
        gross = (exit_px - entry_px) * qty
    else:
        gross = (entry_px - exit_px) * qty
    fee = trade_fee_usdt((entry_px + exit_px) * qty / 2.0, taker_bps=taker_bps)
    return gross - fee


def fetch_df(symbol: str, interval: str, *, days: int, exchange: str = "binance") -> pd.DataFrame:
    sym = norm_symbol(symbol)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - max(1, days) * 86_400_000
    cached = load_klines(sym, interval, start_ms=start_ms, end_ms=end_ms)
    span = 0.0
    if not cached.empty:
        span = (cached["open_time"].max() - cached["open_time"].min()) / 86_400_000
    if cached.empty or span < days * 0.85:
        for ex in (exchange, "bitget", "binance"):
            print(f"[fetch] {sym} {interval} {days}d from {ex} ...")
            rows = fetch_klines_forward(sym, interval, start_ms, end_ms, exchange_id=ex)
            fresh = klines_to_df(rows)
            if not fresh.empty:
                merged = fresh if cached.empty else (
                    pd.concat([cached, fresh]).drop_duplicates("open_time").sort_values("open_time")
                )
                save_klines(sym, interval, merged)
                cached = merged
                break
    return cached.sort_values("open_time").reset_index(drop=True)


def _resample_closes(bars: Sequence[BarOhlc], period_hours: int) -> list[float]:
    c, _, _ = resample_utc(
        [(int(b[0]), b[1], b[2], b[3], b[4]) for b in bars],
        period_hours,
    )
    return c


def simulate_mtfmomo(
    df: pd.DataFrame,
    *,
    symbol: str,
    cfg: MtfMomoConfig,
    window_start_ms: int,
    slippage_bps: float,
    taker_bps: float,
) -> tuple[list[Trade], float, float]:
    equity = float(cfg.equity_usdt)
    peak = equity
    max_dd = 0.0
    trades: list[Trade] = []
    hour_bars: list[HourOhlc] = []
    pos_side = 0
    entry_ms = 0
    entry_px = 0.0
    entry_qty = 0.0
    stop_px = 0.0
    tp_px = 0.0

    for row in df.itertuples(index=False):
        bar_ms = int(row.open_time)
        o, h, l, c = float(row.open), float(row.high), float(row.low), float(row.close)
        hour_bars.append((bar_ms, o, h, l, c))
        max_bars = 24 * max(40, cfg.init_bar_days)
        if len(hour_bars) > max_bars:
            hour_bars = hour_bars[-max_bars:]

        levels = compute_levels(
            hour_bars,
            entry_lb=int(cfg.entry_lb),
            ema_exit=int(cfg.ema_exit),
            ema_4h=int(cfg.ema_4h),
            ema_1d=int(cfg.ema_1d),
        )
        if levels is None:
            continue

        if pos_side != 0:
            hit = momo_bar_hits(side=pos_side, high=h, low=l, stop=stop_px, tp=tp_px)
            reason = hit
            if not hit and should_ema_exit(c, pos_side, levels.ema_exit):
                reason = "ema"
            if reason:
                raw_exit = stop_px if hit == "stop" else (tp_px if hit == "tp" else c)
                exit_px = _apply_slippage(raw_exit, side=pos_side, is_entry=False, bps=slippage_bps)
                net = _close_pnl(
                    side=pos_side, qty=entry_qty, entry_px=entry_px, exit_px=exit_px, taker_bps=taker_bps
                )
                if entry_ms >= window_start_ms:
                    trades.append(
                        Trade(
                            symbol=symbol,
                            side="LONG" if pos_side > 0 else "SHORT",
                            entry_ms=entry_ms,
                            exit_ms=bar_ms,
                            entry_price=entry_px,
                            exit_price=exit_px,
                            exit_reason=reason or "ema",
                            qty=entry_qty,
                            pnl_net=net,
                        )
                    )
                if cfg.compound:
                    equity += net
                    peak = max(peak, equity)
                    max_dd = max(max_dd, peak - equity)
                pos_side = 0
            continue

        sig = momo_entry(c, levels)
        if sig == 0 or bar_ms < window_start_ms:
            continue
        stop, tp = momo_stop_tp(c, sig, levels, stop_atr=float(cfg.stop_atr), tp_atr=float(cfg.tp_atr))
        stop_dist = abs(c - stop)
        eq = equity if cfg.compound else float(cfg.equity_usdt)
        qty = size_for_momo(cfg, c, stop_distance=stop_dist, equity_usdt=eq)
        if qty <= 0:
            continue
        entry_px = _apply_slippage(c, side=sig, is_entry=True, bps=slippage_bps)
        pos_side = sig
        entry_ms = bar_ms
        entry_qty = qty
        stop_px = stop
        tp_px = tp

    return trades, equity, max_dd


def simulate_kama(
    df: pd.DataFrame,
    *,
    symbol: str,
    cfg: KamaTrendConfig,
    window_start_ms: int,
    slippage_bps: float,
    taker_bps: float,
) -> tuple[list[Trade], float, float]:
    equity = float(cfg.equity_usdt)
    peak = equity
    max_dd = 0.0
    trades: list[Trade] = []
    signal_bars: list[BarOhlc] = []
    bar_index = 0
    last_trade_idx = -10_000
    pos_side = 0
    entry_ms = 0
    entry_px = 0.0
    entry_qty = 0.0
    stop_px = 0.0
    tp_px = 0.0
    prev_h = 0.0
    prev_l = 0.0

    for row in df.itertuples(index=False):
        bar_ms = int(row.open_time)
        o, h, l, c = float(row.open), float(row.high), float(row.low), float(row.close)
        if signal_bars and signal_bars[-1][0] == bar_ms:
            signal_bars[-1] = (bar_ms, o, h, l, c)
        else:
            signal_bars.append((bar_ms, o, h, l, c))
            bar_index += 1
        max_bars = (24 * 4) * max(40, cfg.init_bar_days)
        if len(signal_bars) > max_bars:
            trim = len(signal_bars) - max_bars
            signal_bars = signal_bars[trim:]
            bar_index = max(0, bar_index - trim)
            last_trade_idx = max(-10_000, last_trade_idx - trim)

        long_closes = _resample_closes(signal_bars, 4)
        snap = compute_snapshot(
            signal_bars,
            long_tf_closes=long_closes,
            kama_period=int(cfg.kama_period),
            adx_period=int(cfg.adx_period),
            chop_period=int(cfg.chop_period),
            bb_period=int(cfg.bb_period),
        )
        if snap is None:
            prev_h, prev_l = h, l
            continue

        if pos_side != 0:
            hit = kama_bar_hits(
                side=pos_side,
                high=h,
                low=l,
                stop=stop_px,
                tp=tp_px,
                prev_high=prev_h,
                prev_low=prev_l,
            )
            if hit:
                raw_exit = stop_px if hit == "stop" else tp_px
                exit_px = _apply_slippage(raw_exit, side=pos_side, is_entry=False, bps=slippage_bps)
                net = _close_pnl(
                    side=pos_side, qty=entry_qty, entry_px=entry_px, exit_px=exit_px, taker_bps=taker_bps
                )
                if entry_ms >= window_start_ms:
                    trades.append(
                        Trade(
                            symbol=symbol,
                            side="LONG" if pos_side > 0 else "SHORT",
                            entry_ms=entry_ms,
                            exit_ms=bar_ms,
                            entry_price=entry_px,
                            exit_price=exit_px,
                            exit_reason=hit,
                            qty=entry_qty,
                            pnl_net=net,
                        )
                    )
                if cfg.compound:
                    equity += net
                    peak = max(peak, equity)
                    max_dd = max(max_dd, peak - equity)
                pos_side = 0
                last_trade_idx = bar_index
            prev_h, prev_l = h, l
            continue

        bars_since = bar_index - last_trade_idx
        sig = kama_entry(
            c,
            snap,
            adx_min=float(cfg.adx_min),
            chop_max=float(cfg.chop_max),
            bb_width_max_pct=float(cfg.bb_width_max_pct),
            bars_since_trade=bars_since,
            cooldown_bars=int(cfg.cooldown_bars),
        )
        if sig == 0 or bar_ms < window_start_ms:
            prev_h, prev_l = h, l
            continue
        stop, tp = kama_stop_tp(c, sig, snap.atr, stop_atr=float(cfg.stop_atr), tp_atr=float(cfg.tp_atr))
        stop_dist = abs(c - stop)
        eq = equity if cfg.compound else float(cfg.equity_usdt)
        qty = size_for_kama(cfg, c, stop_distance=stop_dist, equity_usdt=eq)
        if qty <= 0:
            prev_h, prev_l = h, l
            continue
        entry_px = _apply_slippage(c, side=sig, is_entry=True, bps=slippage_bps)
        pos_side = sig
        entry_ms = bar_ms
        entry_qty = qty
        stop_px = stop
        tp_px = tp
        prev_h, prev_l = h, l

    return trades, equity, max_dd


def _summarize(name: str, trades: list[Trade], *, equity: float, start_eq: float, max_dd: float) -> None:
    pnl = sum(t.pnl_net for t in trades)
    wins = sum(1 for t in trades if t.pnl_net > 0)
    wr = wins / len(trades) * 100 if trades else 0.0
    ret = pnl / start_eq * 100 if start_eq else 0.0
    print(f"\n=== {name} ===")
    print(f"  trades={len(trades)}  pnl={pnl:+.2f} USDT  ret={ret:+.2f}%  win={wr:.1f}%  max_dd={max_dd:.2f}")
    by_sym: dict[str, list[Trade]] = {}
    for t in trades:
        by_sym.setdefault(t.symbol, []).append(t)
    for sym, ts in sorted(by_sym.items()):
        sp = sum(x.pnl_net for x in ts)
        w = sum(1 for x in ts if x.pnl_net > 0)
        wr2 = w / len(ts) * 100 if ts else 0
        print(f"    {sym}: {len(ts)}t  pnl={sp:+.2f}  win={wr2:.0f}%")
    by_reason: dict[str, float] = {}
    for t in trades:
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0.0) + t.pnl_net
    if by_reason:
        parts = [f"{k} {v:+.1f}" for k, v in sorted(by_reason.items(), key=lambda x: -abs(x[1]))]
        print(f"  exit: {' | '.join(parts)}")
    print(f"  final equity (compound): {equity:.2f} USDT")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--equity", type=float, default=100.0)
    p.add_argument("--slippage-bps", type=float, default=10.0)
    args = p.parse_args()

    days = max(7, int(args.days))
    warmup = 30
    fetch_days = days + warmup
    end_ms = int(time.time() * 1000)
    window_start_ms = end_ms - days * 86_400_000
    taker_bps = fee_taker_bps_from_env()
    slip = float(args.slippage_bps)

    momo_cfg = MtfMomoConfig.from_env()
    momo_cfg = MtfMomoConfig(
        **{**momo_cfg.__dict__, "equity_usdt": float(args.equity), "compound": True}
    )
    kama_cfg = KamaTrendConfig.from_env()
    kama_cfg = KamaTrendConfig(
        **{**kama_cfg.__dict__, "equity_usdt": float(args.equity), "compound": True}
    )

    t0 = pd.Timestamp(window_start_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d")
    t1 = pd.Timestamp(end_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d")
    print(f"Crypto lane backtest  |  window={days}d ({t0} -> {t1})  |  equity={args.equity}/symbol  compound")
    print(f"slippage={slip}bps  taker_fee={taker_bps}bps  warmup={warmup}d")

    all_momo: list[Trade] = []
    momo_start = 0.0
    momo_final = 0.0
    momo_dd = 0.0
    momo_syms = momo_cfg.symbol_list() or ["SOL", "ETH"]
    for raw in momo_syms:
        sym = norm_symbol(raw)
        df = fetch_df(raw, "1h", days=fetch_days)
        if df.empty:
            print(f"[skip] {sym} no 1h data")
            continue
        ts, eq, dd = simulate_mtfmomo(
            df,
            symbol=sym,
            cfg=momo_cfg,
            window_start_ms=window_start_ms,
            slippage_bps=slip,
            taker_bps=taker_bps,
        )
        all_momo.extend(ts)
        momo_start += float(args.equity)
        momo_final += eq
        momo_dd = max(momo_dd, dd)

    all_kama: list[Trade] = []
    kama_start = 0.0
    kama_final = 0.0
    kama_dd = 0.0
    kama_syms = kama_cfg.symbol_list() or ["BTC"]
    interval = f"{max(1, kama_cfg.signal_minutes)}m"
    for raw in kama_syms:
        sym = norm_symbol(raw)
        df = fetch_df(raw, interval, days=fetch_days)
        if df.empty:
            print(f"[skip] {sym} no {interval} data")
            continue
        ts, eq, dd = simulate_kama(
            df,
            symbol=sym,
            cfg=kama_cfg,
            window_start_ms=window_start_ms,
            slippage_bps=slip,
            taker_bps=taker_bps,
        )
        all_kama.extend(ts)
        kama_start += float(args.equity)
        kama_final += eq
        kama_dd = max(kama_dd, dd)

    _summarize("MtfMomo2xA (1h)", all_momo, equity=momo_final, start_eq=momo_start or float(args.equity), max_dd=momo_dd)
    _summarize(
        f"KAMA Trend ({interval})",
        all_kama,
        equity=kama_final,
        start_eq=kama_start or float(args.equity),
        max_dd=kama_dd,
    )

    total_pnl = sum(t.pnl_net for t in all_momo + all_kama)
    total_start = momo_start + kama_start
    print(f"\n=== Combined ===")
    print(f"  total pnl={total_pnl:+.2f} USDT  ret={total_pnl/total_start*100:+.2f}%  trades={len(all_momo)+len(all_kama)}")


if __name__ == "__main__":
    main()
