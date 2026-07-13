#!/usr/bin/env python3
"""IB50 backtest — Initial Balance 50% 机械延续（1m IB + 5m 入场/出场）。"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import pandas as pd

from quant.common.fees import trade_fee_usdt
from quant.common.kline_cache import load_klines, norm_symbol, save_klines
from quant.common.macro_calendar import is_macro_skip_day
from quant.common.resample import resample_ohlcv
from quant.common.session import session_anchor_ms, session_day_str
from quant.common.session_paper import in_regular_session
from quant.engine.eod import should_eod_flat_bar
from quant.ib50.config import Ib50Config
from quant.ib50.core import (
    bar_exit_reason,
    build_ib50_setup,
    finalize_initial_balance,
    ib_complete_at_bar,
    in_ib_window,
    update_ib_range,
    weekday_allowed,
)
from quant.ib50.sizing import fixed_size_for_ib50
from quant.market import fetch_klines_forward, klines_to_df


@dataclass
class Ib50Trade:
    symbol: str
    session_date: str
    side: str
    entry_ms: int
    exit_ms: int
    entry_price: float
    exit_price: float
    exit_reason: str
    first_extreme: str
    ib_high: float
    ib_low: float
    qty: float
    pnl_net: float


def _slip(px: float, *, side: int, is_entry: bool, bps: float) -> float:
    s = max(0.0, bps) / 10_000.0
    if is_entry:
        return px * (1.0 + s) if side > 0 else px * (1.0 - s)
    return px * (1.0 - s) if side > 0 else px * (1.0 + s)


def fetch_1m(symbol: str, *, days: int, exchange: str = "binance") -> pd.DataFrame:
    sym = norm_symbol(symbol)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000
    cached = load_klines(sym, "1m", start_ms=start_ms, end_ms=end_ms)
    span = 0.0
    if not cached.empty:
        span = (cached["open_time"].max() - cached["open_time"].min()) / 86_400_000
    if (os.getenv("BACKTEST_CACHE_ONLY") or "").strip() in ("1", "true", "yes"):
        if not cached.empty:
            return cached.sort_values("open_time").reset_index(drop=True)
    elif cached.empty or span < days * 0.75:
        for ex in (exchange, "bitget", "binance"):
            print(f"[fetch] {sym} 1m {days}d from {ex} ...")
            rows = fetch_klines_forward(sym, "1m", start_ms, end_ms, exchange_id=ex)
            fresh = klines_to_df(rows)
            if not fresh.empty:
                merged = fresh if cached.empty else (
                    pd.concat([cached, fresh]).drop_duplicates("open_time").sort_values("open_time")
                )
                save_klines(sym, "1m", merged)
                cached = merged
                break
    if not cached.empty:
        return cached.sort_values("open_time").reset_index(drop=True)
    df5 = load_klines(sym, "5m", start_ms=start_ms, end_ms=end_ms)
    if df5.empty:
        return pd.DataFrame()
    print(f"[note] {sym} using 5m-expanded 1m (no native 1m cache)")
    rows: list[dict] = []
    for _, r in df5.iterrows():
        base = int(r["open_time"])
        vol = float(r["volume"]) / 5.0
        for i in range(5):
            rows.append(
                {
                    "open_time": base + i * 60_000,
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                    "volume": vol,
                }
            )
    return pd.DataFrame(rows).sort_values("open_time").reset_index(drop=True)


def simulate_symbol(
    df_1m: pd.DataFrame,
    *,
    symbol: str,
    cfg: Ib50Config,
    window_start_ms: int,
    slippage_bps: float,
) -> tuple[list[Ib50Trade], float, float]:
    sess = cfg.session_cfg()
    taker_bps = float(cfg.fee_taker_bps)
    allowed_days = cfg.weekday_filter()
    if df_1m.empty:
        return [], float(cfg.equity_usdt), 0.0

    df_1m = df_1m.copy()
    df_1m["session_day"] = df_1m["open_time"].apply(
        lambda ms: session_day_str(int(ms), tz=sess.session_tz, session_open_time=sess.session_open_time)
    )
    df_5m = resample_ohlcv(df_1m, "5m")
    if df_5m.empty:
        return [], float(cfg.equity_usdt), 0.0
    df_5m["session_day"] = df_5m["open_time"].apply(
        lambda ms: session_day_str(int(ms), tz=sess.session_tz, session_open_time=sess.session_open_time)
    )

    equity = float(cfg.equity_usdt)
    peak = equity
    max_dd = 0.0
    trades: list[Ib50Trade] = []
    days = sorted(df_5m["session_day"].unique())

    pos_side = 0
    entry_ms = 0
    entry_px = 0.0
    entry_qty = 0.0
    stop_px = 0.0
    tp_px = 0.0
    first_extreme = ""
    ib_high = ib_low = 0.0
    traded_today = False
    session_date = ""
    prev_h = prev_l = 0.0

    for day in days:
        if cfg.macro_filter and is_macro_skip_day(day):
            continue
        day_1m = df_1m[df_1m["session_day"] == day]
        day_5m = df_5m[df_5m["session_day"] == day]
        if day_5m.empty:
            continue

        ib_high = ib_low = 0.0
        first_extreme = ""
        ib_ready = False
        traded_today = False
        session_date = day

        src = day_1m if not day_1m.empty else day_5m
        for _, row in src.iterrows():
            ms = int(row["open_time"])
            if not in_regular_session(sess, now_ms=ms):
                continue
            anchor = session_anchor_ms(ms, tz=sess.session_tz, session_open_time=sess.session_open_time)
            if not in_ib_window(ms, anchor_ms=anchor, ib_minutes=int(cfg.ib_minutes)):
                if ib_complete_at_bar(ms, anchor_ms=anchor, ib_minutes=int(cfg.ib_minutes)):
                    ib_ready = ib_high > ib_low
                continue
            ext = first_extreme or None
            ib_high, ib_low, ext = update_ib_range(
                ib_high=ib_high,
                ib_low=ib_low,
                first_extreme=ext,
                open_=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
            )
            if ext:
                first_extreme = ext

        ib_end_min = None
        for _, row in day_5m.iterrows():
            ms = int(row["open_time"])
            if not in_regular_session(sess, now_ms=ms):
                continue
            ts = pd.Timestamp(ms, unit="ms", tz=sess.session_tz)
            if not weekday_allowed(int(ts.weekday()), allowed_days):
                continue
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])

            if pos_side != 0:
                if should_eod_flat_bar(
                    bar_ms=ms,
                    ts=ts,
                    cfg=sess,
                    exit_hour=int(cfg.exit_hour),
                    exit_minute=int(cfg.exit_minute),
                ):
                    reason = "eod_flat"
                    raw_exit = c
                else:
                    reason = bar_exit_reason(
                        side=pos_side,
                        high=h,
                        low=l,
                        stop=stop_px,
                        target=tp_px,
                        prev_high=prev_h,
                        prev_low=prev_l,
                    ) or ""
                    if not reason:
                        prev_h, prev_l = h, l
                        continue
                    raw_exit = stop_px if reason == "stop_loss" else tp_px

                exit_px = _slip(raw_exit, side=pos_side, is_entry=False, bps=slippage_bps)
                gross = (exit_px - entry_px) * entry_qty * pos_side
                fee = trade_fee_usdt((entry_px + exit_px) * entry_qty / 2.0, taker_bps=taker_bps)
                net = gross - fee
                if entry_ms >= window_start_ms:
                    trades.append(
                        Ib50Trade(
                            symbol=symbol,
                            session_date=session_date,
                            side="LONG" if pos_side > 0 else "SHORT",
                            entry_ms=entry_ms,
                            exit_ms=ms,
                            entry_price=entry_px,
                            exit_price=exit_px,
                            exit_reason=reason,
                            first_extreme=first_extreme,
                            ib_high=ib_high,
                            ib_low=ib_low,
                            qty=entry_qty,
                            pnl_net=net,
                        )
                    )
                if cfg.compound:
                    equity += net
                    peak = max(peak, equity)
                    max_dd = max(max_dd, peak - equity)
                pos_side = 0
                prev_h = prev_l = 0.0
                continue

            if not ib_ready or ib_high <= ib_low:
                prev_h, prev_l = h, l
                continue

            anchor = session_anchor_ms(ms, tz=sess.session_tz, session_open_time=sess.session_open_time)
            ib_end_ms = anchor + int(cfg.ib_minutes) * 60_000
            if ms < ib_end_ms:
                prev_h, prev_l = h, l
                continue
            if ib_end_min is None:
                ib_end_ts = pd.Timestamp(ib_end_ms, unit="ms", tz=sess.session_tz)
                ib_end_min = ib_end_ts.hour * 60 + ib_end_ts.minute
            t_min = ts.hour * 60 + ts.minute
            end = int(cfg.entry_end_hour) * 60 + int(cfg.entry_end_minute)
            if ms < window_start_ms or t_min < int(ib_end_min) or t_min > end:
                prev_h, prev_l = h, l
                continue
            if cfg.one_trade_per_session and traded_today:
                prev_h, prev_l = h, l
                continue

            ib = finalize_initial_balance(
                ib_high=ib_high,
                ib_low=ib_low,
                first_extreme=first_extreme or None,
            )
            if ib is None:
                prev_h, prev_l = h, l
                continue

            setup = build_ib50_setup(ib, c, direction_mode=cfg.direction_mode)
            stop_dist = abs(c - setup.stop)
            if stop_dist <= 0:
                prev_h, prev_l = h, l
                continue

            eq = equity if cfg.compound else float(cfg.equity_usdt)
            qty = fixed_size_for_ib50(cfg, symbol, c, stop_distance=stop_dist, equity_usdt=eq)
            if qty <= 0:
                prev_h, prev_l = h, l
                continue

            entry_px = _slip(c, side=setup.side, is_entry=True, bps=slippage_bps)
            pos_side = setup.side
            entry_ms = ms
            entry_qty = qty
            stop_px = setup.stop
            tp_px = setup.target
            traded_today = True
            prev_h, prev_l = h, l

    return trades, equity, max_dd


def _summarize(trades: list[Ib50Trade]) -> dict:
    if not trades:
        return {"trades": 0, "win_rate": 0.0, "net": 0.0, "pf": 0.0}
    wins = [t for t in trades if t.pnl_net > 0]
    losses = [t for t in trades if t.pnl_net <= 0]
    gross_win = sum(t.pnl_net for t in wins)
    gross_loss = abs(sum(t.pnl_net for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    return {
        "trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "net": sum(t.pnl_net for t in trades),
        "pf": pf,
        "avg_win": gross_win / len(wins) if wins else 0.0,
        "avg_loss": -gross_loss / len(losses) if losses else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="IB50 Initial Balance backtest")
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--equity", type=float, default=500.0)
    p.add_argument("--slippage-bps", type=float, default=10.0)
    p.add_argument("--symbols", default="", help="Comma-separated override")
    p.add_argument("--direction", default="", help="continuation|inverse")
    p.add_argument("--weekdays", default="", help="e.g. mon,tue,thu")
    args = p.parse_args()

    days = max(7, int(args.days))
    warmup = 30
    fetch_days = days + warmup
    end_ms = int(time.time() * 1000)
    window_start_ms = end_ms - days * 86_400_000

    cfg = Ib50Config.from_env()
    overrides = {"equity_usdt": float(args.equity), "compound": True}
    if args.direction:
        overrides["direction_mode"] = args.direction
    if args.weekdays:
        overrides["allowed_weekdays"] = args.weekdays
    cfg = Ib50Config(**{**cfg.__dict__, **overrides})

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] or cfg.symbol_list()
    t0 = pd.Timestamp(window_start_ms, unit="ms", tz="America/New_York").strftime("%Y-%m-%d")
    print(f"IB50 backtest {days}d from {t0} | direction={cfg.direction_mode} | weekdays={cfg.allowed_weekdays or 'all'}")

    all_trades: list[Ib50Trade] = []
    for sym in symbols:
        df = fetch_1m(sym, days=fetch_days)
        if df.empty:
            print(f"[skip] {sym} no data")
            continue
        trades, eq, dd = simulate_symbol(
            df,
            symbol=sym,
            cfg=cfg,
            window_start_ms=window_start_ms,
            slippage_bps=float(args.slippage_bps),
        )
        s = _summarize(trades)
        print(
            f"{sym}: trades={s['trades']} wr={s['win_rate']:.1%} net={s['net']:+.2f} "
            f"pf={s['pf']:.2f} eq={eq:.2f} max_dd={dd:.2f}"
        )
        all_trades.extend(trades)

    total = _summarize(all_trades)
    print(
        f"\nTOTAL: trades={total['trades']} wr={total['win_rate']:.1%} "
        f"net={total['net']:+.2f} pf={total['pf']:.2f}"
    )


if __name__ == "__main__":
    main()
