#!/usr/bin/env python3
"""Trading ORB backtest — aligns with trading_orb_vnpy (1m OR + 5m entry/exit)."""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import pandas as pd

from quant.common.config import OrbConfig
from quant.common.fees import trade_fee_usdt
from quant.common.kline_cache import load_klines, norm_symbol, save_klines
from quant.common.macro_calendar import is_macro_skip_day
from quant.common.resample import resample_ohlcv
from quant.common.session import session_day_str, session_anchor_ms
from quant.common.session_paper import in_regular_session
from quant.engine.eod import should_eod_flat_bar
from quant.market import fetch_klines_forward, klines_to_df
from quant.trading_orb.config import OrbVnpyConfig
from quant.trading_orb.sizing import fixed_size_for_orb


@dataclass
class OrbTrade:
    symbol: str
    session_date: str
    side: str
    entry_ms: int
    exit_ms: int
    entry_price: float
    exit_price: float
    exit_reason: str
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
        # fall through to 5m cache expansion
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
    # fallback: expand 5m -> synthetic 1m for OR window
    df5 = load_klines(sym, "5m", start_ms=start_ms, end_ms=end_ms)
    if df5.empty and (os.getenv("BACKTEST_CACHE_ONLY") or "").strip() not in ("1", "true", "yes"):
        for ex in (exchange, "bitget", "binance"):
            print(f"[fetch] {sym} 5m {days}d from {ex} (1m fallback) ...")
            rows = fetch_klines_forward(sym, "5m", start_ms, end_ms, exchange_id=ex)
            df5 = klines_to_df(rows)
            if not df5.empty:
                save_klines(sym, "5m", df5)
                break
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


def _time_key(ms: int, tz: str) -> str:
    return pd.Timestamp(int(ms), unit="ms", tz=tz).strftime("%H:%M")


def _vol_baselines_for_day(
    hist_5m: pd.DataFrame,
    *,
    day: str,
    tz: str,
) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for _, row in hist_5m.iterrows():
        key = _time_key(int(row["open_time"]), tz)
        buckets.setdefault(key, []).append(float(row["volume"]))
    return {k: sum(v) / len(v) for k, v in buckets.items() if v}


def _update_or(
    *,
    or_high: float,
    or_low: float,
    hi: float,
    lo: float,
    bar_ms: int,
    anchor_ms: int,
    or_end_ms: int,
) -> tuple[float, float, float]:
    if bar_ms < anchor_ms or bar_ms >= or_end_ms:
        if or_high > 0 and or_low > 0:
            return or_high, or_low, or_high - or_low
        return or_high, or_low, 0.0
    if or_high <= 0:
        return hi, lo, hi - lo
    return max(or_high, hi), min(or_low, lo), max(or_high, hi) - min(or_low, lo)


def simulate_symbol(
    df_1m: pd.DataFrame,
    *,
    symbol: str,
    cfg: OrbVnpyConfig,
    window_start_ms: int,
    slippage_bps: float,
) -> tuple[list[OrbTrade], float, float]:
    sess = cfg.orb_session_cfg()
    taker_bps = float(cfg.fee_taker_bps)
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
    trades: list[OrbTrade] = []

    days = sorted(df_5m["session_day"].unique())

    pos_side = 0
    entry_ms = 0
    entry_px = 0.0
    entry_qty = 0.0
    stop_px = 0.0
    tp_px = 0.0
    or_range_at_entry = 0.0
    breakeven = False
    traded_today = False
    session_date = ""

    hist_5m = df_5m[df_5m["open_time"].apply(lambda ms: in_regular_session(sess, now_ms=int(ms)))].copy()

    for day in days:
        if cfg.macro_filter and is_macro_skip_day(day):
            continue
        day_1m = df_1m[df_1m["session_day"] == day]
        day_5m = df_5m[df_5m["session_day"] == day]
        if day_5m.empty:
            continue

        vol_base = _vol_baselines_for_day(hist_5m[hist_5m["session_day"] < day], day=day, tz=sess.session_tz)

        or_high = or_low = or_range = 0.0
        traded_today = False
        session_date = day

        # Build OR from 1m (fallback: first 4x5m if no 1m)
        src = day_1m if not day_1m.empty else day_5m
        bar_step = 60_000 if not day_1m.empty else 300_000
        for _, row in src.iterrows():
            ms = int(row["open_time"])
            if not in_regular_session(sess, now_ms=ms):
                continue
            anchor = session_anchor_ms(ms, tz=sess.session_tz, session_open_time=sess.session_open_time)
            or_end = anchor + int(cfg.or_minutes) * 60_000
            or_high, or_low, or_range = _update_or(
                or_high=or_high,
                or_low=or_low,
                hi=float(row["high"]),
                lo=float(row["low"]),
                bar_ms=ms,
                anchor_ms=anchor,
                or_end_ms=or_end,
            )

        for _, row in day_5m.iterrows():
            ms = int(row["open_time"])
            if not in_regular_session(sess, now_ms=ms):
                continue
            ts = pd.Timestamp(ms, unit="ms", tz=sess.session_tz)
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            vol = float(row["volume"])

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
                    reason = ""
                    if pos_side > 0:
                        if l <= stop_px:
                            reason = "stop_loss"
                            raw_exit = stop_px
                        elif tp_px > 0 and h >= tp_px:
                            reason = "target_hit"
                            raw_exit = tp_px
                    else:
                        if h >= stop_px:
                            reason = "stop_loss"
                            raw_exit = stop_px
                        elif tp_px > 0 and l <= tp_px:
                            reason = "target_hit"
                            raw_exit = tp_px
                    if not reason:
                        if not breakeven and or_range_at_entry > 0:
                            fav = (c - entry_px) * pos_side
                            if fav >= float(cfg.breakeven_or_mult) * or_range_at_entry:
                                tick = 0.01
                                if pos_side > 0:
                                    stop_px = max(stop_px, entry_px + tick)
                                else:
                                    stop_px = min(stop_px, entry_px - tick)
                                breakeven = True
                        continue

                exit_px = _slip(raw_exit, side=pos_side, is_entry=False, bps=slippage_bps)
                gross = (exit_px - entry_px) * entry_qty * pos_side
                fee = trade_fee_usdt((entry_px + exit_px) * entry_qty / 2.0, taker_bps=taker_bps)
                net = gross - fee
                if entry_ms >= window_start_ms:
                    trades.append(
                        OrbTrade(
                            symbol=symbol,
                            session_date=session_date,
                            side="LONG" if pos_side > 0 else "SHORT",
                            entry_ms=entry_ms,
                            exit_ms=ms,
                            entry_price=entry_px,
                            exit_price=exit_px,
                            exit_reason=reason,
                            qty=entry_qty,
                            pnl_net=net,
                        )
                    )
                if cfg.compound:
                    equity += net
                    peak = max(peak, equity)
                    max_dd = max(max_dd, peak - equity)
                pos_side = 0
                breakeven = False
                continue

            # entry window 10:00-11:30 ET
            t_min = ts.hour * 60 + ts.minute
            start = int(cfg.entry_start_hour) * 60 + int(cfg.entry_start_minute)
            end = int(cfg.entry_end_hour) * 60 + int(cfg.entry_end_minute)
            if ms < window_start_ms or t_min < start or t_min > end:
                continue
            if cfg.one_trade_per_session and traded_today:
                continue
            if or_range <= 0 or or_high <= or_low:
                continue

            key = _time_key(ms, sess.session_tz)
            base = float(vol_base.get(key, 0.0) or 0.0)
            if base <= 0:
                continue
            rel_vol = vol / base
            if rel_vol < float(cfg.vol_thresh):
                continue

            side = 0
            if c > or_high:
                side = 1
            elif c < or_low:
                side = -1
            else:
                continue

            stop_dist = float(cfg.stop_or_mult) * or_range
            if stop_dist <= 0:
                continue
            or_range_at_entry = or_range
            if side > 0:
                stop_px = c - stop_dist
                tp_px = c + float(cfg.target_or_mult) * or_range
            else:
                stop_px = c + stop_dist
                tp_px = c - float(cfg.target_or_mult) * or_range

            eq = equity if cfg.compound else float(cfg.equity_usdt)
            qty = fixed_size_for_orb(cfg, symbol, c, stop_distance=stop_dist, equity_usdt=eq)
            if qty <= 0:
                continue
            entry_px = _slip(c, side=side, is_entry=True, bps=slippage_bps)
            pos_side = side
            entry_ms = ms
            entry_qty = qty
            traded_today = True
            breakeven = False

    return trades, equity, max_dd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=60)
    p.add_argument("--equity", type=float, default=98.0)
    p.add_argument("--slippage-bps", type=float, default=10.0)
    p.add_argument("--symbols", default="", help="Comma-separated override")
    args = p.parse_args()

    days = max(7, int(args.days))
    warmup = 30
    fetch_days = days + warmup
    end_ms = int(time.time() * 1000)
    window_start_ms = end_ms - days * 86_400_000

    cfg = OrbVnpyConfig.from_env()
    cfg = OrbVnpyConfig(**{**cfg.__dict__, "equity_usdt": float(args.equity), "compound": True})

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] or cfg.symbol_list()
    t0 = pd.Timestamp(window_start_ms, unit="ms", tz="America/New_York").strftime("%Y-%m-%d")
    t1 = pd.Timestamp(end_ms, unit="ms", tz="America/New_York").strftime("%Y-%m-%d")

    print("=" * 60)
    print("Trading ORB 回测")
    print("=" * 60)
    print(f"区间: {t0} -> {t1} ({days}d)  |  标的: {', '.join(symbols)}")
    print(f"每标的 {args.equity}U  compound  OR={cfg.or_minutes}m  vol>{cfg.vol_thresh}")
    print(f"entry {cfg.entry_start_hour:02d}:{cfg.entry_start_minute:02d}-{cfg.entry_end_hour:02d}:{cfg.entry_end_minute:02d} ET  EOD {cfg.exit_hour}:{cfg.exit_minute:02d}")

    all_trades: list[OrbTrade] = []
    start_total = 0.0
    final_total = 0.0
    max_dd = 0.0

    for raw in symbols:
        sym = norm_symbol(raw)
        df = fetch_1m(raw, days=fetch_days)
        if df.empty:
            print(f"[skip] {sym} no 1m data")
            continue
        ts, eq, dd = simulate_symbol(
            df,
            symbol=sym,
            cfg=cfg,
            window_start_ms=window_start_ms,
            slippage_bps=float(args.slippage_bps),
        )
        all_trades.extend(ts)
        start_total += float(args.equity)
        final_total += eq
        max_dd = max(max_dd, dd)
        pnl = sum(t.pnl_net for t in ts)
        n = len(ts)
        w = sum(1 for t in ts if t.pnl_net > 0)
        wr = w / n * 100 if n else 0
        print(f"  {sym}: {n}t  pnl={pnl:+.2f}  win={wr:.0f}%  final={eq:.2f}")

    pnl = sum(t.pnl_net for t in all_trades)
    wins = sum(1 for t in all_trades if t.pnl_net > 0)
    wr = wins / len(all_trades) * 100 if all_trades else 0
    ret = pnl / start_total * 100 if start_total else 0

    print(f"\n=== 合计 ===")
    print(f"  trades={len(all_trades)}  pnl={pnl:+.2f} USDT  ret={ret:+.2f}%  win={wr:.1f}%  max_dd={max_dd:.2f}")
    print(f"  start={start_total:.0f}U  final={final_total:.0f}U")

    by_reason: dict[str, float] = {}
    for t in all_trades:
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0.0) + t.pnl_net
    if by_reason:
        parts = [f"{k} {v:+.1f}" for k, v in sorted(by_reason.items(), key=lambda x: -abs(x[1]))]
        print(f"  exit: {' | '.join(parts)}")


if __name__ == "__main__":
    main()
