#!/usr/bin/env python3
"""Breakout Donchian backtest — 1D execute + 1W confirm + 1H bonus sizing."""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import List, Sequence

import pandas as pd

from quant.breakout_donchian.bars import BarRow, klines_df_to_bars, resample_weekly_from_daily
from quant.breakout_donchian.config import BreakoutDonchianConfig
from quant.breakout_donchian.core import bar_exit_reason, detect_donchian_signal
from quant.breakout_donchian.resonance import evaluate_resonance
from quant.breakout_donchian.sizing import size_for_donchian
from quant.common.fees import fee_taker_bps_from_env, trade_fee_usdt
from quant.common.kline_cache import load_klines, norm_symbol, save_klines
from quant.market import fetch_klines_forward, klines_to_df

_DAY_MS = 86_400_000
_WEEK_MS = 604_800_000


@dataclass
class Trade:
    symbol: str
    side: str
    entry_ms: int
    exit_ms: int
    entry_price: float
    exit_price: float
    exit_reason: str
    tier: str
    qty: float
    pnl_net: float


def _slip(px: float, *, side: int, is_entry: bool, bps: float) -> float:
    s = max(0.0, bps) / 10_000.0
    if is_entry:
        return px * (1.0 + s) if side > 0 else px * (1.0 - s)
    return px * (1.0 - s) if side > 0 else px * (1.0 + s)


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
    start_ms = end_ms - max(1, days) * _DAY_MS
    cached = load_klines(sym, interval, start_ms=start_ms, end_ms=end_ms)
    span = 0.0
    if not cached.empty:
        span = (cached["open_time"].max() - cached["open_time"].min()) / _DAY_MS
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


def _weekly_as_of(daily: Sequence[BarRow]) -> List[BarRow]:
    if not daily:
        return []
    weekly = resample_weekly_from_daily(daily)
    if len(weekly) < 2:
        return weekly
    last_daily_ms = int(daily[-1][0])
    last_week_ms = int(weekly[-1][0])
    if last_daily_ms + _DAY_MS < last_week_ms + _WEEK_MS:
        return weekly[:-1]
    return weekly


def _hourly_as_of(hourly: Sequence[BarRow], day_open_ms: int) -> List[BarRow]:
    cutoff = int(day_open_ms) + _DAY_MS
    return [b for b in hourly if int(b[0]) < cutoff]


def _detect_kwargs(cfg: BreakoutDonchianConfig) -> dict:
    mode = "strict" if str(cfg.breakout_mode).lower() == "strict" else "standard"
    direction = "bullish" if cfg.long_only else None
    return {
        "lookback": int(cfg.lookback),
        "vol_lookback": int(cfg.vol_lookback),
        "vol_mult": float(cfg.vol_mult),
        "strong_close_pct": float(cfg.strong_close_pct),
        "mode": mode,
        "strict_vol_mult": float(cfg.strict_vol_mult),
        "strict_atr_mult": float(cfg.strict_atr_mult),
        "atr_period": int(cfg.atr_period),
        "direction_filter": direction,
        "tp1_rr": float(cfg.tp1_rr),
        "tp2_rr": float(cfg.tp2_rr),
        "tp3_rr": float(cfg.tp3_rr),
        "sl_atr_mult": float(cfg.sl_atr_mult),
        "sl_level_buffer": float(cfg.sl_level_buffer),
    }


def simulate_symbol(
    daily_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    *,
    symbol: str,
    cfg: BreakoutDonchianConfig,
    window_start_ms: int,
    slippage_bps: float,
    taker_bps: float,
) -> tuple[list[Trade], float, float]:
    daily = klines_df_to_bars(daily_df)
    hourly_all = klines_df_to_bars(hourly_df)
    if len(daily) < max(60, cfg.lookback + cfg.vol_lookback + 10):
        return [], float(cfg.equity_usdt), 0.0

    equity = float(cfg.equity_usdt)
    peak = equity
    max_dd = 0.0
    trades: list[Trade] = []
    pos_side = 0
    entry_ms = 0
    entry_px = 0.0
    entry_qty = 0.0
    stop_px = 0.0
    tp1_px = 0.0
    tp2_px = 0.0
    tp3_px = 0.0
    prev_h = 0.0
    prev_l = 0.0
    entry_tier = "dual"

    detect_kw = _detect_kwargs(cfg)

    for i in range(len(daily)):
        bar = daily[i]
        bar_ms, o, h, l, c = int(bar[0]), float(bar[1]), float(bar[2]), float(bar[3]), float(bar[4])
        daily_slice = daily[: i + 1]
        weekly_slice = _weekly_as_of(daily_slice)
        hourly_slice = _hourly_as_of(hourly_all, bar_ms) if cfg.check_1h_bonus else None

        signal = detect_donchian_signal(daily_slice, **detect_kw)
        resonance = evaluate_resonance(cfg, weekly_bars=weekly_slice, hourly_bars=hourly_slice)

        if pos_side != 0:
            flip = (
                signal is not None
                and cfg.signal_flip_exit
                and int(signal.side) != int(pos_side)
            )
            if flip:
                exit_px = _slip(c, side=pos_side, is_entry=False, bps=slippage_bps)
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
                            exit_reason="signal_flip",
                            tier=entry_tier,
                            qty=entry_qty,
                            pnl_net=net,
                        )
                    )
                if cfg.compound:
                    equity += net
                    peak = max(peak, equity)
                    max_dd = max(max_dd, peak - equity)
                pos_side = 0
                prev_h, prev_l = h, l
            else:
                hit = bar_exit_reason(
                    side=pos_side,
                    high=h,
                    low=l,
                    stop=stop_px,
                    tp1=tp1_px,
                    tp2=tp2_px,
                    tp3=tp3_px,
                    prev_high=prev_h,
                    prev_low=prev_l,
                    exit_target=str(cfg.exit_target or "tp1"),
                )
                if hit:
                    if hit == "sl":
                        raw_exit = stop_px
                    elif hit == "tp2":
                        raw_exit = tp2_px
                    elif hit == "tp3":
                        raw_exit = tp3_px
                    else:
                        raw_exit = tp1_px
                    exit_px = _slip(float(raw_exit), side=pos_side, is_entry=False, bps=slippage_bps)
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
                                tier=entry_tier,
                                qty=entry_qty,
                                pnl_net=net,
                            )
                        )
                    if cfg.compound:
                        equity += net
                        peak = max(peak, equity)
                        max_dd = max(max_dd, peak - equity)
                    pos_side = 0
                prev_h, prev_l = h, l

        if pos_side != 0 or bar_ms < window_start_ms:
            if pos_side == 0:
                prev_h, prev_l = h, l
            continue

        if signal is None:
            prev_h, prev_l = h, l
            continue
        if cfg.require_weekly_confirm and not resonance.weekly_ok:
            prev_h, prev_l = h, l
            continue

        risk_mult = resonance.risk_mult if resonance.risk_mult > 0 else float(cfg.risk_mult_base)
        stop_dist = abs(signal.entry - signal.stop)
        eq = equity if cfg.compound else float(cfg.equity_usdt)
        qty = size_for_donchian(
            cfg,
            signal.entry,
            stop_distance=stop_dist,
            equity_usdt=eq,
            risk_mult=risk_mult,
            symbol=symbol,
        )
        if qty <= 0:
            prev_h, prev_l = h, l
            continue

        entry_px = _slip(signal.entry, side=signal.side, is_entry=True, bps=slippage_bps)
        pos_side = signal.side
        entry_ms = bar_ms
        entry_qty = qty
        stop_px = float(signal.stop)
        tp1_px = float(signal.tp1)
        tp2_px = float(signal.tp2)
        tp3_px = float(signal.tp3)
        entry_tier = resonance.tier if resonance.tier != "none" else "dual"
        prev_h, prev_l = h, l

    if pos_side != 0:
        last = daily[-1]
        bar_ms = int(last[0])
        c = float(last[4])
        exit_px = _slip(c, side=pos_side, is_entry=False, bps=slippage_bps)
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
                    exit_reason="eod",
                    tier=entry_tier,
                    qty=entry_qty,
                    pnl_net=net,
                )
            )
        if cfg.compound:
            equity += net

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
    by_tier: dict[str, float] = {}
    for t in trades:
        by_tier[t.tier] = by_tier.get(t.tier, 0.0) + t.pnl_net
    if by_tier:
        parts = [f"{k} {v:+.1f}" for k, v in sorted(by_tier.items())]
        print(f"  tier: {' | '.join(parts)}")
    by_reason: dict[str, float] = {}
    for t in trades:
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0.0) + t.pnl_net
    if by_reason:
        parts = [f"{k} {v:+.1f}" for k, v in sorted(by_reason.items(), key=lambda x: -abs(x[1]))]
        print(f"  exit: {' | '.join(parts)}")
    print(f"  final equity (compound): {equity:.2f} USDT")


def main() -> None:
    p = argparse.ArgumentParser(description="Breakout Donchian 1D+1W+1H backtest")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--equity", type=float, default=100.0)
    p.add_argument("--slippage-bps", type=float, default=10.0)
    p.add_argument("--symbols", default="", help="Comma-separated override")
    p.add_argument("--no-weekly", action="store_true", help="Disable 1W confirm gate")
    p.add_argument("--no-1h", action="store_true", help="Disable 1H bonus sizing")
    p.add_argument(
        "--weekly-mode",
        default="",
        help="trend|pool|strict|off (default from config: trend)",
    )
    args = p.parse_args()

    days = max(30, int(args.days))
    warmup = max(90, int(BreakoutDonchianConfig().init_bar_days))
    fetch_days = days + warmup
    end_ms = int(time.time() * 1000)
    window_start_ms = end_ms - days * _DAY_MS
    taker_bps = fee_taker_bps_from_env()
    slip = float(args.slippage_bps)

    cfg = BreakoutDonchianConfig.from_env()
    overrides = {
        "equity_usdt": float(args.equity),
        "compound": True,
        "use_scanner_watchlist": False,
        "require_weekly_confirm": not args.no_weekly,
        "check_1h_bonus": not args.no_1h,
    }
    if args.weekly_mode:
        overrides["weekly_confirm_mode"] = args.weekly_mode
    cfg = BreakoutDonchianConfig(**{**cfg.__dict__, **overrides})

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] or cfg.symbol_list()
    t0 = pd.Timestamp(window_start_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d")
    t1 = pd.Timestamp(end_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d")
    print(
        f"Breakout Donchian backtest | window={days}d ({t0} -> {t1}) | "
        f"equity={args.equity}/symbol compound | 1W={cfg.require_weekly_confirm}"
        f"({cfg.weekly_confirm_mode}) 1H_bonus={cfg.check_1h_bonus} exit={cfg.exit_target}"
    )
    print(f"slippage={slip}bps  taker_fee={taker_bps}bps  warmup={warmup}d")

    all_trades: list[Trade] = []
    total_start = 0.0
    total_final = 0.0
    max_dd_all = 0.0

    for raw in symbols:
        sym = norm_symbol(raw)
        daily_df = fetch_df(raw, "1d", days=fetch_days)
        if daily_df.empty:
            print(f"[skip] {sym} no 1d data")
            continue
        hdays = max(60, fetch_days * 2)
        hourly_df = fetch_df(raw, "1h", days=hdays)
        trades, eq, dd = simulate_symbol(
            daily_df,
            hourly_df,
            symbol=sym,
            cfg=cfg,
            window_start_ms=window_start_ms,
            slippage_bps=slip,
            taker_bps=taker_bps,
        )
        all_trades.extend(trades)
        total_start += float(args.equity)
        total_final += eq
        max_dd_all = max(max_dd_all, dd)
        sp = sum(t.pnl_net for t in trades)
        print(f"  {sym}: {len(trades)} trades  pnl={sp:+.2f} USDT")

    _summarize(
        "Breakout Donchian (1D+1W+1H)",
        all_trades,
        equity=total_final,
        start_eq=total_start or float(args.equity),
        max_dd=max_dd_all,
    )


if __name__ == "__main__":
    main()
