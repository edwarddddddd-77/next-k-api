#!/usr/bin/env python3
"""Quick IB50 backtest on cached 5m bars (vectorized)."""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import pandas as pd

from quant.common.fees import trade_fee_usdt
from quant.common.kline_cache import load_klines, norm_symbol
from quant.common.macro_calendar import is_macro_skip_day
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


@dataclass
class TradeRow:
    sym: str
    day: str
    side: str
    net: float
    reason: str


def _slip(px: float, side: int, *, is_entry: bool, bps: float = 10.0) -> float:
    s = bps / 10_000.0
    if is_entry:
        return px * (1.0 + s) if side > 0 else px * (1.0 - s)
    return px * (1.0 - s) if side > 0 else px * (1.0 + s)


def _prep_df(sym: str, cfg: Ib50Config, window_start_ms: int) -> pd.DataFrame:
    sess = cfg.session_cfg()
    df = load_klines(norm_symbol(sym), "5m")
    if df.empty:
        return df
    df = df[df.open_time >= window_start_ms - 30 * 86_400_000].copy()
    tz = sess.session_tz
    open_time = sess.session_open_time
    df["day"] = df.open_time.map(lambda ms: session_day_str(int(ms), tz=tz, session_open_time=open_time))
    mask = df.open_time.map(lambda ms: in_regular_session(sess, now_ms=int(ms)))
    return df[mask].reset_index(drop=True)


def simulate(sym: str, cfg: Ib50Config, window_start_ms: int, *, slip_bps: float = 10.0) -> list[TradeRow]:
    df = _prep_df(sym, cfg, window_start_ms)
    if df.empty:
        return []
    sess = cfg.session_cfg()
    taker = float(cfg.fee_taker_bps)
    eq = float(cfg.equity_usdt)
    trades: list[TradeRow] = []
    pos = 0
    entry_ms = entry_px = qty = stop = tp = 0.0
    prev_h = prev_l = 0.0
    allowed = cfg.weekday_filter()

    for day, daydf in df.groupby("day", sort=True):
        if cfg.macro_filter and is_macro_skip_day(day):
            continue
        ib_h = ib_l = 0.0
        first = ""
        ready = False
        traded = False

        for row in daydf.itertuples(index=False):
            ms = int(row.open_time)
            anc = session_anchor_ms(ms, tz=sess.session_tz, session_open_time=sess.session_open_time)
            if in_ib_window(ms, anchor_ms=anc, ib_minutes=int(cfg.ib_minutes)):
                ext = first or None
                ib_h, ib_l, ext = update_ib_range(
                    ib_high=ib_h,
                    ib_low=ib_l,
                    first_extreme=ext,
                    open_=float(row.open),
                    high=float(row.high),
                    low=float(row.low),
                )
                if ext:
                    first = ext
            elif ib_complete_at_bar(ms, anchor_ms=anc, ib_minutes=int(cfg.ib_minutes)):
                ready = ib_h > ib_l

        ib_end_min: int | None = None
        for row in daydf.itertuples(index=False):
            ms = int(row.open_time)
            h, l, c = float(row.high), float(row.low), float(row.close)
            ts = pd.Timestamp(ms, unit="ms", tz=sess.session_tz)
            if not weekday_allowed(int(ts.weekday()), allowed):
                continue

            if pos:
                if should_eod_flat_bar(
                    bar_ms=ms,
                    ts=ts,
                    cfg=sess,
                    exit_hour=int(cfg.exit_hour),
                    exit_minute=int(cfg.exit_minute),
                ):
                    reason = "eod_flat"
                    raw = c
                else:
                    reason = (
                        bar_exit_reason(
                            side=pos,
                            high=h,
                            low=l,
                            stop=stop,
                            target=tp,
                            prev_high=prev_h,
                            prev_low=prev_l,
                        )
                        or ""
                    )
                    if not reason:
                        prev_h, prev_l = h, l
                        continue
                    raw = stop if reason == "stop_loss" else tp
                ex = _slip(raw, pos, is_entry=False, bps=slip_bps)
                gross = (ex - entry_px) * qty * pos
                fee = trade_fee_usdt((entry_px + ex) * qty / 2.0, taker_bps=taker)
                net = gross - fee
                if entry_ms >= window_start_ms:
                    trades.append(TradeRow(sym, day, "LONG" if pos > 0 else "SHORT", net, reason))
                if cfg.compound:
                    eq += net
                pos = 0
                prev_h = prev_l = 0.0
                continue

            if not ready or ib_h <= ib_l or traded:
                prev_h, prev_l = h, l
                continue
            anc = session_anchor_ms(ms, tz=sess.session_tz, session_open_time=sess.session_open_time)
            ib_end = anc + int(cfg.ib_minutes) * 60_000
            if ms < ib_end:
                prev_h, prev_l = h, l
                continue
            if ib_end_min is None:
                t0 = pd.Timestamp(ib_end, unit="ms", tz=sess.session_tz)
                ib_end_min = t0.hour * 60 + t0.minute
            tmin = ts.hour * 60 + ts.minute
            end = int(cfg.entry_end_hour) * 60 + int(cfg.entry_end_minute)
            if ms < window_start_ms or tmin < ib_end_min or tmin > end:
                prev_h, prev_l = h, l
                continue
            ib = finalize_initial_balance(ib_high=ib_h, ib_low=ib_l, first_extreme=first or None)
            if ib is None:
                prev_h, prev_l = h, l
                continue
            setup = build_ib50_setup(ib, c, direction_mode=cfg.direction_mode)
            sd = abs(c - setup.stop)
            if sd <= 0:
                prev_h, prev_l = h, l
                continue
            q = fixed_size_for_ib50(
                cfg,
                sym,
                c,
                stop_distance=sd,
                equity_usdt=eq if cfg.compound else float(cfg.equity_usdt),
            )
            if q <= 0:
                prev_h, prev_l = h, l
                continue
            pos = setup.side
            entry_ms = ms
            entry_px = _slip(c, pos, is_entry=True, bps=slip_bps)
            qty = q
            stop = setup.stop
            tp = setup.target
            traded = True
            prev_h, prev_l = h, l
    return trades


def _summary(trades: list[TradeRow]) -> str:
    if not trades:
        return "trades=0"
    wins = [t for t in trades if t.net > 0]
    loss = [t for t in trades if t.net <= 0]
    gw = sum(t.net for t in wins)
    gl = abs(sum(t.net for t in loss))
    pf = gw / gl if gl else float("inf")
    return (
        f"trades={len(trades)} wr={len(wins)/len(trades):.1%} "
        f"net={sum(t.net for t in trades):+.2f} pf={pf:.2f}"
    )


def _run_label(cfg: Ib50Config, syms: list[str], start_ms: int, label: str) -> list[TradeRow]:
    print(f"--- {label} ---", flush=True)
    all_t: list[TradeRow] = []
    for s in syms:
        t = simulate(s, cfg, start_ms)
        all_t.extend(t)
        print(f"  {s}: {_summary(t)}", flush=True)
    print(f"  POOL: {_summary(all_t)}\n", flush=True)
    return all_t


def main() -> None:
    days = 180
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000
    base = Ib50Config.from_env()
    cfg = Ib50Config(**{**base.__dict__, "equity_usdt": 500.0, "compound": True})
    syms = ["INTC", "COIN", "MSTR", "PLTR", "HOOD", "SOXL"]

    print(f"IB50 backtest {days}d | 5m bars | equity=$500 | risk ~1%/trade", flush=True)
    print(f"Symbols: {', '.join(syms)}\n", flush=True)

    _run_label(cfg, syms, start_ms, "continuation, all weekdays")
    _run_label(
        Ib50Config(**{**cfg.__dict__, "allowed_weekdays": "mon,tue,thu"}),
        syms,
        start_ms,
        "continuation, Mon/Tue/Thu",
    )
    _run_label(
        Ib50Config(**{**cfg.__dict__, "direction_mode": "inverse", "allowed_weekdays": "wed,fri"}),
        syms,
        start_ms,
        "inverse, Wed/Fri",
    )


if __name__ == "__main__":
    main()
