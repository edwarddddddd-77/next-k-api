#!/usr/bin/env python3
"""ICT Master Suite [Trading-IQ] aligned backtest on Binance futures.

Rules ported from the open-source Trading-IQ ICT Library + Master Suite docs:
  - 2022 Model: sweep -> breakout -> FVG nearest 50% range (m2022 UDT)
  - Silver Bullet: NY windows + HTF H/L/H2/L2 bias + session FVG CE
  - Liquidity Raid: 13:30-16:00 NY session raid -> breaker block entry
  - OTE: 0.79 retrace (default), -0.5 fib TP, swing SL
  - Unicorn: FVG + breaker overlap, entry = max(topBlock, FVG.top)

Backtest settings (author sample):
  $1000, 5% position, 0.02% commission, 2-tick slippage/limit verify
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi

load_env_oi()

import numpy as np
import pandas as pd

from binance_fapi import fetch_klines_forward, klines_to_df
from orb.gtl.resample import resample_ohlcv
from tools.cta.ict_tradingiq_core import (
    FVG,
    LIQUIDITY_RAID_SESSION_NY,
    M2022_ENTRY_PCT,
    OTE_LEVEL_DEFAULT,
    SILVER_BULLET_WINDOWS_NY,
    add_daily_levels,
    add_swings,
    build_order_blocks,
    detect_fvgs,
    detect_sweeps,
    in_ny_window,
    merge_htf_columns,
    ms_to_ny,
    nearest_fvg_to_level,
    ote_long_prices,
    ote_short_prices,
    unicorn_long_entry,
    unicorn_short_entry,
)


INITIAL_EQUITY = 1000.0
POSITION_PCT = 0.05
COMMISSION_PCT = 0.02
SLIPPAGE_TICKS = 2
LIMIT_VERIFY_TICKS = 2


def fetch_ohlcv(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = fetch_klines_forward(symbol, interval, start_ms, end_ms)
    df = klines_to_df(rows)
    if df.empty:
        return df
    return (
        df.drop_duplicates(subset=["open_time"], keep="last")
        .sort_values("open_time")
        .reset_index(drop=True)
    )


def tick_size(price: float) -> float:
    if price >= 10_000:
        return 0.1
    if price >= 1_000:
        return 0.01
    return max(price * 0.0001, 0.0001)


@dataclass
class Trade:
    symbol: str
    model: str
    side: str
    entry_ms: int
    entry_px: float
    exit_ms: int
    exit_px: float
    pnl_usdt: float
    pnl_pct: float
    reason: str


@dataclass
class ModelResult:
    symbol: str
    model: str
    trades: List[Trade] = field(default_factory=list)
    equity_start: float = INITIAL_EQUITY
    final_equity_run: float = 0.0

    @property
    def final_equity(self) -> float:
        if self.final_equity_run > 0:
            return self.final_equity_run
        return self.equity_start + sum(t.pnl_usdt for t in self.trades)

    @property
    def total_return_pct(self) -> float:
        return (self.final_equity / self.equity_start - 1.0) * 100.0

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.pnl_usdt > 0) / len(self.trades) * 100.0


@dataclass
class SimConfig:
    commission_pct: float = COMMISSION_PCT
    slippage_ticks: int = SLIPPAGE_TICKS
    limit_verify_ticks: int = LIMIT_VERIFY_TICKS
    position_pct: float = POSITION_PCT
    rr: float = 0.0  # 0 = model-native TP; 2 = fixed 1:2 R:R from entry/SL
    leverage: float = 1.0  # notional = equity * position_pct * leverage


def _tp_from_rr(entry: float, sl: float, side: str, rr: float) -> float:
    risk = abs(entry - sl)
    if risk <= 0:
        return entry
    return entry + rr * risk if side == "long" else entry - rr * risk


def _apply_tp(entry: float, sl: float, side: str, native_tp: float, cfg: SimConfig) -> float:
    if cfg.rr > 0:
        return _tp_from_rr(entry, sl, side, cfg.rr)
    return native_tp


def _position_usdt(equity: float, cfg: SimConfig) -> float:
    return equity * cfg.position_pct * cfg.leverage


def _commission(notional: float, cfg: SimConfig) -> float:
    return notional * cfg.commission_pct / 100.0


def _limit_fillable(px: float, hi: float, lo: float, side: str, cfg: SimConfig) -> bool:
    tick = tick_size(px) * cfg.limit_verify_ticks
    return lo <= px + tick if side == "long" else hi >= px - tick


def _sl_fill(px: float, side: str, cfg: SimConfig) -> float:
    slip = tick_size(px) * cfg.slippage_ticks
    return px - slip if side == "long" else px + slip


def _record_trade(
    res: ModelResult,
    *,
    symbol: str,
    side: str,
    entry_ms: int,
    entry_px: float,
    exit_ms: int,
    exit_px: float,
    size_usdt: float,
    reason: str,
    cfg: SimConfig,
) -> None:
    gross = size_usdt * ((exit_px - entry_px) / entry_px if side == "long" else (entry_px - exit_px) / entry_px)
    fees = _commission(size_usdt, cfg) + _commission(size_usdt * exit_px / entry_px, cfg)
    pnl = gross - fees
    margin = size_usdt / max(cfg.leverage, 1.0)
    res.trades.append(
        Trade(symbol, res.model, side, entry_ms, entry_px, exit_ms, exit_px, pnl,
              pnl / margin * 100.0 if margin > 0 else 0.0, reason)
    )


def _apply_equity(equity: float, delta: float) -> float:
    return max(0.0, equity + delta)


def _simulate_bar(
    res: ModelResult,
    pos: Optional[dict],
    *,
    t: int,
    hi: float,
    lo: float,
    equity: float,
    symbol: str,
    cfg: SimConfig,
) -> Tuple[Optional[dict], float]:
    if pos is None:
        return None, equity

    side = pos["side"]
    if pos.get("pending"):
        if _limit_fillable(pos["entry"], hi, lo, side, cfg):
            pos = {**pos, "pending": False, "entry_ms": t}
        elif t - pos["born_ms"] > pos.get("ttl_ms", 2 * 60 * 60 * 1000):
            return None, equity
        else:
            if (side == "long" and lo <= pos["sl"]) or (side == "short" and hi >= pos["sl"]):
                return None, equity
            return pos, equity

    if side == "long":
        if lo <= pos["sl"]:
            exit_px = _sl_fill(pos["sl"], side, cfg)
            _record_trade(res, symbol=symbol, side=side, entry_ms=pos["entry_ms"], entry_px=pos["entry"],
                          exit_ms=t, exit_px=exit_px, size_usdt=pos["size"], reason="sl", cfg=cfg)
            return None, _apply_equity(equity, res.trades[-1].pnl_usdt)
        if hi >= pos["tp"]:
            _record_trade(res, symbol=symbol, side=side, entry_ms=pos["entry_ms"], entry_px=pos["entry"],
                          exit_ms=t, exit_px=pos["tp"], size_usdt=pos["size"], reason="tp", cfg=cfg)
            return None, _apply_equity(equity, res.trades[-1].pnl_usdt)
    else:
        if hi >= pos["sl"]:
            exit_px = _sl_fill(pos["sl"], side, cfg)
            _record_trade(res, symbol=symbol, side=side, entry_ms=pos["entry_ms"], entry_px=pos["entry"],
                          exit_ms=t, exit_px=exit_px, size_usdt=pos["size"], reason="sl", cfg=cfg)
            return None, _apply_equity(equity, res.trades[-1].pnl_usdt)
        if lo <= pos["tp"]:
            _record_trade(res, symbol=symbol, side=side, entry_ms=pos["entry_ms"], entry_px=pos["entry"],
                          exit_ms=t, exit_px=pos["tp"], size_usdt=pos["size"], reason="tp", cfg=cfg)
            return None, _apply_equity(equity, res.trades[-1].pnl_usdt)
    return pos, equity


def _entry_allowed(fn: Callable[..., bool], t: int, side: str, entry: float) -> bool:
    try:
        return bool(fn(t, side, entry))
    except TypeError:
        return bool(fn(t))


# ---------------------------------------------------------------------------
# 2022 Model -?m2022 UDT
# ---------------------------------------------------------------------------


def backtest_2022_model(
    symbol: str,
    df: pd.DataFrame,
    *,
    test_start_ms: int,
    test_end_ms: int,
    entry_pct: float = M2022_ENTRY_PCT,
    cfg: SimConfig = SimConfig(),
    entry_allow: Optional[Callable[..., bool]] = None,
) -> ModelResult:
    res = ModelResult(symbol=symbol, model="2022_model")
    df = add_swings(df)
    sweep_low, sweep_high = detect_sweeps(df)
    all_fvgs = detect_fvgs(df)
    equity = res.equity_start
    pos: Optional[dict] = None
    setups: List[dict] = []

    for i in range(3, len(df)):
        row = df.iloc[i]
        t = int(row["open_time"])
        if t < test_start_ms:
            continue
        if t > test_end_ms:
            break
        hi, lo, cl = float(row["high"]), float(row["low"]), float(row["close"])
        live_fvgs = [g for g in all_fvgs if not g.invalidated(cl)]

        pos, equity = _simulate_bar(res, pos, t=t, hi=hi, lo=lo, equity=equity, symbol=symbol, cfg=cfg)
        if pos is not None or equity <= 0:
            continue

        if bool(sweep_low.iloc[i]):
            setups.append({"side": "long", "range_lo": lo, "sweep_ms": t, "pre_break_hi": hi,
                           "range_hi": hi, "breakout": False, "active": True})
        if bool(sweep_high.iloc[i]):
            setups.append({"side": "short", "range_hi": hi, "sweep_ms": t, "pre_break_lo": lo,
                           "range_lo": lo, "breakout": False, "active": True})
        setups = setups[-20:]

        for setup in setups:
            if not setup.get("active") or t - setup["sweep_ms"] > 6 * 60 * 60 * 1000:
                setup["active"] = False
                continue
            if setup["side"] == "long":
                if not setup["breakout"]:
                    if cl > setup["pre_break_hi"] and cl > float(row["open"]):
                        setup["breakout"] = True
                        setup["range_hi"] = hi
                        setup["breakout_ms"] = t
                    setup["pre_break_hi"] = max(setup["pre_break_hi"], hi)
                    continue
                if t - setup.get("breakout_ms", t) > 2 * 60 * 60 * 1000:
                    setup["active"] = False
                    continue
                level_50 = setup["range_lo"] + entry_pct * (setup["range_hi"] - setup["range_lo"])
                pool = [g for g in live_fvgs if g.born_ms >= setup["breakout_ms"] and g.direction == 1 and not g.used]
                g = nearest_fvg_to_level(pool, level_50, 1)
                if g is None:
                    continue
                entry = g.mid
                if entry_allow is not None and not _entry_allowed(entry_allow, t, "long", entry):
                    continue
                sl = setup["range_lo"]
                tp = _apply_tp(entry, sl, "long", setup["range_hi"], cfg)
                pos = {"side": "long", "entry": entry, "entry_ms": t, "born_ms": t, "pending": True,
                       "sl": sl, "tp": tp,
                       "size": _position_usdt(equity, cfg), "ttl_ms": 4 * 60 * 60 * 1000}
                g.used = True
                setup["active"] = False
                break
            else:
                if not setup["breakout"]:
                    if cl < setup["pre_break_lo"] and cl < float(row["open"]):
                        setup["breakout"] = True
                        setup["range_lo"] = lo
                        setup["breakout_ms"] = t
                    setup["pre_break_lo"] = min(setup["pre_break_lo"], lo)
                    continue
                if t - setup.get("breakout_ms", t) > 2 * 60 * 60 * 1000:
                    setup["active"] = False
                    continue
                level_50 = setup["range_hi"] - entry_pct * (setup["range_hi"] - setup["range_lo"])
                pool = [g for g in live_fvgs if g.born_ms >= setup["breakout_ms"] and g.direction == -1 and not g.used]
                g = nearest_fvg_to_level(pool, level_50, -1)
                if g is None:
                    continue
                entry = g.mid
                if entry_allow is not None and not _entry_allowed(entry_allow, t, "short", entry):
                    continue
                sl = setup["range_hi"]
                tp = _apply_tp(entry, sl, "short", setup["range_lo"], cfg)
                pos = {"side": "short", "entry": entry, "entry_ms": t, "born_ms": t, "pending": True,
                       "sl": sl, "tp": tp,
                       "size": _position_usdt(equity, cfg), "ttl_ms": 4 * 60 * 60 * 1000}
                g.used = True
                setup["active"] = False
                break

    if pos is not None and not pos.get("pending"):
        last = df[df["open_time"] <= test_end_ms].iloc[-1]
        _record_trade(res, symbol=symbol, side=pos["side"], entry_ms=pos["entry_ms"], entry_px=pos["entry"],
                      exit_ms=int(last["open_time"]), exit_px=float(last["close"]),
                      size_usdt=pos["size"], reason="eod", cfg=cfg)
        equity += res.trades[-1].pnl_usdt
        equity = max(0.0, equity)
    res.final_equity_run = equity
    return res


# ---------------------------------------------------------------------------
# Silver Bullet -?silverBullet() + session FVG CE
# ---------------------------------------------------------------------------


def backtest_silver_bullet(
    symbol: str,
    df: pd.DataFrame,
    df_htf: pd.DataFrame,
    *,
    test_start_ms: int,
    test_end_ms: int,
    cfg: SimConfig = SimConfig(),
) -> ModelResult:
    res = ModelResult(symbol=symbol, model="silver_bullet")
    df = merge_htf_columns(add_swings(add_daily_levels(df)), df_htf)
    all_fvgs = detect_fvgs(df)
    equity = res.equity_start
    pos: Optional[dict] = None
    session_fvgs: List[FVG] = []

    for i in range(2, len(df)):
        row = df.iloc[i]
        t = int(row["open_time"])
        if t < test_start_ms:
            continue
        if t > test_end_ms:
            break
        hi, lo, cl = float(row["high"]), float(row["low"]), float(row["close"])
        in_window = any(in_ny_window(t, a, b) for a, b in SILVER_BULLET_WINDOWS_NY)
        bias = int(row.get("bias", 0))

        if in_window:
            for g in [x for x in all_fvgs if x.born_idx == i and not x.invalidated(cl)]:
                if (g.direction == 1 and bias == 1) or (g.direction == -1 and bias == -1):
                    session_fvgs.append(g)
            session_fvgs = [g for g in session_fvgs if t - g.born_ms <= 60 * 60 * 1000][-10:]

        pos, equity = _simulate_bar(res, pos, t=t, hi=hi, lo=lo, equity=equity, symbol=symbol, cfg=cfg)
        if pos is not None:
            continue
        if equity <= 0 or not in_window:
            continue

        swing_hi = float(row.get("last_swing_hi", np.nan))
        swing_lo = float(row.get("last_swing_lo", np.nan))
        pdh = float(row.get("prev_day_high", np.nan))
        pdl = float(row.get("prev_day_low", np.nan))

        for g in session_fvgs:
            if g.used or t - g.born_ms > 60 * 60 * 1000:
                continue
            if g.direction == 1 and bias == 1:
                native_tp = swing_hi if (not np.isnan(swing_hi) and swing_hi > g.mid) else pdh
                sl = swing_lo if (not np.isnan(swing_lo) and swing_lo < g.mid) else pdl
                if np.isnan(sl) or sl >= g.mid:
                    continue
                if cfg.rr <= 0 and (np.isnan(native_tp) or native_tp <= g.mid):
                    continue
                entry = g.mid
                tp = _apply_tp(entry, sl, "long", native_tp, cfg)
                pos = {"side": "long", "entry": entry, "entry_ms": t, "born_ms": t, "pending": True,
                       "sl": sl, "tp": tp, "size": _position_usdt(equity, cfg), "ttl_ms": 60 * 60 * 1000}
                g.used = True
                break
            if g.direction == -1 and bias == -1:
                native_tp = swing_lo if (not np.isnan(swing_lo) and swing_lo < g.mid) else pdl
                sl = swing_hi if (not np.isnan(swing_hi) and swing_hi > g.mid) else pdh
                if np.isnan(sl) or sl <= g.mid:
                    continue
                if cfg.rr <= 0 and (np.isnan(native_tp) or native_tp >= g.mid):
                    continue
                entry = g.mid
                tp = _apply_tp(entry, sl, "short", native_tp, cfg)
                pos = {"side": "short", "entry": entry, "entry_ms": t, "born_ms": t, "pending": True,
                       "sl": sl, "tp": tp, "size": _position_usdt(equity, cfg), "ttl_ms": 60 * 60 * 1000}
                g.used = True
                break

    res.final_equity_run = equity
    return res


# ---------------------------------------------------------------------------
# Liquidity Raid -?session raid -> breaker block
# ---------------------------------------------------------------------------


def backtest_liquidity_raid(
    symbol: str,
    df: pd.DataFrame,
    *,
    test_start_ms: int,
    test_end_ms: int,
    session_start: str = LIQUIDITY_RAID_SESSION_NY[0],
    session_end: str = LIQUIDITY_RAID_SESSION_NY[1],
    cfg: SimConfig = SimConfig(),
) -> ModelResult:
    res = ModelResult(symbol=symbol, model="liquidity_raid")
    df = add_swings(df)
    bull_bb, bear_bb = build_order_blocks(df)
    equity = res.equity_start
    pos: Optional[dict] = None
    sessions: dict = {}
    raids: List[dict] = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        t = int(row["open_time"])
        if t < test_start_ms:
            continue
        if t > test_end_ms:
            break
        hi, lo = float(row["high"]), float(row["low"])
        day = ms_to_ny(t).strftime("%Y-%m-%d")
        in_sess = in_ny_window(t, session_start, session_end)

        if in_sess:
            s = sessions.setdefault(day, {"hi": hi, "lo": lo, "done": False})
            s["hi"], s["lo"] = max(s["hi"], hi), min(s["lo"], lo)
        elif day in sessions and not sessions[day]["done"]:
            s = sessions[day]
            if hi > s["hi"]:
                raids.append({"dir": -1, "raid_ms": t})
                s["done"] = True
            elif lo < s["lo"]:
                raids.append({"dir": 1, "raid_ms": t})
                s["done"] = True

        pos, equity = _simulate_bar(res, pos, t=t, hi=hi, lo=lo, equity=equity, symbol=symbol, cfg=cfg)
        if pos is not None or equity <= 0:
            continue

        for raid in raids:
            if raid.get("used") or t - raid["raid_ms"] > 4 * 60 * 60 * 1000:
                continue
            if raid["dir"] == 1:
                cands = [b for b in bull_bb if raid["raid_ms"] <= b.born_ms <= t]
                if not cands:
                    continue
                brk = cands[-1]
                entry = (brk.top + brk.bot) / 2.0
                if not (lo <= entry <= hi):
                    continue
                sl = brk.bot
                tp = _apply_tp(entry, sl, "long", float(row.get("last_swing_hi", hi)), cfg)
                pos = {"side": "long", "entry": entry, "entry_ms": t, "born_ms": t, "pending": True,
                       "sl": sl, "tp": tp,
                       "size": _position_usdt(equity, cfg), "ttl_ms": 2 * 60 * 60 * 1000}
                raid["used"] = True
                break
            else:
                cands = [b for b in bear_bb if raid["raid_ms"] <= b.born_ms <= t]
                if not cands:
                    continue
                brk = cands[-1]
                entry = (brk.top + brk.bot) / 2.0
                if not (lo <= entry <= hi):
                    continue
                sl = brk.top
                tp = _apply_tp(entry, sl, "short", float(row.get("last_swing_lo", lo)), cfg)
                pos = {"side": "short", "entry": entry, "entry_ms": t, "born_ms": t, "pending": True,
                       "sl": sl, "tp": tp,
                       "size": _position_usdt(equity, cfg), "ttl_ms": 2 * 60 * 60 * 1000}
                raid["used"] = True
                break
        raids = raids[-40:]

    res.final_equity_run = equity
    return res


# ---------------------------------------------------------------------------
# OTE -?OTEstrat() defaults: 0.79 entry, -0.5 TP, swing SL
# ---------------------------------------------------------------------------


def backtest_ote(
    symbol: str,
    df: pd.DataFrame,
    *,
    test_start_ms: int,
    test_end_ms: int,
    ote_level: float = OTE_LEVEL_DEFAULT,
    cfg: SimConfig = SimConfig(),
) -> ModelResult:
    res = ModelResult(symbol=symbol, model="ote")
    df = add_swings(df)
    equity = res.equity_start
    pos: Optional[dict] = None
    last_hi_idx = last_lo_idx = -1

    for i in range(1, len(df)):
        row = df.iloc[i]
        t = int(row["open_time"])
        if t < test_start_ms:
            continue
        if t > test_end_ms:
            break
        hi, lo, cl = float(row["high"]), float(row["low"]), float(row["close"])

        if bool(row["swing_hi"]):
            last_hi_idx = i
        if bool(row["swing_lo"]):
            last_lo_idx = i

        pos, equity = _simulate_bar(res, pos, t=t, hi=hi, lo=lo, equity=equity, symbol=symbol, cfg=cfg)
        if pos is not None or equity <= 0:
            continue

        if last_lo_idx > 0 and last_hi_idx > last_lo_idx:
            swing_lo = float(df.iloc[last_lo_idx]["low"])
            swing_hi = float(df.iloc[last_hi_idx]["high"])
            if swing_hi > swing_lo:
                entry, sl, native_tp = ote_long_prices(swing_lo, swing_hi, ote_level)
                tp = _apply_tp(entry, sl, "long", native_tp, cfg)
                pos = {"side": "long", "entry": entry, "entry_ms": t, "born_ms": t, "pending": True,
                       "sl": sl, "tp": tp, "size": _position_usdt(equity, cfg), "ttl_ms": 4 * 60 * 60 * 1000}
                last_hi_idx = last_lo_idx = -1
                continue

        if last_hi_idx > 0 and last_lo_idx > last_hi_idx:
            swing_hi = float(df.iloc[last_hi_idx]["high"])
            swing_lo = float(df.iloc[last_lo_idx]["low"])
            if swing_hi > swing_lo:
                entry, sl, native_tp = ote_short_prices(swing_lo, swing_hi, ote_level)
                tp = _apply_tp(entry, sl, "short", native_tp, cfg)
                pos = {"side": "short", "entry": entry, "entry_ms": t, "born_ms": t, "pending": True,
                       "sl": sl, "tp": tp, "size": _position_usdt(equity, cfg), "ttl_ms": 4 * 60 * 60 * 1000}
                last_hi_idx = last_lo_idx = -1

    res.final_equity_run = equity
    return res


# ---------------------------------------------------------------------------
# Unicorn -?entry = max(topBlock, FVG.top) / min(botBlock, FVG.bot)
# ---------------------------------------------------------------------------


def backtest_unicorn(
    symbol: str,
    df: pd.DataFrame,
    *,
    test_start_ms: int,
    test_end_ms: int,
    cfg: SimConfig = SimConfig(),
) -> ModelResult:
    res = ModelResult(symbol=symbol, model="unicorn")
    df = add_swings(df)
    all_fvgs = detect_fvgs(df)
    bull_bb, bear_bb = build_order_blocks(df)
    equity = res.equity_start
    pos: Optional[dict] = None
    swing_seq: List[Tuple[str, float, int]] = []

    for i in range(2, len(df)):
        row = df.iloc[i]
        t = int(row["open_time"])
        if t < test_start_ms:
            continue
        if t > test_end_ms:
            break
        hi, lo, cl = float(row["high"]), float(row["low"]), float(row["close"])
        live_fvgs = [g for g in all_fvgs if not g.invalidated(cl)]

        if bool(row["swing_lo"]):
            swing_seq.append(("lo", float(row["low"]), i))
        if bool(row["swing_hi"]):
            swing_seq.append(("hi", float(row["high"]), i))
        swing_seq = swing_seq[-6:]

        pos, equity = _simulate_bar(res, pos, t=t, hi=hi, lo=lo, equity=equity, symbol=symbol, cfg=cfg)
        if pos is not None or equity <= 0 or len(swing_seq) < 4:
            continue

        kinds = [s[0] for s in swing_seq[-4:]]
        if kinds == ["lo", "hi", "lo", "hi"]:
            lows = [s[1] for s in swing_seq[-4:] if s[0] == "lo"]
            his = [s[1] for s in swing_seq[-4:] if s[0] == "hi"]
            if len(lows) == 2 and len(his) == 2 and lows[1] > lows[0] and his[1] > his[0]:
                seq_start = swing_seq[-4][2]
                seq_ms = int(df.iloc[seq_start]["open_time"])
                for g in [x for x in live_fvgs if x.direction == 1 and x.born_idx >= seq_start and not x.used]:
                    for b in [x for x in bull_bb if x.born_ms >= seq_ms]:
                        entry = unicorn_long_entry(b.top, g.top)
                        if lo <= entry <= hi:
                            sl = lows[0]
                            tp = _apply_tp(entry, sl, "long", his[1], cfg)
                            pos = {"side": "long", "entry": entry, "entry_ms": t, "born_ms": t, "pending": True,
                                   "sl": sl, "tp": tp, "size": _position_usdt(equity, cfg),
                                   "ttl_ms": 2 * 60 * 60 * 1000}
                            g.used = True
                            break
                    if pos:
                        break

        if pos is None and kinds == ["hi", "lo", "hi", "lo"]:
            his = [s[1] for s in swing_seq[-4:] if s[0] == "hi"]
            lows = [s[1] for s in swing_seq[-4:] if s[0] == "lo"]
            if len(lows) == 2 and len(his) == 2 and his[1] < his[0] and lows[1] < lows[0]:
                seq_start = swing_seq[-4][2]
                seq_ms = int(df.iloc[seq_start]["open_time"])
                for g in [x for x in live_fvgs if x.direction == -1 and x.born_idx >= seq_start and not x.used]:
                    for b in [x for x in bear_bb if x.born_ms >= seq_ms]:
                        entry = unicorn_short_entry(b.bot, g.bot)
                        if lo <= entry <= hi:
                            sl = his[0]
                            tp = _apply_tp(entry, sl, "short", lows[1], cfg)
                            pos = {"side": "short", "entry": entry, "entry_ms": t, "born_ms": t, "pending": True,
                                   "sl": sl, "tp": tp, "size": _position_usdt(equity, cfg),
                                   "ttl_ms": 2 * 60 * 60 * 1000}
                            g.used = True
                            break
                    if pos:
                        break

    res.final_equity_run = equity
    return res


MODEL_RUNNERS = {
    "2022_model": backtest_2022_model,
    "silver_bullet": backtest_silver_bullet,
    "liquidity_raid": backtest_liquidity_raid,
    "ote": backtest_ote,
    "unicorn": backtest_unicorn,
}


def _trade_stats(res: ModelResult, cfg: SimConfig = SimConfig()) -> dict:
    if not res.trades:
        return {"gross_usdt": 0.0, "fees_usdt": 0.0, "avg_pct": 0.0}
    gross = fees = 0.0
    for t in res.trades:
        fee = _commission(t.entry_px and (res.equity_start * cfg.position_pct) or 0, cfg)
        # reconstruct from net + fees using stored size implied by pnl_pct
        size = abs(t.pnl_usdt / t.pnl_pct * 100.0) if t.pnl_pct else res.equity_start * cfg.position_pct
        fee = _commission(size, cfg) + _commission(size * t.exit_px / t.entry_px, cfg)
        fees += fee
        gross += t.pnl_usdt + fee
    return {
        "gross_usdt": gross,
        "fees_usdt": fees,
        "avg_pct": sum(t.pnl_pct for t in res.trades) / len(res.trades),
    }


def resolve_window(
    *,
    days: float,
    year: int = 0,
    from_date: str = "",
    to_date: str = "",
) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, str]:
    now = pd.Timestamp.now(tz="UTC")
    if year > 0:
        start = pd.Timestamp(f"{year}-01-01", tz="UTC")
        end = now if now.year == year else pd.Timestamp(f"{year}-12-31 23:59:59", tz="UTC")
        label = f"{year}YTD" if now.year == year else str(year)
    elif from_date.strip():
        start = pd.Timestamp(from_date.strip(), tz="UTC")
        end = pd.Timestamp(to_date.strip(), tz="UTC") if to_date.strip() else now
        label = f"{start.strftime('%Y-%m-%d')}..{end.strftime('%Y-%m-%d')}"
    else:
        end = now
        start = end - pd.Timedelta(days=days)
        label = f"{days:.0f}d"
    warmup = start - pd.Timedelta(days=14)
    return start, end, warmup, label


def run_suite(
    symbol: str,
    *,
    days: float,
    year: int = 0,
    from_date: str = "",
    to_date: str = "",
    tf: str,
    htf: str,
    models: List[str],
    verbose: bool,
    cfg: SimConfig = SimConfig(),
) -> List[ModelResult]:
    start, end, warmup, label = resolve_window(days=days, year=year, from_date=from_date, to_date=to_date)
    start_ms = int(start.value // 1_000_000)
    end_ms = int(end.value // 1_000_000)
    warm_ms = int(warmup.value // 1_000_000)

    df_1m = fetch_ohlcv(symbol, "1m", warm_ms, end_ms)
    if df_1m.empty:
        print(f"{symbol}: no data")
        return []

    df_ltf = resample_ohlcv(df_1m, tf)
    df_htf = resample_ohlcv(df_1m, htf)
    bars = len(df_ltf[df_ltf["open_time"] >= start_ms])
    print(f"\n{symbol} | {label} | ltf={tf} htf={htf} | bars={bars}")
    print_data_coverage(symbol, df_ltf, start_ms, end_ms)

    results: List[ModelResult] = []
    for name in models:
        fn = MODEL_RUNNERS[name]
        if name == "silver_bullet":
            r = fn(symbol, df_ltf, df_htf, test_start_ms=start_ms, test_end_ms=end_ms, cfg=cfg)
        else:
            r = fn(symbol, df_ltf, test_start_ms=start_ms, test_end_ms=end_ms, cfg=cfg)
        results.append(r)
        st = _trade_stats(r, cfg)
        print(
            f"  [{name}] trades={len(r.trades)} win={r.win_rate:.1f}% "
            f"ret={r.total_return_pct:+.2f}% eq=${r.final_equity:.2f} "
            f"| gross${st['gross_usdt']:+.2f} fees${st['fees_usdt']:.2f} avg/trade={st['avg_pct']:+.3f}%"
        )
        if verbose and r.trades:
            for t in r.trades[:4]:
                et = ms_to_ny(t.entry_ms).strftime("%m-%d %H:%M")
                print(f"    {t.side} {t.pnl_pct:+.2f}% {t.reason} entry={et} ET")
            if len(r.trades) > 4:
                print(f"    ... +{len(r.trades) - 4} more")
    return results


def monthly_breakdown(res: ModelResult) -> pd.DataFrame:
    """Compound monthly stats from trade exit times."""
    if not res.trades:
        return pd.DataFrame()
    equity = res.equity_start
    rows: List[dict] = []
    by_month: dict[str, List[Trade]] = {}
    for t in sorted(res.trades, key=lambda x: x.exit_ms):
        m = pd.Timestamp(t.exit_ms, unit="ms", tz="UTC").strftime("%Y-%m")
        by_month.setdefault(m, []).append(t)

    for month in sorted(by_month.keys()):
        chunk = by_month[month]
        start_eq = equity
        pnl = sum(t.pnl_usdt for t in chunk)
        equity += pnl
        wins = sum(1 for t in chunk if t.pnl_usdt > 0)
        rows.append(
            {
                "month": month,
                "trades": len(chunk),
                "wins": wins,
                "win%": round(wins / len(chunk) * 100, 1),
                "pnl$": round(pnl, 2),
                "ret%": round(pnl / start_eq * 100, 2) if start_eq > 0 else 0.0,
                "eq_end$": round(equity, 2),
            }
        )
    return pd.DataFrame(rows)


def print_monthly(results: List[ModelResult]) -> None:
    for r in results:
        df = monthly_breakdown(r)
        if df.empty:
            continue
        print(f"\n--- {r.symbol} / {r.model} 月度明细 (按平仓月, 复利) ---")
        print(df.to_string(index=False))


def print_data_coverage(symbol: str, df_ltf: pd.DataFrame, start_ms: int, end_ms: int) -> None:
    test = df_ltf[(df_ltf["open_time"] >= start_ms) & (df_ltf["open_time"] <= end_ms)]
    if test.empty:
        print(f"  [data] {symbol}: NO bars in test window")
        return
    t0 = pd.Timestamp(int(test.iloc[0]["open_time"]), unit="ms", tz="UTC")
    t1 = pd.Timestamp(int(test.iloc[-1]["open_time"]), unit="ms", tz="UTC")
    span_d = (int(test.iloc[-1]["open_time"]) - int(test.iloc[0]["open_time"])) / 86_400_000
    print(f"  [data] bars={len(test)} | {t0.date()} .. {t1.date()} ({span_d:.0f}d) | 5m_total={len(df_ltf)}")


def print_summary(results: List[ModelResult], *, label: str) -> None:
    rows = [{"symbol": r.symbol, "model": r.model, "trades": len(r.trades),
             "win%": round(r.win_rate, 1), "return%": round(r.total_return_pct, 2),
             "final$": round(r.final_equity, 2)} for r in results]
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("return%", ascending=False)
    print(f"\n{'=' * 72}")
    print(f"ICT Master Suite summary ({label})")
    print(f"{'=' * 72}")
    print(df.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="ICT Master Suite aligned backtest (Trading-IQ)")
    ap.add_argument("--days", type=float, default=30.0, help="rolling window days (ignored if --year/--from)")
    ap.add_argument("--year", type=int, default=0, help="calendar year e.g. 2026 for YTD")
    ap.add_argument("--from", dest="from_date", default="", help="start date YYYY-MM-DD")
    ap.add_argument("--to", dest="to_date", default="", help="end date YYYY-MM-DD")
    ap.add_argument("--commission", type=float, default=COMMISSION_PCT, help="commission %% per fill (0=none)")
    ap.add_argument("--tf", default="5m", help="chart TF (default 5m)")
    ap.add_argument("--htf", default="15m", help="HTF for Silver Bullet bias (default 15m)")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT")
    ap.add_argument("--models", default="2022_model,silver_bullet,liquidity_raid,ote,unicorn")
    ap.add_argument("--position-pct", type=float, default=POSITION_PCT, help="equity fraction per trade (1.0=全仓)")
    ap.add_argument("--rr", type=float, default=0.0, help="fixed R:R TP multiple (1.5=1:1.5); 0=model default")
    ap.add_argument("--leverage", type=float, default=1.0, help="leverage multiplier on notional (2=2x)")
    ap.add_argument("--monthly", action="store_true", help="print per-month PnL breakdown")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    sim_cfg = SimConfig(
        position_pct=args.position_pct,
        rr=args.rr,
        leverage=args.leverage,
        commission_pct=args.commission,
    )

    _, _, _, window_label = resolve_window(
        days=args.days, year=args.year, from_date=args.from_date, to_date=args.to_date
    )

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODEL_RUNNERS]
    if unknown:
        print(f"Unknown models: {unknown}. Available: {list(MODEL_RUNNERS)}")
        sys.exit(1)

    all_results: List[ModelResult] = []
    for sym in [s.strip() for s in args.symbols.split(",") if s.strip()]:
        all_results.extend(
            run_suite(
                sym,
                days=args.days,
                year=args.year,
                from_date=args.from_date,
                to_date=args.to_date,
                tf=args.tf,
                htf=args.htf,
                models=models,
                verbose=args.verbose,
                cfg=sim_cfg,
            )
        )
    print_summary(all_results, label=window_label)
    if args.monthly:
        print_monthly(all_results)
    pos_label = "全仓" if args.position_pct >= 0.99 else f"{args.position_pct*100:.0f}%仓位"
    rr_label = f"1:{args.rr:g}" if args.rr > 0 else "模型默认TP"
    lev_label = f"{args.leverage:g}x" if args.leverage != 1.0 else "1x"
    fee_label = "-? if args.commission <= 0 else f"{args.commission}%/fill"
    print(f"\nSettings: {pos_label}, {lev_label}杠杆, R:R={rr_label}, $1000复利, 手续-?{fee_label}")


if __name__ == "__main__":
    main()
