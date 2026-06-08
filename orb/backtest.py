#!/usr/bin/env python3
"""ORB walk-forward 回测。"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from binance_fapi import fetch_klines_forward, klines_to_df
from orb.config import OrbConfig
from orb.indicators import daily_atr_asof
from orb.resolve import pnl_r, pnl_usdt, resolve_forward
from orb.session import session_day_floor_ms, session_day_str
from orb.signals import classify_signal

_DEFAULT_JSON = str(Path(__file__).resolve().parent.parent / "orb_backtest_last.json")


def _session_key(symbol: str, open_ms: int, cfg: OrbConfig) -> Tuple[str, str]:
    day = session_day_str(int(open_ms), tz=cfg.session_tz, session_open_time=cfg.session_open_time)
    return str(symbol).strip().upper(), day


def _count_open_positions(open_until_ms: Dict[str, int], *, asof_ms: int) -> int:
    return sum(1 for until in open_until_ms.values() if int(until) >= int(asof_ms))


def _load_range(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = fetch_klines_forward(symbol, interval, start_ms, end_ms)
    df = klines_to_df(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time").reset_index(drop=True)


def run_backtest(
    *,
    days: float,
    symbols: List[str],
    cfg: OrbConfig,
    json_path: Optional[str] = _DEFAULT_JSON,
    csv_path: Optional[str] = None,
) -> Dict[str, Any]:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(float(days) * 86_400_000)
    bar_step = cfg.bar_step_ms()
    fetch_start = session_day_floor_ms(start_ms, cfg.session_tz, cfg.session_open_time) - bar_step * 96
    syms = [s.strip().upper() for s in symbols if s.strip()]
    dfs: Dict[str, pd.DataFrame] = {}
    dfs_1m: Dict[str, pd.DataFrame] = {}
    dfs_daily: Dict[str, pd.DataFrame] = {}
    for sym in syms:
        dfs[sym] = _load_range(sym, cfg.signal_interval, fetch_start, end_ms)
        dfs_1m[sym] = _load_range(sym, "1m", fetch_start, end_ms)
        if (cfg.sl_mode or "").strip().lower() == "atr_pct":
            daily_start = fetch_start - int(cfg.atr_period + 5) * 86_400_000
            dfs_daily[sym] = _load_range(sym, "1d", daily_start, end_ms)

    trades: List[Dict[str, Any]] = []
    session_traded: Dict[Tuple[str, str], bool] = {}
    open_until_ms: Dict[str, int] = {}
    for sym in syms:
        df_full = dfs.get(sym)
        df_resolve = dfs_1m.get(sym)
        if df_full is None or df_full.empty or df_resolve is None or df_resolve.empty:
            continue
        df_loop = df_full[df_full["open_time"] >= start_ms].reset_index(drop=True)
        hist_end_5m = int(df_full["open_time"].iloc[-1])
        hist_end_1m = int(df_resolve["open_time"].iloc[-1]) if len(df_resolve) else hist_end_5m
        hist_end_resolve = max(hist_end_5m, hist_end_1m)
        for i in range(len(df_loop)):
            t = int(df_loop.iloc[i]["open_time"])
            if t > hist_end_5m:
                break
            if open_until_ms.get(sym, -1) >= t:
                continue
            if cfg.max_open_positions > 0 and _count_open_positions(open_until_ms, asof_ms=t) >= cfg.max_open_positions:
                continue
            sk = _session_key(sym, t, cfg)
            daily_atr = None
            if (cfg.sl_mode or "").strip().lower() == "atr_pct":
                ddf = dfs_daily.get(sym)
                if ddf is not None and not ddf.empty:
                    daily_atr = daily_atr_asof(
                        ddf, t, period=cfg.atr_period, tz=cfg.session_tz
                    )
            sig = classify_signal(
                sym,
                df_full,
                asof_open_ms=t,
                cfg=cfg,
                session_traded=session_traded.get(sk, False),
                daily_atr=daily_atr,
                bot_equity_usdt=cfg.per_symbol_bot_equity(),
            )
            if sig.side not in ("LONG", "SHORT") or sig.sl_price is None:
                continue
            if (cfg.exit_mode or "").strip().lower() != "eod" and sig.tp_price is None:
                continue
            if cfg.one_trade_per_session:
                session_traded[sk] = True
            out, ex_px, note, bars_seen, exit_bo = resolve_forward(
                df_resolve,
                entry=float(sig.price),
                entry_bar_open_ms=int(sig.entry_bar_open_ms or t),
                side=sig.side,
                sl=float(sig.sl_price),
                tp=float(sig.tp_price) if sig.tp_price is not None else None,
                hist_end_ms=hist_end_resolve,
                bar_step_ms=bar_step,
                cfg=cfg,
            )
            notion = float(sig.paper_notional_usdt or cfg.default_paper_notional())
            row = {
                "symbol": sym,
                "session_date": sig.session_date,
                "side": sig.side,
                "play": sig.play,
                "entry": sig.price,
                "sl": sig.sl_price,
                "tp": sig.tp_price,
                "or_high": sig.or_high,
                "or_low": sig.or_low,
                "volume": sig.volume,
                "signal_open_ms": t,
                "entry_bar_open_ms": sig.entry_bar_open_ms,
            }
            if out is None:
                open_until_ms[sym] = hist_end_resolve
                trades.append({**row, "outcome": None, "exit_price": None, "pnl_usdt": None, "resolve_note": note})
                continue
            open_until_ms[sym] = int(exit_bo) if exit_bo is not None else hist_end_resolve
            trades.append(
                {
                    **row,
                    "outcome": out,
                    "exit_price": ex_px,
                    "exit_bar_open_ms": exit_bo,
                    "pnl_r": round(pnl_r(sig.side, float(sig.price), ex_px, float(sig.sl_price)), 6),
                    "pnl_usdt": round(pnl_usdt(sig.side, float(sig.price), ex_px, notion), 4),
                    "notional_usdt": notion,
                    "resolve_note": note,
                    "bars_seen": bars_seen,
                }
            )

    resolved = [x for x in trades if x.get("outcome")]
    wins = sum(1 for x in resolved if x["outcome"] == "win")
    losses = sum(1 for x in resolved if x["outcome"] == "loss")
    pnl_sum = sum(float(x.get("pnl_usdt") or 0) for x in resolved)
    touch = wins + losses
    summary = {
        "strategy": "orb",
        "days": days,
        "symbols": syms,
        "config": {
            "market": cfg.market,
            "session_tz": cfg.session_tz,
            "session_open_time": cfg.session_open_time,
            "session_close_time": cfg.session_close_time,
            "regular_session_only": cfg.regular_session_only,
            "signal_interval": cfg.signal_interval,
            "or_minutes": cfg.or_minutes,
            "entry_mode": cfg.entry_mode,
            "confirm_bars": cfg.confirm_bars,
            "confirm_no_soften": cfg.confirm_no_soften,
            "vol_mult": cfg.vol_mult,
            "sl_mode": cfg.sl_mode,
            "atr_period": cfg.atr_period,
            "atr_sl_fraction": cfg.atr_sl_fraction,
            "exit_mode": cfg.exit_mode,
            "risk_pct": cfg.risk_pct,
            "symbol_bot_equity_usdt": cfg.per_symbol_bot_equity(),
            "account_equity_usdt": cfg.account_equity_usdt,
            "fixed_notional_usdt": cfg.fixed_notional_usdt,
            "vwap_filter": cfg.vwap_filter,
            "tp_r": cfg.tp_r_multiple,
        },
        "trades_total": len(trades),
        "resolved": len(resolved),
        "win": wins,
        "loss": losses,
        "touch_win_rate": round(wins / touch, 4) if touch else None,
        "sum_pnl_usdt": round(pnl_sum, 4),
        "trades": trades,
    }
    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in summary.items() if k != "trades"}, f, indent=2)
    if csv_path:
        import csv

        fields = list(trades[0].keys()) if trades else ["symbol", "outcome"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for tr in trades:
                w.writerow(tr)
    return summary
