#!/usr/bin/env python3
"""ORB 纸面回放回测 — 与 run_scan_conn 同频扫描、同路径 analyze/resolve。"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from binance_fapi import fetch_klines_forward, klines_to_df
from orb.core.config import OrbConfig
from orb.core.paper import analyze_at_ms, in_regular_session, is_actionable
from orb.core.session import extended_fetch_anchor_ms
from orb.core.resolve import pnl_r, pnl_usdt, resolve_forward
from orb.core.session import session_day_str

_DEFAULT_JSON = str(Path(__file__).resolve().parent.parent / "orb_backtest_last.json")

# 1m 分 chunk：每段 30 天 ≈ 43200 根，低于 fetch_klines_forward 150k 总量 cap
_LOAD_1M_CHUNK_MS = 30 * 86_400_000


def _scan_cron_second_ms() -> int:
    raw = os.getenv("ORB_SCAN_CRON_SECOND", "5")
    try:
        sec = max(0, min(59, int(float(str(raw).strip()))))
    except ValueError:
        sec = 5
    return sec * 1000


def _load_range(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    start_ms = int(start_ms)
    end_ms = int(end_ms)
    if end_ms <= start_ms:
        return pd.DataFrame()
    iv = interval.strip().lower()
    rows: List[Any] = []
    if iv == "1m" and (end_ms - start_ms) > _LOAD_1M_CHUNK_MS:
        cur = start_ms
        while cur <= end_ms:
            chunk_end = min(cur + _LOAD_1M_CHUNK_MS, end_ms)
            rows.extend(fetch_klines_forward(symbol, iv, cur, chunk_end))
            cur = chunk_end + 1
    else:
        rows = fetch_klines_forward(symbol, iv, start_ms, end_ms)
    df = klines_to_df(rows)
    if df.empty:
        return df
    return df.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time").reset_index(drop=True)


def _iter_scan_ms(start_ms: int, end_ms: int, *, bar_step_ms: int) -> List[int]:
    """UTC 5m K 线收盘后 ORB_SCAN_CRON_SECOND 秒触发，与 scheduler */5 对齐。"""
    delay = _scan_cron_second_ms()
    if bar_step_ms <= 0:
        return []
    first = ((start_ms // bar_step_ms) + 1) * bar_step_ms + delay
    out: List[int] = []
    cur = first
    while cur <= end_ms:
        out.append(int(cur))
        cur += bar_step_ms
    return out


def _daily_df_asof(full: pd.DataFrame, now_ms: int) -> pd.DataFrame:
    if full.empty:
        return full
    return full[full["open_time"] <= int(now_ms)].reset_index(drop=True)


def _idle_skip_sim(cfg: OrbConfig, *, now_ms: int, open_count: int) -> Optional[str]:
    if not cfg.regular_session_only:
        return None
    if in_regular_session(cfg, now_ms=now_ms):
        return None
    if open_count > 0:
        return None
    return "outside_regular_session_no_open_positions"


@dataclass
class _SimOpen:
    symbol: str
    side: str
    play: str
    entry: float
    sl: float
    tp: Optional[float]
    entry_bar_open_ms: int
    notional: float
    session_date: str
    scan_open_ms: int


def _resolve_open(
    pos: _SimOpen,
    df1: pd.DataFrame,
    *,
    scan_ms: int,
    cfg: OrbConfig,
) -> Tuple[Optional[str], float, str, Optional[int]]:
    out, ex_px, note, _, exit_bo = resolve_forward(
        df1,
        entry=float(pos.entry),
        entry_bar_open_ms=int(pos.entry_bar_open_ms),
        side=str(pos.side),
        sl=float(pos.sl),
        tp=float(pos.tp) if pos.tp is not None else None,
        hist_end_ms=int(scan_ms),
        bar_step_ms=cfg.bar_step_ms(),
        cfg=cfg,
    )
    return out, ex_px, note, exit_bo


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
    fetch_start = extended_fetch_anchor_ms(start_ms, cfg) - bar_step * 96
    syms = [s.strip().upper() for s in symbols if s.strip()]
    scan_times = _iter_scan_ms(start_ms, end_ms, bar_step_ms=bar_step)

    dfs5: Dict[str, pd.DataFrame] = {}
    dfs1: Dict[str, pd.DataFrame] = {}
    dfs_daily: Dict[str, pd.DataFrame] = {}
    for sym in syms:
        dfs5[sym] = _load_range(sym, cfg.signal_interval, fetch_start, end_ms)
        dfs1[sym] = _load_range(sym, "1m", fetch_start, end_ms)
        if (cfg.sl_mode or "").strip().lower() == "atr_pct":
            dfs_daily[sym] = _load_range(sym, "1d", fetch_start - cfg.daily_atr_warmup_ms(), end_ms)

    wallets: Dict[str, float] = {s: cfg.per_symbol_bot_equity() for s in syms}
    opens: Dict[str, Optional[_SimOpen]] = {s: None for s in syms}
    session_traded: Dict[Tuple[str, str], bool] = {}
    trades: List[Dict[str, Any]] = []

    def _count_open() -> int:
        return sum(1 for s in syms if opens[s] is not None)

    def _close(pos: _SimOpen, *, outcome: str, exit_px: float, exit_bo: Optional[int], note: str, scan_ms: int) -> None:
        sym = pos.symbol
        pu = pnl_usdt(pos.side, float(pos.entry), float(exit_px), float(pos.notional))
        pr = pnl_r(pos.side, float(pos.entry), float(exit_px), float(pos.sl))
        wallets[sym] = round(wallets[sym] + pu, 4)
        trades.append(
            {
                "symbol": sym,
                "session_date": pos.session_date,
                "side": pos.side,
                "play": pos.play,
                "entry": pos.entry,
                "sl": pos.sl,
                "tp": pos.tp,
                "entry_bar_open_ms": pos.entry_bar_open_ms,
                "scan_open_ms": pos.scan_open_ms,
                "scan_close_ms": scan_ms,
                "outcome": outcome,
                "exit_price": ex_px,
                "exit_bar_open_ms": exit_bo,
                "pnl_r": round(pr, 6),
                "pnl_usdt": round(pu, 4),
                "notional_usdt": pos.notional,
                "wallet_after": wallets[sym],
                "resolve_note": note,
            }
        )
        opens[sym] = None

    for scan_ms in scan_times:
        open_n = _count_open()
        if _idle_skip_sim(cfg, now_ms=scan_ms, open_count=open_n):
            continue

        # resolve_pre — 与实盘一致：仅看到 scan_ms 之前的 1m
        for sym in syms:
            pos = opens[sym]
            if pos is None:
                continue
            df1 = dfs1.get(sym)
            if df1 is None or df1.empty:
                continue
            out, ex_px, note, exit_bo = _resolve_open(pos, df1, scan_ms=scan_ms, cfg=cfg)
            if out is not None:
                _close(pos, outcome=out, exit_px=ex_px, exit_bo=exit_bo, note=note, scan_ms=scan_ms)

        if not in_regular_session(cfg, now_ms=scan_ms):
            continue

        for sym in syms:
            wallet = wallets[sym]
            if wallet <= 0:
                continue
            df5 = dfs5.get(sym)
            if df5 is None or df5.empty:
                continue
            sess_day = session_day_str(scan_ms, tz=cfg.session_tz, session_open_time=cfg.session_open_time)
            sk = (sym, sess_day)
            traded = session_traded.get(sk, False) if cfg.one_trade_per_session else False
            ddf = _daily_df_asof(dfs_daily.get(sym, pd.DataFrame()), scan_ms)
            sig = analyze_at_ms(
                sym,
                cfg=cfg,
                now_ms=scan_ms,
                session_traded=traded,
                daily_df=ddf if not ddf.empty else None,
                bot_equity_usdt=wallet,
                df5=df5,
            )
            if not is_actionable(sig, cfg):
                continue
            hold = opens[sym]
            if hold is not None:
                if str(hold.side) == sig.side:
                    continue
                _close(
                    hold,
                    outcome="supersede",
                    exit_px=float(sig.price),
                    exit_bo=None,
                    note="supersede:reverse",
                    scan_ms=scan_ms,
                )
            if cfg.max_open_positions > 0 and _count_open() >= cfg.max_open_positions:
                continue
            entry_bo = int(sig.entry_bar_open_ms or 0)
            if entry_bo <= 0:
                continue
            notion = float(sig.paper_notional_usdt or cfg.default_paper_notional())
            opens[sym] = _SimOpen(
                symbol=sym,
                side=str(sig.side),
                play=str(sig.play),
                entry=float(sig.price),
                sl=float(sig.sl_price),
                tp=float(sig.tp_price) if sig.tp_price is not None else None,
                entry_bar_open_ms=entry_bo,
                notional=notion,
                session_date=str(sig.session_date or sess_day),
                scan_open_ms=int(scan_ms),
            )
            if cfg.one_trade_per_session:
                session_traded[sk] = True

        # resolve_post
        for sym in syms:
            pos = opens[sym]
            if pos is None:
                continue
            df1 = dfs1.get(sym)
            if df1 is None or df1.empty:
                continue
            out, ex_px, note, exit_bo = _resolve_open(pos, df1, scan_ms=scan_ms, cfg=cfg)
            if out is not None:
                _close(pos, outcome=out, exit_px=ex_px, exit_bo=exit_bo, note=note, scan_ms=scan_ms)

    resolved = [x for x in trades if x.get("outcome") and x["outcome"] != "supersede"]
    wins = sum(1 for x in resolved if float(x.get("pnl_usdt") or 0) > 0)
    losses = sum(1 for x in resolved if float(x.get("pnl_usdt") or 0) < 0)
    pnl_sum = sum(float(x.get("pnl_usdt") or 0) for x in resolved)
    pnl_sum_all = sum(float(x.get("pnl_usdt") or 0) for x in trades)
    touch = wins + losses
    summary = {
        "strategy": "orb",
        "engine": "live_scan_sim",
        "days": days,
        "symbols": syms,
        "scans": len(scan_times),
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
            "scan_cron_second": int(_scan_cron_second_ms() / 1000),
        },
        "trades_total": len(trades),
        "resolved": len(resolved),
        "win": wins,
        "loss": losses,
        "touch_win_rate": round(wins / touch, 4) if touch else None,
        "sum_pnl_usdt": round(pnl_sum, 4),
        "sum_pnl_usdt_incl_supersede": round(pnl_sum_all, 4),
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
