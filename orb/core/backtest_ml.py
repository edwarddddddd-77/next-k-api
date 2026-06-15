#!/usr/bin/env python3
"""ORB 回测（ML 专用）：读 data/orb/kline/<SYMBOL>/ 本地 K 线。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from orb.core.backtest import (
    _daily_df_asof,
    _idle_skip_sim,
    _iter_scan_ms,
    _resolve_open,
    _scan_cron_second_ms,
)
from orb.core.config import OrbConfig
from orb.ml.features import BreakoutModel, extract_features, load_model
from orb.core.kline_cache import load_klines  # noqa: E402
from orb.core.paper import analyze_at_ms, in_regular_session, is_actionable
from orb.core.session import extended_fetch_anchor_ms
from orb.core.resolve import pnl_r, pnl_usdt
from orb.core.session import session_day_str


@dataclass
class _SimOpenMl:
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
    features: Optional[Dict[str, float]] = None
    fake_breakout_p: Optional[float] = None


def _load_cached(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    return load_klines(symbol, interval, start_ms=int(start_ms), end_ms=int(end_ms))


def run_backtest(
    *,
    days: float,
    symbols: List[str],
    cfg: OrbConfig,
    record_features: bool = True,
    fake_model: Optional[BreakoutModel] = None,
    fake_max_p: float = 0.65,
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
        dfs5[sym] = _load_cached(sym, cfg.signal_interval, fetch_start, end_ms)
        dfs1[sym] = _load_cached(sym, "1m", fetch_start, end_ms)
        if (cfg.sl_mode or "").strip().lower() == "atr_pct":
            dfs_daily[sym] = _load_cached(sym, "1d", fetch_start - cfg.daily_atr_warmup_ms(), end_ms)

    wallets: Dict[str, float] = {s: cfg.per_symbol_bot_equity() for s in syms}
    opens: Dict[str, Optional[_SimOpenMl]] = {s: None for s in syms}
    session_traded: Dict[Tuple[str, str], bool] = {}
    trades: List[Dict[str, Any]] = []
    skipped_model: List[Dict[str, Any]] = []

    def _count_open() -> int:
        return sum(1 for s in syms if opens[s] is not None)

    def _close(pos: _SimOpenMl, *, outcome: str, exit_px: float, exit_bo: Optional[int], note: str, scan_ms: int) -> None:
        sym = pos.symbol
        pu = pnl_usdt(pos.side, float(pos.entry), float(exit_px), float(pos.notional))
        pr = pnl_r(pos.side, float(pos.entry), float(exit_px), float(pos.sl))
        wallets[sym] = round(wallets[sym] + pu, 4)
        row: Dict[str, Any] = {
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
            "exit_price": exit_px,
            "exit_bar_open_ms": exit_bo,
            "pnl_r": round(pr, 6),
            "pnl_usdt": round(pu, 4),
            "notional_usdt": pos.notional,
            "wallet_after": wallets[sym],
            "resolve_note": note,
        }
        if pos.features is not None:
            row["features"] = dict(pos.features)
        if pos.fake_breakout_p is not None:
            row["fake_breakout_p"] = pos.fake_breakout_p
        trades.append(row)
        opens[sym] = None

    for scan_ms in scan_times:
        if _idle_skip_sim(cfg, now_ms=scan_ms, open_count=_count_open()):
            continue

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

        candidates: List[Tuple[str, Any, str, Tuple[str, str], Any]] = []
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
            candidates.append((sym, sig, sess_day, sk, opens[sym]))

        sync_by_sym: Dict[str, int] = {}
        for sym, sig, *_ in candidates:
            side = str(sig.side)
            sync_by_sym[sym] = sum(1 for s2, g2, *_ in candidates if s2 != sym and str(g2.side) == side)

        for sym, sig, sess_day, sk, hold in candidates:
            if hold is not None:
                if str(hold.side) == sig.side:
                    continue
                _close(hold, outcome="supersede", exit_px=float(sig.price), exit_bo=None, note="supersede:reverse", scan_ms=scan_ms)
            if cfg.max_open_positions > 0 and _count_open() >= cfg.max_open_positions:
                continue
            entry_bo = int(sig.entry_bar_open_ms or 0)
            if entry_bo <= 0:
                continue
            sync_n = int(sync_by_sym.get(sym, 0))
            feat = extract_features(sig, cfg, sync_same_side=sync_n)
            fake_p: Optional[float] = None
            if fake_model is not None:
                fake_p = round(fake_model.predict_proba(feat, symbol=str(sig.symbol or sym)), 4)
                if fake_p >= float(fake_max_p):
                    skipped_model.append(
                        {
                            "symbol": sym,
                            "session_date": str(sig.session_date or sess_day),
                            "side": str(sig.side),
                            "fake_breakout_p": fake_p,
                            "scan_open_ms": int(scan_ms),
                            "sync_same_side": sync_n,
                        }
                    )
                    continue
            notion = float(sig.paper_notional_usdt or cfg.default_paper_notional())
            opens[sym] = _SimOpenMl(
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
                features=feat if record_features else None,
                fake_breakout_p=fake_p,
            )
            if cfg.one_trade_per_session:
                session_traded[sk] = True

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
    touch = wins + losses
    pnl_sum = sum(float(x.get("pnl_usdt") or 0) for x in resolved)
    final_wallet = wallets.get(syms[0], 0.0) if len(syms) == 1 else None
    return {
        "engine": "ml_cached",
        "days": days,
        "symbols": syms,
        "scans": len(scan_times),
        "macro_filter": cfg.macro_filter,
        "fake_filter": fake_model is not None,
        "fake_max_p": fake_max_p if fake_model is not None else None,
        "model_skipped": len(skipped_model),
        "trades_total": len(trades),
        "resolved": len(resolved),
        "touch_win_rate": round(wins / touch, 4) if touch else None,
        "sum_pnl_usdt": round(pnl_sum, 4),
        "final_wallet_usdt": round(final_wallet, 4) if final_wallet is not None else None,
        "trades": trades,
        "skipped": skipped_model,
    }
