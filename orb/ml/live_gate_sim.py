"""Live Gate scan-by-scan 回测模拟（仅回测 / Gate 调参 / eval 工具）。

实盘 scan 走 ``orb/v2/paper.py``，不 import 本模块。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from orb.core.backtest import _daily_df_asof, _iter_scan_ms, _resolve_open, _SimOpen
from orb.ml.ranker import BreakoutRanker
from orb.core.config import OrbConfig
from orb.ml.features import extract_features, label_is_true_breakout
from orb.core.kline_cache import has_kline_cache, load_klines
from orb.core.signals import compute_position_notional
from orb.v2.robots import (
    next_free_robot as _next_free_robot,
    next_robot_index as _next_robot_index,
    release_robots_through as _release_robots_through,
    robot_equity_for_signals as _robot_equity_for_signals,
)
from orb.ml.gate import LiveGateConfig, LiveGateDayState, evaluate_open_decision, rollback_open_decision, summarize_live_gate_day
from orb.core.paper import analyze_at_ms, in_regular_session, is_actionable
from orb.core.session import (
    extended_fetch_anchor_ms,
    session_anchor_ms,
    session_close_ms,
    session_day_str,
)
from orb.core.macro_calendar import is_macro_skip_day, macro_events_for_day
from orb.core.us_equity_calendar import is_us_equity_trading_day
from orb.ml.model.paths import resolve_samples_path


def ml_cfg(
    *,
    compound_per_symbol: bool = False,
    fixed_notional: float = 0.0,
    respect_env_filters: bool = True,
) -> OrbConfig:
    if not respect_env_filters:
        os.environ["ORB_MACRO_FILTER"] = "0"
    cfg = OrbConfig.from_env()
    if not respect_env_filters:
        cfg.macro_filter = False
    cfg.max_open_positions = 999
    if compound_per_symbol:
        cfg.fixed_notional_usdt = 0.0
    elif fixed_notional > 0:
        cfg.fixed_notional_usdt = float(fixed_notional)
    return cfg


def cached_symbols(symbols: List[str]) -> List[str]:
    return [s for s in symbols if has_kline_cache(s, "5m")]


def init_symbol_wallets(symbols: List[str], cfg: OrbConfig) -> Dict[str, float]:
    eq = float(cfg.per_symbol_bot_equity())
    return {str(s).strip().upper(): eq for s in symbols if str(s).strip()}


def trading_dates_from_samples(
    *,
    last_sessions: int = 0,
    end_date: str = "",
    samples_path: Path | None = None,
) -> List[str]:
    path = samples_path or resolve_samples_path()
    if not path.is_file():
        return []
    rows = json.loads(path.read_text(encoding="utf-8")).get("rows") or []
    all_dates = sorted({str(r.get("session_date") or "") for r in rows if r.get("session_date")})
    trading = [d for d in all_dates if is_us_equity_trading_day(d)]
    if not trading:
        return []
    end = (end_date or trading[-1]).strip()
    if end:
        trading = [d for d in trading if d <= end]
    n = int(last_sessions)
    if n > 0:
        return trading[-n:]
    return trading


def _first_resolve_scan_ms(
    pos: _SimOpen,
    df1: pd.DataFrame,
    scans: List[int],
    *,
    open_scan_ms: int,
    cfg: OrbConfig,
) -> Optional[int]:
    """与 paper.py resolve_pre 一致：开仓 scan 之后首次能结算的 scan。"""
    for s in scans:
        if s <= int(open_scan_ms):
            continue
        out, _, _, _ = _resolve_open(pos, df1, scan_ms=s, cfg=cfg)
        if out is not None:
            return int(s)
    return None


def _resolve_trade_row(
    *,
    sym: str,
    sig: Any,
    session_date: str,
    scan_ms: int,
    entry_bo: int,
    df1: pd.DataFrame,
    close_ms: int,
    bar: int,
    cfg: OrbConfig,
    notional: float,
    wallet_before: Optional[float] = None,
    robot_id: Optional[int] = None,
    scans: Optional[List[int]] = None,
) -> Optional[Dict[str, Any]]:
    if entry_bo <= 0 or df1 is None or df1.empty:
        return None
    pos = _SimOpen(
        symbol=sym,
        side=str(sig.side),
        play=str(sig.play),
        entry=float(sig.price),
        sl=float(sig.sl_price),
        tp=float(sig.tp_price) if sig.tp_price is not None else None,
        entry_bar_open_ms=entry_bo,
        notional=float(notional),
        session_date=session_date,
        scan_open_ms=int(scan_ms),
    )
    out, ex_px, _note, exit_bo = _resolve_open(pos, df1, scan_ms=close_ms + bar, cfg=cfg)
    if not out or out == "supersede":
        return None
    from orb.core.resolve import pnl_usdt

    pnl = float(pnl_usdt(pos.side, pos.entry, ex_px, pos.notional))
    if scans:
        release_ms = _first_resolve_scan_ms(pos, df1, scans, open_scan_ms=scan_ms, cfg=cfg)
        exit_ms = release_ms if release_ms is not None else int(close_ms + bar)
    else:
        exit_ms = int(exit_bo) + int(bar) if exit_bo else int(close_ms + bar)
    row: Dict[str, Any] = {
        "session_date": session_date,
        "scan_open_ms": int(scan_ms),
        "side": pos.side,
        "entry": pos.entry,
        "notional_usdt": round(float(pos.notional), 4),
        "pnl_usdt": round(pnl, 4),
        "true_breakout": label_is_true_breakout(out, pnl),
        "outcome": out,
        "exit_ms": exit_ms,
        "exit_bar_open_ms": exit_bo,
    }
    if wallet_before is not None:
        row["wallet_before"] = round(float(wallet_before), 4)
    if robot_id is not None:
        row["robot_id"] = int(robot_id)
    return row


def _score_and_sort_candidates(
    candidates: List[Tuple[str, Any]],
    sync_by_sym: Dict[str, int],
    cfg: OrbConfig,
    ranker: BreakoutRanker,
) -> List[Tuple[str, Any, int, Dict[str, float], float]]:
    scored: List[Tuple[str, Any, int, Dict[str, float], float]] = []
    for sym, sig in candidates:
        sync_n = int(sync_by_sym.get(sym, 0))
        feat = extract_features(sig, cfg, sync_same_side=sync_n)
        p_true = float(ranker.predict_true(feat, symbol=sym))
        scored.append((sym, sig, sync_n, feat, p_true))
    scored.sort(key=lambda x: x[4], reverse=True)
    return scored


def _rollback_open(state: LiveGateDayState, decision: Dict[str, Any]) -> None:
    rollback_open_decision(state, symbol=str(decision.get("symbol") or ""))


def _finalize_blocked_open(decision: Dict[str, Any], state: LiveGateDayState, reason: str) -> None:
    _rollback_open(state, decision)
    decision["opened"] = False
    decision["reason"] = reason
    state.skipped.append(dict(decision))


def _merge_trade_into_opened(state: LiveGateDayState, trade_row: Dict[str, Any]) -> None:
    if not state.opened:
        return
    row = state.opened[-1]
    for key in (
        "notional_usdt",
        "side",
        "entry",
        "pnl_usdt",
        "true_breakout",
        "outcome",
        "exit_ms",
        "exit_bar_open_ms",
        "robot_id",
        "wallet_before",
    ):
        if key in trade_row and trade_row[key] is not None:
            row[key] = trade_row[key]
    if trade_row.get("pnl_usdt") is not None and trade_row.get("wallet_before") is not None:
        row["wallet_after"] = round(float(trade_row["wallet_before"]) + float(trade_row["pnl_usdt"]), 4)


def simulate_live_gate_day(
    session_date: str,
    symbols: List[str],
    cfg: OrbConfig,
    ranker: BreakoutRanker,
    gate: LiveGateConfig,
    *,
    wallets: Optional[Dict[str, float]] = None,
    robot_wallets: Optional[List[float]] = None,
) -> Dict[str, Any]:
    tz = cfg.session_tz
    ts = pd.Timestamp(f"{session_date} 12:00:00", tz=tz)
    anchor = session_anchor_ms(int(ts.value // 1_000_000), tz=tz, session_open_time=cfg.session_open_time)
    close = session_close_ms(anchor, tz=tz, session_close_time=cfg.session_close_time)
    if close is None:
        close = anchor + 6 * 60 * 60 * 1000
    bar = cfg.bar_step_ms()
    scans = [
        s
        for s in _iter_scan_ms(anchor, close, bar_step_ms=bar)
        if session_day_str(s, tz=tz, session_open_time=cfg.session_open_time) == session_date
    ]

    warmup = cfg.daily_atr_warmup_ms() + bar * 96
    fetch_start = extended_fetch_anchor_ms(anchor, cfg) - warmup
    end_ms = close + bar * 4
    dfs5, dfs1, dfs_daily = {}, {}, {}
    for sym in symbols:
        dfs5[sym] = load_klines(sym, cfg.signal_interval, start_ms=fetch_start, end_ms=end_ms)
        dfs1[sym] = load_klines(sym, "1m", start_ms=fetch_start, end_ms=end_ms)
        if (cfg.sl_mode or "").strip().lower() == "atr_pct":
            dfs_daily[sym] = load_klines(sym, "1d", start_ms=fetch_start - cfg.daily_atr_warmup_ms(), end_ms=end_ms)

    macro_skip_day = bool(cfg.macro_filter and is_macro_skip_day(session_date))
    macro_events = macro_events_for_day(session_date) if cfg.macro_filter else ()

    state = LiveGateDayState()
    # 与 orb/v2/paper.py 的 v2_session_traded 一致：仅成功开仓后锁定，Gate 拒绝可继续扫描
    session_opened: Dict[str, bool] = {}
    robots_used_today: set[int] = set()
    robot_busy: Dict[int, Dict[str, Any]] = {}
    robot_reuse = bool(gate.robot_reuse_after_exit and robot_wallets is not None)
    timeline: List[Dict[str, Any]] = []

    for scan_ms in scans:
        if not in_regular_session(cfg, now_ms=scan_ms):
            continue
        if robot_reuse and robot_wallets is not None:
            _release_robots_through(robot_busy, robot_wallets, scan_ms)
        signal_equity = (
            _robot_equity_for_signals(robot_wallets, cfg)
            if robot_wallets is not None
            else cfg.per_symbol_bot_equity()
        )
        candidates: List[Tuple[str, Any]] = []
        for sym in symbols:
            if session_opened.get(sym):
                continue
            if wallets is not None and float(wallets.get(sym, 0) or 0) <= 0:
                continue
            df5 = dfs5.get(sym)
            if df5 is None or df5.empty:
                continue
            ddf = _daily_df_asof(dfs_daily.get(sym, pd.DataFrame()), scan_ms)
            if wallets is not None:
                bot_eq = float(wallets.get(sym, cfg.per_symbol_bot_equity()))
            else:
                bot_eq = signal_equity
            sig = analyze_at_ms(
                sym,
                cfg=cfg,
                now_ms=scan_ms,
                session_traded=False,
                daily_df=ddf if not ddf.empty else None,
                bot_equity_usdt=bot_eq,
                df5=df5,
            )
            if not is_actionable(sig, cfg):
                continue
            candidates.append((sym, sig))

        if not candidates:
            continue

        sync_by_sym: Dict[str, int] = {}
        for sym, sig in candidates:
            side = str(sig.side)
            sync_by_sym[sym] = sum(1 for s2, g2 in candidates if s2 != sym and str(g2.side) == side)

        ranked = _score_and_sort_candidates(candidates, sync_by_sym, cfg, ranker)
        scan_et = pd.Timestamp(scan_ms, unit="ms", tz=tz).strftime("%H:%M")

        for sym, sig, sync_n, feat, p_rank in ranked:
            if session_opened.get(sym):
                continue
            if robot_reuse and robot_wallets is not None:
                _release_robots_through(robot_busy, robot_wallets, scan_ms)
                if len(robot_busy) >= gate.max_opens_per_day:
                    break
            elif not robot_reuse and state.opens >= gate.max_opens_per_day:
                break

            decision = evaluate_open_decision(
                ranker,
                symbol=sym,
                feat=feat,
                sync=sync_n,
                state=state,
                gate=gate,
                p_true=p_rank,
                p_fake=float(ranker.predict_fake(feat, symbol=sym)),
            )
            decision["scan_et"] = scan_et
            decision["scan_open_ms"] = int(scan_ms)

            if not decision.get("opened"):
                timeline.append(decision)
                continue

            ridx: Optional[int] = None
            if robot_wallets is not None:
                if robot_reuse:
                    ridx = _next_free_robot(robot_busy, robot_wallets)
                else:
                    ridx = _next_robot_index(robots_used_today, robot_wallets)

            if robot_wallets is not None and ridx is None:
                _finalize_blocked_open(decision, state, "no_robot_slot")
                timeline.append(decision)
                continue

            entry_bo = int(sig.entry_bar_open_ms or 0)
            trade_row: Optional[Dict[str, Any]] = None
            if entry_bo > 0:
                df1 = dfs1.get(sym)
                if robot_wallets is not None and ridx is not None:
                    notion = compute_position_notional(
                        entry=float(sig.price),
                        sl=float(sig.sl_price),
                        cfg=cfg,
                        bot_equity_usdt=robot_wallets[ridx],
                    )
                    trade_row = _resolve_trade_row(
                        sym=sym,
                        sig=sig,
                        session_date=session_date,
                        scan_ms=scan_ms,
                        entry_bo=entry_bo,
                        df1=df1,
                        close_ms=close,
                        bar=bar,
                        cfg=cfg,
                        notional=notion,
                        wallet_before=robot_wallets[ridx],
                        robot_id=ridx + 1,
                        scans=scans,
                    )
                elif df1 is not None and not df1.empty:
                    notion = float(sig.paper_notional_usdt or cfg.default_paper_notional())
                    wb = float(wallets.get(sym, 0) or 0) if wallets is not None else None
                    trade_row = _resolve_trade_row(
                        sym=sym,
                        sig=sig,
                        session_date=session_date,
                        scan_ms=scan_ms,
                        entry_bo=entry_bo,
                        df1=df1,
                        close_ms=close,
                        bar=bar,
                        cfg=cfg,
                        notional=notion,
                        wallet_before=wb,
                        scans=scans,
                    )

            if not trade_row:
                _finalize_blocked_open(decision, state, "no_trade_row")
                timeline.append(decision)
                continue

            _merge_trade_into_opened(state, trade_row)
            if robot_wallets is not None and ridx is not None and trade_row.get("pnl_usdt") is not None:
                if robot_reuse:
                    robot_busy[ridx] = {
                        "symbol": sym,
                        "exit_ms": int(trade_row.get("exit_ms") or scan_ms),
                        "pnl_usdt": float(trade_row["pnl_usdt"]),
                    }
                else:
                    robot_wallets[ridx] = round(float(robot_wallets[ridx]) + float(trade_row["pnl_usdt"]), 4)
                    robots_used_today.add(ridx)
            elif wallets is not None and trade_row.get("pnl_usdt") is not None:
                wallets[sym] = round(float(wallets.get(sym, 0) or 0) + float(trade_row["pnl_usdt"]), 4)

            session_opened[sym] = True
            timeline.append(decision)

    if robot_reuse and robot_wallets is not None and robot_busy:
        flush_ms = int(close + bar)
        _release_robots_through(robot_busy, robot_wallets, flush_ms)

    summary = summarize_live_gate_day(state, gate)
    return {
        "session_date": session_date,
        "macro_skip_day": macro_skip_day,
        "macro_events": list(macro_events),
        "gate": {
            "max_opens": gate.max_opens_per_day,
            "min_p_true": gate.min_p_true,
            "min_goal_true": gate.min_goal_true,
        },
        **summary,
        "opened": state.opened,
        "skipped_sample": state.skipped[:12],
        "timeline": timeline,
    }


def run_gate_eval_sessions(
    *,
    dates: List[str],
    symbols: List[str],
    gate: LiveGateConfig,
    ranker: BreakoutRanker,
    respect_env_filters: bool = True,
    robot_wallets: Optional[List[float]] = None,
    label: str = "",
) -> List[Dict[str, Any]]:
    cfg = ml_cfg(respect_env_filters=respect_env_filters)
    days: List[Dict[str, Any]] = []
    for i, d in enumerate(dates, 1):
        if label:
            print(f"[gate_tune] {label} [{i}/{len(dates)}] {d} ...", flush=True)
        days.append(
            simulate_live_gate_day(
                d,
                symbols,
                cfg,
                ranker,
                gate,
                robot_wallets=robot_wallets,
            )
        )
    return days
