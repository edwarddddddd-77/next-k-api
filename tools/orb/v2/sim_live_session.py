#!/usr/bin/env python3
"""单日/区间 session 实盘逻辑 replay（对齐 orb/v2/paper.py + df5_for_breakout_score）。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402
from orb.core.backtest import _daily_df_asof, _iter_scan_ms  # noqa: E402
from orb.core.breakout_score import breakout_kline_range_ms  # noqa: E402
from orb.core.config import OrbConfig  # noqa: E402
from orb.core.kline_cache import load_klines  # noqa: E402
from orb.core.macro_calendar import is_macro_skip_day  # noqa: E402
from orb.core.or_entry_fill import resolve_entry_fill  # noqa: E402
from orb.core.paper import (  # noqa: E402
    _signal_df_from_bars,
    analyze_at_ms,
    in_regular_session,
    is_actionable,
)
from orb.core.session import session_anchor_ms, session_close_ms, session_day_str  # noqa: E402
from orb.core.signals import compute_position_notional  # noqa: E402
from orb.ml.features import extract_features  # noqa: E402
from orb.ml.gate import LiveGateConfig, LiveGateDayState, evaluate_open_decision, evaluate_open_decision_without_ml, gate_ml_enabled_from_env, rollback_open_decision  # noqa: E402
from orb.ml.model import BreakoutModelBundle  # noqa: E402
from orb.ml.samples import parse_symbol_list  # noqa: E402
from orb.v2.paper import _paper_breakout_score  # noqa: E402
from orb.v2.paths import resolve_gate_config_path, resolve_symbols_path  # noqa: E402
from orb.v2.robots import (  # noqa: E402
    bound_robot_index_available,
    init_robot_wallets,
    next_free_robot as _next_free_robot,
    release_robots_through as _release_robots_through,
    robot_bound_mode,
    robot_equity_for_signals as _robot_equity_for_signals,
    robot_count_from_env,
    robot_equity_from_env,
    robot_symbol_bindings,
    symbol_to_robot_index,
)
from tools.orb.ml.eval_live_gate import _ml_cfg  # noqa: E402
from tools.orb.v2.backtest_universe import universe_session_dates  # noqa: E402

import pandas as pd  # noqa: E402

DEFAULT_FEE_BPS_PER_SIDE = 4.0  # 单边费率 bps；往返开平 ×2


def trade_fee_usdt(notional_usdt: float, *, fee_bps_per_side: float) -> float:
    """按名义本金计费：fee = notional × (bps/10000) × 2（开+平）。"""
    n = max(0.0, float(notional_usdt or 0.0))
    bps = max(0.0, float(fee_bps_per_side))
    return round(n * (bps / 10000.0) * 2.0, 4)


TRADE_CSV_FIELDS = [
    "session_date",
    "scan_et",
    "scan_open_ms",
    "symbol",
    "side",
    "entry",
    "notional_usdt",
    "p_true",
    "breakout_score",
    "robot_id",
    "wallet_before",
    "wallet_after",
    "pnl_usdt_gross",
    "fee_usdt",
    "pnl_usdt",
    "true_breakout",
    "outcome",
    "exit_ms",
    "entry_mode",
    "signal_entry",
    "fill_bar_open_ms",
    "chase_slip",
]

DAILY_CSV_FIELDS = [
    "session_date",
    "macro_skip_day",
    "opens",
    "gate_skips",
    "bs_skips",
    "gross_pnl_usdt",
    "fees_usdt",
    "net_pnl_usdt",
    "robot_wallets_end",
]


def _sim_load_signal_df(full_df5: pd.DataFrame, cfg: OrbConfig, *, now_ms: int) -> pd.DataFrame:
    """等价于实盘 scan 时刻的 _load_signal_df（9:30..now_ms）。"""
    return _signal_df_from_bars(full_df5, cfg, now_ms=int(now_ms))


def simulate_live_session(
    session_date: str,
    symbols: List[str],
    *,
    gate: LiveGateConfig,
    ranker,
    cfg: OrbConfig,
    robot_wallets: List[float],
    respect_env_filters: bool = True,
    fee_bps_per_side: float = DEFAULT_FEE_BPS_PER_SIDE,
    entry_fill: str = "signal",
    ml_enabled: bool = True,
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

    macro_skip = bool(respect_env_filters and cfg.macro_filter and is_macro_skip_day(session_date))

    fetch_start, end_ms = breakout_kline_range_ms(session_date, cfg)
    dfs5: Dict[str, pd.DataFrame] = {}
    dfs1: Dict[str, pd.DataFrame] = {}
    dfs_daily: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        dfs5[sym] = load_klines(sym, cfg.signal_interval, start_ms=fetch_start, end_ms=end_ms)
        dfs1[sym] = load_klines(sym, "1m", start_ms=fetch_start, end_ms=end_ms)
        if (cfg.sl_mode or "").strip().lower() == "atr_pct":
            dfs_daily[sym] = load_klines(
                sym, "1d", start_ms=fetch_start - cfg.daily_atr_warmup_ms(), end_ms=end_ms
            )

    gate_state = LiveGateDayState()
    session_opened: Dict[str, bool] = {}
    robot_busy: Dict[int, Dict[str, Any]] = {}
    robot_reuse = bool(gate.robot_reuse_after_exit)
    robot_bound = robot_bound_mode(symbol_count=len(symbols), robot_count=len(robot_wallets))
    need_breakout_score = ml_enabled and float(gate.min_breakout_score or 0) > 0

    timeline: List[Dict[str, Any]] = []
    trades: List[Dict[str, Any]] = []
    gate_skips: List[Dict[str, Any]] = []

    for scan_ms in scans:
        if not in_regular_session(cfg, now_ms=scan_ms):
            continue
        if macro_skip:
            continue
        if robot_reuse:
            _release_robots_through(robot_busy, robot_wallets, scan_ms)
        signal_equity = _robot_equity_for_signals(robot_wallets, cfg)
        scan_et = pd.Timestamp(scan_ms, unit="ms", tz=tz).strftime("%H:%M")

        # 每个 scan 独立 df5_cache（与 paper.py 单次 run_scan 一致）
        df5_cache: Dict[str, Any] = {}

        candidates: List[Tuple[str, Any]] = []
        for sym in symbols:
            if session_opened.get(sym):
                continue
            if robot_bound:
                ridx = symbol_to_robot_index(sym, symbols)
                if ridx is None or ridx >= len(robot_wallets) or float(robot_wallets[ridx]) <= 0:
                    continue
                bot_equity = float(robot_wallets[ridx])
            else:
                if signal_equity <= 0:
                    continue
                bot_equity = signal_equity
            full5 = dfs5.get(sym)
            if full5 is None or full5.empty:
                continue
            df5_cache[sym] = _sim_load_signal_df(full5, cfg, now_ms=int(scan_ms))
            ddf = _daily_df_asof(dfs_daily.get(sym, pd.DataFrame()), scan_ms)
            sig = analyze_at_ms(
                sym,
                cfg=cfg,
                now_ms=int(scan_ms),
                session_traded=False,
                daily_df=ddf if not ddf.empty else None,
                bot_equity_usdt=bot_equity,
                df5=df5_cache[sym],
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

        scored: List[Tuple[float, str, Any, int, Dict[str, float]]] = []
        for sym, sig in candidates:
            sync_n = int(sync_by_sym.get(sym, 0))
            feat = extract_features(sig, cfg, sync_same_side=sync_n)
            if ml_enabled and ranker is not None:
                p_true = float(ranker.predict_true(feat, symbol=sym))
            else:
                p_true = 1.0
            scored.append((p_true, sym, sig, sync_n, feat))
        scored.sort(key=lambda x: x[0], reverse=True)

        for p_true, sym, sig, sync_n, feat in scored:
            if session_opened.get(sym):
                continue
            if robot_bound:
                if robot_reuse:
                    _release_robots_through(robot_busy, robot_wallets, scan_ms)
            elif robot_reuse:
                _release_robots_through(robot_busy, robot_wallets, scan_ms)
                if len(robot_busy) >= gate.max_opens_per_day:
                    break
            elif gate_state.opens >= gate.max_opens_per_day:
                break

            breakout_score: Optional[float] = None
            if need_breakout_score:
                breakout_score = _paper_breakout_score(
                    sym,
                    sig,
                    cfg,
                    session_day=session_date,
                    now_ms=int(scan_ms),
                    df5_cache=df5_cache,
                )

            if ml_enabled and ranker is not None:
                decision = evaluate_open_decision(
                    ranker,
                    symbol=sym,
                    feat=feat,
                    sync=sync_n,
                    state=gate_state,
                    gate=gate,
                    p_true=p_true,
                    p_fake=float(ranker.predict_fake(feat, symbol=sym)),
                    breakout_score=breakout_score,
                )
            else:
                decision = evaluate_open_decision_without_ml(
                    symbol=sym,
                    feat=feat,
                    sync=sync_n,
                    state=gate_state,
                    gate=gate,
                    breakout_score=breakout_score,
                )
            decision["scan_et"] = scan_et
            decision["scan_open_ms"] = int(scan_ms)
            decision["side"] = str(sig.side)
            decision["entry"] = float(sig.price)

            if not decision.get("opened"):
                gate_skips.append(
                    {
                        "scan_et": scan_et,
                        "symbol": sym,
                        "side": sig.side,
                        "p_true": p_true,
                        "breakout_score": breakout_score,
                        "sync": sync_n,
                        "reason": decision.get("reason"),
                    }
                )
                timeline.append(decision)
                continue

            if robot_bound:
                ridx = bound_robot_index_available(robot_busy, robot_wallets, sym, symbols)
            else:
                ridx = _next_free_robot(robot_busy, robot_wallets)
            if ridx is None:
                rollback_open_decision(gate_state, symbol=sym)
                decision["opened"] = False
                decision["reason"] = "robot_busy" if robot_bound else "no_robot_slot"
                gate_skips.append({**decision, "sync": sync_n})
                timeline.append(decision)
                continue

            entry_bo = int(sig.entry_bar_open_ms or 0)
            trade_row = None
            fill_reason = "ok"
            if entry_bo > 0:
                notion = compute_position_notional(
                    entry=float(sig.price),
                    sl=float(sig.sl_price),
                    cfg=cfg,
                    bot_equity_usdt=robot_wallets[ridx],
                )
                trade_row, fill_reason = resolve_entry_fill(
                    mode=entry_fill,
                    sym=sym,
                    sig=sig,
                    session_date=session_date,
                    scan_ms=scan_ms,
                    df1=dfs1.get(sym),
                    df5=df5_cache.get(sym),
                    close_ms=close,
                    bar=bar,
                    cfg=cfg,
                    notional=notion,
                    wallet_before=robot_wallets[ridx],
                    robot_id=ridx + 1,
                    scans=scans,
                )
            if not trade_row:
                rollback_open_decision(gate_state, symbol=sym)
                decision["opened"] = False
                decision["reason"] = fill_reason if entry_bo > 0 else "no_entry_bar"
                gate_skips.append({**decision, "sync": sync_n})
                timeline.append(decision)
                continue

            gross = float(trade_row.get("pnl_usdt") or 0)
            fee = trade_fee_usdt(
                float(trade_row.get("notional_usdt") or 0),
                fee_bps_per_side=fee_bps_per_side,
            )
            net = round(gross - fee, 2)
            trade_row["pnl_usdt_gross"] = round(gross, 2)
            trade_row["pnl_usdt"] = net
            trade_row["fee_usdt"] = fee
            trade_row["breakout_score"] = breakout_score
            trade_row["p_true"] = p_true
            trade_row["scan_et"] = scan_et
            trade_row["symbol"] = sym
            trades.append(trade_row)

            robot_busy[ridx] = {
                "symbol": sym,
                "exit_ms": int(trade_row.get("exit_ms") or scan_ms),
                "pnl_usdt": net,
            }
            robot_wallets[ridx] = round(float(trade_row.get("wallet_after") or robot_wallets[ridx]), 2)
            decision["robot_id"] = ridx + 1
            decision["pnl_usdt"] = net
            decision["outcome"] = trade_row.get("outcome")
            session_opened[sym] = True
            timeline.append(decision)

    if robot_busy:
        _release_robots_through(robot_busy, robot_wallets, int(close + bar))

    gross_pnl = round(sum(float(t.get("pnl_usdt_gross") or 0) for t in trades), 2)
    net_pnl = round(sum(float(t.get("pnl_usdt") or 0) for t in trades), 2)
    fees_pnl = round(sum(float(t.get("fee_usdt") or 0) for t in trades), 2)
    bs_skips = sum(1 for g in gate_skips if str(g.get("reason") or "").startswith("breakout_score"))
    fill_skips = sum(1 for g in gate_skips if str(g.get("reason") or "") == "or_limit_not_filled")

    return {
        "session_date": session_date,
        "macro_skip_day": macro_skip,
        "entry_fill": entry_fill,
        "fill_skips": fill_skips,
        "gate": gate.__dict__,
        "robot_bound": robot_bound,
        "robot_bindings": robot_symbol_bindings(symbols) if robot_bound else None,
        "robot_equity_usdt": robot_equity_from_env(),
        "robot_count": len(robot_wallets),
        "fee_bps_per_side": float(fee_bps_per_side),
        "opens": len(trades),
        "gate_skips": len(gate_skips),
        "bs_skips": bs_skips,
        "gross_pnl_usdt": gross_pnl,
        "net_pnl_usdt": net_pnl,
        "fees_usdt": fees_pnl,
        "trades": trades,
        "gate_skip_detail": gate_skips,
        "timeline_len": len(timeline),
        "robot_wallets_end": {f"R{i + 1}": round(w, 2) for i, w in enumerate(robot_wallets)},
    }


def simulate_live_sessions(
    dates: List[str],
    symbols: List[str],
    *,
    gate: LiveGateConfig,
    ranker,
    cfg: OrbConfig,
    robot_wallets: List[float],
    respect_env_filters: bool = True,
    fee_bps_per_side: float = DEFAULT_FEE_BPS_PER_SIDE,
    entry_fill: str = "signal",
    ml_enabled: bool = True,
) -> List[Dict[str, Any]]:
    days: List[Dict[str, Any]] = []
    for d in dates:
        day = simulate_live_session(
            d,
            symbols,
            gate=gate,
            ranker=ranker,
            cfg=cfg,
            robot_wallets=robot_wallets,
            respect_env_filters=respect_env_filters,
            fee_bps_per_side=fee_bps_per_side,
            entry_fill=entry_fill,
            ml_enabled=ml_enabled,
        )
        days.append(day)
    return days


def _write_trades_csv(days: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TRADE_CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for day in days:
            for t in day.get("trades") or []:
                row = dict(t)
                row["session_date"] = day["session_date"]
                w.writerow(row)


def _write_daily_csv(days: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DAILY_CSV_FIELDS)
        w.writeheader()
        for day in days:
            w.writerow(
                {
                    "session_date": day["session_date"],
                    "macro_skip_day": day.get("macro_skip_day"),
                    "opens": day.get("opens"),
                    "gate_skips": day.get("gate_skips"),
                    "bs_skips": day.get("bs_skips"),
                    "gross_pnl_usdt": day.get("gross_pnl_usdt"),
                    "fees_usdt": day.get("fees_usdt"),
                    "net_pnl_usdt": day.get("net_pnl_usdt"),
                    "robot_wallets_end": json.dumps(day.get("robot_wallets_end") or {}, ensure_ascii=False),
                }
            )


def _resolve_dates(args, syms: List[str], cfg: OrbConfig) -> List[str]:
    all_dates = universe_session_dates(syms, cfg)
    if not all_dates:
        return []
    d0 = (args.from_date or "").strip()
    d1 = (args.to_date or "").strip()
    if d0 or d1:
        lo = d0 or all_dates[0]
        hi = d1 or all_dates[-1]
        return [d for d in all_dates if lo <= d <= hi]
    if (args.date or "").strip():
        return [args.date.strip()]
    return all_dates


def _print_day_trades(day: Dict[str, Any]) -> None:
    sd = day["session_date"]
    print(
        f"## {sd}  net={float(day.get('net_pnl_usdt') or 0):+.2f}U  "
        f"opens={day.get('opens')}  bs_skips={day.get('bs_skips')}"
        + ("  [macro skip]" if day.get("macro_skip_day") else "")
    )
    for t in day.get("trades") or []:
        print(
            f"  {t.get('scan_et',''):<5} {str(t.get('symbol','')):<10} {str(t.get('side','')):<5} "
            f"P={float(t.get('p_true') or 0):.2f} BS={float(t.get('breakout_score') or 0):.1f} "
            f"R{int(t.get('robot_id') or 0)} {float(t.get('pnl_usdt') or 0):+.2f}U "
            f"{t.get('outcome','')}"
        )


def main() -> int:
    load_env_oi()
    ap = argparse.ArgumentParser(description="Simulate session(s) with live paper.py logic")
    ap.add_argument("--date", default="", help="单日 YYYY-MM-DD")
    ap.add_argument("--from-date", default="")
    ap.add_argument("--to-date", default="")
    ap.add_argument("--symbols-file", default=str(resolve_symbols_path()))
    ap.add_argument("--symbols", default="", help="逗号分隔标的，如 COIN（覆盖 symbols-file）")
    ap.add_argument("--gate-config", default=str(resolve_gate_config_path()))
    ap.add_argument("--min-p", type=float, default=None, help="覆盖 gate min_p_true")
    ap.add_argument("--min-bs", type=float, default=None, help="覆盖 gate min_breakout_score")
    ap.add_argument("--max-opens", type=int, default=None, help="覆盖 gate max_opens_per_day")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--csv-out", default="")
    ap.add_argument("--daily-csv-out", default="")
    ap.add_argument("--robot-equity", type=float, default=0.0, help="每 robot 初始资金 U（覆盖 .env.oi）")
    ap.add_argument(
        "--fee-bps",
        type=float,
        default=DEFAULT_FEE_BPS_PER_SIDE,
        help="单边手续费 bps，往返开平按 notional×2 计费（默认 4=0.04%%）",
    )
    ap.add_argument("--no-live-filters", action="store_true")
    ap.add_argument(
        "--entry-fill",
        default="signal",
        choices=("signal", "stoplimit", "stoplimit_gap", "stoplimit_honest", "stoplimit_gap_honest", "market"),
        help="成交模型：signal=理想信号价 | stoplimit_gap=OR Stop-Limit+gap | market=5m收盘价追单",
    )
    ap.add_argument("--robot-count", type=int, default=0, help="覆盖 robot 数量（0=env）")
    ap.add_argument("--or-minutes", type=int, default=0, help="覆盖 OR 窗口分钟（0=env，如 15/30）")
    ap.add_argument("--no-robot-reset", action="store_true", help="关闭 robot cap/floor 提现重置")
    ap.add_argument("--no-gate-ml", action="store_true", help="绕过 ML p / BS / early-trap（等同 ORB_V2_GATE_ML=0）")
    ap.add_argument("--quiet-detail", action="store_true")
    args = ap.parse_args()

    if args.no_robot_reset:
        import os

        os.environ["ORB_V2_ROBOT_RESET_CAP"] = "0"
    if args.no_gate_ml:
        import os

        os.environ["ORB_V2_GATE_ML"] = "0"

    if (args.symbols or "").strip():
        from orb.core.kline_cache import norm_symbol

        syms = [norm_symbol(s.strip()) for s in args.symbols.split(",") if s.strip()]
    else:
        syms = parse_symbol_list(Path(args.symbols_file).read_text(encoding="utf-8"))
    gate = LiveGateConfig.from_json(Path(args.gate_config))
    if args.min_p is not None:
        gate.min_p_true = float(args.min_p)
    if args.min_bs is not None:
        gate.min_breakout_score = float(args.min_bs)
    if args.max_opens is not None:
        gate.max_opens_per_day = int(args.max_opens)
    if args.no_gate_ml or not gate_ml_enabled_from_env():
        from orb.ml.gate import gate_with_ml_bypass

        gate = gate_with_ml_bypass(gate)
    cfg = _ml_cfg(compound_per_symbol=True, respect_env_filters=not bool(args.no_live_filters))
    if int(args.or_minutes) > 0:
        cfg.or_minutes = int(args.or_minutes)
    ml_enabled = gate_ml_enabled_from_env() and not args.no_gate_ml
    ranker = None
    if ml_enabled:
        model = BreakoutModelBundle.load_production()
        if not model.is_ready:
            print("ML model not ready")
            return 1
        ranker = model.ranker

    dates = _resolve_dates(args, syms, cfg)
    if not dates:
        print("No session dates in cache for range")
        return 1

    rc = int(args.robot_count) if int(args.robot_count) > 0 else robot_count_from_env()
    if robot_bound_mode(symbol_count=len(syms), robot_count=rc):
        rc = len(syms)
    re = float(args.robot_equity) if float(args.robot_equity) > 0 else robot_equity_from_env()
    fee_bps = max(0.0, float(args.fee_bps))
    wallets = init_robot_wallets(count=rc, equity_usdt=re)
    bound = robot_bound_mode(symbol_count=len(syms), robot_count=rc)

    print(
        f"[live sim] {dates[0]} .. {dates[-1]} | {len(dates)} sessions | "
        f"{len(syms)} syms | ml={'on' if ml_enabled else 'off'} | "
        f"gate p>={gate.min_p_true} bs>={gate.min_breakout_score:.0f} | "
        f"fill={args.entry_fill} or={cfg.or_minutes}m risk={cfg.risk_pct} | "
        f"robots={rc}x{re}U bound={bound} reset={not bool(args.no_robot_reset)} | fee={fee_bps}bps/side×2",
        flush=True,
    )

    t0 = time.time()
    days: List[Dict[str, Any]] = []
    for i, d in enumerate(dates, 1):
        print(f"[{i}/{len(dates)}] {d} ...", flush=True)
        day = simulate_live_session(
            d,
            syms,
            gate=gate,
            ranker=ranker,
            cfg=cfg,
            robot_wallets=wallets,
            respect_env_filters=not bool(args.no_live_filters),
            fee_bps_per_side=fee_bps,
            entry_fill=str(args.entry_fill),
            ml_enabled=ml_enabled,
        )
        days.append(day)
        if not args.quiet_detail:
            _print_day_trades(day)

    total_opens = sum(int(d.get("opens") or 0) for d in days)
    total_net = round(sum(float(d.get("net_pnl_usdt") or 0) for d in days), 2)
    total_gross = round(sum(float(d.get("gross_pnl_usdt") or 0) for d in days), 2)
    total_fees = round(sum(float(d.get("fees_usdt") or 0) for d in days), 2)
    total_bs_skips = sum(int(d.get("bs_skips") or 0) for d in days)
    total_fill_skips = sum(int(d.get("fill_skips") or 0) for d in days)

    tag = dates[0] if len(dates) == 1 else f"{dates[0]}_{dates[-1]}"
    eq_tag = f"_eq{int(re)}" if float(args.robot_equity) > 0 else ""
    sym_tag = ""
    if (args.symbols or "").strip():
        sym_tag = "_" + "_".join(s.replace("USDT", "") for s in syms[:3])
        if len(syms) > 3:
            sym_tag += f"_x{len(syms)}"
    fill_tag = f"_{args.entry_fill}" if args.entry_fill != "signal" else ""
    filter_tag = "_filter" if ml_enabled else "_no_filter"
    reset_tag = "_no_reset" if args.no_robot_reset else ""
    out_dir = ROOT / "output" / "orb" / "v2" / "eval"
    json_path = (
        Path(args.json_out)
        if args.json_out.strip()
        else out_dir / f"live_sim{fill_tag}{sym_tag}_{tag}{eq_tag}{filter_tag}{reset_tag}.json"
    )
    csv_path = (
        Path(args.csv_out)
        if args.csv_out.strip()
        else out_dir / f"live_sim{fill_tag}{sym_tag}_{tag}{eq_tag}{filter_tag}{reset_tag}.trades.csv"
    )
    daily_csv_path = (
        Path(args.daily_csv_out)
        if args.daily_csv_out.strip()
        else out_dir / f"live_sim{fill_tag}{sym_tag}_{tag}{eq_tag}{filter_tag}{reset_tag}.daily.csv"
    )

    payload = {
        "rule": "live_paper_v2_replay",
        "entry_fill": args.entry_fill,
        "gate_ml_enabled": ml_enabled,
        "or_minutes": cfg.or_minutes,
        "robot_reset": not bool(args.no_robot_reset),
        "risk_pct": cfg.risk_pct,
        "symbols": syms,
        "date_range": {"from": dates[0], "to": dates[-1], "sessions": len(dates)},
        "gate": gate.__dict__,
        "robot_equity_usdt": re,
        "robot_count": rc,
        "robot_bound": bound,
        "robot_bindings": robot_symbol_bindings(syms) if bound else None,
        "fee_model": {
            "bps_per_side": fee_bps,
            "round_trip": True,
            "formula": "notional * (bps/10000) * 2",
        },
        "summary": {
            "total_opens": total_opens,
            "total_gross_pnl_usdt": total_gross,
            "total_fees_usdt": total_fees,
            "total_net_pnl_usdt": total_net,
            "total_bs_skips": total_bs_skips,
            "total_fill_skips": total_fill_skips,
            "robot_wallets_end": {f"R{i + 1}": round(w, 2) for i, w in enumerate(wallets)},
            "elapsed_sec": round(time.time() - t0, 1),
        },
        "days": days,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_trades_csv(days, csv_path)
    _write_daily_csv(days, daily_csv_path)

    print()
    print(
        f"SUMMARY {len(dates)} sessions | opens={total_opens} | "
        f"gross={total_gross}U net={total_net}U fees={total_fees}U "
        f"bs_skips={total_bs_skips} fill_skips={total_fill_skips}"
    )
    print(f"json       -> {json_path}")
    print(f"trades csv -> {csv_path}")
    print(f"daily csv  -> {daily_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
