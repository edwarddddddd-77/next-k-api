#!/usr/bin/env python3
"""对比 V2 回测引擎 vs Live 开单逻辑 replay（同 K 线、同 Gate，不含 PnL）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402
from orb.ml.gate import (  # noqa: E402
    LiveGateConfig,
    LiveGateDayState,
    evaluate_open_decision,
    rollback_open_decision,
)
from orb.ml.features import extract_features  # noqa: E402
from orb.ml.ranker import BreakoutRanker  # noqa: E402
from orb.ml.samples import parse_symbol_list  # noqa: E402
from orb.core.backtest import _daily_df_asof, _iter_scan_ms  # noqa: E402
from orb.core.paper import analyze_at_ms, in_regular_session, is_actionable  # noqa: E402
from orb.core.session import extended_fetch_anchor_ms  # noqa: E402
from orb.core.session import session_anchor_ms, session_close_ms, session_day_str  # noqa: E402
from orb.core.signals import compute_position_notional  # noqa: E402
from orb.core.kline_cache import kline_path, load_klines, symbol_cache_dir  # noqa: E402
from orb.v2.paths import resolve_gate_config_path, resolve_gbm_path, resolve_profiles_path, resolve_symbols_path  # noqa: E402
from orb.v2.robots import (  # noqa: E402
    init_robot_wallets,
    next_free_robot as _next_free_robot,
    release_robots_through as _release_robots_through,
    robot_equity_for_signals as _robot_equity_for_signals,
)
from tools.orb.ml.eval_live_gate import (  # noqa: E402
    _cached_symbols,
    _ml_cfg,
    _resolve_trade_row,
    simulate_live_gate_day,
)

import pandas as pd  # noqa: E402


def _decision_key(d: Dict[str, Any]) -> Tuple:
    return (
        int(d.get("scan_open_ms") or 0),
        str(d.get("symbol") or ""),
        bool(d.get("opened")),
        str(d.get("reason") or ""),
        round(float(d.get("p_true") or 0), 4),
    )


def replay_live_open_timeline(
    session_date: str,
    symbols: List[str],
    cfg,
    ranker: BreakoutRanker,
    gate: LiveGateConfig,
    robot_wallets: List[float],
) -> List[Dict[str, Any]]:
    """按 paper.py 顺序 replay（analyze_at_ms + Gate 后分 robot），状态 in-memory。"""
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

    session_seen: Dict[str, bool] = {}
    robot_busy: Dict[int, Dict[str, Any]] = {}
    gate_state = LiveGateDayState()
    timeline: List[Dict[str, Any]] = []
    robot_reuse = bool(gate.robot_reuse_after_exit)

    for scan_ms in scans:
        if not in_regular_session(cfg, now_ms=scan_ms):
            continue
        if robot_reuse:
            _release_robots_through(robot_busy, robot_wallets, scan_ms)
        signal_equity = _robot_equity_for_signals(robot_wallets, cfg)
        scan_et = pd.Timestamp(scan_ms, unit="ms", tz=tz).strftime("%H:%M")

        candidates: List[Tuple[str, Any]] = []
        for sym in symbols:
            if session_seen.get(sym):
                continue
            if signal_equity <= 0:
                continue
            df5 = dfs5.get(sym)
            if df5 is None or df5.empty:
                continue
            ddf = _daily_df_asof(dfs_daily.get(sym, pd.DataFrame()), scan_ms)
            sig = analyze_at_ms(
                sym,
                cfg=cfg,
                now_ms=scan_ms,
                session_traded=False,
                daily_df=ddf if not ddf.empty else None,
                bot_equity_usdt=signal_equity,
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

        scored: List[Tuple[float, str, Any, int, Dict[str, float]]] = []
        for sym, sig in candidates:
            sync_n = int(sync_by_sym.get(sym, 0))
            feat = extract_features(sig, cfg, sync_same_side=sync_n)
            p_true = float(ranker.predict_true(feat, symbol=sym))
            scored.append((p_true, sym, sig, sync_n, feat))
        scored.sort(key=lambda x: x[0], reverse=True)

        for p_true, sym, sig, sync_n, feat in scored:
            if robot_reuse and len(robot_busy) >= gate.max_opens_per_day:
                break
            if not robot_reuse and gate_state.opens >= gate.max_opens_per_day:
                break

            decision = evaluate_open_decision(
                ranker,
                symbol=sym,
                feat=feat,
                sync=sync_n,
                state=gate_state,
                gate=gate,
                p_true=p_true,
                p_fake=float(ranker.predict_fake(feat, symbol=sym)),
            )
            decision["scan_et"] = scan_et
            decision["scan_open_ms"] = int(scan_ms)

            if not decision.get("opened"):
                session_seen[sym] = True
                timeline.append(decision)
                continue

            ridx = _next_free_robot(robot_busy, robot_wallets)
            if ridx is None:
                rollback_open_decision(gate_state, symbol=sym)
                decision["opened"] = False
                decision["reason"] = "no_robot_slot"
                session_seen[sym] = True
                timeline.append(decision)
                continue

            entry_bo = int(sig.entry_bar_open_ms or 0)
            trade_row = None
            if entry_bo > 0:
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
                    df1=dfs1.get(sym),
                    close_ms=close,
                    bar=bar,
                    cfg=cfg,
                    notional=notion,
                    wallet_before=robot_wallets[ridx],
                    robot_id=ridx + 1,
                )
            if not trade_row:
                rollback_open_decision(gate_state, symbol=sym)
                decision["opened"] = False
                decision["reason"] = "no_trade_row"
                session_seen[sym] = True
                timeline.append(decision)
                continue

            robot_busy[ridx] = {
                "symbol": sym,
                "exit_ms": int(trade_row.get("exit_ms") or scan_ms),
                "pnl_usdt": float(trade_row.get("pnl_usdt") or 0),
            }
            decision["robot_id"] = ridx + 1
            session_seen[sym] = True
            timeline.append(decision)

    if robot_busy:
        _release_robots_through(robot_busy, robot_wallets, int(close + bar))
    return timeline


def compare_day(
    session_date: str,
    symbols: List[str],
    cfg,
    ranker: BreakoutRanker,
    gate: LiveGateConfig,
) -> Dict[str, Any]:
    wallets = init_robot_wallets(count=8, equity_usdt=10_000.0)
    bt = simulate_live_gate_day(
        session_date, symbols, cfg, ranker, gate, robot_wallets=list(wallets)
    )
    live = replay_live_open_timeline(
        session_date, symbols, cfg, ranker, gate, robot_wallets=list(wallets)
    )

    bt_tl = bt.get("timeline") or []
    bt_keys = [_decision_key(d) for d in bt_tl]
    live_keys = [_decision_key(d) for d in live]

    bt_opens = {
        (d.get("symbol"), round(float(d.get("p_true") or 0), 4))
        for d in bt.get("opened") or []
    }
    live_opens = {
        (d.get("symbol"), round(float(d.get("p_true") or 0), 4))
        for d in live
        if d.get("opened")
    }

    mismatches = []
    n = max(len(bt_keys), len(live_keys))
    for i in range(n):
        bk = bt_keys[i] if i < len(bt_keys) else None
        lk = live_keys[i] if i < len(live_keys) else None
        if bk != lk:
            mismatches.append({"idx": i, "backtest": bk, "live_replay": lk})

    return {
        "session_date": session_date,
        "backtest_timeline": len(bt_tl),
        "live_timeline": len(live),
        "backtest_opens": len(bt.get("opened") or []),
        "live_opens": len(live_opens),
        "opens_match": bt_opens == live_opens,
        "timeline_match": bt_keys == live_keys,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
        "backtest_open_symbols": sorted({str(x[0]) for x in bt_opens}),
        "live_open_symbols": sorted({str(x[0]) for x in live_opens}),
    }


def main() -> int:
    load_env_oi()
    ap = argparse.ArgumentParser(description="Verify backtest vs live open-rule parity")
    ap.add_argument("--from-date", default="2026-03-19")
    ap.add_argument("--to-date", default="2026-06-12")
    ap.add_argument("--symbols-file", default=str(resolve_symbols_path()))
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    syms = _cached_symbols(parse_symbol_list(Path(args.symbols_file).read_text(encoding="utf-8")))
    if not syms:
        print("No cached symbols")
        return 1

    cfg = _ml_cfg(compound_per_symbol=True, respect_env_filters=True)
    gate = LiveGateConfig.from_json(Path(resolve_gate_config_path()))
    ranker = BreakoutRanker.load(
        gbm_path=resolve_gbm_path(),
        profiles_path=resolve_profiles_path(),
    )

    from tools.orb.v2.backtest_symbol import session_dates_from_cache

    all_dates = session_dates_from_cache(syms[0], cfg)
    dates = [d for d in all_dates if args.from_date <= d <= args.to_date]
    if not dates:
        print("No dates")
        return 1

    print(
        f"parity check | {len(syms)} syms | {dates[0]}..{dates[-1]} | "
        f"{len(dates)} days | macro={cfg.macro_filter}",
        flush=True,
    )

    results = []
    ok_days = 0
    for i, d in enumerate(dates, 1):
        r = compare_day(d, syms, cfg, ranker, gate)
        results.append(r)
        if r["timeline_match"] and r["opens_match"]:
            ok_days += 1
        else:
            print(
                f"  MISMATCH {d}: tl={r['backtest_timeline']}/{r['live_timeline']} "
                f"opens={r['backtest_opens']}/{r['live_opens']} "
                f"diffs={r['mismatch_count']}",
                flush=True,
            )
        if i % 10 == 0:
            print(f"  [{i}/{len(dates)}] ok={ok_days}", flush=True)

    summary = {
        "date_range": {"from": dates[0], "to": dates[-1], "days": len(dates)},
        "filters": {"macro": cfg.macro_filter},
        "days_total": len(dates),
        "days_timeline_match": ok_days,
        "days_timeline_mismatch": len(dates) - ok_days,
        "mismatch_days": [r for r in results if not r["timeline_match"]],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    out = Path(args.json_out) if args.json_out.strip() else ROOT / "output" / "orb" / "v2" / "eval" / "live_backtest_parity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "days": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nfull -> {out}")
    return 0 if ok_days == len(dates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
