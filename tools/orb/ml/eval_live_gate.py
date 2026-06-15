#!/usr/bin/env python3
"""scan-by-scan 模拟 live_gate：突破当下打分，过线才开（robot 模式可 SL/TP 后复用 slot）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402
from orb.ml.model import BreakoutModelBundle  # noqa: E402
from orb.ml.samples import parse_symbol_list  # noqa: E402
from orb.ml.gate import LiveGateConfig  # noqa: E402
from orb.ml.paths import V2_LIVE_GATE_EVAL, ensure_v2_dirs  # noqa: E402
from orb.v2.paths import resolve_gate_config_path, resolve_symbols_path  # noqa: E402
from orb.ml.live_gate_sim import (  # noqa: E402
    cached_symbols as _cached_symbols,
    init_symbol_wallets,
    ml_cfg as _ml_cfg,
    run_gate_eval_sessions,
    simulate_live_gate_day,
    trading_dates_from_samples as _trading_dates_from_samples,
)
from orb.v2.robots import init_robot_wallets  # noqa: E402
from orb.v2.robots import robot_count_from_env, robot_equity_from_env  # noqa: E402

# 向后兼容：backtest / parity 从此模块 import
__all__ = [
    "_ml_cfg",
    "_cached_symbols",
    "simulate_live_gate_day",
    "init_robot_wallets",
    "init_symbol_wallets",
    "_trading_dates_from_samples",
]


def main() -> int:
    load_env_oi()
    ap = argparse.ArgumentParser(description="Live gate scan-by-scan simulation")
    ap.add_argument("--date", default="")
    ap.add_argument(
        "--dates",
        default="2026-05-29,2026-06-01,2026-06-02,2026-06-03,2026-06-04,2026-06-05,2026-06-08,2026-06-09,2026-06-10,2026-06-11",
    )
    ap.add_argument("--symbols-file", default=str(resolve_symbols_path()))
    ap.add_argument("--last-sessions", type=int, default=0, help="最近 N 个 NYSE 交易日（从样本推断）")
    ap.add_argument("--last-days", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--end-date", default="", help="配合 --last-sessions，默认样本最后交易日")
    ap.add_argument("--gate-config", default=str(resolve_gate_config_path()))
    ap.add_argument("--min-p", type=float, default=None, help="override min_p_true")
    ap.add_argument("--json-out", default=str(V2_LIVE_GATE_EVAL))
    ap.add_argument(
        "--no-live-filters",
        action="store_true",
        help="关闭 env 宏观过滤（旧回测口径）",
    )
    ap.add_argument("--robot-count", type=int, default=0)
    ap.add_argument("--robot-equity", type=float, default=0.0)
    args = ap.parse_args()

    ensure_v2_dirs()
    gate = LiveGateConfig.from_json(Path(args.gate_config))
    if args.min_p is not None:
        gate.min_p_true = float(args.min_p)

    syms = _cached_symbols(parse_symbol_list(Path(args.symbols_file).read_text(encoding="utf-8")))
    model = BreakoutModelBundle.load_production()
    ranker = model.ranker
    if not model.is_ready:
        print("ML model not ready — check data/orb/live/", flush=True)
        return 1
    last_n = int(args.last_sessions) or int(args.last_days)
    if last_n > 0:
        dates = _trading_dates_from_samples(last_sessions=last_n, end_date=args.end_date.strip())
    elif args.date.strip():
        dates = [args.date.strip()]
    else:
        dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    if not dates:
        print("No session dates")
        return 1
    print(f"evaluating {len(dates)} days: {dates[0]} .. {dates[-1]}", flush=True)

    rc = int(args.robot_count) if int(args.robot_count) > 0 else robot_count_from_env()
    re = float(args.robot_equity) if float(args.robot_equity) > 0 else robot_equity_from_env()
    robots = init_robot_wallets(count=rc, equity_usdt=re)
    days = run_gate_eval_sessions(
        dates=dates,
        symbols=syms,
        gate=gate,
        ranker=ranker,
        respect_env_filters=not bool(args.no_live_filters),
        robot_wallets=robots,
    )
    hit_min = sum(1 for x in days if x["goal_met_min"])
    hit_tgt = sum(1 for x in days if x["goal_met_target"])
    avg_opens = round(sum(x["opens"] for x in days) / len(days), 2) if days else 0
    avg_true = round(sum(x["true_opens"] for x in days) / len(days), 2) if days else 0

    out = {
        "rule": "breakout_scan_score_then_open_if_p>=min_p",
        "ranker": ranker.kind,
        "date_range": {"from": dates[0], "to": dates[-1], "days": len(days)},
        "gate": gate.__dict__,
        "summary": {
            "days": len(days),
            "goal_min_hit_days": hit_min,
            "goal_target_hit_days": hit_tgt,
            "avg_opens_per_day": avg_opens,
            "avg_true_opens_per_day": avg_true,
        },
        "days": days,
    }
    Path(args.json_out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(out["summary"], indent=2, ensure_ascii=False))
    cap = (
        f"8 并发 robot，SL/TP 后复用 slot"
        if gate.robot_reuse_after_exit
        else f"日最多 {gate.max_opens_per_day} 单"
    )
    print(f"\n规则: 突破 scan 当下打分 | P(true)>={gate.min_p_true} 才开 | {cap}")
    print()
    for d in days:
        ok = "OK" if d["goal_met_min"] else "MISS"
        opened = ", ".join(
            f"{r['symbol'].replace('USDT','')}@{'T' if r.get('true_breakout') else 'F'}(P={r['p_true']:.2f})"
            for r in d["opened"]
        ) or "none"
        print(f"{d['session_date']} [{ok}] opens={d['opens']} true={d['true_opens']} aborted={d['day_aborted']}")
        print(f"  opened: {opened}")
    print(f"\nfull -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
