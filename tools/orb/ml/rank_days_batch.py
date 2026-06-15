#!/usr/bin/env python3
"""批量：某日 Top-K 选股（GBM + 先验 + 日级 gate）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402
from tools.rank_day_breakouts import (  # noqa: E402
    DAY_GATE_MIN_EST,
    DEFAULT_MIN_HITS,
    DEFAULT_PICK_K,
    EARLY_TRAP_MAX_SYNC,
    EARLY_TRAP_MINUTES,
    EARLY_TRAP_MIN_SYNC,
    _cached_symbols,
    _ml_cfg,
    evaluate_day_pick6,
)
from orb.ml.samples import parse_symbol_list  # noqa: E402
from orb.ml.ranker import BreakoutRanker  # noqa: E402


def run_day_pick6(
    session_date: str,
    syms: list[str],
    cfg,
    ranker: BreakoutRanker,
    *,
    k: int,
    min_hits: int,
    early_minutes: float,
    min_sync: int,
    max_sync: int,
    min_est: float,
) -> dict:
    r = evaluate_day_pick6(
        session_date,
        syms,
        cfg,
        ranker,
        k=k,
        min_hits=min_hits,
        early_minutes=early_minutes,
        min_sync=min_sync,
        max_sync=max_sync,
        min_est=min_est,
    )
    gate = r.get("day_gate") or {}
    return {
        "date": session_date,
        "breakouts": r["breakouts_that_day"],
        "eligible_n": r["eligible_n"],
        "excluded_n": r["excluded_n"],
        "true_in_pool": r["true_in_pool"],
        "est_true_sum": gate.get("est_true_sum"),
        "skipped_by_gate": bool(r.get("skipped_by_gate")),
        "pick_k": k,
        "pick_hits": r["pick_hits"],
        "goal_met": bool(r["goal_met"]) and not bool(r.get("skipped_by_gate")),
        "picked": r["picked"],
        "missed_true": r["missed_true"][:8],
    }


def main() -> int:
    load_env_oi()
    ap = argparse.ArgumentParser(description="Batch day-level Top-K pick evaluation")
    ap.add_argument(
        "--dates",
        default="2026-05-29,2026-06-01,2026-06-02,2026-06-03,2026-06-04,2026-06-05,2026-06-08,2026-06-09,2026-06-10,2026-06-11",
    )
    ap.add_argument("--pick-k", type=int, default=DEFAULT_PICK_K)
    ap.add_argument("--min-hits", type=int, default=DEFAULT_MIN_HITS)
    ap.add_argument("--early-minutes", type=float, default=EARLY_TRAP_MINUTES)
    ap.add_argument("--early-min-sync", type=int, default=EARLY_TRAP_MIN_SYNC)
    ap.add_argument("--early-max-sync", type=int, default=EARLY_TRAP_MAX_SYNC)
    ap.add_argument("--day-gate-min-est", type=float, default=DAY_GATE_MIN_EST)
    ap.add_argument("--json-out", default=str(ROOT / "output" / "rank_days_batch.json"))
    args = ap.parse_args()

    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    syms = _cached_symbols(parse_symbol_list((ROOT / "config" / "orb" / "v2" / "symbols.txt").read_text()))
    cfg = _ml_cfg()
    ranker = BreakoutRanker.load(use_prior=True)
    if ranker.gbm is None and ranker.logistic is None:
        print("Missing ranker model")
        return 1

    k = max(1, int(args.pick_k))
    min_hits = max(0, int(args.min_hits))
    days = [
        run_day_pick6(
            d,
            syms,
            cfg,
            ranker,
            k=k,
            min_hits=min_hits,
            early_minutes=float(args.early_minutes),
            min_sync=int(args.early_min_sync),
            max_sync=int(args.early_max_sync),
            min_est=float(args.day_gate_min_est),
        )
        for d in dates
    ]

    goal_days = sum(1 for x in days if x["goal_met"])
    gate_days = sum(1 for x in days if x["skipped_by_gate"])
    avg_hits = round(sum(x["pick_hits"] for x in days) / len(days), 2) if days else 0
    summary = {
        "goal": f"Top-{k} 至少 {min_hits} 真突破",
        "ranker": ranker.kind,
        "early_filter": f"{args.early_minutes}min & sync={args.early_min_sync}-{args.early_max_sync}",
        "day_gate_min_est": float(args.day_gate_min_est),
        "days": len(days),
        "goal_met_days": goal_days,
        "goal_met_rate": round(goal_days / len(days), 3) if days else 0,
        "gate_skip_days": gate_days,
        "avg_hits_per_day": avg_hits,
        "pick_k": k,
        "min_hits": min_hits,
    }
    out = {"summary": summary, "days": days}
    Path(args.json_out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    for d in days:
        ok = "OK" if d["goal_met"] else ("GATE" if d["skipped_by_gate"] else "MISS")
        picked = ", ".join(f"{p['symbol']}{'(T)' if p['true'] else '(F)'}" for p in d["picked"])
        print(
            f"{d['date']} [{ok}] {d['pick_hits']}/{k}  pool={d['breakouts']} true={d['true_in_pool']} "
            f"est={d['est_true_sum']}"
        )
        print(f"  pick: {picked}")
        if d.get("missed_true"):
            missed = ", ".join(f"{m['symbol']}@#{m['rank']}" for m in d["missed_true"][:4])
            print(f"  missed: {missed}")
        print()
    print(f"full -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
