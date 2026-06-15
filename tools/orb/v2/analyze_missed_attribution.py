#!/usr/bin/env python3
"""漏捕 hold30 真突破的原因归因。"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tools.orb.v2.analyze_true_breakout_capture import _first_breakout_of_day  # noqa: E402
from tools.orb.ml.eval_live_gate import _ml_cfg  # noqa: E402
from orb.ml.samples import parse_symbol_list  # noqa: E402
from orb.v2.paths import resolve_symbols_path  # noqa: E402


def _norm_reason(reason: str) -> str:
    r = str(reason or "").strip()
    if r.startswith("p_true<"):
        return "p_below_min"
    if r == "early_sync_trap":
        return "early_sync_trap"
    if r == "max_opens_reached":
        return "max_opens_reached"
    if r == "no_robot_slot":
        return "no_robot_slot"
    if r == "day_aborted":
        return "day_aborted"
    return r or "unknown"


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backtest-json",
        default=str(ROOT / "output/orb/v2/eval/universe_60d_eight-robots_v22_backtest.json"),
    )
    args = ap.parse_args()
    bt_path = Path(args.backtest_json)
    bt = json.loads(bt_path.read_text(encoding="utf-8"))
    dates = [d["session_date"] for d in bt.get("days") or []]

    # timeline index: (date, symbol) -> decision row (first only)
    decisions: dict[tuple[str, str], dict] = {}
    for day in bt.get("days") or []:
        sd = day["session_date"]
        opens_so_far = 0
        seen_sym: set[str] = set()
        for row in day.get("timeline") or []:
            sym = str(row.get("symbol") or "").upper()
            if sym in seen_sym:
                continue
            seen_sym.add(sym)
            rec = dict(row)
            rec["_opens_before"] = opens_so_far
            decisions[(sd, sym)] = rec
            if row.get("opened"):
                opens_so_far += 1

    opened_keys = set()
    for day in bt.get("days") or []:
        sd = day["session_date"]
        for r in day.get("opened") or []:
            opened_keys.add((sd, str(r.get("symbol") or "").upper()))

    syms = parse_symbol_list(resolve_symbols_path().read_text(encoding="utf-8"))
    cfg = _ml_cfg(compound_per_symbol=True)

    print("Building pool (reuse scan) ...", flush=True)
    pool = []
    for i, sd in enumerate(dates, 1):
        if i % 15 == 0:
            print(f"  {i}/{len(dates)}", flush=True)
        for sym in syms:
            row = _first_breakout_of_day(sd, sym, cfg)
            if row:
                pool.append(row)

    pool_h30 = [r for r in pool if r["hold30_true"]]
    missed = [r for r in pool_h30 if (r["session_date"], r["symbol"]) not in opened_keys]
    captured = [r for r in pool_h30 if (r["session_date"], r["symbol"]) in opened_keys]

    by_reason = Counter()
    by_reason_pnl = defaultdict(float)
    p_buckets = Counter()
    trap_detail = Counter()
    max8_detail = 0
    no_timeline = 0
    rows_detail = []

    for r in missed:
        key = (r["session_date"], r["symbol"])
        dec = decisions.get(key)
        pnl = float(r.get("pnl_hold30") or 0)
        if not dec:
            no_timeline += 1
            by_reason["no_timeline"] += 1
            by_reason_pnl["no_timeline"] += pnl
            continue
        reason = _norm_reason(str(dec.get("reason") or ""))
        by_reason[reason] += 1
        by_reason_pnl[reason] += pnl
        p = float(dec.get("p_true") or 0)
        if reason == "p_below_min":
            if p < 0.25:
                p_buckets["p<0.25"] += 1
            elif p < 0.30:
                p_buckets["p 0.25-0.30"] += 1
            elif p < 0.35:
                p_buckets["p 0.30-0.35"] += 1
            else:
                p_buckets["p>=0.35 (tier C?)"] += 1
        if reason == "early_sync_trap":
            sync = int(dec.get("sync_same_side") or 0)
            mins = float(dec.get("minutes_after_or") or 0)
            trap_detail[f"sync={sync}, {mins:.0f}min"] += 1
        if reason == "max_opens_reached":
            max8_detail += 1
        rows_detail.append(
            {
                "date": r["session_date"],
                "symbol": r["symbol"].replace("USDT", ""),
                "reason": dec.get("reason"),
                "p_true": round(p, 4),
                "sync": dec.get("sync_same_side"),
                "min_after_or": dec.get("minutes_after_or"),
                "opens_before": dec.get("_opens_before"),
                "pnl_hold30_1k": round(pnl, 2),
            }
        )

    # 若 p>=0.35 仍被拦（非 max8）
    high_p_missed = [
        x for x in rows_detail
        if float(x["p_true"]) >= 0.35 and _norm_reason(str(x["reason"])) != "max_opens_reached"
    ]

    pool_map = {(x["session_date"], x["symbol"]): x for x in pool}
    opened_false = []
    for day in bt.get("days") or []:
        sd = day["session_date"]
        for r in day.get("opened") or []:
            sym = str(r.get("symbol") or "").upper()
            p = pool_map.get((sd, sym))
            if p and not p["hold30_true"]:
                opened_false.append(p)

    summary = {
        "pool_hold30_true": len(pool_h30),
        "captured_hold30": len(captured),
        "missed_hold30": len(missed),
        "missed_by_reason_count": dict(by_reason.most_common()),
        "missed_by_reason_pnl_hold30_1k": {k: round(v, 2) for k, v in sorted(by_reason_pnl.items(), key=lambda x: -x[1])},
        "missed_p_below_min_buckets": dict(p_buckets),
        "missed_early_trap_patterns": dict(trap_detail.most_common(8)),
        "missed_high_p_not_max8": len(high_p_missed),
        "opened_but_not_hold30_true": len(opened_false),
    }

    print("\n=== 漏捕 290 个 hold30 真突破 — 原因归因 ===\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n--- Top 15 漏捕真突破（按 hold30 pnl @1kU）---")
    rows_detail.sort(key=lambda x: -x["pnl_hold30_1k"])
    for x in rows_detail[:15]:
        print(
            f"  {x['date']} {x['symbol']:6} {x['reason']:22} p={x['p_true']:.3f} "
            f"sync={x['sync']} {x['min_after_or']}min opens_before={x['opens_before']} "
            f"pnl@1k={x['pnl_hold30_1k']:+.1f}U"
        )

    print("\n--- p>=0.35 仍被拦（非 8 单满）Top 10 ---")
    high_p_missed.sort(key=lambda x: -x["pnl_hold30_1k"])
    for x in high_p_missed[:10]:
        print(f"  {x['date']} {x['symbol']:6} {x['reason']} p={x['p_true']:.3f} pnl@1k={x['pnl_hold30_1k']:+.1f}U")

    out = ROOT / "output/orb/v2/eval/universe_60d_missed_attribution.json"
    out.write_text(
        json.dumps({"summary": summary, "top_missed": rows_detail[:50], "high_p_missed": high_p_missed[:30]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
