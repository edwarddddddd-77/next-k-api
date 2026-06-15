#!/usr/bin/env python3
"""真突破特征画像 + 某日 Top-K 选股命中率。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402
from orb.ml.samples import parse_symbol_list, split_holdout_by_date  # noqa: E402
from orb.ml.features import (  # noqa: E402
    FEATURE_NAMES,
    default_shared_fake_model_path,
    default_shared_samples_path,
    default_shared_true_model_path,
    label_is_true_breakout,
    load_model,
)
from tools.rank_day_breakouts import (  # noqa: E402
    DEFAULT_MIN_HITS,
    DEFAULT_PICK_K,
    EARLY_TRAP_MINUTES,
    EARLY_TRAP_MIN_SYNC,
    _cached_symbols,
    _ml_cfg,
    evaluate_day_pick6,
)


def _feat_row(r: dict) -> dict[str, float]:
    return {k.replace("f_", "", 1): float(r.get(f"f_{k}", 0) or 0) for k in FEATURE_NAMES}


def feature_profile(rows: List[dict], *, label_key: str = "true_breakout") -> dict[str, dict]:
    buckets: dict[int, list[dict]] = {0: [], 1: []}
    for r in rows:
        y = int(r.get(label_key) or label_is_true_breakout(str(r.get("outcome", "")), float(r.get("pnl_usdt") or 0)))
        buckets[y].append(_feat_row(r))

    out: dict[str, dict] = {}
    for tag, lbl in (("fake", 0), ("true", 1)):
        grp = buckets[lbl]
        if not grp:
            continue
        stats: dict[str, dict] = {}
        for k in FEATURE_NAMES:
            vals = np.array([g[k] for g in grp], dtype=np.float64)
            stats[k] = {
                "mean": round(float(vals.mean()), 4),
                "median": round(float(np.median(vals)), 4),
                "p25": round(float(np.percentile(vals, 25)), 4),
                "p75": round(float(np.percentile(vals, 75)), 4),
            }
        out[tag] = {"n": len(grp), "features": stats}
    return out


def feature_separation(rows: List[dict]) -> List[dict]:
    prof = feature_profile(rows)
    if "true" not in prof or "fake" not in prof:
        return []
    scored: list[dict] = []
    for k in FEATURE_NAMES:
        tm = prof["true"]["features"][k]["mean"]
        fm = prof["fake"]["features"][k]["mean"]
        tv = [r.get(f"f_{k}", 0) for r in rows if int(r.get("true_breakout", 0))]
        fv = [r.get(f"f_{k}", 0) for r in rows if not int(r.get("true_breakout", 0))]
        if not tv or not fv:
            continue
        pooled = float(np.sqrt((np.var(tv) + np.var(fv)) / 2)) or 1e-9
        scored.append(
            {
                "feature": k,
                "true_mean": tm,
                "fake_mean": fm,
                "delta": round(tm - fm, 4),
                "effect_size": round(abs(tm - fm) / pooled, 4),
                "direction": "true_higher" if tm > fm else "true_lower",
            }
        )
    return sorted(scored, key=lambda x: x["effect_size"], reverse=True)


def main() -> int:
    load_env_oi()
    ap = argparse.ArgumentParser(description="True breakout feature profile + pick-K eval")
    ap.add_argument("--samples", default=str(default_shared_samples_path()))
    ap.add_argument(
        "--dates",
        default="2026-05-29,2026-06-01,2026-06-02,2026-06-03,2026-06-04,2026-06-05,2026-06-08,2026-06-09,2026-06-10,2026-06-11",
    )
    ap.add_argument("--pick-k", type=int, default=DEFAULT_PICK_K)
    ap.add_argument("--min-hits", type=int, default=DEFAULT_MIN_HITS)
    ap.add_argument("--json-out", default=str(ROOT / "output" / "true_breakout_profile.json"))
    args = ap.parse_args()

    data = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    rows = list(data.get("rows") or [])
    train_rows, _ = split_holdout_by_date(rows, holdout_days=10)

    profile_all = feature_profile(rows)
    separation = feature_separation(rows)

    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    syms = _cached_symbols(parse_symbol_list((ROOT / "config" / "orb" / "v2" / "symbols.txt").read_text()))
    cfg = _ml_cfg()
    fake_m = load_model(default_shared_fake_model_path())
    true_m = load_model(default_shared_true_model_path())

    k = max(1, int(args.pick_k))
    min_hits = max(0, int(args.min_hits))
    per_day = [
        evaluate_day_pick6(d, syms, cfg, fake_m, true_m, k=k, min_hits=min_hits)
        for d in dates
    ]
    rank_of_trues: list[int] = []
    for day in per_day:
        for i, r in enumerate(day["ranked_all"], 1):
            if r["true_breakout"]:
                rank_of_trues.append(i)

    goal_days = sum(1 for x in per_day if x["goal_met"])
    avg_hits = round(sum(x["pick_hits"] for x in per_day) / len(per_day), 2) if per_day else 0

    out = {
        "goal": f"某日 Top-{k} 至少 {min_hits} 个真突破",
        "early_filter": f"{EARLY_TRAP_MINUTES}min & sync={EARLY_TRAP_MIN_SYNC}-{EARLY_TRAP_MAX_SYNC}",
        "samples_n": len(rows),
        "true_rate_pct": round(sum(int(r.get("true_breakout", 0)) for r in rows) / len(rows) * 100, 1),
        "feature_separation_ranked": separation[:8],
        "profile_all": profile_all,
        "true_breakout_rank_median": int(np.median(rank_of_trues)) if rank_of_trues else None,
        "pick_eval": {
            "k": k,
            "min_hits": min_hits,
            "days": len(per_day),
            "goal_met_days": goal_days,
            "goal_met_rate": round(goal_days / len(per_day), 3) if per_day else 0,
            "avg_hits_per_day": avg_hits,
            "per_day": [
                {
                    "date": d["session_date"],
                    "breakouts_that_day": d["breakouts_that_day"],
                    "eligible_n": d["eligible_n"],
                    "excluded_n": d["excluded_n"],
                    "true_in_pool": d["true_in_pool"],
                    "pick_hits": d["pick_hits"],
                    "goal_met": d["goal_met"],
                    "picked": d["picked"],
                    "missed_true": d["missed_true"][:8],
                }
                for d in per_day
            ],
        },
    }

    Path(args.json_out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== feature separation (top 6) ===")
    for s in separation[:6]:
        print(f"  {s['feature']:<18} true={s['true_mean']:+.3f} fake={s['fake_mean']:+.3f} d={s['delta']:+.3f}")

    print(f"\n=== Top-{k} pick (>={min_hits} true) ===")
    print(f"  {goal_days}/{len(per_day)} days OK  avg {avg_hits}/{k}  true rank median #{out['true_breakout_rank_median']}")
    for day in out["pick_eval"]["per_day"]:
        ok = "OK" if day["goal_met"] else "MISS"
        picked = ", ".join(f"{p['symbol']}{'(T)' if p['true'] else '(F)'}" for p in day["picked"])
        print(f"  {day['date']} [{ok}] {day['pick_hits']}/{k}  pool={day['breakouts_that_day']} true={day['true_in_pool']}")
        print(f"    {picked}")

    print(f"\nfull -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
