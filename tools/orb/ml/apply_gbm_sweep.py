#!/usr/bin/env python3
"""读取 gbm_sweep.json 最优配置，重训并写入 staging（不覆盖 orb_live）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from orb.ml.gbm import GbmHyperParams, save_gbm, score_gbm_holdout, train_gbm  # noqa: E402
from orb.ml.paths import V2_GBM_SWEEP, default_shared_samples_path  # noqa: E402
from orb.ml.model.paths import ensure_model_dirs, staging_train_report_path  # noqa: E402
from orb.ml.model.paths import STAGING_GBM_META, STAGING_GBM_PKL, STAGING_GBM_TRAIN_REPORT  # noqa: E402
from orb.ml.samples import split_holdout_by_date  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default=str(V2_GBM_SWEEP))
    ap.add_argument("--samples", default=str(default_shared_samples_path()))
    ap.add_argument("--holdout-days", type=int, default=10)
    ap.add_argument("--use-gate-best", action="store_true", help="优先 gate_eval 中 PnL 最高项")
    ap.add_argument("--label-mode", default="", help="强制标签（默认 hold_30m 或 sweep best_holdout）")
    args = ap.parse_args()

    sweep_path = Path(args.sweep)
    if not sweep_path.is_file():
        print(f"Missing sweep result: {sweep_path}")
        print("Run tools/orb/ml/sweep_breakout_gbm.py first.")
        return 1

    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    pick = sweep.get("best_holdout") or sweep.get("best") or {}
    if args.use_gate_best and sweep.get("gate_eval"):
        pick = sweep["gate_eval"][0]
    if args.label_mode.strip():
        label_mode = args.label_mode.strip()
        holdout_rows = sweep.get("top10") or []
        label_pick = next((r for r in holdout_rows if r.get("label_mode") == label_mode), None)
        pick = label_pick or pick
        hp = GbmHyperParams.from_dict(pick.get("hyperparams") or {})
    else:
        # 生产默认：hold_30m + holdout 最优（与 Gate min_p 口径一致）
        holdout_rows = sweep.get("top10") or []
        hold30 = next((r for r in holdout_rows if r.get("label_mode") == "hold_30m"), None)
        pick = hold30 or pick
        label_mode = str(pick.get("label_mode") or "hold_30m")
        hp = GbmHyperParams.from_dict(pick.get("hyperparams") or {})

    samples_path = Path(args.samples)
    rows = list(json.loads(samples_path.read_text(encoding="utf-8")).get("rows") or [])
    train_rows, test_rows = split_holdout_by_date(rows, holdout_days=max(0, int(args.holdout_days)))

    from orb.ml.gbm import rows_to_xy_gbm

    X, y = rows_to_xy_gbm(train_rows, label_mode=label_mode)
    model = train_gbm(X, y, label_mode=label_mode, hyperparams=hp)
    holdout = score_gbm_holdout(model, test_rows)

    ensure_model_dirs()
    save_gbm(model, STAGING_GBM_PKL, STAGING_GBM_META)

    report = {
        "kind": "gbm",
        "label_mode": label_mode,
        "hyperparams": hp.to_dict(),
        "samples_total": len(rows),
        "train_n": len(train_rows),
        "holdout_n": len(test_rows),
        **model.metrics,
        **holdout,
        "source_sweep": str(sweep_path),
    }
    staging_train_report_path().write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    STAGING_GBM_TRAIN_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nstaging model -> {STAGING_GBM_PKL}")
    print("Review then copy to orb_live/ or run promote pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
