#!/usr/bin/env python3
"""训练共享 GBM 突破排序模型（hold_30m 标签）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from orb.ml.gbm import DEFAULT_GBM_META, DEFAULT_GBM_PATH, rows_to_xy_gbm, save_gbm, score_gbm_holdout, train_gbm  # noqa: E402
from orb.ml.samples import split_holdout_by_date  # noqa: E402
from orb.ml.paths import DEFAULT_GBM_TRAIN_REPORT, default_shared_samples_path, ensure_v1_dirs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(default_shared_samples_path()))
    ap.add_argument("--holdout-days", type=int, default=10)
    ap.add_argument("--label-mode", default="hold_30m", choices=("hold_30m", "true_breakout", "quality"))
    ap.add_argument("--out", default=str(DEFAULT_GBM_PATH))
    ap.add_argument("--meta-out", default=str(DEFAULT_GBM_META))
    ap.add_argument("--report-out", default="", help="训练报告 JSON（默认 v1 output 路径）")
    args = ap.parse_args()

    ensure_v1_dirs()
    data = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    rows = list(data.get("rows") or [])
    if args.label_mode == "hold_30m" and not any("hold30_true" in r for r in rows):
        print("Missing hold30_true — run tools/relabel_hold30_samples.py first")
        return 1

    train_rows, test_rows = split_holdout_by_date(rows, holdout_days=max(0, int(args.holdout_days)))
    X, y = rows_to_xy_gbm(train_rows, label_mode=args.label_mode)
    model = train_gbm(X, y, label_mode=args.label_mode)
    save_gbm(model, Path(args.out), Path(args.meta_out))

    report = {
        "kind": "gbm",
        "label_mode": args.label_mode,
        "samples_total": len(rows),
        "train_n": len(train_rows),
        "holdout_n": len(test_rows),
        **model.metrics,
    }
    if test_rows:
        report.update(score_gbm_holdout(model, test_rows))
    rep_path = Path(args.report_out) if args.report_out.strip() else DEFAULT_GBM_TRAIN_REPORT
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
