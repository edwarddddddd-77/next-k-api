#!/usr/bin/env python3
"""训练共享 ORB 真/假突破模型。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from orb.ml.samples import split_holdout_by_date  # noqa: E402
from orb.ml.features import (  # noqa: E402
    LABEL_MODES,
    RANK_FEATURE_NAMES,
    build_symbol_vocab,
    default_shared_fake_model_path,
    default_shared_samples_path,
    default_shared_true_model_path,
    feature_names_for,
    rows_to_xy,
    save_model,
    score_rows,
    train_model,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train shared ORB breakout models")
    ap.add_argument("--samples", type=str, default=str(default_shared_samples_path()))
    ap.add_argument("--holdout-days", type=int, default=10)
    ap.add_argument("--no-symbol-oh", action="store_true", help="不用 symbol one-hot（解决 OOV）")
    ap.add_argument("--rank-only", action="store_true", help="训练排序模型：特征不含 sync")
    ap.add_argument("--label-mode", choices=LABEL_MODES, default="quality", help="标签定义")
    ap.add_argument("--true-out", type=str, default="")
    ap.add_argument("--fake-out", type=str, default="")
    args = ap.parse_args()

    path = Path(args.samples)
    if not path.is_file():
        print(f"Missing samples: {path}")
        return 1

    data = json.loads(path.read_text(encoding="utf-8"))
    rows = list(data.get("rows") or [])
    summary = data.get("summary") or {}
    if len(rows) < 50:
        print(f"Too few samples: {len(rows)}")
        return 1

    train_rows, test_rows = split_holdout_by_date(rows, holdout_days=max(0, int(args.holdout_days)))
    vocab = [] if args.no_symbol_oh else build_symbol_vocab(train_rows)
    feat_names = feature_names_for(rank_only=bool(args.rank_only))
    true_out = Path(args.true_out) if args.true_out.strip() else default_shared_true_model_path()
    fake_out = Path(args.fake_out) if args.fake_out.strip() else default_shared_fake_model_path()

    results: dict = {
        "samples": len(rows),
        "train": len(train_rows),
        "holdout": len(test_rows),
        "symbols_oh": len(vocab) + (1 if vocab else 0),
        "feature_names": list(feat_names),
        "rank_only": bool(args.rank_only),
        "no_symbol_oh": bool(args.no_symbol_oh),
        "label_mode": args.label_mode,
        "macro_filter": summary.get("macro_filter"),
    }

    for target, out in (("true", true_out), ("fake", fake_out)):
        X, y = rows_to_xy(
            train_rows,
            target=target,
            vocab=vocab,
            rank_only=bool(args.rank_only),
            label_mode=args.label_mode,
        )
        model = train_model(X, y, target=target, symbol_vocab=vocab, feature_names=feat_names)
        save_model(model, out)
        entry = {**model.metrics, "path": str(out)}
        if test_rows:
            scored = score_rows(
                model,
                test_rows,
                label_mode=args.label_mode,
                rank_only=bool(args.rank_only),
            )
            hits = sum(1 for s in scored if (s["p"] >= 0.5) == bool(s["actual"]))
            entry["holdout_accuracy"] = round(hits / len(scored), 4)
            entry["holdout_n"] = len(scored)
            pos = [s for s in scored if s["actual"]]
            neg = [s for s in scored if not s["actual"]]
            if pos and neg:
                entry["holdout_p_mean_pos"] = round(sum(s["p"] for s in pos) / len(pos), 4)
                entry["holdout_p_mean_neg"] = round(sum(s["p"] for s in neg) / len(neg), 4)
        results[target] = entry
        print(f"[{target}] saved {out}")

    report = path.with_name("orb_shared_breakout_train_report.json")
    report.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
