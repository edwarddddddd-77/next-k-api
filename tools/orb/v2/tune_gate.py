#!/usr/bin/env python3
"""手动触发 Gate 调参（与 promote 后逻辑相同）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402
from orb.ml.model.auto_config import MlAutoConfig  # noqa: E402
from orb.ml.model.gate_tune import tune_gate_after_promote  # noqa: E402
from orb.ml.model.paths import GBM_TRAIN_REPORT  # noqa: E402


def main() -> int:
    load_env_oi()
    ap = argparse.ArgumentParser(description="ORB V2 gate tune suggest/apply")
    ap.add_argument("--apply", action="store_true", help="等同 ORB_ML_AUTO_GATE_APPLY=1")
    ap.add_argument("--force-sweep", action="store_true", help="忽略 trigger，强制 sweep")
    args = ap.parse_args()

    cfg = MlAutoConfig.from_env()
    if args.apply:
        cfg.auto_gate_apply = True
    if args.force_sweep:
        cfg.gate_always_sweep = True

    train = {}
    if GBM_TRAIN_REPORT.is_file():
        train = json.loads(GBM_TRAIN_REPORT.read_text(encoding="utf-8"))

    report = tune_gate_after_promote(train_report=train, cfg=cfg)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
