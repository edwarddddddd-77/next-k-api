#!/usr/bin/env python3
"""ORB 2.0 月度训练 CLI → orb.ml.model.run_training_pipeline（全自动 staging/验收/promote）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402
from orb.ml.model import run_training_pipeline  # noqa: E402
from orb.ml.model.train import TrainingValidationError  # noqa: E402


def main() -> int:
    load_env_oi()
    ap = argparse.ArgumentParser(description="ORB 2.0 monthly ML retrain (auto fetch/validate/promote)")
    ap.add_argument("--days", type=float, default=180.0, help="样本回看天数")
    ap.add_argument("--holdout-days", type=int, default=10)
    ap.add_argument("--bootstrap-only", action="store_true", help="仅从 legacy 复制，不训练")
    ap.add_argument("--skip-collect", action="store_true")
    ap.add_argument("--skip-archive", action="store_true")
    ap.add_argument("--skip-fetch-klines", action="store_true")
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--skip-promote", action="store_true", help="只写 staging，不覆盖 production")
    ap.add_argument("--skip-gate-tune", action="store_true", help="promote 后跳过 Gate 建议/应用")
    ap.add_argument("--no-auto", action="store_true", help="旧行为：直接覆盖 production，不 staging")
    ap.add_argument("--tag", default="", help="归档目录名，默认 YYYYMM")
    args = ap.parse_args()

    try:
        report = run_training_pipeline(
            days=float(args.days),
            holdout_days=int(args.holdout_days),
            bootstrap_only=bool(args.bootstrap_only),
            skip_collect=bool(args.skip_collect),
            skip_archive=bool(args.skip_archive),
            skip_fetch_klines=bool(args.skip_fetch_klines),
            skip_validate=bool(args.skip_validate),
            skip_promote=bool(args.skip_promote),
            skip_gate_tune=bool(args.skip_gate_tune),
            tag=args.tag.strip(),
            auto=not args.no_auto,
        )
    except TrainingValidationError as exc:
        print(json.dumps({"ok": False, "reasons": exc.reasons, "detail": exc.detail}, indent=2), file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
