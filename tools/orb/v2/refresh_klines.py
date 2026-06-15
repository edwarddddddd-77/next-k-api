#!/usr/bin/env python3
"""定时刷新 universe K 线缓存 → data/orb/kline/。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402
from orb.data.kline_fetch import fetch_universe_klines  # noqa: E402
from orb.ml.model.auto_config import MlAutoConfig  # noqa: E402
from orb.ml.model.paths import resolve_train_symbols_path  # noqa: E402


def main() -> int:
    load_env_oi()
    auto_cfg = MlAutoConfig.from_env()
    ap = argparse.ArgumentParser(description="Refresh ORB universe kline cache")
    ap.add_argument("--symbols-file", default="", help="默认 ORB_V2_SYMBOLS_FILE / universe.txt")
    ap.add_argument("--days", type=float, default=0.0, help="0=读 ORB_ML_KLINE_DAYS")
    ap.add_argument("--skip-existing", action="store_true", help="跳过已有完整缓存")
    ap.add_argument("--force-refresh", action="store_true", help="忽略 ORB_ML_KLINE_SKIP_EXISTING，全量重拉")
    args = ap.parse_args()

    sym_file = Path(args.symbols_file) if args.symbols_file.strip() else resolve_train_symbols_path()
    days = float(args.days) if args.days > 0 else float(auto_cfg.kline_days)
    if args.force_refresh:
        skip_existing = False
    elif args.skip_existing:
        skip_existing = True
    else:
        skip_existing = auto_cfg.kline_skip_existing

    summary = fetch_universe_klines(
        symbols_file=sym_file,
        days=days,
        skip_existing=skip_existing,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary.get("errors") and auto_cfg.fail_on_kline_errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
