#!/usr/bin/env python3
"""初始化 data/orb/live/ 人工替换包（从当前 Gate + 模型复制）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402
from orb.ml.live_bundle import bootstrap_from_legacy, bundle_status, live_bundle_root  # noqa: E402


def main() -> int:
    load_env_oi()
    import argparse

    ap = argparse.ArgumentParser(description="Bootstrap data/orb/live manual bundle")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    copied = bootstrap_from_legacy(overwrite=bool(args.overwrite))
    print(
        json.dumps(
            {
                "live_bundle": str(live_bundle_root()),
                "copied": copied,
                "status": bundle_status(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
