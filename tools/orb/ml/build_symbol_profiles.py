#!/usr/bin/env python3
"""生成 43 标的突破画像。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from orb.ml.features import default_shared_samples_path  # noqa: E402
from orb.ml.profiles import DEFAULT_PROFILES_PATH, build_profiles, save_profiles  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default=str(default_shared_samples_path()))
    ap.add_argument("--json-out", default=str(DEFAULT_PROFILES_PATH))
    args = ap.parse_args()

    data = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    rows = list(data.get("rows") or [])
    prof = build_profiles(rows)
    out = save_profiles(prof, Path(args.json_out))
    print(json.dumps({"path": str(out), "symbols": prof["symbols"], "global_true_rate": prof["global_true_rate"]}, indent=2))
    tier_a = sum(1 for p in prof["profiles"].values() if p["tier"] == "A")
    tier_b = sum(1 for p in prof["profiles"].values() if p["tier"] == "B")
    tier_c = sum(1 for p in prof["profiles"].values() if p["tier"] == "C")
    print(f"tier A(>=30)={tier_a} B(15-29)={tier_b} C(<15)={tier_c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
