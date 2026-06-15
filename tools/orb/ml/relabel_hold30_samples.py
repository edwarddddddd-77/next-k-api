#!/usr/bin/env python3
"""为样本添加 30min hold 标签（读本地 1m 缓存）。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi  # noqa: E402
from orb.ml.horizon import relabel_row_horizon  # noqa: E402
from orb.core.config import OrbConfig  # noqa: E402
from orb.ml.features import default_shared_samples_path  # noqa: E402


def _ml_cfg() -> OrbConfig:
    os.environ["ORB_MACRO_FILTER"] = "0"
    cfg = OrbConfig.from_env()
    cfg.macro_filter = False
    return cfg


def main() -> int:
    load_env_oi()
    ap = argparse.ArgumentParser(description="Add hold_30m labels to breakout samples")
    ap.add_argument("--samples", default=str(default_shared_samples_path()))
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    path = Path(args.samples)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = list(data.get("rows") or [])
    cfg = _ml_cfg()
    t0 = time.time()
    ok = 0
    for i, r in enumerate(rows):
        extra = relabel_row_horizon(r, cfg=cfg)
        if extra:
            r.update(extra)
            ok += 1
        if (i + 1) % 100 == 0:
            print(f"[relabel] {i+1}/{len(rows)} ok={ok}", flush=True)

    hold30_true = sum(int(r.get("hold30_true") or 0) for r in rows)
    summary = dict(data.get("summary") or {})
    summary.update(
        {
            "hold30_labeled": ok,
            "hold30_true": hold30_true,
            "hold30_true_rate_pct": round(hold30_true / len(rows) * 100, 1) if rows else 0,
            "relabel_sec": round(time.time() - t0, 1),
        }
    )
    out_path = Path(args.json_out) if args.json_out.strip() else path
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
