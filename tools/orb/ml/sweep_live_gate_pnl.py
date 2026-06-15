#!/usr/bin/env python3
"""从 live_gate timeline 重放参数，优化总盈利。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from orb.ml.paths import V2_LIVE_GATE_LAST30D, V2_LIVE_GATE_SWEEP, ensure_v2_dirs  # noqa: E402
from orb.ml.gate import LiveGateConfig  # noqa: E402
from orb.ml.gate_replay import replay_day  # noqa: E402


def main() -> int:
    path = V2_LIVE_GATE_LAST30D if V2_LIVE_GATE_LAST30D.is_file() else ROOT / "output" / "live_gate_last30d.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    days = data["days"]
    gate = LiveGateConfig()

    configs: list[tuple[str, dict]] = []
    for min_p in (0.30, 0.32, 0.35, 0.38, 0.40, 0.42, 0.45):
        for max_o in (6, 8):
            for abort in (True, False):
                label = f"p>={min_p:.2f} max={max_o} abort={'Y' if abort else 'N'}"
                configs.append((label, {"min_p": min_p, "max_opens": max_o, "day_abort": abort}))

    rows = []
    for label, cfg in configs:
        total = 0.0
        win = loss = 0
        opens_total = 0
        for d in days:
            r = replay_day(d["timeline"], gate=gate, **cfg)
            total += r["pnl"]
            opens_total += r["opens"]
            if r["pnl"] > 0:
                win += 1
            elif r["pnl"] < 0:
                loss += 1
        rows.append(
            {
                "label": label,
                **cfg,
                "total_pnl": round(total, 1),
                "avg_daily": round(total / len(days), 1),
                "win_days": win,
                "loss_days": loss,
                "avg_opens": round(opens_total / len(days), 1),
            }
        )

    rows.sort(key=lambda x: x["total_pnl"], reverse=True)
    print("=== 参数扫描（按总PnL排序 Top10）===")
    print(f"{'label':<40} {'total':>9} {'avg/d':>8} {'W/L':>7} {'avg_opens':>9}")
    for r in rows[:10]:
        print(
            f"{r['label']:<40} {r['total_pnl']:+9.1f}U {r['avg_daily']:+8.1f}U "
            f"{r['win_days']}/{r['loss_days']:<3} {r['avg_opens']:9.1f}"
        )

    best = rows[0]
    out = {"best": best, "top10": rows[:10]}
    ensure_v2_dirs()
    V2_LIVE_GATE_SWEEP.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nbest -> {best['label']}  total {best['total_pnl']:+.1f}U")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
