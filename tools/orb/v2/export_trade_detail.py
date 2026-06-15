#!/usr/bin/env python3
"""Export backtest JSON to readable trade detail text."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = json.loads(Path(args.json).read_text(encoding="utf-8"))
    lines: list[str] = []
    s = d["summary"]
    lines.append(
        f"# 60d backtest {s['total_pnl_usdt']:+.0f}U | opens={s['total_opens']} true={s['total_true_opens']}"
    )
    lines.append("")
    lines.append("## Daily trades")
    for day in d["days"]:
        sd = day["session_date"]
        pnl = sum(float(r.get("pnl_usdt") or 0) for r in day.get("opened", []))
        macro = ""
        if day.get("macro_skip_day"):
            ev = ",".join(day.get("macro_events") or [])
            macro = f" [MACRO:{ev}]"
        lines.append(f"{sd}  {pnl:+.1f}U  opens={day.get('opens')} true={day.get('true_opens')}{macro}")
        if day.get("macro_skip_day"):
            lines.append("  (no trades — macro filter blocked entire session)")
            continue
        for r in day.get("opened", []):
            sym = str(r.get("symbol", "")).replace("USDT", "")
            tb = "T" if r.get("true_breakout") else "F"
            lines.append(
                f"  {sym:6} R{r.get('robot_id')} {str(r.get('side', '')):5} "
                f"P={float(r.get('p_true', 0)):.3f} pnl={float(r.get('pnl_usdt', 0)):+.1f} "
                f"{tb} @{r.get('scan_et', '')} {r.get('outcome', '')}"
            )
    lines.append("")
    lines.append("## Macro blocked (counterfactual, if macro off)")
    for row in d.get("macro_impact", []):
        sd = row["session_date"]
        ev = ",".join(row.get("macro_events") or [])
        lines.append(
            f"{sd} [{ev}] cf_pnl={float(row.get('counterfactual_pnl_usdt') or 0):+.1f}U "
            f"cf_opens={row.get('counterfactual_opens')}"
        )
        for t in row.get("counterfactual_trades", []):
            sym = str(t.get("symbol", "")).replace("USDT", "")
            tb = "T" if t.get("true_breakout") else "F"
            lines.append(
                f"  [BLOCKED] {sym:6} R{t.get('robot_id')} {str(t.get('side', '')):5} "
                f"P={float(t.get('p_true', 0)):.3f} pnl={float(t.get('pnl_usdt', 0)):+.1f} "
                f"{tb} @{t.get('scan_et', '')} {t.get('outcome', '')}"
            )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"written {out} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
