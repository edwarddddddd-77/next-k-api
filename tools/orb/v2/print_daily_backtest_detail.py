#!/usr/bin/env python3
"""从 universe backtest JSON 导出 60 日每日明细（txt + csv）。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _day_pnl(day: dict) -> float:
    return round(sum(float(r.get("pnl_usdt") or 0) for r in day.get("opened") or []), 2)


def _fmt_exit(r: dict, tz_hint: str = "") -> str:
    ex = r.get("exit_ms")
    if ex:
        try:
            import pandas as pd

            return pd.Timestamp(int(ex), unit="ms", tz="America/New_York").strftime("%H:%M")
        except Exception:
            return str(ex)
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json",
        default=str(ROOT / "output/orb/v2/eval/universe_60d_eight-robots_v22_backtest.json"),
    )
    ap.add_argument("--txt-out", default="")
    ap.add_argument("--csv-out", default="")
    ap.add_argument("--day-csv-out", default="")
    args = ap.parse_args()

    src = Path(args.json)
    if not src.is_file():
        print(f"Missing: {src}")
        return 1
    bt = json.loads(src.read_text(encoding="utf-8"))
    days = bt.get("days") or []
    stem = src.stem

    txt_out = Path(args.txt_out) if args.txt_out else src.with_name(f"{stem}_daily.txt")
    csv_out = Path(args.csv_out) if args.csv_out else src.with_name(f"{stem}_daily_trades.csv")
    day_csv = Path(args.day_csv_out) if args.day_csv_out else src.with_name(f"{stem}_daily_summary.csv")

    lines: list[str] = []
    gate = bt.get("gate") or {}
    sm = bt.get("summary") or {}
    lines.append(f"ORB V2 backtest 每日明细")
    lines.append(f"source: {src.name}")
    lines.append(
        f"range: {bt.get('date_range', {}).get('from')} .. {bt.get('date_range', {}).get('to')} "
        f"({len(days)} sessions)"
    )
    lines.append(
        f"gate: p>={gate.get('min_p_true')} robot_reuse={gate.get('robot_reuse_after_exit')} "
        f"trap_bypass_p>={gate.get('early_trap_bypass_min_p')}"
    )
    lines.append(
        f"total: PnL={sm.get('total_pnl_usdt'):+.1f}U return={sm.get('return_pct')}% "
        f"opens={sm.get('total_opens')} true_eod={sm.get('total_true_opens')}"
    )
    lines.append("")

    trade_fields = [
        "session_date",
        "robot_id",
        "symbol",
        "side",
        "scan_et",
        "exit_et",
        "entry",
        "notional_usdt",
        "p_true",
        "pnl_usdt",
        "true_breakout",
        "outcome",
        "reason",
        "wallet_before",
        "wallet_after",
        "sync_same_side",
        "minutes_after_or",
    ]
    trade_rows: list[dict] = []
    day_rows: list[dict] = []
    cum = 0.0

    for day in days:
        sd = day["session_date"]
        opens = day.get("opened") or []
        pnl = _day_pnl(day)
        cum = round(cum + pnl, 2)
        true_n = sum(1 for r in opens if r.get("true_breakout"))
        goal = "OK" if day.get("goal_met_min") else "—"
        lines.append(
            f"## {sd}  PnL={pnl:+.1f}U  cum={cum:+.1f}U  "
            f"opens={len(opens)}  true_eod={true_n}  goal={goal}"
        )
        day_rows.append(
            {
                "session_date": sd,
                "day_pnl_usdt": pnl,
                "cum_pnl_usdt": cum,
                "opens": len(opens),
                "true_eod": true_n,
                "goal_met": goal,
                "skipped_gate": int(day.get("skipped") or 0),
            }
        )
        if not opens:
            lines.append("  (无开单)\n")
            continue
        for r in opens:
            sym = str(r.get("symbol", "")).replace("USDT", "")
            tb = "T" if r.get("true_breakout") else "F"
            rid = r.get("robot_id")
            bot = f"R{rid}" if rid else "  "
            notion = float(r.get("notional_usdt") or 0)
            exit_et = _fmt_exit(r)
            outcome = str(r.get("outcome") or "")
            lines.append(
                f"  {bot} {sym:6} {str(r.get('side', '')):5} "
                f"@{r.get('scan_et', '')}->{exit_et or '?'} "
                f"entry={r.get('entry')} P={float(r.get('p_true', 0)):.3f} "
                f"n={notion:.0f}U pnl={float(r.get('pnl_usdt', 0)):+.1f}U "
                f"{tb} {outcome} {r.get('reason', '')}"
            )
            trade_rows.append(
                {
                    "session_date": sd,
                    "robot_id": rid,
                    "symbol": sym,
                    "side": r.get("side"),
                    "scan_et": r.get("scan_et"),
                    "exit_et": exit_et,
                    "entry": r.get("entry"),
                    "notional_usdt": round(notion, 2),
                    "p_true": round(float(r.get("p_true") or 0), 4),
                    "pnl_usdt": round(float(r.get("pnl_usdt") or 0), 2),
                    "true_breakout": int(bool(r.get("true_breakout"))),
                    "outcome": outcome,
                    "reason": r.get("reason"),
                    "wallet_before": r.get("wallet_before"),
                    "wallet_after": r.get("wallet_after"),
                    "sync_same_side": r.get("sync_same_side"),
                    "minutes_after_or": r.get("minutes_after_or"),
                }
            )
        lines.append("")

    txt_out.write_text("\n".join(lines), encoding="utf-8")

    day_csv.parent.mkdir(parents=True, exist_ok=True)
    with day_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(day_rows[0].keys()) if day_rows else [])
        w.writeheader()
        w.writerows(day_rows)

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=trade_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(trade_rows)

    print(f"days: {len(day_rows)}")
    print(f"txt  -> {txt_out}")
    print(f"day  -> {day_csv}")
    print(f"trades -> {csv_out}")
    print("\n=== 60日汇总 ===")
    print(f"{'date':12} {'pnl':>10} {'cum':>12} {'opens':>6} {'true':>5} goal")
    for r in day_rows:
        print(
            f"{r['session_date']:12} {r['day_pnl_usdt']:>+10.1f} {r['cum_pnl_usdt']:>+12.1f} "
            f"{r['opens']:>6} {r['true_eod']:>5} {r['goal_met']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
