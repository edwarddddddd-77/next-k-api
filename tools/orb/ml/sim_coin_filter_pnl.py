#!/usr/bin/env python3
"""COIN 单标：假突破过滤 vs  baseline PnL 对比。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from orb.ml.samples import split_holdout_by_date
from orb.ml.features import BreakoutModel, score_rows


def report(subset: list[dict], fake_m: BreakoutModel, label: str) -> None:
    if not subset:
        print(f"=== {label}: no trades ===\n")
        return
    scored = score_rows(fake_m, subset)
    base = sum(float(r["pnl_usdt"] or 0) for r in subset)
    wins = sum(1 for r in subset if float(r["pnl_usdt"] or 0) > 0)
    losses = sum(1 for r in subset if float(r["pnl_usdt"] or 0) < 0)
    print(f"=== {label} ({len(subset)} trades) ===")
    print(f"baseline: {base:+.1f}U | win {wins} loss {losses} | avg {base/len(subset):+.2f}U/trade")
    best_th, best_pnl, best_delta = None, base, 0.0
    for th in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
        kept = [s for s in scored if s["p"] < th]
        pnl = sum(float(s["pnl_usdt"] or 0) for s in kept)
        skip = len(scored) - len(kept)
        delta = pnl - base
        if pnl > best_pnl:
            best_th, best_pnl, best_delta = th, pnl, delta
        kw = sum(1 for s in kept if float(s["pnl_usdt"] or 0) > 0)
        kl = sum(1 for s in kept if float(s["pnl_usdt"] or 0) < 0)
        wr = kw / (kw + kl) if kw + kl else 0
        print(f"  P(fake)<{th:.2f}: keep {len(kept):2d} skip {skip:2d} | {pnl:+8.1f}U ({delta:+6.1f}) | WR {wr:.0%}")
    if best_th is not None:
        print(f"  best threshold: {best_th:.2f} -> {best_pnl:+.1f}U ({best_delta:+.1f} vs baseline)")
    print()


def main() -> int:
    rows = json.loads((ROOT / "output" / "orb_shared_breakout_samples.json").read_text(encoding="utf-8"))["rows"]
    coin = [r for r in rows if r.get("symbol") == "COINUSDT"]
    train, holdout = split_holdout_by_date(coin, holdout_days=10)
    fake_m = BreakoutModel.from_dict(
        json.loads((ROOT / "output" / "orb_shared_fake_breakout_model.json").read_text(encoding="utf-8"))
    )

    print("Shared model | macro=off | same config as training\n")
    report(coin, fake_m, "COIN full 180d")
    report(holdout, fake_m, "COIN holdout last 10d")
    report(train, fake_m, "COIN train")

    d611 = [r for r in coin if r.get("session_date") == "2026-06-11"]
    if d611:
        report(d611, fake_m, "COIN 2026-06-11")
        for s in score_rows(fake_m, d611):
            print(f"  {s['side']:5s} P(fake)={s['p']:.3f} pnl={float(s['pnl_usdt']):+.1f}U sync={s.get('sync')}")

    # 明细：holdout 每笔
    print("--- COIN holdout trades ---")
    for s in sorted(score_rows(fake_m, holdout), key=lambda x: -x["p"]):
        sym = "COIN"
        print(
            f"{s['session_date']} {sym} {s['side']:5s} P(fake)={s['p']:.3f} "
            f"pnl={float(s['pnl_usdt']):+6.1f}U {s['outcome']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
