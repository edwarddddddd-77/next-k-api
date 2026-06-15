#!/usr/bin/env python3
"""同一扫描时刻多标突破：模型能否把真突破标的排到前面。"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from orb.ml.features import BreakoutModel, score_rows


def main() -> int:
    rows = json.loads((ROOT / "output" / "orb_shared_breakout_samples.json").read_text(encoding="utf-8"))["rows"]
    fake_m = BreakoutModel.from_dict(
        json.loads((ROOT / "output" / "orb_shared_fake_breakout_model.json").read_text(encoding="utf-8"))
    )
    true_m = BreakoutModel.from_dict(
        json.loads((ROOT / "output" / "orb_shared_true_breakout_model.json").read_text(encoding="utf-8"))
    )

    buckets: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        scan = int(r.get("scan_open_ms") or r.get("entry_bar_open_ms") or 0)
        buckets[(str(r["session_date"]), scan)].append(r)

    multi = {k: v for k, v in buckets.items() if len(v) >= 2}
    print(f"multi-symbol scans: {len(multi)} buckets, {sum(len(v) for v in multi.values())} trades\n")

    def eval_rank(model: BreakoutModel, *, ascending_fake: bool) -> tuple[int, int, int]:
        hit1 = hit3 = n = 0
        for subset in multi.values():
            wins = [r for r in subset if float(r.get("pnl_usdt") or 0) > 0]
            if not wins:
                continue
            sc = score_rows(model, subset)
            if ascending_fake:
                ranked = sorted(sc, key=lambda x: x["p"])
            else:
                for s in sc:
                    feat = {
                        k.replace("f_", "", 1): v
                        for k, v in next(r for r in subset if r["symbol"] == s["symbol"]).items()
                        if str(k).startswith("f_")
                    }
                    s["p_true"] = model.predict_proba(feat, symbol=str(s["symbol"]))
                ranked = sorted(sc, key=lambda x: x["p_true"], reverse=True)
            win_syms = {w["symbol"] for w in wins}
            top3 = {s["symbol"] for s in ranked[:3]}
            hit1 += int(ranked[0]["symbol"] in win_syms)
            hit3 += int(bool(win_syms & top3))
            n += 1
        return hit1, hit3, n

    h1, h3, n = eval_rank(fake_m, ascending_fake=True)
    print("Pick lowest P(fake) among co-breakouts:")
    print(f"  top-1 is a winner: {h1}/{n} = {h1/n:.1%}" if n else "  n/a")
    print(f"  top-3 has a winner: {h3}/{n} = {h3/n:.1%}\n" if n else "")

    h1, h3, n = eval_rank(true_m, ascending_fake=False)
    print("Pick highest P(true) among co-breakouts:")
    print(f"  top-1 is a winner: {h1}/{n} = {h1/n:.1%}" if n else "  n/a")
    print(f"  top-3 has a winner: {h3}/{n} = {h3/n:.1%}\n" if n else "")

    # naive baseline: pick widest OR
    hit1 = hit3 = n = 0
    for subset in multi.values():
        wins = [r for r in subset if float(r.get("pnl_usdt") or 0) > 0]
        if not wins:
            continue
        ranked = sorted(subset, key=lambda r: float(r.get("f_or_width_pct") or 0), reverse=True)
        win_syms = {w["symbol"] for w in wins}
        top3 = {r["symbol"] for r in ranked[:3]}
        hit1 += int(ranked[0]["symbol"] in win_syms)
        hit3 += int(bool(win_syms & top3))
        n += 1
    print("Baseline: widest OR among co-breakouts:")
    print(f"  top-1 is a winner: {hit1}/{n} = {hit1/n:.1%}" if n else "  n/a")
    print(f"  top-3 has a winner: {hit3}/{n} = {h3/n:.1%}\n" if n else "")

    print("=== 2026-06-11 co-breakout ranking ===")
    d611 = [r for r in rows if r.get("session_date") == "2026-06-11"]
    by_scan: dict[int, list[dict]] = defaultdict(list)
    for r in d611:
        by_scan[int(r.get("scan_open_ms") or 0)].append(r)
    for scan, subset in sorted(by_scan.items()):
        if len(subset) < 2:
            continue
        sc = sorted(score_rows(fake_m, subset), key=lambda x: x["p"])
        print(f"scan {scan} ({len(subset)} symbols, lowest P(fake) first):")
        for s in sc:
            sym = str(s["symbol"]).replace("USDT", "")
            pnl = float(s["pnl_usdt"] or 0)
            tag = "WIN" if pnl > 0 else "LOSS"
            print(f"  {sym:5s} P(fake)={s['p']:.3f} pnl={pnl:+7.1f}U {tag} sync={s.get('sync')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
