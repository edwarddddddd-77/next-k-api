#!/usr/bin/env python3
"""诊断模型排序失准原因。"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    data = json.loads((ROOT / "output" / "rank_days_batch.json").read_text(encoding="utf-8"))
    vocab = set(json.loads((ROOT / "output" / "orb_shared_fake_breakout_model.json").read_text())["symbol_vocab"])

    rows = []
    for d in data["days"]:
        for r in d["ranked"]:
            rows.append({**r, "date": d["date"]})

    print("=== 1. FLNC/CBRS 为何常占 Top-1 ===")
    for d in data["days"]:
        t1 = d.get("top1") or {}
        if t1.get("symbol") in ("FLNC", "CBRS", "GOOGL"):
            r1 = d["ranked"][0]
            in_vocab = f"{t1['symbol']}USDT" in vocab
            syncs = [r["sync"] for r in d["ranked"]]
            print(
                f"{d['date']} Top1={t1['symbol']} sync={r1['sync']} side={r1['side']} "
                f"p_true={r1['p_true']:.3f} true={r1['true']} in_vocab={in_vocab} "
                f"day_sync[min,med,max]=[{min(syncs)},{st.median(syncs):.0f},{max(syncs)}]"
            )

    print("\n=== 2. sync 与 P(true) ===")
    for lo, hi, label in [(0, 0, "sync=0"), (1, 9, "sync 1-9"), (10, 16, "sync 10-16"), (17, 99, "sync>=17")]:
        sub = [r for r in rows if lo <= r["sync"] <= hi]
        if not sub:
            continue
        tr = sum(1 for r in sub if r["true"]) / len(sub)
        print(f"{label:12s} n={len(sub):3d} true_rate={tr:5.1%} avg_p_true={st.mean(r['p_true'] for r in sub):.3f}")

    print("\n=== 3. 训练 vocab 未覆盖的标 (18个) ===")
    all_syms = {r["symbol"] for r in rows}
    oov = sorted(s for s in all_syms if f"{s}USDT" not in vocab)
    print("OOV symbols seen in 10d:", ", ".join(oov))
    oov_top1 = sum(1 for d in data["days"] if d.get("top1") and d["top1"]["symbol"] in oov)
    print(f"OOV as Top-1: {oov_top1}/10 days")

    print("\n=== 4. P(true) 分离度（10天全样本）===")
    t = [r["p_true"] for r in rows if r["true"]]
    f = [r["p_true"] for r in rows if not r["true"]]
    print(f"TRUE  mean={st.mean(t):.4f}  FAKE mean={st.mean(f):.4f}  gap={st.mean(t)-st.mean(f):.4f}")

    print("\n=== 5. 同日「先突破 sync=0」vs「晚突破 sync高」===")
    early_fake = late_true = 0
    for d in data["days"]:
        ranked = d["ranked"]
        if not ranked:
            continue
        early = [r for r in ranked if r["sync"] == 0]
        late = [r for r in ranked if r["sync"] >= 10]
        if early:
            early_fake += sum(1 for r in early if not r["true"]) / len(early)
        if late:
            late_true += sum(1 for r in late if r["true"]) / len(late)
    n_days = len(data["days"])
    print(f"avg fake rate when sync=0 (first movers): {early_fake/n_days:.1%}")
    print(f"avg true rate when sync>=10 (late cohort): {late_true/n_days:.1%}")

    print("\n=== 6. 6/03 高 sync 赢家被压分 ===")
    d603 = next(d for d in data["days"] if d["date"] == "2026-06-03")
    for r in d603["ranked"]:
        if r["true"] and r["sync"] >= 20:
            print(f"  {r['symbol']:5s} rank={r['rank']:2d} p_true={r['p_true']:.3f} pnl={r['pnl_usdt']:+.1f}U sync={r['sync']}")


if __name__ == "__main__":
    main()
