"""单币：L1 寻优 + L3 最近 1500 根选参。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env_loader import load_env_oi

load_env_oi()

from moss_quant import config as cfg
from moss_quant.optimize_service import run_strategy_optimize
from moss_quant.recent_window_pick import apply_recent_pick_to_best

SYM = "SUIUSDT"


def main() -> None:
    cap = float(cfg.MOSS_QUANT_DEFAULT_CAPITAL)
    print(f">>> {SYM} L1 optimize + L3 recent_{cfg.MOSS_QUANT_RECENT_PICK_BARS}")
    opt = run_strategy_optimize(symbol=SYM, capital=cap, refresh_klines=True, top_n=1)
    best = opt.get("best")
    if not best:
        print("L1 无结果")
        return
    best = apply_recent_pick_to_best(best, SYM, capital=cap, refresh_klines=False)
    sm = best.get("summary") or {}
    print(json.dumps(
        {
            "l1": {
                "template": sm.get("l1_template"),
                "train_pct": round(float(sm.get("l1_train_return") or 0) * 100, 2),
                "val_pct": round(float(sm.get("l1_val_return") or 0) * 100, 2),
            },
            "recent_pick": sm.get("recent_pick"),
            "final": {
                "template": best.get("template"),
                "tactical": best.get("tactical_params"),
                "param_source": sm.get("param_source"),
                "recent_applied": sm.get("recent_applied"),
            },
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
