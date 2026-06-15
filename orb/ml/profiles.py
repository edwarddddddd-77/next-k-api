#!/usr/bin/env python3
"""标的突破画像 + 先验校准。"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from orb.ml.features import FEATURE_NAMES, label_is_true_breakout

from orb.ml.paths import V1_OUTPUT, default_profiles_path

DEFAULT_PROFILES_PATH = V1_OUTPUT / "symbol_breakout_profiles.json"

PRIOR_MIN_SAMPLES = 15
PRIOR_FULL_SAMPLES = 60


def _feat(row: dict, key: str) -> float:
    return float(row.get(f"f_{key}") or row.get(key) or 0.0)


def build_profiles(rows: List[Dict[str, Any]], *, global_true_rate: Optional[float] = None) -> Dict[str, Any]:
    by: dict[str, list] = defaultdict(list)
    for r in rows:
        sym = str(r.get("symbol") or "").upper()
        if sym:
            by[sym].append(r)

    if global_true_rate is None:
        global_true_rate = sum(int(r.get("true_breakout") or label_is_true_breakout(str(r.get("outcome", "")), float(r.get("pnl_usdt") or 0))) for r in rows) / max(len(rows), 1)

    profiles: dict[str, dict] = {}
    for sym, grp in sorted(by.items()):
        n = len(grp)
        true_n = sum(int(r.get("true_breakout") or label_is_true_breakout(str(r.get("outcome", "")), float(r.get("pnl_usdt") or 0))) for r in grp)
        hold30_n = sum(int(r.get("hold30_true") or 0) for r in grp)
        pnl = [float(r.get("pnl_usdt") or 0) for r in grp]
        mins = [_feat(r, "minutes_after_or") for r in grp]
        syncs = [_feat(r, "sync_same_side") for r in grp]
        tier = "A" if n >= 30 else ("B" if n >= PRIOR_MIN_SAMPLES else "C")
        profiles[sym] = {
            "symbol": sym.replace("USDT", ""),
            "n": n,
            "tier": tier,
            "true_n": true_n,
            "true_rate": round(true_n / n, 4) if n else 0.0,
            "hold30_n": hold30_n,
            "hold30_rate": round(hold30_n / n, 4) if n else 0.0,
            "avg_pnl_usdt": round(sum(pnl) / n, 2) if n else 0.0,
            "median_minutes_after_or": round(sorted(mins)[len(mins) // 2], 1) if mins else 0.0,
            "median_sync": round(sorted(syncs)[len(syncs) // 2], 1) if syncs else 0.0,
        }

    return {
        "global_true_rate": round(global_true_rate, 4),
        "symbols": len(profiles),
        "profiles": profiles,
    }


def save_profiles(data: Dict[str, Any], path: Optional[Path] = None) -> Path:
    out = path or DEFAULT_PROFILES_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def load_profiles(path: Optional[Path] = None) -> Dict[str, Any]:
    from orb.ml.paths import default_profiles_path

    p = path or default_profiles_path()
    if not p.is_file():
        return {"profiles": {}, "global_true_rate": 0.2}
    return json.loads(p.read_text(encoding="utf-8"))


def symbol_prior_factor(sym: str, profiles: Dict[str, Any]) -> float:
    """0~1 先验：高历史真率标的加权。"""
    key = str(sym or "").upper()
    if not key.endswith("USDT"):
        key = key + "USDT"
    prof = (profiles.get("profiles") or {}).get(key)
    if not prof:
        return 0.5
    n = int(prof.get("n") or 0)
    if n < PRIOR_MIN_SAMPLES:
        return 0.5
    rate = float(prof.get("hold30_rate") or prof.get("true_rate") or 0.5)
    w = min(1.0, n / PRIOR_FULL_SAMPLES)
    return 0.5 + w * (rate - 0.5)


def apply_symbol_prior(p_model: float, sym: str, profiles: Dict[str, Any], *, model_weight: float = 0.75) -> float:
    prior = symbol_prior_factor(sym, profiles)
    p = max(1e-6, min(1 - 1e-6, float(p_model)))
    pr = max(1e-6, min(1 - 1e-6, prior))
    logit = math.log(p / (1 - p))
    pl = math.log(pr / (1 - pr))
    mw = max(0.0, min(1.0, model_weight))
    z = mw * logit + (1 - mw) * pl
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
