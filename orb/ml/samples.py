#!/usr/bin/env python3
"""ORB 突破训练样本工具。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from orb.ml.features import FEATURE_NAMES, label_is_fake, label_is_true_breakout

MAX_LOOKBACK_DAYS = 180.0
MIN_SYMBOL_SAMPLES = 8


def norm_symbol(raw: str) -> str:
    s = str(raw).strip().upper()
    if not s:
        return ""
    return s if s.endswith("USDT") else s + "USDT"


def parse_symbol_list(text: str) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for part in text.replace("\n", ",").split(","):
        raw = part.strip()
        if not raw or raw.startswith("#"):
            continue
        sym = norm_symbol(raw)
        if sym and sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def trade_to_row(t: Dict[str, Any]) -> Dict[str, Any]:
    if t.get("outcome") in (None, "supersede") or not t.get("features"):
        return {}
    feat = dict(t["features"])
    outcome = str(t.get("outcome") or "")
    pnl = float(t.get("pnl_usdt") or 0)
    return {
        "session_date": t.get("session_date", ""),
        "symbol": str(t.get("symbol") or "").upper(),
        "side": t.get("side", ""),
        "outcome": outcome,
        "entry": t.get("entry"),
        "sl": t.get("sl"),
        "exit_price": t.get("exit_price"),
        "pnl_usdt": pnl,
        "pnl_r": t.get("pnl_r"),
        "scan_open_ms": t.get("scan_open_ms"),
        "entry_bar_open_ms": t.get("entry_bar_open_ms"),
        "fake_breakout": label_is_fake(outcome, pnl),
        "true_breakout": label_is_true_breakout(outcome, pnl),
        **{f"f_{k}": feat.get(k) for k in FEATURE_NAMES},
    }


def collect_rows(raw: dict) -> List[Dict[str, Any]]:
    return [r for r in (trade_to_row(t) for t in raw.get("trades") or []) if r]


def recompute_sync(rows: List[Dict[str, Any]]) -> None:
    buckets: dict[tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        scan_ms = int(r.get("scan_open_ms") or r.get("entry_bar_open_ms") or 0)
        buckets[(str(r.get("session_date") or ""), scan_ms)].append(r)
    for group in buckets.values():
        for r in group:
            side = str(r.get("side") or "").upper()
            sym = str(r.get("symbol") or "").upper()
            sync = sum(
                1
                for o in group
                if str(o.get("symbol") or "").upper() != sym
                and str(o.get("side") or "").upper() == side
            )
            r["f_sync_same_side"] = float(sync)


def filter_low_sample_symbols(
    rows: List[Dict[str, Any]], *, min_samples: int = MIN_SYMBOL_SAMPLES
) -> tuple[List[Dict[str, Any]], List[str]]:
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        counts[str(r.get("symbol") or "").upper()] += 1
    dropped = sorted(s for s, n in counts.items() if n < min_samples)
    kept = {s for s, n in counts.items() if n >= min_samples}
    return [r for r in rows if str(r.get("symbol") or "").upper() in kept], dropped


def split_holdout_by_date(
    rows: List[Dict[str, Any]], *, holdout_days: int = 10
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dates = sorted({str(r.get("session_date") or "") for r in rows if r.get("session_date")})
    if holdout_days <= 0 or len(dates) <= holdout_days:
        return rows, []
    hold = set(dates[-holdout_days:])
    train = [r for r in rows if str(r.get("session_date") or "") not in hold]
    test = [r for r in rows if str(r.get("session_date") or "") in hold]
    return train, test
