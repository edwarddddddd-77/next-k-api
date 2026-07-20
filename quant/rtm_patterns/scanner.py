"""Scan OHLC data for RTM institutional price action patterns."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from quant.rtm_patterns.config import RTMConfig
from quant.rtm_patterns.patterns import detect_all_patterns
from quant.rtm_patterns.pivots import find_pivots
from quant.rtm_patterns.types import PatternHit

# All cheat-sheet pattern IDs
RTM_PATTERN_IDS: tuple[str, ...] = (
    "qm_quick_retest",
    "qm_late_retest",
    "qm_shadow",
    "qm_reentry",
    "qm_continuation",
    "qm_ignored",
    "fakeout_2r",
    "fakeout_2s",
    "ruler",
    "v_twin",
    "mpl",
    "flag_b",
    "flag_ab",
    "fakeout_v1",
    "fakeout_v2_sr_flip",
    "fakeout_v3_diamond",
    "double_ssr",
    "compression",
    "cplq",
    "three_drive",
    "cancan",
    "cancan_fakeout",
)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c.lower(): c for c in df.columns}
    required = ["open", "high", "low", "close"]
    out = df.copy()
    rename: dict[str, str] = {}
    for req in required:
        if req not in mapping:
            raise ValueError(f"DataFrame missing column: {req}")
        rename[mapping[req]] = req
    return out.rename(columns=rename)


def scan_rtm_patterns(
    df: pd.DataFrame,
    *,
    config: RTMConfig | None = None,
    patterns: Sequence[str] | None = None,
    dedupe: bool = True,
) -> list[PatternHit]:
    """
    Detect RTM patterns on OHLC data.

    Parameters
    ----------
    df : DataFrame with open, high, low, close columns (case-insensitive).
    config : Detection thresholds.
    patterns : Optional filter — only return these pattern IDs.
    dedupe : Collapse duplicate (pattern, bar_index, direction) hits.
    """
    cfg = config or RTMConfig()
    norm = _normalize_columns(df)
    open_ = norm["open"].to_numpy(dtype=float)
    high = norm["high"].to_numpy(dtype=float)
    low = norm["low"].to_numpy(dtype=float)
    close = norm["close"].to_numpy(dtype=float)

    pivots = find_pivots(high, low, left=cfg.pivot_left, right=cfg.pivot_right)
    hits = detect_all_patterns(open_, high, low, close, pivots, cfg)

    allowed = set(patterns) if patterns else None
    if allowed:
        hits = [h for h in hits if h.pattern in allowed]

    if dedupe:
        seen: set[tuple[str, int, str]] = set()
        unique: list[PatternHit] = []
        for h in hits:
            key = (h.pattern, h.bar_index, h.direction)
            if key not in seen:
                seen.add(key)
                unique.append(h)
        hits = unique

    hits.sort(key=lambda h: (h.bar_index, h.pattern))
    return hits


def hits_to_dataframe(hits: Sequence[PatternHit]) -> pd.DataFrame:
    """Convert pattern hits to a summary DataFrame."""
    rows = [
        {
            "bar_index": h.bar_index,
            "pattern": h.pattern,
            "direction": h.direction,
            "qml_level": h.qml_level,
            "entry_level": h.entry_level,
            "stop_level": h.stop_level,
            "target_level": h.target_level,
            "pivot_indices": h.pivot_indices,
            "quality": h.meta.get("quality"),
            "zone_source": h.meta.get("zone_source"),
            **{f"meta_{k}": v for k, v in h.meta.items()},
        }
        for h in hits
    ]
    return pd.DataFrame(rows)


def pattern_counts(hits: Sequence[PatternHit]) -> dict[str, int]:
    counts: dict[str, int] = {p: 0 for p in RTM_PATTERN_IDS}
    for h in hits:
        counts[h.pattern] = counts.get(h.pattern, 0) + 1
    return {k: v for k, v in counts.items() if v > 0}
