"""Supply/Demand zone engine — maps to cheat-sheet red/green boxes."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from quant.rtm_patterns.config import RTMConfig
from quant.rtm_patterns.pivots import (
    alternating_sequence,
    atr,
    bearish_qm_structure,
    bullish_qm_structure,
    equal_level_clusters,
    is_equal,
)
from quant.rtm_patterns.types import Pivot, SupplyDemandZone, ZoneKind


def _zone_from_level(
    kind: ZoneKind,
    level: float,
    bar: int,
    atr_val: float,
    mult: float,
    source: str,
) -> SupplyDemandZone:
    half = max(atr_val * mult, level * 0.0005)
    if kind == "supply":
        return SupplyDemandZone(kind, level + half, level - half, bar, source)
    return SupplyDemandZone(kind, level + half, level - half, bar, source)


def build_supply_demand_zones(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    pivots: Sequence[Pivot],
    cfg: RTMConfig,
) -> list[SupplyDemandZone]:
    """Build S/D zones from QM shoulders, liquidity pools, OB, and consolidation."""
    zones: list[SupplyDemandZone] = []
    atr_vals = atr(high, low, close)
    n = len(close)

    for i in range(cfg.pivot_right * 2, n):
        confirmed = [p for p in pivots if p.index <= i - cfg.pivot_right]
        atr_val = atr_vals[i]
        if atr_val is None or np.isnan(atr_val) or atr_val <= 0:
            continue

        seq4 = alternating_sequence(confirmed, 4)
        if seq4:
            bear = bearish_qm_structure(seq4)
            if bear:
                qml = bear[0]
                zones.append(_zone_from_level("supply", qml, seq4[0].index, atr_val, cfg.zone_atr_mult, "qm_shoulder"))
            bull = bullish_qm_structure(seq4)
            if bull:
                qml = bull[0]
                zones.append(_zone_from_level("demand", qml, seq4[0].index, atr_val, cfg.zone_atr_mult, "qm_shoulder"))

        for kind, zkind in (("high", "supply"), ("low", "demand")):
            for level, group in equal_level_clusters(
                confirmed,
                kind=kind,
                tolerance_pct=cfg.eq_tolerance_pct,
                min_touches=2,
                max_touches=cfg.max_ruler_touches,
            ):
                zones.append(
                    _zone_from_level(zkind, level, group[-1].index, atr_val, cfg.zone_atr_mult, "liquidity_pool")
                )

        lb = cfg.compression_lookback
        if i >= lb:
            box_h = float(np.max(high[i - lb : i]))
            box_l = float(np.min(low[i - lb : i]))
            box_mid = (box_h + box_l) / 2.0
            if close[i] < box_mid:
                zones.append(_zone_from_level("supply", box_h, i - 1, atr_val, cfg.zone_atr_mult, "consolidation"))
            else:
                zones.append(_zone_from_level("demand", box_l, i - 1, atr_val, cfg.zone_atr_mult, "consolidation"))

        if i >= 2:
            move = close[i] - close[i - 2]
            if abs(move) >= atr_val * cfg.displacement_atr_mult:
                if move > 0 and close[i - 1] < open_[i - 1]:
                    zones.append(
                        _zone_from_level("demand", low[i - 1], i - 1, atr_val, cfg.zone_atr_mult, "order_block")
                    )
                if move < 0 and close[i - 1] > open_[i - 1]:
                    zones.append(
                        _zone_from_level("supply", high[i - 1], i - 1, atr_val, cfg.zone_atr_mult, "order_block")
                    )

    return _dedupe_zones(zones, cfg)


def _dedupe_zones(zones: Sequence[SupplyDemandZone], cfg: RTMConfig) -> list[SupplyDemandZone]:
    out: list[SupplyDemandZone] = []
    for z in sorted(zones, key=lambda x: x.formed_bar):
        dup = False
        for existing in out:
            if existing.kind != z.kind:
                continue
            ref = (existing.top + existing.bottom) / 2.0
            mid = (z.top + z.bottom) / 2.0
            if is_equal(ref, mid, ref, cfg.eq_tolerance_pct * 1.5):
                dup = True
                break
        if not dup:
            out.append(z)
    return out


def active_zones(zones: Sequence[SupplyDemandZone], bar_index: int, max_age: int) -> list[SupplyDemandZone]:
    return [z for z in zones if z.formed_bar <= bar_index <= z.formed_bar + max_age]


def bar_touches_zone(high: float, low: float, zone: SupplyDemandZone) -> bool:
    return low <= zone.top and high >= zone.bottom


def find_zone_at(
    zones: Sequence[SupplyDemandZone],
    bar_index: int,
    high: float,
    low: float,
    kind: ZoneKind,
    max_age: int,
) -> SupplyDemandZone | None:
    for z in reversed(active_zones(zones, bar_index, max_age)):
        if z.kind != kind:
            continue
        if bar_touches_zone(high, low, z):
            return z
    return None
