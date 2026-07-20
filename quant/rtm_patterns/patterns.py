"""RTM pattern detectors v2 — S/D zones, rejection, quality scoring."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from quant.rtm_patterns.config import RTMConfig
from quant.rtm_patterns.pivots import (
    alternating_sequence,
    atr,
    bearish_qm_structure,
    bullish_qm_structure,
    compute_sr_flips,
    count_level_role_flips,
    count_zigzags,
    equal_level_clusters,
    is_bearish_rejection,
    is_bullish_rejection,
    is_equal,
    pivots_before,
    range_contraction,
    trend_direction,
)
from quant.rtm_patterns.quality import attach_quality, passes_quality, score_hit
from quant.rtm_patterns.types import PatternHit, Pivot, SRFlipLevel, SupplyDemandZone
from quant.rtm_patterns.zones import active_zones, bar_touches_zone, build_supply_demand_zones, find_zone_at


def _emit(
    hit: PatternHit,
    *,
    zone: SupplyDemandZone | None,
    rejection: bool,
    structure_ok: bool = True,
    liquidity_sweep: bool = False,
    trend_aligned: bool = False,
    cfg: RTMConfig,
    **extra,
) -> PatternHit | None:
    q = score_hit(
        zone=zone,
        rejection=rejection,
        structure_ok=structure_ok,
        liquidity_sweep=liquidity_sweep,
        trend_aligned=trend_aligned,
    )
    out = attach_quality(hit, q, zone_source=zone.source if zone else None, **extra)
    return out if passes_quality(out, cfg.min_quality_score) else None


# ── QM family ──────────────────────────────────────────────────────────────


def detect_qm_patterns(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    pivots: Sequence[Pivot],
    zones: Sequence[SupplyDemandZone],
    cfg: RTMConfig,
    atr_vals: np.ndarray,
) -> list[PatternHit]:
    hits: list[PatternHit] = []
    n = len(close)

    for i in range(n):
        confirmed = pivots_before(pivots, i - cfg.pivot_right)
        seq4 = alternating_sequence(confirmed, 4)
        if seq4 is None:
            continue
        bars_since = i - seq4[-1].index

        bear = bearish_qm_structure(seq4)
        if bear is not None:
            qml, head, ll, l1, pidx = bear
            supply = find_zone_at(zones, i, high[i], low[i], "supply", cfg.zone_max_age_bars)
            rej = is_bearish_rejection(
                open_, high, low, close, i, qml,
                body_ratio=cfg.engulfing_body_ratio,
                wick_ratio=cfg.rejection_wick_ratio,
            )
            in_zone = supply is not None and bar_touches_zone(high[i], low[i], supply)

            if bars_since <= cfg.qm_late_retest_bars and low[i] <= qml <= high[i]:
                if not cfg.require_zone_for_qm or in_zone:
                    if not cfg.require_rejection_candle or rej:
                        pat = "qm_quick_retest" if bars_since <= cfg.qm_quick_retest_bars else "qm_late_retest"
                        h = _emit(
                            PatternHit(
                                pat, "short", i,
                                qml_level=qml, entry_level=close[i], stop_level=head, target_level=l1,
                                pivot_indices=pidx, meta={"bars_since": bars_since},
                            ),
                            zone=supply, rejection=rej, liquidity_sweep=high[i] > qml, cfg=cfg,
                        )
                        if h:
                            hits.append(h)

            if rej and abs(high[i] - head) <= (atr_vals[i] or 0) * cfg.zone_atr_mult * 2:
                h = _emit(
                    PatternHit(
                        "qm_shadow", "short", i,
                        qml_level=qml, entry_level=close[i], stop_level=head, target_level=l1, pivot_indices=pidx,
                    ),
                    zone=supply, rejection=True, cfg=cfg,
                )
                if h:
                    hits.append(h)

            if trend_direction(close, i) < 0 and in_zone and rej:
                h = _emit(
                    PatternHit(
                        "qm_continuation", "short", i,
                        qml_level=qml, entry_level=close[i], stop_level=head, pivot_indices=pidx,
                    ),
                    zone=supply, rejection=rej, trend_aligned=True, cfg=cfg,
                )
                if h:
                    hits.append(h)

            if close[i] > qml and seq4[-1].index < i and high[i] > qml:
                h = _emit(
                    PatternHit(
                        "qm_ignored", "long", i,
                        qml_level=qml, entry_level=close[i], stop_level=ll, pivot_indices=pidx, meta={"flip": True},
                    ),
                    zone=supply, rejection=False, liquidity_sweep=True, cfg=cfg,
                )
                if h:
                    hits.append(h)

        bull = bullish_qm_structure(seq4)
        if bull is not None:
            qml, head, hh, h1, pidx = bull
            demand = find_zone_at(zones, i, high[i], low[i], "demand", cfg.zone_max_age_bars)
            rej = is_bullish_rejection(
                open_, high, low, close, i, qml,
                body_ratio=cfg.engulfing_body_ratio,
                wick_ratio=cfg.rejection_wick_ratio,
            )
            in_zone = demand is not None and bar_touches_zone(high[i], low[i], demand)

            if bars_since <= cfg.qm_late_retest_bars and low[i] <= qml <= high[i]:
                if not cfg.require_zone_for_qm or in_zone:
                    if not cfg.require_rejection_candle or rej:
                        pat = "qm_quick_retest" if bars_since <= cfg.qm_quick_retest_bars else "qm_late_retest"
                        h = _emit(
                            PatternHit(
                                pat, "long", i,
                                qml_level=qml, entry_level=close[i], stop_level=head, target_level=h1,
                                pivot_indices=pidx, meta={"bars_since": bars_since},
                            ),
                            zone=demand, rejection=rej, liquidity_sweep=low[i] < qml, cfg=cfg,
                        )
                        if h:
                            hits.append(h)

            if rej and abs(low[i] - head) <= (atr_vals[i] or 0) * cfg.zone_atr_mult * 2:
                h = _emit(
                    PatternHit(
                        "qm_shadow", "long", i,
                        qml_level=qml, entry_level=close[i], stop_level=head, target_level=h1, pivot_indices=pidx,
                    ),
                    zone=demand, rejection=True, cfg=cfg,
                )
                if h:
                    hits.append(h)

            if trend_direction(close, i) > 0 and in_zone and rej:
                h = _emit(
                    PatternHit(
                        "qm_continuation", "long", i,
                        qml_level=qml, entry_level=close[i], stop_level=head, pivot_indices=pidx,
                    ),
                    zone=demand, rejection=rej, trend_aligned=True, cfg=cfg,
                )
                if h:
                    hits.append(h)

            if close[i] < qml and seq4[-1].index < i and low[i] < qml:
                h = _emit(
                    PatternHit(
                        "qm_ignored", "short", i,
                        qml_level=qml, entry_level=close[i], stop_level=hh, pivot_indices=pidx, meta={"flip": True},
                    ),
                    zone=demand, rejection=False, liquidity_sweep=True, cfg=cfg,
                )
                if h:
                    hits.append(h)

    for i in range(n):
        confirmed = pivots_before(pivots, i - cfg.pivot_right)
        if len(confirmed) < 6:
            continue
        outer = alternating_sequence(confirmed[-6:-2], 4)
        inner = alternating_sequence(confirmed[-4:], 4)
        if outer is None or inner is None:
            continue

        ob = bearish_qm_structure(outer)
        ib = bearish_qm_structure(inner)
        if ob and ib and is_equal(ob[0], ib[0], ob[0], cfg.eq_tolerance_pct):
            supply = find_zone_at(zones, i, high[i], low[i], "supply", cfg.zone_max_age_bars)
            rej = is_bearish_rejection(open_, high, low, close, i, ob[0], body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
            if rej:
                h = _emit(
                    PatternHit(
                        "qm_reentry", "short", i,
                        qml_level=ob[0], entry_level=close[i], stop_level=ib[1],
                        pivot_indices=ib[4], meta={"main_qml": ob[0]},
                    ),
                    zone=supply, rejection=rej, structure_ok=True, cfg=cfg,
                )
                if h:
                    hits.append(h)

        ob2 = bullish_qm_structure(outer)
        ib2 = bullish_qm_structure(inner)
        if ob2 and ib2 and is_equal(ob2[0], ib2[0], ob2[0], cfg.eq_tolerance_pct):
            demand = find_zone_at(zones, i, high[i], low[i], "demand", cfg.zone_max_age_bars)
            rej = is_bullish_rejection(open_, high, low, close, i, ob2[0], body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
            if rej:
                h = _emit(
                    PatternHit(
                        "qm_reentry", "long", i,
                        qml_level=ob2[0], entry_level=close[i], stop_level=ib2[1],
                        pivot_indices=ib2[4], meta={"main_qml": ob2[0]},
                    ),
                    zone=demand, rejection=rej, structure_ok=True, cfg=cfg,
                )
                if h:
                    hits.append(h)

    return hits


# ── Fakeout family ───────────────────────────────────────────────────────────


def detect_fakeout_patterns(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    pivots: Sequence[Pivot],
    zones: Sequence[SupplyDemandZone],
    sr_flips: Sequence[SRFlipLevel],
    cfg: RTMConfig,
) -> list[PatternHit]:
    hits: list[PatternHit] = []
    n = len(close)

    for i in range(1, n):
        confirmed = pivots_before(pivots, i - 1)

        high_clusters = equal_level_clusters(confirmed, kind="high", tolerance_pct=cfg.eq_tolerance_pct, min_touches=2, max_touches=2)
        low_clusters = equal_level_clusters(confirmed, kind="low", tolerance_pct=cfg.eq_tolerance_pct, min_touches=2, max_touches=2)

        for level, group in high_clusters:
            swept = high[i] > level and close[i] < level
            if not swept:
                continue
            supply = find_zone_at(zones, i, high[i], low[i], "supply", cfg.zone_max_age_bars)
            rej = is_bearish_rejection(open_, high, low, close, i, level, body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
            if cfg.require_zone_for_fakeout and supply is None:
                continue
            if cfg.require_rejection_candle and not rej:
                continue
            for pat in ("fakeout_2r", "fakeout_v1"):
                h = _emit(
                    PatternHit(pat, "short", i, entry_level=close[i], stop_level=high[i], target_level=level, pivot_indices=tuple(p.index for p in group)),
                    zone=supply, rejection=rej, liquidity_sweep=True, cfg=cfg,
                )
                if h:
                    hits.append(h)

        for level, group in low_clusters:
            swept = low[i] < level and close[i] > level
            if not swept:
                continue
            demand = find_zone_at(zones, i, high[i], low[i], "demand", cfg.zone_max_age_bars)
            rej = is_bullish_rejection(open_, high, low, close, i, level, body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
            if cfg.require_zone_for_fakeout and demand is None:
                continue
            if cfg.require_rejection_candle and not rej:
                continue
            for pat in ("fakeout_2s", "fakeout_v1"):
                h = _emit(
                    PatternHit(pat, "long", i, entry_level=close[i], stop_level=low[i], target_level=level, pivot_indices=tuple(p.index for p in group)),
                    zone=demand, rejection=rej, liquidity_sweep=True, cfg=cfg,
                )
                if h:
                    hits.append(h)

        for kind, direction, sweep_hi in (("high", "short", True), ("low", "long", False)):
            clusters = equal_level_clusters(
                confirmed, kind=kind, tolerance_pct=cfg.eq_tolerance_pct,
                min_touches=cfg.min_ruler_touches, max_touches=cfg.max_ruler_touches,
            )
            for level, group in clusters:
                swept = (sweep_hi and high[i] > level and close[i] < level) or (not sweep_hi and low[i] < level and close[i] > level)
                if not swept:
                    continue
                zkind = "supply" if sweep_hi else "demand"
                zone = find_zone_at(zones, i, high[i], low[i], zkind, cfg.zone_max_age_bars)
                rej_fn = is_bearish_rejection if sweep_hi else is_bullish_rejection
                rej = rej_fn(open_, high, low, close, i, level, body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
                if not rej:
                    continue
                h = _emit(
                    PatternHit(
                        "ruler", direction, i,
                        entry_level=close[i], stop_level=high[i] if sweep_hi else low[i],
                        pivot_indices=tuple(p.index for p in group), meta={"touches": len(group)},
                    ),
                    zone=zone, rejection=rej, liquidity_sweep=True, cfg=cfg,
                )
                if h:
                    hits.append(h)

        if len(confirmed) >= 3:
            p1, p2, p3 = confirmed[-3], confirmed[-2], confirmed[-1]
            if p1.kind == "low" and p2.kind == "high" and p3.kind == "low":
                sharp = (p2.price - p1.price) > (p2.price - p3.price) * 0.8
                if sharp and high[i] > p2.price and close[i] < p2.price:
                    supply = find_zone_at(zones, i, high[i], low[i], "supply", cfg.zone_max_age_bars)
                    rej = is_bearish_rejection(open_, high, low, close, i, p2.price, body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
                    if rej:
                        h = _emit(
                            PatternHit(
                                "v_twin", "short", i,
                                entry_level=close[i], stop_level=high[i], pivot_indices=(p1.index, p2.index, p3.index),
                            ),
                            zone=supply, rejection=rej, liquidity_sweep=True, cfg=cfg,
                        )
                        if h:
                            hits.append(h)
            if p1.kind == "high" and p2.kind == "low" and p3.kind == "high":
                sharp = (p1.price - p2.price) > (p3.price - p2.price) * 0.8
                if sharp and low[i] < p2.price and close[i] > p2.price:
                    demand = find_zone_at(zones, i, high[i], low[i], "demand", cfg.zone_max_age_bars)
                    rej = is_bullish_rejection(open_, high, low, close, i, p2.price, body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
                    if rej:
                        h = _emit(
                            PatternHit(
                                "v_twin", "long", i,
                                entry_level=close[i], stop_level=low[i], pivot_indices=(p1.index, p2.index, p3.index),
                            ),
                            zone=demand, rejection=rej, liquidity_sweep=True, cfg=cfg,
                        )
                        if h:
                            hits.append(h)

        for flip in sr_flips:
            if flip.broken_bar >= i or i - flip.broken_bar > cfg.sr_flip_max_retest_bars:
                continue
            level = flip.level
            if flip.original == "support":
                retest = high[i] >= level and close[i] < level
                rej = is_bearish_rejection(open_, high, low, close, i, level, body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
                supply = find_zone_at(zones, i, high[i], low[i], "supply", cfg.zone_max_age_bars)
                if retest and rej and supply is not None:
                    h = _emit(
                        PatternHit(
                            "fakeout_v2_sr_flip", "short", i,
                            entry_level=close[i], stop_level=high[i],
                            pivot_indices=(flip.pivot_index,), meta={"flip": "S->R"},
                        ),
                        zone=supply, rejection=rej, liquidity_sweep=True, cfg=cfg,
                    )
                    if h:
                        hits.append(h)
            else:
                retest = low[i] <= level and close[i] > level
                rej = is_bullish_rejection(open_, high, low, close, i, level, body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
                demand = find_zone_at(zones, i, high[i], low[i], "demand", cfg.zone_max_age_bars)
                if retest and rej and demand is not None:
                    h = _emit(
                        PatternHit(
                            "fakeout_v2_sr_flip", "long", i,
                            entry_level=close[i], stop_level=low[i],
                            pivot_indices=(flip.pivot_index,), meta={"flip": "R->S"},
                        ),
                        zone=demand, rejection=rej, liquidity_sweep=True, cfg=cfg,
                    )
                    if h:
                        hits.append(h)

        seq4 = alternating_sequence(confirmed, 4)
        if seq4:
            bq = bearish_qm_structure(seq4)
            if bq:
                head = bq[1]
                for level, group in high_clusters:
                    if is_equal(head, level, head, cfg.eq_tolerance_pct):
                        rej = is_bearish_rejection(open_, high, low, close, i, bq[0], body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
                        if low[i] <= bq[0] <= high[i] and rej:
                            supply = find_zone_at(zones, i, high[i], low[i], "supply", cfg.zone_max_age_bars)
                            h = _emit(
                                PatternHit(
                                    "fakeout_v3_diamond", "short", i,
                                    qml_level=bq[0], entry_level=close[i], stop_level=head,
                                    pivot_indices=bq[4], meta={"head_level": level},
                                ),
                                zone=supply, rejection=rej, structure_ok=True, cfg=cfg,
                            )
                            if h:
                                hits.append(h)
            bu = bullish_qm_structure(seq4)
            if bu:
                head = bu[1]
                for level, group in low_clusters:
                    if is_equal(head, level, head, cfg.eq_tolerance_pct):
                        rej = is_bullish_rejection(open_, high, low, close, i, bu[0], body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
                        if low[i] <= bu[0] <= high[i] and rej:
                            demand = find_zone_at(zones, i, high[i], low[i], "demand", cfg.zone_max_age_bars)
                            h = _emit(
                                PatternHit(
                                    "fakeout_v3_diamond", "long", i,
                                    qml_level=bu[0], entry_level=close[i], stop_level=head,
                                    pivot_indices=bu[4], meta={"head_level": level},
                                ),
                                zone=demand, rejection=rej, structure_ok=True, cfg=cfg,
                            )
                            if h:
                                hits.append(h)

        for level, group in high_clusters:
            if high[i] > level and close[i] < level:
                prior_lows = [p.price for p in confirmed if p.kind == "low" and p.index < i]
                if prior_lows and close[i] < min(prior_lows) - (level * cfg.mpl_engulf_atr / 100):
                    supply = find_zone_at(zones, i, high[i], low[i], "supply", cfg.zone_max_age_bars)
                    rej = is_bearish_rejection(open_, high, low, close, i, level, body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
                    if rej:
                        h = _emit(
                            PatternHit(
                                "mpl", "short", i,
                                entry_level=close[i], stop_level=high[i],
                                pivot_indices=tuple(p.index for p in group), meta={"engulfed_low": min(prior_lows)},
                            ),
                            zone=supply, rejection=rej, liquidity_sweep=True, cfg=cfg,
                        )
                        if h:
                            hits.append(h)

        for level, group in low_clusters:
            if low[i] < level and close[i] > level:
                prior_highs = [p.price for p in confirmed if p.kind == "high" and p.index < i]
                if prior_highs and close[i] > max(prior_highs) + (level * cfg.mpl_engulf_atr / 100):
                    demand = find_zone_at(zones, i, high[i], low[i], "demand", cfg.zone_max_age_bars)
                    rej = is_bullish_rejection(open_, high, low, close, i, level, body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
                    if rej:
                        h = _emit(
                            PatternHit(
                                "mpl", "long", i,
                                entry_level=close[i], stop_level=low[i],
                                pivot_indices=tuple(p.index for p in group), meta={"engulfed_high": max(prior_highs)},
                            ),
                            zone=demand, rejection=rej, liquidity_sweep=True, cfg=cfg,
                        )
                        if h:
                            hits.append(h)

        seen_levels: set[float] = set()
        for p in confirmed[-25:]:
            level = p.price
            if level in seen_levels:
                continue
            seen_levels.add(level)
            hi_t, lo_t = count_level_role_flips(confirmed, level, cfg.eq_tolerance_pct * 2)
            if hi_t >= cfg.double_ssr_min_flips and lo_t >= cfg.double_ssr_min_flips:
                if high[i] > level and close[i] < level:
                    supply = find_zone_at(zones, i, high[i], low[i], "supply", cfg.zone_max_age_bars)
                    rej = is_bearish_rejection(open_, high, low, close, i, level, body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
                    if rej:
                        h = _emit(
                            PatternHit(
                                "double_ssr", "short", i,
                                entry_level=close[i], stop_level=high[i],
                                pivot_indices=(p.index,), meta={"high_touches": hi_t, "low_touches": lo_t},
                            ),
                            zone=supply, rejection=rej, liquidity_sweep=True, cfg=cfg,
                        )
                        if h:
                            hits.append(h)
                if low[i] < level and close[i] > level:
                    demand = find_zone_at(zones, i, high[i], low[i], "demand", cfg.zone_max_age_bars)
                    rej = is_bullish_rejection(open_, high, low, close, i, level, body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
                    if rej:
                        h = _emit(
                            PatternHit(
                                "double_ssr", "long", i,
                                entry_level=close[i], stop_level=low[i],
                                pivot_indices=(p.index,), meta={"high_touches": hi_t, "low_touches": lo_t},
                            ),
                            zone=demand, rejection=rej, liquidity_sweep=True, cfg=cfg,
                        )
                        if h:
                            hits.append(h)

    return hits


# ── Compression / Drive / Can-Can ────────────────────────────────────────────


def detect_compression_patterns(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    pivots: Sequence[Pivot],
    zones: Sequence[SupplyDemandZone],
    cfg: RTMConfig,
    atr_vals: np.ndarray,
) -> list[PatternHit]:
    hits: list[PatternHit] = []
    n = len(close)
    lb = cfg.compression_lookback

    for i in range(lb, n):
        atr_val = atr_vals[i]
        if atr_val is None or np.isnan(atr_val) or atr_val <= 0:
            continue

        window_h = high[i - lb : i + 1]
        window_l = low[i - lb : i + 1]
        window_range = float(np.max(window_h) - np.min(window_l))
        compressed = window_range <= atr_val * cfg.compression_atr_ratio * lb / 2.5
        contracting = range_contraction(high, low, i, lb)
        zigzags = count_zigzags(pivots, i - lb, i)
        if not (compressed and contracting and zigzags >= cfg.compression_min_zigzags):
            continue

        zone_high = float(np.max(window_h))
        zone_low = float(np.min(window_l))

        if high[i] >= zone_high - atr_val * 0.2:
            supply = find_zone_at(zones, i, high[i], low[i], "supply", cfg.zone_max_age_bars)
            rej = is_bearish_rejection(open_, high, low, close, i, zone_high, body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
            if rej and close[i] < close[i - 1]:
                h = _emit(
                    PatternHit(
                        "compression", "short", i,
                        entry_level=close[i], stop_level=zone_high, target_level=zone_low, meta={"zigzags": zigzags},
                    ),
                    zone=supply, rejection=rej, structure_ok=True, cfg=cfg,
                )
                if h:
                    hits.append(h)
            if high[i] > zone_high and close[i] < zone_high and rej:
                h = _emit(
                    PatternHit("cplq", "short", i, entry_level=close[i], stop_level=high[i], meta={"liquidity_grab": True}),
                    zone=supply, rejection=rej, liquidity_sweep=True, cfg=cfg,
                )
                if h:
                    hits.append(h)

        if low[i] <= zone_low + atr_val * 0.2:
            demand = find_zone_at(zones, i, high[i], low[i], "demand", cfg.zone_max_age_bars)
            rej = is_bullish_rejection(open_, high, low, close, i, zone_low, body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
            if rej and close[i] > close[i - 1]:
                h = _emit(
                    PatternHit(
                        "compression", "long", i,
                        entry_level=close[i], stop_level=zone_low, target_level=zone_high, meta={"zigzags": zigzags},
                    ),
                    zone=demand, rejection=rej, structure_ok=True, cfg=cfg,
                )
                if h:
                    hits.append(h)
            if low[i] < zone_low and close[i] > zone_low and rej:
                h = _emit(
                    PatternHit("cplq", "long", i, entry_level=close[i], stop_level=low[i], meta={"liquidity_grab": True}),
                    zone=demand, rejection=rej, liquidity_sweep=True, cfg=cfg,
                )
                if h:
                    hits.append(h)

        confirmed = pivots_before(pivots, i)
        highs = [p for p in confirmed[-8:] if p.kind == "high"]
        lows = [p for p in confirmed[-8:] if p.kind == "low"]
        if len(highs) >= 3:
            h3 = highs[-3:]
            if h3[0].price < h3[1].price < h3[2].price:
                supply = find_zone_at(zones, i, high[i], low[i], "supply", cfg.zone_max_age_bars)
                rej = is_bearish_rejection(open_, high, low, close, i, h3[2].price, body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
                if rej:
                    h = _emit(
                        PatternHit(
                            "three_drive", "short", i,
                            entry_level=close[i], stop_level=h3[2].price, pivot_indices=tuple(p.index for p in h3),
                        ),
                        zone=supply, rejection=rej, structure_ok=True, cfg=cfg,
                    )
                    if h:
                        hits.append(h)
        if len(lows) >= 3:
            l3 = lows[-3:]
            if l3[0].price > l3[1].price > l3[2].price:
                demand = find_zone_at(zones, i, high[i], low[i], "demand", cfg.zone_max_age_bars)
                rej = is_bullish_rejection(open_, high, low, close, i, l3[2].price, body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
                if rej:
                    h = _emit(
                        PatternHit(
                            "three_drive", "long", i,
                            entry_level=close[i], stop_level=l3[2].price, pivot_indices=tuple(p.index for p in l3),
                        ),
                        zone=demand, rejection=rej, structure_ok=True, cfg=cfg,
                    )
                    if h:
                        hits.append(h)

        if zigzags >= cfg.cancan_min_zigzags:
            if close[i] < close[i - 1]:
                supply = find_zone_at(zones, i, high[i], low[i], "supply", cfg.zone_max_age_bars)
                rej = is_bearish_rejection(open_, high, low, close, i, zone_high, body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
                if rej:
                    h = _emit(
                        PatternHit("cancan", "short", i, entry_level=close[i], stop_level=zone_high, meta={"zigzags": zigzags}),
                        zone=supply, rejection=rej, structure_ok=True, cfg=cfg,
                    )
                    if h:
                        hits.append(h)
                if high[i] > zone_high and close[i] < zone_high and rej:
                    h = _emit(
                        PatternHit("cancan_fakeout", "short", i, entry_level=close[i], stop_level=high[i], meta={"zigzags": zigzags}),
                        zone=supply, rejection=rej, liquidity_sweep=True, cfg=cfg,
                    )
                    if h:
                        hits.append(h)
            else:
                demand = find_zone_at(zones, i, high[i], low[i], "demand", cfg.zone_max_age_bars)
                rej = is_bullish_rejection(open_, high, low, close, i, zone_low, body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
                if rej:
                    h = _emit(
                        PatternHit("cancan", "long", i, entry_level=close[i], stop_level=zone_low, meta={"zigzags": zigzags}),
                        zone=demand, rejection=rej, structure_ok=True, cfg=cfg,
                    )
                    if h:
                        hits.append(h)
                if low[i] < zone_low and close[i] > zone_low and rej:
                    h = _emit(
                        PatternHit("cancan_fakeout", "long", i, entry_level=close[i], stop_level=low[i], meta={"zigzags": zigzags}),
                        zone=demand, rejection=rej, liquidity_sweep=True, cfg=cfg,
                    )
                    if h:
                        hits.append(h)

    return hits


# ── Flag family ──────────────────────────────────────────────────────────────


def detect_flag_patterns(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    zones: Sequence[SupplyDemandZone],
    cfg: RTMConfig,
    atr_vals: np.ndarray,
) -> list[PatternHit]:
    hits: list[PatternHit] = []
    n = len(close)
    lb = cfg.compression_lookback

    for i in range(lb, n):
        atr_val = atr_vals[i]
        if atr_val is None or np.isnan(atr_val):
            continue
        box_h = float(np.max(high[i - lb : i]))
        box_l = float(np.min(low[i - lb : i]))
        if box_h - box_l > atr_val * 2.0:
            continue

        if high[i] > box_h and close[i] < box_l:
            supply = find_zone_at(zones, i, high[i], low[i], "supply", cfg.zone_max_age_bars)
            rej = is_bearish_rejection(open_, high, low, close, i, box_h, body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
            if rej:
                h = _emit(
                    PatternHit(
                        "flag_b", "short", i,
                        entry_level=close[i], stop_level=high[i], target_level=box_l, meta={"box": (box_l, box_h)},
                    ),
                    zone=supply, rejection=rej, liquidity_sweep=True, cfg=cfg,
                )
                if h:
                    hits.append(h)

        if low[i] < box_l and close[i] > box_h:
            demand = find_zone_at(zones, i, high[i], low[i], "demand", cfg.zone_max_age_bars)
            rej = is_bullish_rejection(open_, high, low, close, i, box_l, body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
            if rej:
                h = _emit(
                    PatternHit(
                        "flag_b", "long", i,
                        entry_level=close[i], stop_level=low[i], target_level=box_h, meta={"box": (box_l, box_h)},
                    ),
                    zone=demand, rejection=rej, liquidity_sweep=True, cfg=cfg,
                )
                if h:
                    hits.append(h)

        half = lb // 2
        if half < 3:
            continue
        box1_h = float(np.max(high[i - lb : i - half]))
        box1_l = float(np.min(low[i - lb : i - half]))
        box2_h = float(np.max(high[i - half : i]))
        box2_l = float(np.min(low[i - half : i]))
        nested = box2_h <= box1_h and box2_l >= box1_l
        if nested and high[i] > box2_h and close[i] < box1_l:
            supply = find_zone_at(zones, i, high[i], low[i], "supply", cfg.zone_max_age_bars)
            rej = is_bearish_rejection(open_, high, low, close, i, box2_h, body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
            if rej:
                h = _emit(
                    PatternHit(
                        "flag_ab", "short", i,
                        entry_level=close[i], stop_level=high[i],
                        meta={"outer": (box1_l, box1_h), "inner": (box2_l, box2_h)},
                    ),
                    zone=supply, rejection=rej, liquidity_sweep=True, cfg=cfg,
                )
                if h:
                    hits.append(h)
        if nested and low[i] < box2_l and close[i] > box1_h:
            demand = find_zone_at(zones, i, high[i], low[i], "demand", cfg.zone_max_age_bars)
            rej = is_bullish_rejection(open_, high, low, close, i, box2_l, body_ratio=cfg.engulfing_body_ratio, wick_ratio=cfg.rejection_wick_ratio)
            if rej:
                h = _emit(
                    PatternHit(
                        "flag_ab", "long", i,
                        entry_level=close[i], stop_level=low[i],
                        meta={"outer": (box1_l, box1_h), "inner": (box2_l, box2_h)},
                    ),
                    zone=demand, rejection=rej, liquidity_sweep=True, cfg=cfg,
                )
                if h:
                    hits.append(h)

    return hits


def detect_all_patterns(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    pivots: Sequence[Pivot],
    cfg: RTMConfig,
) -> list[PatternHit]:
    atr_vals = atr(high, low, close)
    zones = build_supply_demand_zones(open_, high, low, close, pivots, cfg)
    sr_flips = compute_sr_flips(close, pivots, atr_vals, break_atr_mult=cfg.sr_flip_break_atr)

    hits: list[PatternHit] = []
    hits.extend(detect_qm_patterns(open_, high, low, close, pivots, zones, cfg, atr_vals))
    hits.extend(detect_fakeout_patterns(open_, high, low, close, pivots, zones, sr_flips, cfg))
    hits.extend(detect_compression_patterns(open_, high, low, close, pivots, zones, cfg, atr_vals))
    hits.extend(detect_flag_patterns(open_, high, low, close, zones, cfg, atr_vals))
    return hits
