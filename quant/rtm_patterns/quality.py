"""Pattern quality scoring — filters weak signals toward ~80% cheat-sheet fidelity."""

from __future__ import annotations

from quant.rtm_patterns.types import PatternHit, SupplyDemandZone


def score_hit(
    *,
    zone: SupplyDemandZone | None,
    rejection: bool,
    structure_ok: bool,
    liquidity_sweep: bool,
    trend_aligned: bool = False,
) -> float:
    score = 0.0
    if zone is not None:
        score += 0.30
    if rejection:
        score += 0.30
    if structure_ok:
        score += 0.25
    if liquidity_sweep:
        score += 0.10
    if trend_aligned:
        score += 0.05
    return min(score, 1.0)


def attach_quality(hit: PatternHit, quality: float, **extra) -> PatternHit:
    meta = dict(hit.meta)
    meta["quality"] = round(quality, 3)
    meta.update(extra)
    return PatternHit(
        pattern=hit.pattern,
        direction=hit.direction,
        bar_index=hit.bar_index,
        qml_level=hit.qml_level,
        entry_level=hit.entry_level,
        stop_level=hit.stop_level,
        target_level=hit.target_level,
        pivot_indices=hit.pivot_indices,
        meta=meta,
    )


def passes_quality(hit: PatternHit, min_score: float) -> bool:
    return float(hit.meta.get("quality", 0.0)) >= min_score
