"""Anchor Drift 信号（BQuant drift_engine 规则）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DriftSignal = Literal["LONG", "SHORT", "FLAT", "CONVERGED"]


@dataclass(frozen=True)
class DriftSignalResult:
    signal: DriftSignal
    confidence: int
    reasoning: str


def calculate_drift(*, anchor_price: float, current_price: float) -> float | None:
    anchor = float(anchor_price or 0.0)
    current = float(current_price or 0.0)
    if anchor <= 0 or current <= 0:
        return None
    return (current - anchor) / anchor


def generate_signal(
    drift_pct: float,
    *,
    signal_threshold: float = 0.015,
    converge_threshold: float = 0.003,
) -> DriftSignalResult:
    """|drift| > signal_threshold → 反向；|drift| < converge_threshold → 收敛平仓。"""
    abs_drift = abs(float(drift_pct))
    pct_display = float(drift_pct) * 100.0
    sig = max(0.0, float(signal_threshold))
    conv = max(0.0, float(converge_threshold))

    if abs_drift < conv:
        return DriftSignalResult(
            signal="CONVERGED",
            confidence=90,
            reasoning=(
                f"Drift {pct_display:+.2f}% below convergence threshold ({conv * 100:.1f}%); exit."
            ),
        )

    if abs_drift > sig:
        direction: DriftSignal = "SHORT" if drift_pct > 0 else "LONG"
        excess = (abs_drift - sig) / max(sig * 2.0, 1e-9)
        confidence = round(60 + 25 * min(excess, 1.0))
        return DriftSignalResult(
            signal=direction,
            confidence=int(confidence),
            reasoning=(
                f"Drift {pct_display:+.2f}% exceeds ±{sig * 100:.1f}% threshold; "
                f"mean-reversion signal {direction}."
            ),
        )

    return DriftSignalResult(
        signal="FLAT",
        confidence=50,
        reasoning=(
            f"Drift {pct_display:+.2f}% between convergence ({conv * 100:.1f}%) "
            f"and signal ({sig * 100:.1f}%) thresholds."
        ),
    )


def adverse_drift_stop(
    drift_pct: float,
    *,
    side: int,
    signal_threshold: float,
    max_adverse_extension: float,
) -> bool:
    """漂移相对入场方向继续扩大超过 extension 则止损。"""
    if side == 0:
        return False
    limit = float(signal_threshold) + max(0.0, float(max_adverse_extension))
    if side < 0:
        return float(drift_pct) > limit
    return float(drift_pct) < -limit
