"""RTM pattern data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

PatternDirection = Literal["long", "short"]
PivotKind = Literal["high", "low"]
ZoneKind = Literal["supply", "demand"]


@dataclass(frozen=True)
class Pivot:
    index: int
    price: float
    kind: PivotKind


@dataclass(frozen=True)
class SupplyDemandZone:
    """Red/green box from cheat sheet — institutional reaction zone."""

    kind: ZoneKind
    top: float
    bottom: float
    formed_bar: int
    source: str


@dataclass(frozen=True)
class SRFlipLevel:
    """Support/resistance that broke and flipped role."""

    level: float
    original: Literal["support", "resistance"]
    pivot_index: int
    broken_bar: int


@dataclass(frozen=True)
class PatternHit:
    """A detected RTM pattern occurrence."""

    pattern: str
    direction: PatternDirection
    bar_index: int
    qml_level: float | None = None
    entry_level: float | None = None
    stop_level: float | None = None
    target_level: float | None = None
    pivot_indices: tuple[int, ...] = ()
    meta: dict[str, Any] = field(default_factory=dict)
