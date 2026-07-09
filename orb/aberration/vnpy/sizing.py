"""Aberration 定仓：notional = equity × position_pct × leverage。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orb.aberration.config import AberrationVnpyConfig


def round_order_volume(volume: float, price: float) -> float:
    px = max(1e-9, float(price or 1.0))
    raw = max(0.001, float(volume))
    if px >= 1000:
        step = 0.001
    elif px >= 100:
        step = 0.01
    elif px >= 10:
        step = 0.1
    else:
        step = 1.0
    return max(step, math.floor(raw / step) * step)


def fixed_size_for_aberration(
    cfg: "AberrationVnpyConfig",
    price: float,
    *,
    equity_usdt: float | None = None,
) -> float:
    px = max(1e-9, float(price or 0.0))
    eq = float(equity_usdt if equity_usdt is not None else cfg.equity_usdt)
    lev = float(cfg.live_leverage if cfg.live_enabled else cfg.leverage)
    notion = eq * float(cfg.position_pct) * lev
    cap = float(cfg.max_notional_usdt or 0.0)
    if cap > 0:
        notion = min(notion, cap)
    return round_order_volume(notion / px, px)
