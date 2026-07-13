"""Anchor Drift 定仓。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant.anchor_drift.config import AnchorDriftConfig


def round_order_volume(volume: float, price: float) -> float:
    px = max(1e-9, float(price or 1.0))
    raw = max(0.001, float(volume))
    if px >= 1000:
        step = 0.001
    elif px >= 100:
        step = 0.01
    else:
        step = 0.1
    return max(step, math.floor(raw / step) * step)


def risk_budget_usdt(cfg: "AnchorDriftConfig", *, equity_usdt: float | None = None) -> float:
    eq = float(equity_usdt if equity_usdt is not None else (cfg.equity_usdt or 14.0))
    return max(0.0, eq * float(cfg.risk_pct or 0.01))


def size_for_drift(
    cfg: "AnchorDriftConfig",
    price: float,
    *,
    anchor_price: float,
    equity_usdt: float | None = None,
) -> float:
    px = max(1e-9, float(price or 0.0))
    anchor = max(1e-9, float(anchor_price or px))
    stop_dist = max(1e-9, anchor * float(cfg.signal_threshold or 0.015))
    risk_usd = risk_budget_usdt(cfg, equity_usdt=equity_usdt)
    qty = risk_usd / stop_dist
    notion = qty * px
    cap = float(cfg.max_notional_usdt or 0.0)
    if cap > 0:
        notion = min(notion, cap)
    eq = float(equity_usdt if equity_usdt is not None else (cfg.equity_usdt or 14.0))
    notion = min(notion, eq)
    return round_order_volume(notion / px, px)
