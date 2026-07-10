"""Smart Breakout 定仓。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant.squeeze_breakout.config import SqueezeBreakoutConfig


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


def risk_budget_usdt(cfg: "SqueezeBreakoutConfig", *, equity_usdt: float | None = None) -> float:
    eq = float(equity_usdt if equity_usdt is not None else (cfg.equity_usdt or 100.0))
    return max(0.0, eq * float(cfg.risk_pct or 0.02))


def size_for_breakout(
    cfg: "SqueezeBreakoutConfig",
    price: float,
    *,
    stop_distance: float,
    equity_usdt: float | None = None,
) -> float:
    px = max(1e-9, float(price or 0.0))
    risk_per_unit = max(1e-9, float(stop_distance or 0.0))
    risk_usd = risk_budget_usdt(cfg, equity_usdt=equity_usdt)
    vol = risk_usd / risk_per_unit
    notion = vol * px
    cap = float(cfg.max_notional_usdt or 0.0)
    if cap > 0:
        notion = min(notion, cap)
    eq = float(equity_usdt if equity_usdt is not None else (cfg.equity_usdt or 100.0))
    notion = min(notion, eq)
    return round_order_volume(notion / px, px)
