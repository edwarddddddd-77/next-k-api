"""IB50 定仓（按 IB 止损距离定风险）。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quant.ib50.config import Ib50Config


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


def risk_budget_usdt(cfg: "Ib50Config", *, equity_usdt: float | None = None) -> float:
    fixed = float(cfg.risk_per_trade_usdt or 0.0)
    if fixed > 0:
        return fixed
    eq = float(equity_usdt if equity_usdt is not None else (cfg.equity_usdt or 50.0))
    return max(0.0, eq * float(cfg.risk_pct or 0.01))


def fixed_size_for_ib50(
    cfg: "Ib50Config",
    symbol: str,
    price: float,
    *,
    stop_distance: float,
    equity_usdt: float | None = None,
) -> float:
    _ = symbol
    px = max(1e-9, float(price or 0.0))
    risk_per_share = max(1e-9, float(stop_distance or 0.0))
    risk_usd = risk_budget_usdt(cfg, equity_usdt=equity_usdt)
    shares = risk_usd / risk_per_share
    notion = shares * px
    cap = float(cfg.max_notional_usdt or 0.0)
    if cap > 0:
        notion = min(notion, cap)
    vol = notion / px
    return round_order_volume(vol, px)
