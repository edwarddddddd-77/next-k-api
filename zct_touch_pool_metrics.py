#!/usr/bin/env python3
"""触轨池筛选指标：扣摩擦后的 profit factor、周期末连续亏损等。"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import zct_vwap_signal_scanner as z

# 摩擦（与触轨池主筛 PF 一致）：
# - Taker 万分之四 = 4 bps / 边
# - 滑点万 1.5 = 1.5 bps / 边
# 往返成本 = notional × (taker+slip)_bps × 2 / 10000
_DEFAULT_TAKER_BPS = 4.0
_DEFAULT_SLIP_BPS = 1.5


def taker_bps_per_side() -> float:
    try:
        return max(0.0, float(os.getenv("ZCT_TOUCH_POOL_TAKER_BPS", str(_DEFAULT_TAKER_BPS)).strip()))
    except ValueError:
        return _DEFAULT_TAKER_BPS


def slippage_bps_per_side() -> float:
    try:
        return max(0.0, float(os.getenv("ZCT_TOUCH_POOL_SLIPPAGE_BPS", str(_DEFAULT_SLIP_BPS)).strip()))
    except ValueError:
        return _DEFAULT_SLIP_BPS


def friction_bps_per_side() -> float:
    return taker_bps_per_side() + slippage_bps_per_side()


def round_trip_friction_usdt(notional_usdt: float) -> float:
    """双边 (开仓+平仓) × (Taker + 滑点) bps。"""
    bps = friction_bps_per_side()
    return float(notional_usdt) * 2.0 * (bps / 10_000.0)


def net_pnl_after_friction(
    side: str,
    entry: float,
    exit_px: float,
    notional_usdt: float,
) -> float:
    raw = z._pnl_usdt(side, float(entry), float(exit_px), float(notional_usdt))
    return raw - round_trip_friction_usdt(notional_usdt)


def trailing_consecutive_losses_at_end(
    ordered_touch_rows: List[Dict[str, Any]],
) -> int:
    """按时间升序的 win/loss 序列，返回周期末连续亏损笔数。"""
    streak = 0
    for r in ordered_touch_rows:
        oc = str(r.get("outcome") or "").lower()
        if oc == "loss":
            streak += 1
        elif oc == "win":
            streak = 0
    return streak


def symbol_touch_metrics(
    trades: List[Dict[str, Any]],
    symbol: str,
    *,
    default_notional: float,
) -> Dict[str, Any]:
    su = str(symbol).strip().upper()
    touch_rows = [
        r
        for r in trades
        if str(r.get("symbol", "")).strip().upper() == su
        and r.get("outcome") in ("win", "loss")
        and r.get("exit_price") is not None
    ]
    touch_rows.sort(key=lambda x: int(x.get("signal_open_ms") or 0))

    gross_profit = 0.0
    gross_loss = 0.0
    for r in touch_rows:
        notion = float(r.get("notional_usdt") or default_notional)
        net = net_pnl_after_friction(
            str(r.get("side") or "LONG"),
            float(r["entry"]),
            float(r["exit_price"]),
            notion,
        )
        if net > 0:
            gross_profit += net
        elif net < 0:
            gross_loss += abs(net)

    if gross_loss > 1e-12:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    return {
        "profit_factor_net": round(profit_factor, 6)
        if profit_factor != float("inf")
        else None,
        "profit_factor_net_display": round(profit_factor, 4)
        if profit_factor != float("inf")
        else "inf",
        "gross_profit_net_usdt": round(gross_profit, 4),
        "gross_loss_net_usdt": round(gross_loss, 4),
        "consecutive_losses_at_end": trailing_consecutive_losses_at_end(touch_rows),
        "taker_bps_per_side": taker_bps_per_side(),
        "slippage_bps_per_side": slippage_bps_per_side(),
        "friction_bps_per_side": friction_bps_per_side(),
        "friction_round_trip_bps": friction_bps_per_side() * 2.0,
    }


def enrich_per_symbol_stats(
    per_symbol: Dict[str, Dict[str, Any]],
    trades: List[Dict[str, Any]],
    *,
    default_notional: float,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for sym, row in per_symbol.items():
        merged = dict(row)
        merged.update(
            symbol_touch_metrics(trades, sym, default_notional=default_notional)
        )
        out[sym] = merged
    return out
