"""Supertrend 持仓利润保护：浮盈回撤平仓 + ATR 跟踪止损。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import supertrend_config as cfg


@dataclass
class ProtectState:
    mfe_price: float
    peak_pnl_pct: float
    trail_armed: bool
    trail_stop: Optional[float] = None


def parse_meta(raw: Any) -> Dict[str, Any]:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    try:
        return json.loads(str(raw))
    except (json.JSONDecodeError, TypeError):
        return {}


def _protect_from_meta(meta: Dict[str, Any], *, entry: float) -> ProtectState:
    prot = meta.get("protect") if isinstance(meta.get("protect"), dict) else {}
    mfe = float(prot.get("mfe_price") or entry or 0)
    peak = float(prot.get("peak_pnl_pct") or 0)
    armed = bool(prot.get("trail_armed"))
    ts = prot.get("trail_stop")
    trail_stop = float(ts) if ts is not None else None
    return ProtectState(
        mfe_price=mfe,
        peak_pnl_pct=peak,
        trail_armed=armed,
        trail_stop=trail_stop,
    )


def pnl_pct(side: str, entry: float, price: float) -> float:
    if entry <= 0:
        return 0.0
    if side == "LONG":
        return (price - entry) / entry
    return (entry - price) / entry


def update_protect_state(
    state: ProtectState,
    *,
    side: str,
    entry: float,
    high: float,
    low: float,
    close: float,
    atr: float,
) -> ProtectState:
    """用本根 K 线更新 MFE（影线）、峰值浮盈（默认收盘）、跟踪止损线。"""
    if entry <= 0:
        return state

    if side == "LONG":
        mfe = max(state.mfe_price, high)
    else:
        mfe = min(state.mfe_price, low) if state.mfe_price > 0 else low

    peak = max(state.peak_pnl_pct, pnl_pct(side, entry, close))
    if not cfg.ST_GIVEBACK_PEAK_USE_CLOSE:
        peak = max(peak, pnl_pct(side, entry, mfe))

    trail_armed = state.trail_armed
    trail_stop = state.trail_stop

    if atr > 0:
        fav_move = (mfe - entry) if side == "LONG" else (entry - mfe)
        if not trail_armed and cfg.ST_TRAIL_ARM_ATR > 0 and fav_move >= cfg.ST_TRAIL_ARM_ATR * atr:
            trail_armed = True

        if trail_armed and "trail_atr" in cfg.st_exit_modes_enabled():
            mult = cfg.ST_TRAIL_ATR_MULT
            if side == "LONG":
                trail_stop = mfe - mult * atr
            else:
                trail_stop = mfe + mult * atr

    return ProtectState(
        mfe_price=mfe,
        peak_pnl_pct=peak,
        trail_armed=trail_armed,
        trail_stop=trail_stop,
    )


def protect_state_to_meta(meta: Dict[str, Any], state: ProtectState) -> Dict[str, Any]:
    out = dict(meta)
    out["protect"] = {
        "mfe_price": state.mfe_price,
        "peak_pnl_pct": state.peak_pnl_pct,
        "trail_armed": state.trail_armed,
        "trail_stop": state.trail_stop,
    }
    return out


def evaluate_profit_exit(
    state: ProtectState,
    *,
    side: str,
    entry: float,
    high: float,
    low: float,
    close: float,
    atr: float,
) -> Optional[str]:
    """
    检查是否触发利润保护平仓。顺序：giveback → trail_atr。
    跟踪止损用影线触发（多：low，空：high）。
    """
    modes = set(cfg.st_exit_modes_enabled())
    cur_pct = pnl_pct(side, entry, close)

    if "giveback" in modes and cfg.ST_GIVEBACK_PCT > 0:
        min_peak = cfg.ST_GIVEBACK_MIN_PEAK_PCT
        if state.peak_pnl_pct >= min_peak > 0:
            floor = state.peak_pnl_pct * (1.0 - cfg.ST_GIVEBACK_PCT)
            if cur_pct <= floor:
                if not cfg.ST_GIVEBACK_REQUIRE_POSITIVE_PCT or cur_pct > 0:
                    return "giveback"

    if "trail_atr" in modes and state.trail_armed and state.trail_stop is not None:
        ts = state.trail_stop
        if side == "LONG" and low <= ts:
            return "trail_atr"
        if side == "SHORT" and high >= ts:
            return "trail_atr"

    return None


def profit_exit_fill_price(
    exit_rule: str,
    *,
    side: str,
    close: float,
    high: float,
    low: float,
    state: ProtectState,
) -> float:
    """纸面成交价：跟踪止损触及线价，其余用收盘。"""
    if exit_rule != "trail_atr" or state.trail_stop is None:
        return close
    ts = state.trail_stop
    if side == "LONG" and low <= ts:
        return float(ts)
    if side == "SHORT" and high >= ts:
        return float(ts)
    return close


def run_profit_protection(
    row: Any,
    *,
    high: float,
    low: float,
    close: float,
    atr: float,
) -> Tuple[Optional[str], Dict[str, Any], float]:
    """
    从持仓行更新保护状态并判断是否平仓。
    返回 (exit_rule, updated_meta, fill_price)。
    """
    meta = parse_meta(row["meta_json"] if row else None)
    side = str(row["side"])
    entry = float(row["entry_price"] or 0)
    state = _protect_from_meta(meta, entry=entry)
    state = update_protect_state(
        state,
        side=side,
        entry=entry,
        high=high,
        low=low,
        close=close,
        atr=atr,
    )
    meta = protect_state_to_meta(meta, state)
    exit_rule = evaluate_profit_exit(
        state,
        side=side,
        entry=entry,
        high=high,
        low=low,
        close=close,
        atr=atr,
    )
    fill = profit_exit_fill_price(
        exit_rule or "",
        side=side,
        close=close,
        high=high,
        low=low,
        state=state,
    ) if exit_rule else close
    return exit_rule, meta, fill
