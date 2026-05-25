"""动量纸面 — 开仓过滤（topMovers 事件 + vp_regime 5m，不含费率）。"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import momentum_config as cfg


def _parse_price_change(event_raw: Dict[str, Any] | None) -> Optional[float]:
    if not event_raw:
        return None
    raw = event_raw.get("priceChange")
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _event_age_minutes(event_raw: Dict[str, Any] | None, *, now_ms: int) -> Optional[float]:
    if not event_raw:
        return None
    try:
        ts = int(event_raw.get("createTimestamp") or 0)
    except (TypeError, ValueError):
        return None
    if ts <= 0:
        return None
    return max(0.0, (now_ms - ts) / 60_000.0)


def _check_event_filters(
    side: str,
    event_raw: Dict[str, Any] | None,
    *,
    now_ms: int,
) -> Tuple[bool, str]:
    if not cfg.mom_filter_enabled():
        return True, ""

    age = _event_age_minutes(event_raw, now_ms=now_ms)
    if age is None:
        return False, "filter:event:no_timestamp"
    if age > cfg.MOM_MAX_EVENT_AGE_MIN:
        return False, f"filter:event:stale:{age:.0f}m"

    pc = _parse_price_change(event_raw)
    if pc is None:
        return False, "filter:event:no_price_change"

    if side.upper() == "LONG":
        if pc < cfg.MOM_MIN_PULLBACK_PCT:
            return False, f"filter:event:pullback_small:{pc:.4f}"
        return True, ""

    if side.upper() == "SHORT":
        if pc > -cfg.MOM_MIN_RALLY_PCT:
            return False, f"filter:event:rally_small:{pc:.4f}"
        return True, ""

    return False, "filter:event:bad_side"


def _check_vp_filter(side: str, symbol: str) -> Tuple[bool, str]:
    if not cfg.mom_filter_enabled() or not cfg.MOM_VP_FILTER:
        return True, ""

    from vp_regime_scanner import analyze_symbol_vp

    res = analyze_symbol_vp(symbol, interval=cfg.MOM_VP_INTERVAL)
    if res is None:
        if cfg.MOM_VP_STRICT:
            return False, "filter:vp:no_data"
        return True, ""

    scheme = str(res.scheme or "").upper()
    allowed = (
        cfg.MOM_LONG_VP_SCHEMES if side.upper() == "LONG" else cfg.MOM_SHORT_VP_SCHEMES
    )
    if scheme not in allowed:
        return False, f"filter:vp:scheme:{scheme}"
    return True, ""


def check_open_allowed(
    *,
    side: str,
    symbol: str,
    event_raw: Dict[str, Any] | None,
    peer_symbol: Optional[str] = None,
    now_ms: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    开仓前过滤。返回 (allowed, skip_reason)。
    skip_reason 为空表示通过。
    """
    if not cfg.mom_filter_enabled():
        return True, ""

    sym = str(symbol or "").strip().upper()
    if not sym:
        return False, "filter:no_symbol"

    peer = str(peer_symbol or "").strip().upper()
    if peer and sym == peer:
        return False, "filter:same_symbol_both_legs"

    ts_ms = now_ms if now_ms is not None else int(time.time() * 1000)

    ok, reason = _check_event_filters(side, event_raw, now_ms=ts_ms)
    if not ok:
        return False, reason

    ok, reason = _check_vp_filter(side, sym)
    if not ok:
        return False, reason

    return True, ""
