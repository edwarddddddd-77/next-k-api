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


def filter_reason_label(reason: str) -> str:
    """将 skip_reason 转为可读中文（用于日志）。"""
    if not reason:
        return "通过"
    labels = {
        "filter:no_symbol": "无标的",
        "filter:same_symbol_both_legs": "多空同标的",
        "filter:event:no_timestamp": "事件无时间戳",
        "filter:event:no_price_change": "事件无 priceChange",
        "filter:event:bad_side": "方向无效",
        "filter:vp:no_data": "VP 无数据",
    }
    if reason in labels:
        return labels[reason]
    if reason.startswith("filter:event:stale:"):
        return f"事件过旧({reason.split(':')[-1]})"
    if reason.startswith("filter:event:pullback_small:"):
        return f"回调幅度不足({reason.split(':')[-1]})"
    if reason.startswith("filter:event:rally_small:"):
        return f"反弹幅度不足({reason.split(':')[-1]})"
    if reason.startswith("filter:vp:scheme:"):
        return f"VP 方案不允许({reason.split(':')[-1]})"
    return reason


def _check_event_filters(
    side: str,
    event_raw: Dict[str, Any] | None,
    *,
    now_ms: int,
) -> Tuple[bool, str]:
    if not cfg.mom_filter_enabled():
        return True, ""

    if cfg.MOM_EVENT_AGE_FILTER:
        age = _event_age_minutes(event_raw, now_ms=now_ms)
        if age is None:
            return False, "filter:event:no_timestamp"
        if age > cfg.MOM_MAX_EVENT_AGE_MIN:
            return False, f"filter:event:stale:{age:.0f}m"

    if not cfg.MOM_PRICE_CHANGE_FILTER:
        return True, ""

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


def _check_vp_filter(side: str, symbol: str) -> Tuple[bool, str, Optional[str]]:
    """返回 (ok, reason, vp_scheme)。"""
    if not cfg.mom_filter_enabled() or not cfg.MOM_VP_FILTER:
        return True, "", None

    from vp_regime_scanner import analyze_symbol_vp

    res = analyze_symbol_vp(symbol, interval=cfg.MOM_VP_INTERVAL)
    if res is None:
        if cfg.MOM_VP_STRICT:
            return False, "filter:vp:no_data", None
        return True, "", None

    scheme = str(res.scheme or "").upper()
    allowed = (
        cfg.MOM_LONG_VP_SCHEMES if side.upper() == "LONG" else cfg.MOM_SHORT_VP_SCHEMES
    )
    if scheme not in allowed:
        return False, f"filter:vp:scheme:{scheme}", scheme
    return True, "", scheme


def inspect_open_filter(
    *,
    side: str,
    symbol: str,
    event_raw: Dict[str, Any] | None,
    peer_symbol: Optional[str] = None,
    now_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """开仓过滤检查明细（供日志 / mom_runs.detail_json）。"""
    sym = str(symbol or "").strip().upper()
    peer = str(peer_symbol or "").strip().upper() or None
    ts_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    age = _event_age_minutes(event_raw, now_ms=ts_ms)
    pc = _parse_price_change(event_raw)
    vp_allowed = (
        sorted(cfg.MOM_LONG_VP_SCHEMES)
        if side.upper() == "LONG"
        else sorted(cfg.MOM_SHORT_VP_SCHEMES)
    )
    out: Dict[str, Any] = {
        "side": side.upper(),
        "symbol": sym,
        "peer_symbol": peer,
        "filter_enabled": cfg.mom_filter_enabled(),
        "vp_filter": cfg.MOM_VP_FILTER if cfg.mom_filter_enabled() else False,
        "vp_interval": cfg.MOM_VP_INTERVAL,
        "vp_allowed_schemes": vp_allowed,
        "event_type": (event_raw or {}).get("eventType"),
        "price_change": pc,
        "event_age_min": round(age, 1) if age is not None else None,
        "price_change_filter": cfg.MOM_PRICE_CHANGE_FILTER,
        "event_age_filter": cfg.MOM_EVENT_AGE_FILTER,
        "min_pullback_pct": cfg.MOM_MIN_PULLBACK_PCT,
        "min_rally_pct": cfg.MOM_MIN_RALLY_PCT,
        "max_event_age_min": cfg.MOM_MAX_EVENT_AGE_MIN,
        "vp_scheme": None,
        "allowed": True,
        "reason": "",
        "reason_cn": "通过",
    }
    if not cfg.mom_filter_enabled():
        return out

    if not sym:
        out.update(allowed=False, reason="filter:no_symbol", reason_cn=filter_reason_label("filter:no_symbol"))
        return out
    if peer and sym == peer:
        out.update(
            allowed=False,
            reason="filter:same_symbol_both_legs",
            reason_cn=filter_reason_label("filter:same_symbol_both_legs"),
        )
        return out

    ok, reason = _check_event_filters(side, event_raw, now_ms=ts_ms)
    if not ok:
        out.update(allowed=False, reason=reason, reason_cn=filter_reason_label(reason))
        return out

    ok, reason, scheme = _check_vp_filter(side, sym)
    out["vp_scheme"] = scheme
    if not ok:
        out.update(allowed=False, reason=reason, reason_cn=filter_reason_label(reason))
        return out
    if scheme:
        out["vp_scheme"] = scheme
    return out


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
    detail = inspect_open_filter(
        side=side,
        symbol=symbol,
        event_raw=event_raw,
        peer_symbol=peer_symbol,
        now_ms=now_ms,
    )
    if detail["allowed"]:
        return True, ""
    return False, str(detail["reason"])
