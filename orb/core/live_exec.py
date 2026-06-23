"""ORB 纸面信号 → Next-k-protocol 实盘执行（开/平/止损）。"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from orb.core.config import OrbConfig
from orb.core.protocol_client import SOURCE_ORB, ingest_signals, protocol_configured
from orb.core.signals import OrbSignal

logger = logging.getLogger(__name__)


def live_enabled(cfg: OrbConfig) -> bool:
    return bool(getattr(cfg, "live_enabled", False)) and protocol_configured()


def _leverage(cfg: OrbConfig) -> float:
    lev = float(getattr(cfg, "live_leverage", 0.0) or 0.0)
    if lev > 0:
        return lev
    return max(1.0, float(cfg.leverage or 1.0))


def _margin_from_notional(notional_usdt: float, cfg: OrbConfig) -> float:
    lev = _leverage(cfg)
    n = max(0.0, float(notional_usdt or 0.0))
    if n > 0 and lev > 0:
        return n / lev
    return max(0.0, float(cfg.margin_usdt or 0.0))


def _open_signal_id(sig: OrbSignal) -> str:
    bar = int(sig.entry_bar_open_ms or 0)
    sess = (sig.session_date or "").strip()
    return f"orb:open:{sig.symbol}:{sess}:{bar}"


def _close_signal_id(symbol: str, *, tag: str) -> str:
    sym = str(symbol).strip().upper()
    return f"orb:close:{sym}:{tag}:{int(time.time() * 1000)}"


def build_open_payload(sig: OrbSignal, cfg: OrbConfig) -> Dict[str, Any]:
    notional = float(sig.paper_notional_usdt or cfg.default_paper_notional())
    lev = _leverage(cfg)
    margin = _margin_from_notional(notional, cfg)
    if notional > 0 and lev > 0:
        implied = round(margin * lev, 4)
        if abs(implied - notional) > max(1.0, notional * 0.001):
            logger.warning(
                "[orb] live open margin×lev drift: %s notional=%.4f implied=%.4f lev=%s",
                sig.symbol,
                notional,
                implied,
                lev,
            )
    payload: Dict[str, Any] = {
        "source": SOURCE_ORB,
        "api_signal_id": _open_signal_id(sig),
        "symbol": str(sig.symbol).strip().upper(),
        "side": str(sig.side).upper(),
        "margin_usdt": round(margin, 4),
        "leverage": lev,
        "entry_price": float(sig.price) if sig.price else None,
        "sl_price": float(sig.sl_price) if sig.sl_price is not None else None,
        "tp_price": float(sig.tp_price) if sig.tp_price is not None else None,
        "play": sig.play or "ORB",
        "confidence": sig.confidence or "high",
        "action": "open",
        "client_ref": _open_signal_id(sig),
    }
    return payload


def build_close_payload(
    symbol: str,
    side: str,
    *,
    close_price: Optional[float] = None,
    play: Optional[str] = None,
    tag: str = "resolve",
) -> Dict[str, Any]:
    sym = str(symbol).strip().upper()
    side_u = str(side).upper()
    payload: Dict[str, Any] = {
        "source": SOURCE_ORB,
        "api_signal_id": _close_signal_id(sym, tag=tag),
        "symbol": sym,
        "side": side_u,
        "action": "close",
        "play": play or "ORB",
    }
    if close_price is not None and close_price > 0 and str(tag) != "session_close":
        payload["close_price"] = float(close_price)
    return payload


def live_ingest_succeeded(result: Optional[Dict[str, Any]]) -> bool:
    """Return True when protocol ingest traded the signal (or live was not attempted)."""
    if result is None:
        return True
    if result.get("skipped"):
        return True
    if result.get("error"):
        return False
    if int(result.get("errors") or 0) > 0:
        return False
    if int(result.get("traded") or 0) >= 1:
        return True
    for detail in result.get("details") or []:
        action = str(detail.get("action") or "").lower()
        if action == "traded":
            return True
        if action == "error":
            return False
    return False


def notify_open(sig: OrbSignal, cfg: OrbConfig) -> Dict[str, Any]:
    if not live_enabled(cfg):
        return {"skipped": True, "reason": "live_disabled"}
    if str(sig.side).upper() not in ("LONG", "SHORT"):
        return {"skipped": True, "reason": "not_actionable"}
    payload = build_open_payload(sig, cfg)
    return ingest_signals([payload])


def notify_close(
    symbol: str,
    side: str,
    cfg: OrbConfig,
    *,
    close_price: Optional[float] = None,
    play: Optional[str] = None,
    tag: str = "resolve",
) -> Dict[str, Any]:
    if not live_enabled(cfg):
        return {"skipped": True, "reason": "live_disabled"}
    payload = build_close_payload(
        symbol, side, close_price=close_price, play=play, tag=tag
    )
    return ingest_signals([payload])
