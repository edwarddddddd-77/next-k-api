"""ORB 纸面信号 → Next-k-protocol 实盘执行（开/平/止损）。"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, Optional

from orb.core.config import OrbConfig
from orb.core.kline_cache import norm_symbol
from orb.core.protocol_client import (
    LIVE_PENDING_NOTE,
    SOURCE_ORB,
    ingest_signals,
    lookup_signal,
    protocol_configured,
    reconcile_pending_entries,
)
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


def _close_signal_id(symbol: str, *, signal_id: int, tag: str) -> str:
    sym = norm_symbol(symbol)
    return f"orb:close:{sym}:{int(signal_id)}:{str(tag or 'resolve').strip().lower()}"


def _live_entry_type(cfg: OrbConfig) -> str:
    raw = str(getattr(cfg, "live_entry_type", "") or "stoplimit_gap").strip().lower()
    if raw in ("stoplimit_gap", "stoplimit", "stop_limit", "stop-limit"):
        return "STOP_LIMIT"
    if raw in ("market", ""):
        return "MARKET"
    return raw.upper()


def ingest_detail_action(result: Optional[Dict[str, Any]]) -> str:
    if not isinstance(result, dict):
        return ""
    for detail in result.get("details") or []:
        action = str(detail.get("action") or "").lower()
        if action:
            return action
    return ""


def live_open_is_pending(result: Optional[Dict[str, Any]]) -> bool:
    return ingest_detail_action(result) == "submitted"


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
        "entry_type": _live_entry_type(cfg),
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
    signal_id: Optional[int] = None,
) -> Dict[str, Any]:
    sym = norm_symbol(symbol)
    side_u = str(side).upper()
    sid = int(signal_id or 0)
    tag_s = str(tag or "resolve").strip().lower()
    api_id = _close_signal_id(sym, signal_id=sid, tag=tag_s) if sid > 0 else f"orb:close:{sym}:{tag_s}"
    payload: Dict[str, Any] = {
        "source": SOURCE_ORB,
        "api_signal_id": api_id,
        "symbol": sym,
        "side": side_u,
        "action": "close",
        "play": play or "ORB",
    }
    if close_price is not None and close_price > 0 and tag_s != "session_close":
        payload["close_price"] = float(close_price)
    return payload


def live_ingest_succeeded(result: Optional[Dict[str, Any]]) -> bool:
    """Return True when protocol ingest traded/submitted the signal (or live was not attempted)."""
    if result is None:
        return True
    action = ingest_detail_action(result)
    if action == "duplicate":
        return False
    if result.get("skipped") is True:
        return True
    if result.get("error"):
        return False
    if int(result.get("errors") or 0) > 0:
        return False
    if int(result.get("traded") or 0) >= 1:
        return True
    if action in ("traded", "submitted"):
        return True
    for detail in result.get("details") or []:
        act = str(detail.get("action") or "").lower()
        if act in ("traded", "submitted"):
            return True
        if act == "error":
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
    signal_id: Optional[int] = None,
) -> Dict[str, Any]:
    if not live_enabled(cfg):
        return {"skipped": True, "reason": "live_disabled"}
    payload = build_close_payload(
        symbol,
        side,
        close_price=close_price,
        play=play,
        tag=tag,
        signal_id=signal_id,
    )
    return ingest_signals([payload])


def sync_live_pending_entries(conn: sqlite3.Connection, cfg: OrbConfig) -> int:
    """对账 pending STOP 入场：成交清标记，取消则回滚纸面持仓。"""
    if not live_enabled(cfg):
        return 0
    try:
        reconcile_pending_entries()
    except Exception as exc:
        logger.warning("[orb] protocol reconcile failed: %s", exc)

    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, symbol, session_date, entry_bar_open_ms
        FROM orb_signals
        WHERE outcome IS NULL AND COALESCE(notes, '') = ?
          AND side IN ('LONG', 'SHORT') AND sl_price IS NOT NULL
        """,
        (LIVE_PENDING_NOTE,),
    )
    rows = cur.fetchall()
    changed = 0
    for sid, sym, session_date, entry_bar in rows:
        api_id = f"orb:open:{str(sym).strip().upper()}:{session_date or ''}:{int(entry_bar or 0)}"
        try:
            proto = lookup_signal(source=SOURCE_ORB, api_signal_id=api_id)
        except Exception as exc:
            logger.warning("[orb] protocol lookup %s failed: %s", api_id, exc)
            continue
        if not proto:
            continue
        status = str(proto.get("status") or "").lower()
        if status == "traded":
            cur.execute("UPDATE orb_signals SET notes=NULL WHERE id=?", (int(sid),))
            changed += 1
        elif status in ("cancelled", "error"):
            cur.execute(
                """
                DELETE FROM orb_signals
                WHERE id=? AND outcome IS NULL AND COALESCE(notes, '') = ?
                """,
                (int(sid), LIVE_PENDING_NOTE),
            )
            changed += 1
            logger.info("[orb] rolled back pending paper open id=%s symbol=%s status=%s", sid, sym, status)
    if changed:
        conn.commit()
    return changed
