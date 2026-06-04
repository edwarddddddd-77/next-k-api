"""Moss2 Protocol 信号发送。"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from moss2.protocol_client import Moss2ProtocolClient

logger = logging.getLogger(__name__)


def _client() -> Moss2ProtocolClient:
    return Moss2ProtocolClient.from_env()


def is_real_mode() -> bool:
    from moss2.config import real_mode_enabled

    return real_mode_enabled() and _client().enabled()


def send_open(
    *,
    symbol: str,
    side: str,
    entry_price: float,
    margin_usdt: float,
    leverage: float,
    profile_id: int,
    params_version: str = "v1",
    sl_price: Optional[float] = None,
    tp_price: Optional[float] = None,
    composite: float = 0.0,
    regime: str = "",
) -> Dict[str, Any]:
    if not is_real_mode():
        return {"ok": False, "error": "moss2_real_mode_disabled"}
    try:
        return _client().send_open(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            margin_usdt=margin_usdt,
            leverage=leverage,
            profile_id=profile_id,
            params_version=params_version,
            sl_price=sl_price,
            tp_price=tp_price,
            composite=composite,
            regime=regime,
        )
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500] if exc.response else ""
        logger.error("[moss2] send_open HTTP: %s", detail)
        return {"ok": False, "error": detail or str(exc)}
    except Exception as exc:
        logger.error("[moss2] send_open failed: %s", exc)
        return {"ok": False, "error": str(exc)}


def send_close(
    *,
    symbol: str,
    side: str,
    exit_rule: str,
    close_price: float,
    profile_id: int,
    params_version: str = "v1",
) -> Dict[str, Any]:
    if not is_real_mode():
        return {"ok": False, "error": "moss2_real_mode_disabled"}
    try:
        return _client().send_close(
            symbol=symbol,
            side=side,
            exit_rule=exit_rule,
            close_price=close_price,
            profile_id=profile_id,
            params_version=params_version,
        )
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500] if exc.response else ""
        logger.error("[moss2] send_close HTTP: %s", detail)
        return {"ok": False, "error": detail or str(exc)}
    except Exception as exc:
        logger.error("[moss2] send_close failed: %s", exc)
        return {"ok": False, "error": str(exc)}
