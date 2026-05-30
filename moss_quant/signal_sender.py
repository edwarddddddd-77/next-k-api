"""Moss Quant 信号发送器：将纸面信号推送到 Next-k-protocol 实盘执行。"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from moss_quant.protocol_client import ProtocolClient

logger = logging.getLogger(__name__)

SOURCE = "moss_quant"


def _client(timeout: float = 30.0) -> ProtocolClient:
    client = ProtocolClient.from_env()
    client.timeout = timeout
    return client


def _proto_url() -> str:
    return _client().base_url


def _proto_token() -> str:
    return _client().token


def _headers() -> Dict[str, str]:
    return _client().headers()


def _post(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    client = _client()
    url = f"{client.base_url}{path}"
    try:
        return client._post(path, body)
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500] if exc.response else ""
        logger.error("[moss_quant] POST %s HTTP %s: %s", url, exc.response.status_code, detail)
        return {"ok": False, "error": detail or str(exc)}
    except Exception as exc:
        logger.error("[moss_quant] POST %s failed: %s", url, exc)
        return {"ok": False, "error": str(exc)}


def _put(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    client = _client()
    url = f"{client.base_url}{path}"
    try:
        return client._put(path, body)
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500] if exc.response else ""
        logger.error("[moss_quant] PUT %s HTTP %s: %s", url, exc.response.status_code, detail)
        return {"ok": False, "error": detail or str(exc)}
    except Exception as exc:
        logger.error("[moss_quant] PUT %s failed: %s", url, exc)
        return {"ok": False, "error": str(exc)}


def is_real_mode() -> bool:
    from moss_quant.config import MOSS_QUANT_REAL_MODE
    return MOSS_QUANT_REAL_MODE and bool(_proto_url())


def send_open(
    *,
    symbol: str,
    side: str,
    entry_price: float,
    sl_price: float,
    tp_price: Optional[float],
    margin_usdt: float,
    leverage: float,
    profile_id: int,
    play: str = "",
    composite: float = 0.0,
    regime: str = "",
) -> Dict[str, Any]:
    """发送开仓信号 → POST /api/binance/signals/ingest"""
    if not is_real_mode():
        logger.debug("[moss_quant] real mode disabled, skip send_open")
        return {"ok": False, "error": "real_mode_disabled"}

    logger.info("[moss_quant] send_open: symbol=%s side=%s margin=%.2f lev=%.2f sl=%.4f tp=%s",
                symbol, side, margin_usdt, leverage, sl_price, tp_price)
    try:
        return _client().send_open(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            margin_usdt=margin_usdt,
            leverage=leverage,
            profile_id=profile_id,
            play=play,
            composite=composite,
            regime=regime,
        )
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500] if exc.response else ""
        logger.error("[moss_quant] send_open HTTP %s: %s", exc.response.status_code, detail)
        return {"ok": False, "error": detail or str(exc)}
    except Exception as exc:
        logger.error("[moss_quant] send_open failed: %s", exc)
        return {"ok": False, "error": str(exc)}


def send_close(
    *,
    symbol: str,
    side: str,
    exit_rule: str,
    close_price: float,
    profile_id: int,
) -> Dict[str, Any]:
    """发送平仓信号 → POST /api/binance/signals/ingest"""
    if not is_real_mode():
        logger.debug("[moss_quant] real mode disabled, skip send_close")
        return {"ok": False, "error": "real_mode_disabled"}

    logger.info("[moss_quant] send_close: symbol=%s side=%s rule=%s price=%.4f",
                symbol, side, exit_rule, close_price)
    try:
        return _client().send_close(
            symbol=symbol,
            side=side,
            exit_rule=exit_rule,
            close_price=close_price,
            profile_id=profile_id,
        )
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500] if exc.response else ""
        logger.error("[moss_quant] send_close HTTP %s: %s", exc.response.status_code, detail)
        return {"ok": False, "error": detail or str(exc)}
    except Exception as exc:
        logger.error("[moss_quant] send_close failed: %s", exc)
        return {"ok": False, "error": str(exc)}


def send_update_sl(
    *,
    symbol: str,
    side: str,
    new_sl_price: float,
    profile_id: Optional[int] = None,
) -> Dict[str, Any]:
    """更新移动止损 → POST /api/binance/signals/ingest"""
    if not is_real_mode():
        logger.debug("[moss_quant] real mode disabled, skip send_update_sl")
        return {"ok": False, "error": "real_mode_disabled"}

    logger.info("[moss_quant] send_update_sl: symbol=%s side=%s new_sl=%.4f",
                symbol, side, new_sl_price)
    try:
        return _client().send_update_sl(
            symbol=symbol,
            side=side,
            new_sl_price=new_sl_price,
            profile_id=profile_id,
        )
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500] if exc.response else ""
        logger.error("[moss_quant] send_update_sl HTTP %s: %s", exc.response.status_code, detail)
        return {"ok": False, "error": detail or str(exc)}
    except Exception as exc:
        logger.error("[moss_quant] send_update_sl failed: %s", exc)
        return {"ok": False, "error": str(exc)}


def send_rolling(
    *,
    symbol: str,
    side: str,
    margin_usdt: float,
    leverage: float,
    profile_id: int,
    play: str = "",
    sl_price: float,
    tp_price: Optional[float],
    rolling_count: int = 0,
) -> Dict[str, Any]:
    """滚仓加仓 → POST /api/binance/signals/ingest (play 标记为 rolling)"""
    if not is_real_mode():
        logger.debug("[moss_quant] real mode disabled, skip send_rolling")
        return {"ok": False, "error": "real_mode_disabled"}

    logger.info("[moss_quant] send_rolling: symbol=%s side=%s margin=%.2f lev=%.2f",
                symbol, side, margin_usdt, leverage)
    try:
        return _client().send_open(
            symbol=symbol,
            side=side,
            entry_price=None,
            sl_price=sl_price,
            tp_price=tp_price,
            margin_usdt=margin_usdt,
            leverage=leverage,
            profile_id=profile_id,
            play=f"{play}_rolling" if play else "rolling",
            composite=0.0,
            regime="",
            action="rolling",
        )
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500] if exc.response else ""
        logger.error("[moss_quant] send_rolling HTTP %s: %s", exc.response.status_code, detail)
        return {"ok": False, "error": detail or str(exc)}
    except Exception as exc:
        logger.error("[moss_quant] send_rolling failed: %s", exc)
        return {"ok": False, "error": str(exc)}
