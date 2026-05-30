"""Moss Quant 信号发送器：将纸面信号推送到 Next-k-protocol 实盘执行。"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import httpx

from moss_quant.protocol_client import ProtocolClient

logger = logging.getLogger(__name__)

SOURCE = "moss_quant"

# 内存缓存：profile_id → protocol position_id
_pos_id_cache: Dict[int, int] = {}


def get_cached_position_id(profile_id: int) -> int:
    return _pos_id_cache.get(profile_id, 0)


def set_cached_position_id(profile_id: int, position_id: int) -> None:
    _pos_id_cache[profile_id] = position_id


def fetch_and_cache_position_id(symbol: str, profile_id: int) -> int:
    """查询 protocol 该 symbol+source 的最新 open 仓位 ID 并缓存。"""
    try:
        positions = _client(timeout=10).get_moss_positions(status="open", limit=200)
        # API 返回 ORDER BY id DESC，取第一个匹配（最新仓位）
        for pos in positions:
            if (
                pos.get("symbol") == symbol
                and pos.get("source") == SOURCE
                and int(pos.get("profile_id") or 0) == int(profile_id)
            ):
                pid = int(pos["id"])
                _pos_id_cache[profile_id] = pid
                return pid
    except Exception as exc:
        logger.warning("[moss_quant] fetch position_id for %s failed: %s", symbol, exc)
    return 0


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
    notional: float,
    profile_id: int,
    play: str = "",
    composite: float = 0.0,
    regime: str = "",
) -> Dict[str, Any]:
    """发送开仓信号 → POST /api/binance/signals/ingest"""
    if not is_real_mode():
        logger.debug("[moss_quant] real mode disabled, skip send_open")
        return {"ok": False, "error": "real_mode_disabled"}

    logger.info("[moss_quant] send_open: symbol=%s side=%s notional=%.2f sl=%.4f tp=%s",
                symbol, side, notional, sl_price, tp_price)
    try:
        return _client().send_open(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            notional=notional,
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
    position_id: int = 0,
) -> Dict[str, Any]:
    """发送平仓信号 → POST /api/binance/positions/close"""
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
            position_id=position_id,
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
    position_id: int,
    new_sl_price: float,
    profile_id: Optional[int] = None,
) -> Dict[str, Any]:
    """更新移动止损 → PUT /api/binance/positions/{id}/sl"""
    if not is_real_mode():
        logger.debug("[moss_quant] real mode disabled, skip send_update_sl")
        return {"ok": False, "error": "real_mode_disabled"}

    logger.info("[moss_quant] send_update_sl: pos_id=%s new_sl=%.4f",
                position_id, new_sl_price)
    try:
        return _client().send_update_sl(
            position_id=position_id,
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
    notional: float,
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

    logger.info("[moss_quant] send_rolling: symbol=%s side=%s notional=%.2f",
                symbol, side, notional)
    try:
        return _client().send_open(
            symbol=symbol,
            side=side,
            entry_price=None,
            sl_price=sl_price,
            tp_price=tp_price,
            notional=notional,
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
