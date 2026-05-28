"""Moss Quant 信号发送器：将纸面信号推送到 Next-k-protocol 实盘执行。"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import httpx

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
    url = f"{_proto_url()}/api/binance/positions?status=open&limit=200"
    try:
        resp = httpx.get(url, headers=_headers(), timeout=10)
        resp.raise_for_status()
        positions = resp.json()
        # API 返回 ORDER BY id DESC，取第一个匹配（最新仓位）
        for pos in positions:
            if pos.get("symbol") == symbol and pos.get("source") == SOURCE:
                pid = int(pos["id"])
                _pos_id_cache[profile_id] = pid
                return pid
    except Exception as exc:
        logger.warning("[moss_quant] fetch position_id for %s failed: %s", symbol, exc)
    return 0


def _proto_url() -> str:
    return os.getenv("PROTOCOL_API_URL", "").strip().rstrip("/")


def _proto_token() -> str:
    return os.getenv("PROTOCOL_MAINTENANCE_TOKEN", "").strip()


def _headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Maintenance-Token": _proto_token(),
    }


def _post(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{_proto_url()}{path}"
    try:
        resp = httpx.post(url, json=body, headers=_headers(), timeout=30)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:500] if exc.response else ""
        logger.error("[moss_quant] POST %s HTTP %s: %s", url, exc.response.status_code, detail)
        return {"ok": False, "error": detail or str(exc)}
    except Exception as exc:
        logger.error("[moss_quant] POST %s failed: %s", url, exc)
        return {"ok": False, "error": str(exc)}


def _put(path: str, body: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{_proto_url()}{path}"
    try:
        resp = httpx.put(url, json=body, headers=_headers(), timeout=30)
        resp.raise_for_status()
        return resp.json()
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

    api_signal_id = f"moss_{profile_id}_{int(time.time() * 1000)}"
    signal = {
        "source": SOURCE,
        "api_signal_id": api_signal_id,
        "symbol": symbol,
        "side": side,
        "entry_price": entry_price,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "notional_usdt": round(notional, 2),
        "play": play,
        "regime": regime,
    }
    body = {"signals": [signal]}
    logger.info("[moss_quant] send_open: symbol=%s side=%s notional=%.2f sl=%.4f tp=%s",
                symbol, side, notional, sl_price, tp_price)
    return _post("/api/binance/signals/ingest", body)


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

    body = {
        "source": SOURCE,
        "api_signal_id": f"moss_{profile_id}_close_{int(time.time() * 1000)}",
        "symbol": symbol,
        "side": side,
        "exit_rule": exit_rule,
        "close_price": close_price,
    }
    if position_id:
        body["position_id"] = position_id
    logger.info("[moss_quant] send_close: symbol=%s side=%s rule=%s price=%.4f",
                symbol, side, exit_rule, close_price)
    return _post("/api/binance/positions/close", body)


def send_update_sl(
    *,
    position_id: int,
    new_sl_price: float,
) -> Dict[str, Any]:
    """更新移动止损 → PUT /api/binance/positions/{id}/sl"""
    if not is_real_mode():
        logger.debug("[moss_quant] real mode disabled, skip send_update_sl")
        return {"ok": False, "error": "real_mode_disabled"}

    body = {"new_sl_price": new_sl_price}
    logger.info("[moss_quant] send_update_sl: pos_id=%s new_sl=%.4f",
                position_id, new_sl_price)
    return _put(f"/api/binance/positions/{position_id}/sl", body)


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

    api_signal_id = f"moss_{profile_id}_rolling_{rolling_count}_{int(time.time() * 1000)}"
    signal = {
        "source": SOURCE,
        "api_signal_id": api_signal_id,
        "symbol": symbol,
        "side": side,
        "entry_price": None,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "notional_usdt": round(notional, 2),
        "play": f"{play}_rolling" if play else "rolling",
    }
    body = {"signals": [signal]}
    logger.info("[moss_quant] send_rolling: symbol=%s side=%s notional=%.2f",
                symbol, side, notional)
    return _post("/api/binance/signals/ingest", body)
