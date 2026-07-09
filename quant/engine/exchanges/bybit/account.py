"""Bybit Linear 账户 REST（vnpy 实盘）。"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
import urllib.parse
from typing import Any, Dict, List, Optional

import requests

from quant.common.kline_cache import norm_symbol

logger = logging.getLogger(__name__)

_BYBIT_BASES = {
    "REAL": "https://api.bybit.com",
    "TESTNET": "https://api-testnet.bybit.com",
    "DEMO": "https://api-demo.bybit.com",
}


def _api_key() -> str:
    return (os.getenv("BYBIT_API_KEY") or "").strip()


def _api_secret() -> str:
    return (os.getenv("BYBIT_API_SECRET") or "").strip()


def _base_url() -> str:
    server = (os.getenv("BYBIT_SERVER") or "REAL").strip().upper()
    return _BYBIT_BASES.get(server, _BYBIT_BASES["REAL"])


def _signed_request(method: str, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    key = _api_key()
    secret = _api_secret()
    if not key or not secret:
        raise RuntimeError("BYBIT_API_KEY/SECRET not configured")
    payload = dict(params or {})
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    query = urllib.parse.urlencode(sorted(payload.items()))
    sign_payload = f"{timestamp}{key}{recv_window}{query}"
    sign = hmac.new(secret.encode(), sign_payload.encode(), hashlib.sha256).hexdigest()
    headers = {
        "X-BAPI-API-KEY": key,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
        "X-BAPI-SIGN": sign,
    }
    url = f"{_base_url()}{path}"
    if method.upper() == "GET":
        resp = requests.get(f"{url}?{query}" if query else url, headers=headers, timeout=15)
    else:
        resp = requests.post(url, headers=headers, data=query, timeout=15)
    if resp.status_code >= 400:
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        raise RuntimeError(f"Bybit {path} HTTP {resp.status_code}: {body}")
    data = resp.json()
    if int(data.get("retCode", 0)) != 0:
        raise RuntimeError(f"Bybit {path} retCode={data.get('retCode')} {data.get('retMsg')}")
    return data.get("result", data)


def set_symbol_leverage(symbol: str, leverage: int) -> None:
    sym = norm_symbol(symbol)
    lev = str(max(1, int(leverage)))
    try:
        _signed_request(
            "POST",
            "/v5/position/set-leverage",
            {"category": "linear", "symbol": sym, "buyLeverage": lev, "sellLeverage": lev},
        )
        logger.info("[vnpy] bybit leverage %s -> %sx", sym, lev)
    except RuntimeError as exc:
        msg = str(exc)
        if "110043" in msg or "leverage not modified" in msg.lower():
            return
        raise


def ensure_one_way_mode() -> None:
    """Bybit 统一账户默认单向；显式切换 position mode。"""
    try:
        _signed_request("POST", "/v5/position/switch-mode", {"category": "linear", "mode": 0})
        logger.info("[vnpy] bybit position mode -> one-way")
    except RuntimeError as exc:
        msg = str(exc)
        if "110025" in msg or "position mode is not modified" in msg.lower():
            return
        raise


def fetch_position_snapshots(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    """symbol -> {amount, entry}（仅非零持仓）。"""
    want = {norm_symbol(s) for s in symbols}
    out: Dict[str, Dict[str, float]] = {}
    for sym in sorted(want):
        try:
            result = _signed_request(
                "GET",
                "/v5/position/list",
                {"category": "linear", "symbol": sym},
            )
        except RuntimeError as exc:
            logger.warning("[vnpy] bybit position list %s: %s", sym, exc)
            continue
        rows = result.get("list") if isinstance(result, dict) else None
        if not isinstance(rows, list):
            continue
        for row in rows:
            if norm_symbol(str(row.get("symbol") or "")) != sym:
                continue
            side = str(row.get("side") or "").lower()
            size = float(row.get("size") or 0.0)
            if abs(size) < 1e-12 or side == "none":
                continue
            signed = size if side == "buy" else -size
            out[sym] = {
                "amount": signed,
                "entry": float(row.get("avgPrice") or row.get("entryPrice") or 0.0),
            }
    return out


def fetch_position_amounts(symbols: List[str]) -> Dict[str, float]:
    return {sym: float(snap.get("amount") or 0.0) for sym, snap in fetch_position_snapshots(symbols).items()}


def _lane_leverage(cfg) -> int:
    lev = float(getattr(cfg, "live_leverage", 0.0) or 0.0)
    if lev > 0:
        return int(lev)
    return 5


def ensure_pool_leverage(symbols: List[str], cfg) -> None:
    ensure_one_way_mode()
    lev = _lane_leverage(cfg)
    for raw in symbols:
        set_symbol_leverage(norm_symbol(raw), lev)
