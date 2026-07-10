"""Bitget USDT 永续账户 REST（vnpy 实盘）。"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

from quant.common.kline_cache import norm_symbol

logger = logging.getLogger(__name__)

_PRODUCT_TYPE = "USDT-FUTURES"
_MARGIN_COIN = "USDT"

_BITGET_BASES = {
    "REAL": "https://api.bitget.com",
    "DEMO": "https://api.bitget.com",
}


def _api_key() -> str:
    return (os.getenv("BITGET_API_KEY") or "").strip()


def _api_secret() -> str:
    return (os.getenv("BITGET_API_SECRET") or "").strip()


def _passphrase() -> str:
    return (os.getenv("BITGET_PASSPHRASE") or os.getenv("BITGET_API_PASSPHRASE") or "").strip()


def _base_url() -> str:
    server = (os.getenv("BITGET_SERVER") or "REAL").strip().upper()
    return _BITGET_BASES.get(server, _BITGET_BASES["REAL"])


def _sign(timestamp: str, method: str, path: str, body: str) -> str:
    payload = f"{timestamp}{method.upper()}{path}{body}"
    digest = hmac.new(_api_secret().encode(), payload.encode(), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


def _headers(timestamp: str, sign: str) -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "ACCESS-KEY": _api_key(),
        "ACCESS-SIGN": sign,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": _passphrase(),
    }


def _signed_request(method: str, path: str, params: Optional[Dict[str, Any]] = None, body: Optional[Dict[str, Any]] = None) -> Any:
    if not _api_key() or not _api_secret() or not _passphrase():
        raise RuntimeError("BITGET_API_KEY/SECRET/PASSPHRASE not configured")
    query = ""
    req_path = path
    if method.upper() == "GET" and params:
        parts = [f"{k}={v}" for k, v in sorted(params.items())]
        query = "?" + "&".join(parts)
        req_path = path + query
    body_s = json.dumps(body or {}, separators=(",", ":")) if method.upper() != "GET" else ""
    ts = str(int(time.time() * 1000))
    sign = _sign(ts, method, req_path, body_s)
    url = f"{_base_url()}{req_path}"
    headers = _headers(ts, sign)
    if method.upper() == "GET":
        resp = requests.get(url, headers=headers, timeout=15)
    else:
        resp = requests.post(url, headers=headers, data=body_s, timeout=15)
    if resp.status_code >= 400:
        try:
            payload = resp.json()
        except Exception:
            payload = resp.text
        raise RuntimeError(f"Bitget {path} HTTP {resp.status_code}: {payload}")
    data = resp.json()
    if str(data.get("code", "")) != "00000":
        raise RuntimeError(f"Bitget {path} code={data.get('code')} {data.get('msg')}")
    return data.get("data", data)


def set_symbol_leverage(symbol: str, leverage: int) -> None:
    sym = norm_symbol(symbol)
    lev = str(max(1, int(leverage)))
    try:
        _signed_request(
            "POST",
            "/api/v2/mix/account/set-leverage",
            body={
                "symbol": sym,
                "productType": _PRODUCT_TYPE,
                "marginCoin": _MARGIN_COIN,
                "leverage": lev,
            },
        )
        logger.info("[vnpy] bitget leverage %s -> %sx", sym, lev)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "no change" in msg or "not modified" in msg:
            return
        raise


def ensure_one_way_mode() -> None:
    try:
        _signed_request(
            "POST",
            "/api/v2/mix/account/set-position-mode",
            body={"productType": _PRODUCT_TYPE, "posMode": "one_way_mode"},
        )
        logger.info("[vnpy] bitget position mode -> one-way")
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "no change" in msg or "not modified" in msg:
            return
        raise


def fetch_position_snapshots(symbols: List[str]) -> Dict[str, Dict[str, float]]:
    want = {norm_symbol(s) for s in symbols}
    out: Dict[str, Dict[str, float]] = {}
    try:
        rows = _signed_request(
            "GET",
            "/api/v2/mix/position/all-position",
            params={"productType": _PRODUCT_TYPE, "marginCoin": _MARGIN_COIN},
        )
    except RuntimeError as exc:
        logger.warning("[vnpy] bitget position list failed: %s", exc)
        return out
    if not isinstance(rows, list):
        return out
    for row in rows:
        sym = norm_symbol(str(row.get("symbol") or ""))
        if sym not in want:
            continue
        total = float(row.get("total") or row.get("available") or 0.0)
        if abs(total) < 1e-12:
            continue
        side = str(row.get("holdSide") or "").lower()
        signed = total if side != "short" else -total
        out[sym] = {
            "amount": signed,
            "entry": float(row.get("openPriceAvg") or row.get("averageOpenPrice") or 0.0),
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
