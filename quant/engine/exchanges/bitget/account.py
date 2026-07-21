"""Bitget USDT 永续账户 REST（vnpy 实盘）。

Supports optional per-call credentials via ``bitget_creds()`` context
(for HL sub-account routing). Default = process env BITGET_*.
"""

from __future__ import annotations

import base64
import contextvars
import hashlib
import hmac
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import requests

from quant.common.kline_cache import norm_symbol

logger = logging.getLogger(__name__)

_PRODUCT_TYPE = "USDT-FUTURES"
_MARGIN_COIN = "USDT"

_BITGET_BASES = {
    "REAL": "https://api.bitget.com",
    "DEMO": "https://api.bitget.com",
}


@dataclass(frozen=True)
class BitgetCreds:
    api_key: str
    api_secret: str
    passphrase: str
    server: str = "REAL"

    def ok(self) -> bool:
        return bool(self.api_key and self.api_secret and self.passphrase)


_creds_ctx: contextvars.ContextVar[Optional[BitgetCreds]] = contextvars.ContextVar(
    "bitget_creds", default=None
)


@contextmanager
def bitget_creds(creds: Optional[BitgetCreds]) -> Iterator[None]:
    """Temporarily use sub-account (or other) API keys for REST calls."""
    token = _creds_ctx.set(creds)
    try:
        yield
    finally:
        _creds_ctx.reset(token)


def load_creds_from_env(prefix: str = "") -> BitgetCreds:
    """Load keys from env. prefix='' → BITGET_*; prefix='BITGET_SUB_BTC' → BITGET_SUB_BTC_API_KEY etc."""
    p = (prefix or "").strip().rstrip("_")
    if p:
        key = (os.getenv(f"{p}_API_KEY") or "").strip()
        sec = (os.getenv(f"{p}_API_SECRET") or "").strip()
        pwd = (
            os.getenv(f"{p}_PASSPHRASE")
            or os.getenv(f"{p}_API_PASSPHRASE")
            or ""
        ).strip()
        server = (os.getenv(f"{p}_SERVER") or os.getenv("BITGET_SERVER") or "REAL").strip().upper()
    else:
        key = (os.getenv("BITGET_API_KEY") or "").strip()
        sec = (os.getenv("BITGET_API_SECRET") or "").strip()
        pwd = (os.getenv("BITGET_PASSPHRASE") or os.getenv("BITGET_API_PASSPHRASE") or "").strip()
        server = (os.getenv("BITGET_SERVER") or "REAL").strip().upper()
    if server not in _BITGET_BASES:
        server = "REAL"
    return BitgetCreds(api_key=key, api_secret=sec, passphrase=pwd, server=server)


def _active_creds() -> BitgetCreds:
    override = _creds_ctx.get()
    if override is not None:
        return override
    return load_creds_from_env("")


def _api_key() -> str:
    return _active_creds().api_key


def _api_secret() -> str:
    return _active_creds().api_secret


def _passphrase() -> str:
    return _active_creds().passphrase


def _base_url() -> str:
    server = _active_creds().server.strip().upper()
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
    creds = _active_creds()
    if not creds.ok():
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


def fetch_all_signed_positions() -> Dict[str, float]:
    """All non-zero one-way signed sizes on the active Bitget account."""
    out: Dict[str, float] = {}
    try:
        rows = _signed_request(
            "GET",
            "/api/v2/mix/position/all-position",
            params={"productType": _PRODUCT_TYPE, "marginCoin": _MARGIN_COIN},
        )
    except RuntimeError as exc:
        logger.warning("[vnpy] bitget all-position failed: %s", exc)
        return out
    if not isinstance(rows, list):
        return out
    for row in rows:
        sym = norm_symbol(str(row.get("symbol") or ""))
        if not sym:
            continue
        total = float(row.get("total") or row.get("available") or 0.0)
        if abs(total) < 1e-12:
            continue
        side = str(row.get("holdSide") or "").lower()
        signed = total if side != "short" else -total
        out[sym] = signed
    return out


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


def fetch_signed_position(symbol: str) -> float:
    """One-way signed base size for a single symbol (long +, short -)."""
    sym = norm_symbol(symbol)
    snaps = fetch_position_snapshots([sym])
    return float((snaps.get(sym) or {}).get("amount") or 0.0)


def get_order_by_client_oid(symbol: str, client_oid: str) -> Optional[Dict[str, Any]]:
    """Return order detail if clientOid already used; None if not found."""
    sym = norm_symbol(symbol)
    oid = str(client_oid or "").strip()
    if not oid:
        return None
    try:
        data = _signed_request(
            "GET",
            "/api/v2/mix/order/detail",
            params={
                "symbol": sym,
                "productType": _PRODUCT_TYPE,
                "clientOid": oid,
            },
        )
    except RuntimeError as exc:
        msg = str(exc).lower()
        if any(
            k in msg
            for k in (
                "not exist",
                "not found",
                "does not exist",
                "no order",
                "order id",
                "40015",  # common bitget not-found-ish
                "43001",
            )
        ):
            return None
        # Ambiguous: do not place duplicate blindly — surface error to caller
        raise
    if not data:
        return None
    if isinstance(data, dict):
        # Some responses wrap empty
        if not data.get("clientOid") and not data.get("orderId") and not data.get("orderid"):
            return None
        return data
    return None


def place_market_order(
    *,
    symbol: str,
    side: str,
    size: float,
    client_oid: str,
    reduce_only: bool = False,
    leverage: Optional[int] = None,
) -> Dict[str, Any]:
    """Place USDT-M market order (same REST path as vnpy Bitget gateway).

    side: buy|sell (one-way mode)
    size: base-coin quantity
    client_oid: idempotent id (Bitget typically ≤64; keep ≤32)
    """
    sym = norm_symbol(symbol)
    side_l = str(side or "").strip().lower()
    if side_l not in ("buy", "sell"):
        raise ValueError(f"invalid side: {side}")
    qty = float(size)
    if qty <= 0:
        raise ValueError("size must be > 0")
    oid = str(client_oid or "").strip()
    if not oid:
        raise ValueError("client_oid required")
    if len(oid) > 32:
        oid = oid[:32]

    existing = get_order_by_client_oid(sym, oid)
    if existing:
        return {"deduped": True, "order": existing, "clientOid": oid, "symbol": sym}

    if leverage is not None and int(leverage) > 0 and not reduce_only:
        try:
            set_symbol_leverage(sym, int(leverage))
        except Exception as exc:
            logger.warning("[vnpy] bitget set leverage skip %s: %s", sym, exc)

    # Size string: trim trailing zeros; Bitget rejects excess precision per symbol.
    size_s = f"{qty:.8f}".rstrip("0").rstrip(".")
    if not size_s or size_s == "0":
        raise ValueError(f"size rounds to zero: {qty}")

    body = {
        "symbol": sym,
        "productType": _PRODUCT_TYPE,
        "marginCoin": _MARGIN_COIN,
        "marginMode": "crossed",
        "side": side_l,
        "orderType": "market",
        "size": size_s,
        "clientOid": oid,
        "reduceOnly": "YES" if reduce_only else "NO",
    }
    data = _signed_request("POST", "/api/v2/mix/order/place-order", body=body)
    logger.info(
        "[vnpy] bitget place %s %s size=%s reduceOnly=%s oid=%s -> %s",
        sym,
        side_l,
        size_s,
        body["reduceOnly"],
        oid,
        data,
    )
    return {"deduped": False, "order": data, "clientOid": oid, "symbol": sym, "size": size_s, "side": side_l}


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
