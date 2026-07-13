"""Bitget 现货 REST 签名请求（vnpy 实盘）。"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, Optional

import requests

_BITGET_BASES = {
    "REAL": "https://api.bitget.com",
    "DEMO": "https://api.bitget.com",
}


def api_key() -> str:
    return (os.getenv("BITGET_API_KEY") or "").strip()


def api_secret() -> str:
    return (os.getenv("BITGET_API_SECRET") or "").strip()


def passphrase() -> str:
    return (os.getenv("BITGET_PASSPHRASE") or os.getenv("BITGET_API_PASSPHRASE") or "").strip()


def credentials_configured() -> bool:
    return bool(api_key() and api_secret() and passphrase())


def base_url() -> str:
    server = (os.getenv("BITGET_SERVER") or "REAL").strip().upper()
    return _BITGET_BASES.get(server, _BITGET_BASES["REAL"])


def _sign(timestamp: str, method: str, path: str, body: str) -> str:
    payload = f"{timestamp}{method.upper()}{path}{body}"
    digest = hmac.new(api_secret().encode(), payload.encode(), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


def _headers(timestamp: str, sign: str) -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "ACCESS-KEY": api_key(),
        "ACCESS-SIGN": sign,
        "ACCESS-TIMESTAMP": timestamp,
        "ACCESS-PASSPHRASE": passphrase(),
    }


def signed_request(
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
    *,
    timeout: float = 15.0,
) -> Any:
    if not credentials_configured():
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
    url = f"{base_url()}{req_path}"
    headers = _headers(ts, sign)
    if method.upper() == "GET":
        resp = requests.get(url, headers=headers, timeout=timeout)
    else:
        resp = requests.post(url, headers=headers, data=body_s, timeout=timeout)
    if resp.status_code >= 400:
        try:
            payload = resp.json()
        except Exception:
            payload = resp.text
        raise RuntimeError(f"Bitget spot {path} HTTP {resp.status_code}: {payload}")
    data = resp.json()
    if str(data.get("code", "")) != "00000":
        raise RuntimeError(f"Bitget spot {path} code={data.get('code')} {data.get('msg')}")
    return data.get("data", data)
