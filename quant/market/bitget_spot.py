"""Bitget 现货公开行情（V2 REST）。"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests

from quant.market.klines import interval_step_ms, klines_to_df

logger = logging.getLogger(__name__)

PROVIDER_BITGET_SPOT = "bitget_spot"

_BITGET_BASES = {
    "REAL": "https://api.bitget.com",
    "DEMO": "https://api.bitget.com",
}

_BITGET_INTERVAL = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1d": "1day",
    "1w": "1week",
}


def _norm_pair(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def _base_url() -> str:
    server = (os.getenv("BITGET_SERVER") or os.getenv("BITGET_MARKET_SERVER") or "REAL").strip().upper()
    return _BITGET_BASES.get(server, _BITGET_BASES["REAL"])


def _timeout_sec() -> float:
    try:
        return max(3.0, float(os.getenv("BITGET_MARKET_TIMEOUT_SEC", "12") or 12))
    except ValueError:
        return 12.0


def _map_interval(interval: str) -> str:
    key = interval.strip().lower()
    if key in _BITGET_INTERVAL:
        return _BITGET_INTERVAL[key]
    if key.endswith("m") and key[:-1].isdigit():
        return f"{key[:-1]}min"
    return "1min"


def _public_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{_base_url()}{path}"
    backoff = (0.5, 1.5, 3.0)
    for delay in backoff:
        try:
            resp = requests.get(url, params=params or {}, timeout=_timeout_sec())
            if resp.status_code in (418, 429):
                logger.warning("[bitget_spot] market %s rate limited (%s)", path, resp.status_code)
                time.sleep(delay)
                continue
            if resp.status_code >= 400:
                return {}
            data = resp.json()
            if str(data.get("code", "")) != "00000":
                logger.warning("[bitget_spot] market %s code=%s %s", path, data.get("code"), data.get("msg"))
                return {}
            payload = data.get("data")
            return payload if isinstance(payload, (dict, list)) else {}
        except Exception as exc:
            logger.warning("[bitget_spot] market %s error: %s", path, exc)
            time.sleep(delay)
    return {}


def check_connectivity(*, timeout_sec: float | None = None) -> tuple[bool, str]:
    t = float(timeout_sec if timeout_sec is not None else _timeout_sec())
    try:
        resp = requests.get(f"{_base_url()}/api/v2/public/time", timeout=t)
        if resp.status_code == 200:
            return True, f"ok ({_base_url()})"
        return False, f"time status={resp.status_code} ({_base_url()})"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc} ({_base_url()})"


def fetch_mark_price(symbol: str) -> Optional[float]:
    sym = _norm_pair(symbol)
    rows = _public_get("/api/v2/spot/market/tickers", {"symbol": sym})
    if isinstance(rows, list) and rows:
        row = rows[0]
    elif isinstance(rows, dict):
        row = rows
    else:
        return None
    for key in ("lastPr", "close", "buyOne", "sellOne", "last"):
        raw = row.get(key)
        if raw is None:
            continue
        try:
            px = float(raw)
            if px > 0:
                return px
        except (TypeError, ValueError):
            continue
    return None


def fetch_klines_forward(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: Optional[int] = None,
) -> List[List[Any]]:
    sym = _norm_pair(symbol)
    if end_ms is None:
        end_ms = int(time.time() * 1000)
    out: List[List[Any]] = []
    cur = int(start_ms)
    cap = 150_000
    gran = _map_interval(interval)
    while cur <= int(end_ms) and len(out) < cap:
        params = {
            "symbol": sym,
            "granularity": gran,
            "startTime": str(cur),
            "endTime": str(int(end_ms)),
            "limit": "1000",
        }
        batch = _public_get("/api/v2/spot/market/candles", params)
        if not isinstance(batch, list) or not batch:
            break
        rows_asc = sorted(batch, key=lambda r: int(r[0]))
        appended = 0
        for row in rows_asc:
            if not row or len(row) < 6:
                continue
            ot = int(row[0])
            if ot < int(start_ms) or ot > int(end_ms):
                continue
            normalized = [ot, row[1], row[2], row[3], row[4], row[5]]
            if out and out[-1][0] == ot:
                out[-1] = normalized
            else:
                out.append(normalized)
                appended += 1
        if appended == 0:
            break
        last_open = int(rows_asc[-1][0])
        nxt = last_open + max(1, interval_step_ms(interval))
        if nxt <= cur:
            break
        cur = nxt
        if len(batch) < 1000:
            break
    return out


def fetch_symbol_info(symbol: str) -> dict:
    sym = _norm_pair(symbol)
    row = _public_get("/api/v2/spot/public/symbols", {"symbol": sym})
    if isinstance(row, list) and row:
        return row[0] if isinstance(row[0], dict) else {}
    return row if isinstance(row, dict) else {}


__all__ = [
    "PROVIDER_BITGET_SPOT",
    "check_connectivity",
    "fetch_klines_forward",
    "fetch_mark_price",
    "fetch_symbol_info",
    "klines_to_df",
]
