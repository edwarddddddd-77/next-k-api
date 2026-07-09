"""Bybit Linear 公开行情（V5 REST）。"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from quant.common.kline_cache import norm_symbol
from quant.market.klines import interval_step_ms, klines_to_df

logger = logging.getLogger(__name__)

PROVIDER_BYBIT = "bybit"

_BYBIT_BASES = {
    "REAL": "https://api.bybit.com",
    "TESTNET": "https://api-testnet.bybit.com",
    "DEMO": "https://api-demo.bybit.com",
}

_BYBIT_INTERVAL = {
    "1m": "1",
    "3m": "3",
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "1h": "60",
    "2h": "120",
    "4h": "240",
    "1d": "D",
    "1w": "W",
}


def _base_url() -> str:
    server = (os.getenv("BYBIT_SERVER") or os.getenv("BYBIT_MARKET_SERVER") or "REAL").strip().upper()
    return _BYBIT_BASES.get(server, _BYBIT_BASES["REAL"])


def _timeout_sec() -> float:
    try:
        return max(3.0, float(os.getenv("BYBIT_MARKET_TIMEOUT_SEC", "12") or 12))
    except ValueError:
        return 12.0


def _map_interval(interval: str) -> str:
    key = interval.strip().lower()
    if key in _BYBIT_INTERVAL:
        return _BYBIT_INTERVAL[key]
    if key.endswith("m") and key[:-1].isdigit():
        return key[:-1]
    return "1"


def _public_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{_base_url()}{path}"
    backoff = (0.5, 1.5, 3.0)
    for delay in backoff:
        try:
            resp = requests.get(url, params=params or {}, timeout=_timeout_sec())
            if resp.status_code in (418, 429):
                logger.warning("[bybit] market %s rate limited (%s)", path, resp.status_code)
                time.sleep(delay)
                continue
            if resp.status_code >= 400:
                return {}
            data = resp.json()
            if int(data.get("retCode", 0)) != 0:
                logger.warning("[bybit] market %s retCode=%s %s", path, data.get("retCode"), data.get("retMsg"))
                return {}
            result = data.get("result")
            return result if isinstance(result, dict) else {}
        except Exception as exc:
            logger.warning("[bybit] market %s error: %s", path, exc)
            time.sleep(delay)
    return {}


def check_connectivity(*, timeout_sec: float | None = None) -> tuple[bool, str]:
    t = float(timeout_sec if timeout_sec is not None else _timeout_sec())
    try:
        resp = requests.get(f"{_base_url()}/v5/market/time", timeout=t)
        if resp.status_code == 200:
            return True, f"ok ({_base_url()})"
        return False, f"time status={resp.status_code} ({_base_url()})"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc} ({_base_url()})"


def fetch_mark_price(symbol: str) -> Optional[float]:
    sym = norm_symbol(symbol)
    result = _public_get("/v5/market/tickers", {"category": "linear", "symbol": sym})
    rows = result.get("list") if isinstance(result, dict) else None
    if not isinstance(rows, list) or not rows:
        return None
    row = rows[0]
    for key in ("lastPrice", "markPrice", "indexPrice"):
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
    sym = norm_symbol(symbol)
    if end_ms is None:
        end_ms = int(time.time() * 1000)
    out: List[List[Any]] = []
    cur = int(start_ms)
    cap = 150_000
    bybit_interval = _map_interval(interval)
    while cur <= int(end_ms) and len(out) < cap:
        result = _public_get(
            "/v5/market/kline",
            {
                "category": "linear",
                "symbol": sym,
                "interval": bybit_interval,
                "start": cur,
                "end": int(end_ms),
                "limit": 1000,
            },
        )
        batch = result.get("list") if isinstance(result, dict) else None
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


__all__ = [
    "PROVIDER_BYBIT",
    "check_connectivity",
    "fetch_klines_forward",
    "fetch_mark_price",
    "klines_to_df",
]
