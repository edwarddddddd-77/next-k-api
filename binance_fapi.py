"""币安 U 本位 fapi REST（K 线 / 现价 — 供 quant.market 与 accumulation 共用）。"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from utils.rate_limit import MinIntervalGuard

logger = logging.getLogger(__name__)

FAPI = (os.getenv("BINANCE_FAPI_BASE") or "https://fapi.binance.com").strip().rstrip("/")

_KLINE_WEIGHT_BY_LIMIT: Tuple[Tuple[int, int], ...] = (
    (100, 1),
    (500, 2),
    (1000, 5),
    (10_000, 10),
)

_kline_slot_guard = MinIntervalGuard("NEXT_K_BINANCE_KLINE_MIN_INTERVAL_SEC", 0.4)
_weight_lock = threading.Lock()
_last_weight_1m: int = 0


def kline_request_weight(limit: int) -> int:
    n = max(1, int(limit))
    for cap, w in _KLINE_WEIGHT_BY_LIMIT:
        if n <= cap:
            return w
    return 10


def _weight_soft_cap() -> int:
    try:
        return max(500, int(os.getenv("BINANCE_FAPI_WEIGHT_SOFT_CAP", "2000") or 2000))
    except ValueError:
        return 2000


def _weight_backoff_sec() -> float:
    try:
        return max(0.5, float(os.getenv("BINANCE_FAPI_WEIGHT_BACKOFF_SEC", "3") or 3))
    except ValueError:
        return 3.0


def _note_weight_headers(headers: Any) -> None:
    global _last_weight_1m
    if headers is None:
        return
    raw = headers.get("X-MBX-USED-WEIGHT-1M") or headers.get("X-MBX-USED-WEIGHT-1m")
    if raw is None:
        return
    try:
        used = int(raw)
    except (TypeError, ValueError):
        return
    with _weight_lock:
        _last_weight_1m = used
    cap = _weight_soft_cap()
    if used >= cap:
        wait = _weight_backoff_sec()
        logger.warning(
            "[binance] fapi 1m weight %s/%s，暂停 %.1fs",
            used,
            cap,
            wait,
        )
        time.sleep(wait)


def _wait_kline_slot(estimated_weight: int) -> None:
    while True:
        ok, wait_sec = _kline_slot_guard.check_allow()
        if not ok:
            time.sleep(max(wait_sec, 0.05))
            continue
        with _weight_lock:
            projected = _last_weight_1m + max(1, int(estimated_weight))
        if projected >= _weight_soft_cap():
            time.sleep(_weight_backoff_sec())
            continue
        break


def _api_timeout_sec() -> float:
    try:
        return max(3.0, float(os.getenv("BINANCE_FAPI_TIMEOUT_SEC", "12") or 12))
    except ValueError:
        return 12.0


def _requests_proxies() -> Optional[Dict[str, str]]:
    http = (os.getenv("HTTP_PROXY") or os.getenv("http_proxy") or "").strip()
    https = (os.getenv("HTTPS_PROXY") or os.getenv("https_proxy") or http).strip()
    if not http and not https:
        return None
    out: Dict[str, str] = {}
    if http:
        out["http"] = http
    if https:
        out["https"] = https
    return out


def check_fapi_connectivity(*, timeout_sec: Optional[float] = None) -> tuple[bool, str]:
    t = float(timeout_sec if timeout_sec is not None else _api_timeout_sec())
    try:
        r = requests.get(
            f"{FAPI}/fapi/v1/ping",
            timeout=t,
            proxies=_requests_proxies(),
        )
        if r.status_code == 200:
            return True, f"ok ({FAPI})"
        return False, f"ping status={r.status_code} ({FAPI})"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc} ({FAPI})"


def api_get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    data, _status = api_get_raw(endpoint, params)
    return data


def api_get_raw(
    endpoint: str, params: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[Any], int]:
    url = f"{FAPI}{endpoint}"
    backoff = (0.8, 2.0, 4.0)
    proxies = _requests_proxies()
    timeout = _api_timeout_sec()
    for delay in backoff:
        try:
            r = requests.get(url, params=params or {}, timeout=timeout, proxies=proxies)
            _note_weight_headers(r.headers)
            if r.status_code == 200:
                return r.json(), 200
            if r.status_code in (418, 429):
                logger.warning(
                    "[binance] %s %s rate limited (%s), retry in %.1fs",
                    endpoint,
                    (params or {}).get("symbol"),
                    r.status_code,
                    delay,
                )
                time.sleep(delay)
                continue
            return None, r.status_code
        except Exception as e:
            logger.warning("[binance] %s request error: %s", endpoint, e)
            time.sleep(delay)
    return None, 0


def fetch_mark_price(symbol: str) -> Optional[float]:
    data = api_get("/fapi/v1/ticker/price", {"symbol": symbol})
    if isinstance(data, dict) and data.get("price") is not None:
        try:
            return float(data["price"])
        except (TypeError, ValueError):
            return None
    return None


def fetch_klines_forward(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: Optional[int] = None,
) -> List[List[Any]]:
    """从 start_ms 起分页拉取 K 线（含 start 根），直到 end_ms。"""
    if end_ms is None:
        end_ms = int(time.time() * 1000)
    out: List[List[Any]] = []
    cur = int(start_ms)
    cap = 150_000
    while cur <= end_ms and len(out) < cap:
        est_w = kline_request_weight(1500)
        _wait_kline_slot(est_w)
        try:
            data, status = api_get_raw(
                "/fapi/v1/klines",
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": int(cur),
                    "endTime": int(end_ms),
                    "limit": 1500,
                },
            )
        finally:
            _kline_slot_guard.mark_used()
        if status not in (200,):
            logger.warning(
                "[binance] klines forward %s %s status=%s start=%s",
                symbol,
                interval,
                status,
                cur,
            )
        batch = data if isinstance(data, list) else []
        if not batch:
            break
        for row in batch:
            if not row:
                continue
            ot = int(row[0])
            if ot < int(start_ms):
                continue
            if ot > int(end_ms):
                break
            out.append(row)
        last_open = int(batch[-1][0])
        nxt = last_open + 1
        if nxt <= cur:
            break
        cur = nxt
        if len(batch) < 1500:
            break
    return out
