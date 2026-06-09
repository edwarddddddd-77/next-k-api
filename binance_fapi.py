"""币安 U 本位 fapi REST（K 线 / 现价等轻量脚本共用）。"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from utils.rate_limit import MinIntervalGuard

logger = logging.getLogger(__name__)

FAPI = "https://fapi.binance.com"

# /fapi/v1/klines 单次 limit 上限
BINANCE_KLINE_MAX_PER_REQUEST = 1500

# /fapi/v1/klines 权重（limit 与 IP 1 分钟 2400 上限）：1500 根 ≈ 10
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


def last_binance_weight_1m() -> int:
    with _weight_lock:
        return _last_weight_1m


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
    """K 线请求前：最小间隔 + 已用权重过高时主动让路。"""
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


def api_get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    data, _status = api_get_raw(endpoint, params)
    return data


def api_get_raw(
    endpoint: str, params: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[Any], int]:
    url = f"{FAPI}{endpoint}"
    backoff = (0.8, 1.5, 3.0, 6.0, 12.0)
    for delay in backoff:
        try:
            r = requests.get(url, params=params or {}, timeout=20)
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
            logger.debug("[binance] %s request error: %s", endpoint, e)
            time.sleep(delay)
    return None, 0


def fetch_klines(
    symbol: str,
    interval: str,
    limit: int,
    *,
    end_time_ms: Optional[int] = None,
) -> List[List[Any]]:
    lim = max(1, int(limit))
    params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": lim}
    if end_time_ms is not None:
        params["endTime"] = int(end_time_ms)

    est_w = kline_request_weight(lim)
    _wait_kline_slot(est_w)
    try:
        data, status = api_get_raw("/fapi/v1/klines", params)
    finally:
        _kline_slot_guard.mark_used()

    if status in (418, 429):
        logger.warning(
            "[binance] klines %s limit=%s weight≈%s still limited",
            symbol,
            lim,
            est_w,
        )
    if isinstance(data, list):
        return data
    return []


def fetch_klines_history(
    symbol: str,
    interval: str,
    total_bars: int,
    *,
    end_time_ms: Optional[int] = None,
) -> List[List[Any]]:
    """
    分页拉取最多 total_bars 根 K 线（时间升序，保留最近一段）。
    通过 endTime 向历史回溯；单次请求不超过 BINANCE_KLINE_MAX_PER_REQUEST。
    """
    total = max(1, int(total_bars))
    if total <= BINANCE_KLINE_MAX_PER_REQUEST:
        return fetch_klines(symbol, interval, total, end_time_ms=end_time_ms)

    collected: List[List[Any]] = []
    end_ms = end_time_ms
    max_pages = (total // BINANCE_KLINE_MAX_PER_REQUEST) + 3

    for _ in range(max_pages):
        if len(collected) >= total:
            break
        chunk_size = min(
            BINANCE_KLINE_MAX_PER_REQUEST, total - len(collected)
        )
        batch = fetch_klines(symbol, interval, chunk_size, end_time_ms=end_ms)
        if not batch:
            break

        if collected:
            first_open = int(collected[0][0])
            older = [r for r in batch if int(r[0]) < first_open]
            if not older:
                break
            collected = older + collected
        else:
            collected = list(batch)

        if len(batch) < chunk_size:
            break
        end_ms = int(batch[0][0]) - 1

    if len(collected) > total:
        collected = collected[-total:]
    return collected


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
            data, _status = api_get_raw(
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


def klines_to_df(rows: List[List[Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df[[0, 1, 2, 3, 4, 5]].copy()
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df["open_time"] = df["open_time"].astype("int64")
    return df
