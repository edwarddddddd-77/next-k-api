"""币安 U 本位 fapi REST（K 线 / 现价，Supertrend 等轻量脚本共用）。"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from supertrend_config import FAPI


def api_get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    url = f"{FAPI}{endpoint}"
    backoff = (0.8, 1.5, 3.0, 6.0, 12.0)
    for delay in backoff:
        try:
            r = requests.get(url, params=params or {}, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(delay)
                continue
            return None
        except Exception:
            time.sleep(delay)
    return None


def fetch_klines(
    symbol: str,
    interval: str,
    limit: int,
    *,
    end_time_ms: Optional[int] = None,
) -> List[List[Any]]:
    params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time_ms is not None:
        params["endTime"] = int(end_time_ms)
    data = api_get("/fapi/v1/klines", params)
    return data if isinstance(data, list) else []


def fetch_mark_price(symbol: str) -> Optional[float]:
    data = api_get("/fapi/v1/ticker/price", {"symbol": symbol})
    if isinstance(data, dict) and data.get("price") is not None:
        try:
            return float(data["price"])
        except (TypeError, ValueError):
            return None
    return None


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
