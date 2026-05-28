"""Moss K 线缓存：币安或 Hyperliquid（见 MOSS_QUANT_DATA_SOURCE）。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from binance_fapi import fetch_klines
from moss_quant import config as cfg

logger = logging.getLogger(__name__)


def _cache_path(symbol: str, interval: str) -> Path:
    cfg.MOSS_QUANT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = symbol.replace("/", "_")
    return cfg.MOSS_QUANT_CACHE_DIR / f"{safe}_{interval}.csv"


def klines_to_df(rows: List[List[Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
    df = pd.DataFrame(rows)
    df = df[[0, 1, 2, 3, 4, 5]].copy()
    df.columns = ["open_time", "open", "high", "low", "close", "volume"]
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def fetch_and_cache(
    symbol: str,
    *,
    interval: Optional[str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    interval = interval or cfg.MOSS_QUANT_KLINE_INTERVAL
    limit = limit or cfg.MOSS_QUANT_KLINE_LIMIT
    from binance_fapi import kline_request_weight

    logger.debug(
        "[moss] binance klines %s %s limit=%s weight≈%s",
        symbol,
        interval,
        limit,
        kline_request_weight(limit),
    )
    rows = fetch_klines(symbol, interval, limit)
    if not rows:
        raise RuntimeError(f"no klines for {symbol} {interval}")
    df = klines_to_df(rows)
    path = _cache_path(symbol, interval)
    df.to_csv(path, index=False)
    return df


def _kline_stale(df: pd.DataFrame) -> bool:
    """最后一根 K 线是否过旧（与 Hyperliquid 路径一致）。"""
    if df is None or df.empty:
        return True
    last = pd.Timestamp(df["timestamp"].iloc[-1])
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    else:
        last = last.tz_convert("UTC")
    age_min = (pd.Timestamp.now(tz="UTC") - last).total_seconds() / 60.0
    return age_min > float(cfg.MOSS_QUANT_KLINE_STALE_MINUTES)


def _use_binance_klines(symbol: str) -> bool:
    """全局 binance 源，或非 Moss 内置标的（回测任意币安永续默认走币安 K 线）。"""
    if cfg.MOSS_QUANT_DATA_SOURCE == "binance":
        return True
    from moss_quant.universe import is_symbol_allowed, normalize_usdt_perp_symbol

    sym = normalize_usdt_perp_symbol(symbol)
    return bool(sym) and not is_symbol_allowed(sym)


def _load_binance_cached(
    symbol: str,
    *,
    interval: str,
    refresh: bool,
) -> pd.DataFrame:
    path = _cache_path(symbol, interval)
    if refresh or not path.is_file():
        return fetch_and_cache(symbol, interval=interval)
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df.empty or _kline_stale(df):
        return fetch_and_cache(symbol, interval=interval)
    return df


def load_cached(
    symbol: str,
    *,
    interval: Optional[str] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    interval = interval or cfg.MOSS_QUANT_KLINE_INTERVAL
    if _use_binance_klines(symbol):
        return _load_binance_cached(symbol, interval=interval, refresh=refresh)
    if cfg.MOSS_QUANT_DATA_SOURCE == "hyperliquid":
        from moss_quant.hyperliquid_klines import load_hyperliquid_cached

        return load_hyperliquid_cached(
            symbol, interval=interval, refresh=refresh
        )
    return _load_binance_cached(symbol, interval=interval, refresh=refresh)


def catalog_entry(symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
    if cfg.MOSS_QUANT_DATA_SOURCE == "hyperliquid":
        from moss_quant.hyperliquid_klines import catalog_entry as hl_catalog

        return hl_catalog(symbol, df)
    if df.empty:
        return {"symbol": symbol, "bars": 0, "data_source": "binance"}
    ts0 = df["timestamp"].iloc[0]
    ts1 = df["timestamp"].iloc[-1]
    return {
        "symbol": symbol,
        "bars": len(df),
        "start": ts0.isoformat().replace("+00:00", "Z"),
        "end": ts1.isoformat().replace("+00:00", "Z"),
        "csv_path": str(_cache_path(symbol, cfg.MOSS_QUANT_KLINE_INTERVAL)),
        "data_source": "binance",
    }


def update_kline_meta(conn, symbol: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    t0 = int(df["timestamp"].iloc[0].timestamp() * 1000)
    t1 = int(df["timestamp"].iloc[-1].timestamp() * 1000)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    conn.execute(
        """INSERT INTO moss_kline_meta(symbol, interval, first_open_time_ms, last_open_time_ms, bar_count, updated_at_utc)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(symbol, interval) DO UPDATE SET
             first_open_time_ms=excluded.first_open_time_ms,
             last_open_time_ms=excluded.last_open_time_ms,
             bar_count=excluded.bar_count,
             updated_at_utc=excluded.updated_at_utc""",
        (symbol, cfg.MOSS_QUANT_KLINE_INTERVAL, t0, t1, len(df), now),
    )
