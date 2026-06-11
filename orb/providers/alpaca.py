"""Alpaca Market Data — 美股盘前/日线（PMH、Gap、RVOL）。"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from orb.config import OrbConfig
from orb.premarket import premarket_anchor_ms
from orb.session import session_anchor_ms, session_day_str
from utils.rate_limit import MinIntervalGuard

logger = logging.getLogger(__name__)

DEFAULT_DATA_URL = "https://data.alpaca.markets"
_BAR_COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]

_slot_guard = MinIntervalGuard("ALPACA_DATA_MIN_INTERVAL_SEC", 0.25)
_cache_lock = threading.Lock()
# 实盘按 (symbol, session_date) 缓存，避免每 5m 重复拉 20 天数据
_live_day_cache: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}

# Binance 代币 → Alpaca ticker（少数非标准映射）
_SYMBOL_MAP: Dict[str, str] = {
    "COINUSDT": "COIN",
    "INTCUSDT": "INTC",
    "PAYPUSDT": "PYPL",
    "MSTRUSDT": "MSTR",
    "PLTRUSDT": "PLTR",
    "EWYUSDT": "EWY",
    "GOOGLUSDT": "GOOGL",
    "QQQUSDT": "QQQ",
}

_TIMEFRAME = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "1d": "1Day",
}


def binance_to_alpaca_symbol(binance_symbol: str) -> str:
    sym = str(binance_symbol or "").strip().upper()
    if sym in _SYMBOL_MAP:
        return _SYMBOL_MAP[sym]
    if sym.endswith("USDT"):
        return sym[:-4]
    return sym


def clear_alpaca_live_cache() -> None:
    """测试或跨日调试时清空实盘日缓存。"""
    with _cache_lock:
        _live_day_cache.clear()


def _credentials() -> Tuple[Optional[str], Optional[str]]:
    key = (os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID") or "").strip()
    secret = (os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET_KEY") or "").strip()
    if not key or not secret:
        return None, None
    return key, secret


def alpaca_configured() -> bool:
    key, secret = _credentials()
    return bool(key and secret)


def _data_base_url() -> str:
    return (os.getenv("ALPACA_DATA_URL") or DEFAULT_DATA_URL).strip().rstrip("/")


def _data_feed(cfg: Optional[OrbConfig] = None) -> str:
    if cfg is not None and (cfg.alpaca_data_feed or "").strip():
        return cfg.alpaca_data_feed.strip().lower()
    return (os.getenv("ALPACA_DATA_FEED") or "iex").strip().lower()


def _headers() -> Dict[str, str]:
    key, secret = _credentials()
    if not key or not secret:
        raise RuntimeError("Alpaca API credentials missing (ALPACA_API_KEY / ALPACA_API_SECRET)")
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    }


def _ms_to_rfc3339(ms: int) -> str:
    return pd.Timestamp(int(ms), unit="ms", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _slice_bars_to_asof(df: pd.DataFrame, asof_ms: int) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["open_time"] <= int(asof_ms)].copy().reset_index(drop=True)


def _bar_rows_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_BAR_COLUMNS)
    out = []
    for b in rows:
        ts = pd.Timestamp(b["t"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        out.append(
            {
                "open_time": int(ts.value // 1_000_000),
                "open": float(b["o"]),
                "high": float(b["h"]),
                "low": float(b["l"]),
                "close": float(b["c"]),
                "volume": float(b.get("v") or 0),
            }
        )
    df = pd.DataFrame(out)
    return (
        df.drop_duplicates(subset=["open_time"], keep="last")
        .sort_values("open_time")
        .reset_index(drop=True)
    )


def _fetch_stock_bars_once(
    alpaca_symbol: str,
    *,
    start_ms: int,
    end_ms: int,
    interval: str = "5m",
    feed: str = "iex",
    adjustment: str = "raw",
) -> pd.DataFrame:
    tf = _TIMEFRAME.get((interval or "5m").strip().lower())
    if not tf:
        raise ValueError(f"unsupported alpaca interval: {interval}")
    sym = str(alpaca_symbol).strip().upper()
    url = f"{_data_base_url()}/v2/stocks/bars"
    base_params: Dict[str, Any] = {
        "symbols": sym,
        "timeframe": tf,
        "start": _ms_to_rfc3339(int(start_ms)),
        "end": _ms_to_rfc3339(int(end_ms)),
        "limit": 10000,
        "feed": feed,
        "adjustment": adjustment,
        "sort": "asc",
    }
    params: Dict[str, Any] = dict(base_params)
    all_rows: List[Dict[str, Any]] = []
    page = 0
    while True:
        ok, wait_sec = _slot_guard.check_allow()
        if not ok:
            time.sleep(wait_sec)
        resp = requests.get(url, headers=_headers(), params=params, timeout=30)
        _slot_guard.mark_used()
        if resp.status_code == 401:
            raise RuntimeError("Alpaca API unauthorized — check ALPACA_API_KEY / ALPACA_API_SECRET")
        if resp.status_code == 403:
            raise RuntimeError(
                f"Alpaca data feed '{feed}' forbidden — upgrade plan or set ALPACA_DATA_FEED=iex"
            )
        resp.raise_for_status()
        payload = resp.json()
        chunk = (payload.get("bars") or {}).get(sym) or []
        all_rows.extend(chunk)
        token = payload.get("next_page_token")
        if not token:
            break
        params = {
            "symbols": sym,
            "timeframe": tf,
            "feed": feed,
            "page_token": token,
            "limit": 10000,
        }
        page += 1
        if page > 500:
            logger.warning("[alpaca] pagination stopped after 500 pages for %s", sym)
            break
    return _bar_rows_to_df(all_rows)


def fetch_stock_bars(
    alpaca_symbol: str,
    *,
    start_ms: int,
    end_ms: int,
    interval: str = "5m",
    feed: str = "iex",
    adjustment: str = "raw",
) -> pd.DataFrame:
    """拉取 [start_ms, end_ms] 内的 Alpaca bars（含 extended hours）。"""
    feed_u = (feed or "iex").strip().lower()
    try:
        return _fetch_stock_bars_once(
            alpaca_symbol,
            start_ms=start_ms,
            end_ms=end_ms,
            interval=interval,
            feed=feed_u,
            adjustment=adjustment,
        )
    except RuntimeError as exc:
        if feed_u == "sip" and "forbidden" in str(exc).lower():
            logger.warning("[alpaca] feed=sip forbidden for %s, fallback to iex", alpaca_symbol)
            return _fetch_stock_bars_once(
                alpaca_symbol,
                start_ms=start_ms,
                end_ms=end_ms,
                interval=interval,
                feed="iex",
                adjustment=adjustment,
            )
        raise


def fetch_daily_bars(
    alpaca_symbol: str,
    *,
    start_ms: int,
    end_ms: int,
    feed: str = "iex",
) -> pd.DataFrame:
    return fetch_stock_bars(
        alpaca_symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        interval="1d",
        feed=feed,
    )


def _premarket_history_window_ms(asof_ms: int, cfg: OrbConfig) -> Tuple[int, int]:
    lookback_days = max(1, int(cfg.premarket_rvol_lookback))
    rth = session_anchor_ms(
        asof_ms, tz=cfg.session_tz, session_open_time=cfg.session_open_time
    )
    pm0 = premarket_anchor_ms(
        asof_ms,
        tz=cfg.session_tz,
        session_open_time=cfg.session_open_time,
        premarket_open_time=cfg.premarket_open_time,
    )
    start = pm0 - lookback_days * 86_400_000
    end = max(int(asof_ms), rth)
    return int(start), int(end)


def _live_cache_key(binance_symbol: str, session_date: str) -> str:
    return f"{binance_symbol.strip().upper()}:{session_date}"


def _get_live_cached(
    binance_symbol: str,
    cfg: OrbConfig,
    *,
    asof_ms: int,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    sess_day = session_day_str(
        asof_ms, tz=cfg.session_tz, session_open_time=cfg.session_open_time
    )
    key = _live_cache_key(binance_symbol, sess_day)
    with _cache_lock:
        hit = _live_day_cache.get(key)
    if hit is None:
        return None
    bars, daily = hit
    if bars.empty and daily.empty:
        return None
    return hit


def _put_live_cache(
    binance_symbol: str,
    cfg: OrbConfig,
    *,
    asof_ms: int,
    bars: pd.DataFrame,
    daily: pd.DataFrame,
) -> None:
    sess_day = session_day_str(
        asof_ms, tz=cfg.session_tz, session_open_time=cfg.session_open_time
    )
    key = _live_cache_key(binance_symbol, sess_day)
    with _cache_lock:
        _live_day_cache[key] = (bars, daily)


def load_premarket_history(
    binance_symbol: str,
    cfg: OrbConfig,
    *,
    asof_ms: int,
    cached_bars: Optional[pd.DataFrame] = None,
    cached_daily: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    返回 (intraday_bars, daily_bars) 供盘前指标计算，均截断至 asof_ms。
    cached_* 非空：回测预加载；否则走实盘日缓存 + API。
    """
    asof_ms = int(asof_ms)
    empty = pd.DataFrame(columns=_BAR_COLUMNS)

    if cached_bars is not None:
        bars = cached_bars if not cached_bars.empty else empty
        daily = (
            cached_daily
            if cached_daily is not None and not cached_daily.empty
            else empty
        )
        return _slice_bars_to_asof(bars, asof_ms), _slice_bars_to_asof(daily, asof_ms)

    live = _get_live_cached(binance_symbol, cfg, asof_ms=asof_ms)
    if live is not None:
        bars, daily = live
        return _slice_bars_to_asof(bars, asof_ms), _slice_bars_to_asof(daily, asof_ms)

    if not alpaca_configured():
        logger.warning(
            "[alpaca] credentials missing — premarket stats unavailable for %s",
            binance_symbol,
        )
        return empty.copy(), empty.copy()

    alpaca_sym = binance_to_alpaca_symbol(binance_symbol)
    start_ms, end_ms = _premarket_history_window_ms(asof_ms, cfg)
    feed = _data_feed(cfg)
    try:
        bars = fetch_stock_bars(
            alpaca_sym,
            start_ms=start_ms,
            end_ms=end_ms,
            interval=cfg.signal_interval,
            feed=feed,
        )
    except Exception as exc:
        logger.warning("[alpaca] fetch bars failed for %s: %s", alpaca_sym, exc)
        bars = empty.copy()

    lookback = max(30, int(cfg.premarket_rvol_lookback) + 10)
    daily_start = int(asof_ms) - lookback * 86_400_000
    try:
        daily_full = fetch_daily_bars(
            alpaca_sym,
            start_ms=daily_start,
            end_ms=int(asof_ms),
            feed=feed,
        )
    except Exception as exc:
        logger.warning("[alpaca] fetch daily failed for %s: %s", alpaca_sym, exc)
        daily_full = empty.copy()

    _put_live_cache(binance_symbol, cfg, asof_ms=asof_ms, bars=bars, daily=daily_full)
    return _slice_bars_to_asof(bars, asof_ms), _slice_bars_to_asof(daily_full, asof_ms)


def alpaca_rth_open(
    bars_df: pd.DataFrame,
    asof_open_ms: int,
    *,
    cfg: OrbConfig,
) -> Optional[float]:
    """Alpaca 数据源上的 RTH 首根 K 线开盘价（Gap 与 PM 同源）。"""
    if bars_df.empty:
        return None
    rth = session_anchor_ms(
        int(asof_open_ms), tz=cfg.session_tz, session_open_time=cfg.session_open_time
    )
    sub = bars_df[bars_df["open_time"] >= rth]
    if sub.empty:
        return None
    val = float(sub["open"].iloc[0])
    return val if val > 0 else None


def preload_premarket_for_backtest(
    binance_symbols: List[str],
    cfg: OrbConfig,
    *,
    start_ms: int,
    end_ms: int,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """回测启动时一次性拉取各标 Alpaca bars + daily。"""
    bars_map: Dict[str, pd.DataFrame] = {}
    daily_map: Dict[str, pd.DataFrame] = {}
    if not cfg.premarket_filter:
        return bars_map, daily_map
    if (cfg.premarket_source or "alpaca").strip().lower() != "alpaca":
        return bars_map, daily_map
    if not alpaca_configured():
        logger.warning("[alpaca] backtest premarket enabled but credentials missing")
        return bars_map, daily_map

    lookback = max(1, int(cfg.premarket_rvol_lookback))
    fetch_start = int(start_ms) - lookback * 86_400_000
    feed = _data_feed(cfg)
    daily_start = fetch_start - 30 * 86_400_000

    for sym in binance_symbols:
        alpaca_sym = binance_to_alpaca_symbol(sym)
        try:
            bars_map[sym] = fetch_stock_bars(
                alpaca_sym,
                start_ms=fetch_start,
                end_ms=int(end_ms),
                interval=cfg.signal_interval,
                feed=feed,
            )
            daily_map[sym] = fetch_daily_bars(
                alpaca_sym,
                start_ms=daily_start,
                end_ms=int(end_ms),
                feed=feed,
            )
            logger.info(
                "[alpaca] preloaded %s (%s): %d bars, %d daily",
                sym,
                alpaca_sym,
                len(bars_map[sym]),
                len(daily_map[sym]),
            )
        except Exception as exc:
            logger.warning("[alpaca] preload failed for %s: %s", sym, exc)
            bars_map[sym] = pd.DataFrame(columns=_BAR_COLUMNS)
            daily_map[sym] = pd.DataFrame(columns=_BAR_COLUMNS)
    return bars_map, daily_map
