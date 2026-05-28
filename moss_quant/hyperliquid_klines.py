"""Hyperliquid 15m K 线：官方工厂 CSV 或 ccxt 实时拉取。"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from glob import glob
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from moss_quant import config as cfg
from moss_quant.universe import symbol_to_base

logger = logging.getLogger(__name__)

try:
    import ccxt
except ImportError:
    ccxt = None  # type: ignore

_CCXT_MAX_PAGES = 50


def _cache_path(symbol: str, interval: str) -> Path:
    cfg.MOSS_QUANT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = symbol.replace("/", "_").upper()
    return cfg.MOSS_QUANT_CACHE_DIR / f"hl_{safe}_{interval}.csv"


def symbol_to_ccxt(symbol: str) -> str:
    """BTCUSDT / BTCUSDC → Hyperliquid ccxt: BTC/USDC:USDC"""
    base = symbol_to_base(symbol)
    return f"{base}/USDC:USDC"


def _ts_iso(ts: Any) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t.isoformat().replace("+00:00", "Z")


def _kline_stale(df: pd.DataFrame) -> bool:
    """最后一根 K 线是否过旧（需拉新）。"""
    if df is None or df.empty:
        return True
    last = pd.Timestamp(df["timestamp"].iloc[-1])
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    else:
        last = last.tz_convert("UTC")
    age_min = (pd.Timestamp.now(tz="UTC") - last).total_seconds() / 60.0
    return age_min > float(cfg.MOSS_QUANT_KLINE_STALE_MINUTES)


def _factory_cache_dir() -> Optional[Path]:
    raw = (cfg.MOSS_QUANT_HL_FACTORY_CACHE or "").strip()
    if raw:
        p = Path(raw)
        return p if p.is_dir() else None
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = (
            parent
            / "moss-trade-bot-skills-main"
            / "moss-trade-bot-factory-1.0.24"
            / "scripts"
            / "data_cache"
        )
        if candidate.is_dir():
            return candidate
    return None


def _factory_csv_path(symbol: str) -> Optional[Path]:
    """匹配官方 data_cache：hyperliquid_BTCUSDC_15m_*.csv"""
    root = _factory_cache_dir()
    if not root:
        return None
    base = symbol_to_base(symbol)
    pattern = str(root / f"hyperliquid_{base}USDC_15m_*.csv")
    matches = sorted(glob(pattern))
    return Path(matches[-1]) if matches else None


def _trim_limit(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(df) <= limit:
        return df.reset_index(drop=True)
    return df.tail(limit).reset_index(drop=True)


def _load_factory_csv(symbol: str, limit: int) -> Optional[pd.DataFrame]:
    path = _factory_csv_path(symbol)
    if not path or not path.is_file():
        return None
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df.empty:
        return None
    logger.info("[moss] HL factory CSV %s (%s bars)", path.name, len(df))
    return _trim_limit(df, limit)


def fetch_hyperliquid(
    symbol: str,
    *,
    interval: Optional[str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """ccxt 拉取 Hyperliquid 永续 K 线并写入 hl_* 缓存。"""
    if ccxt is None:
        raise ImportError("ccxt required for Hyperliquid klines: pip install ccxt")
    interval = interval or cfg.MOSS_QUANT_KLINE_INTERVAL
    limit = limit or cfg.MOSS_QUANT_KLINE_LIMIT
    ccxt_sym = symbol_to_ccxt(symbol)
    exchange = ccxt.hyperliquid({"enableRateLimit": True})

    days = max(cfg.MOSS_QUANT_HL_FETCH_DAYS, int(limit * 15 / (24 * 60)) + 5)
    since = exchange.parse8601(
        (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    )
    all_rows: list = []
    batch = 5000
    pages = 0
    while pages < _CCXT_MAX_PAGES:
        pages += 1
        try:
            ohlcv = exchange.fetch_ohlcv(ccxt_sym, interval, since=since, limit=batch)
        except Exception as e:
            logger.warning("[moss] HL fetch page %s failed %s: %s", pages, ccxt_sym, e)
            break
        if not ohlcv:
            break
        all_rows.extend(ohlcv)
        last_ts = ohlcv[-1][0]
        if last_ts <= since:
            break
        since = last_ts + 1
        if len(ohlcv) < batch:
            break
        time.sleep(max(exchange.rateLimit / 1000, 0.05))

    if not all_rows:
        raise RuntimeError(f"no hyperliquid ohlcv for {symbol} ({ccxt_sym})")

    df = pd.DataFrame(
        all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = _trim_limit(df, limit)
    path = _cache_path(symbol, interval)
    df.to_csv(path, index=False)
    logger.info("[moss] HL ccxt cached %s -> %s (%s bars)", ccxt_sym, path.name, len(df))
    return df


def _bootstrap_from_factory(symbol: str, interval: str, limit: int) -> Optional[pd.DataFrame]:
    fac = _load_factory_csv(symbol, limit)
    if fac is None or fac.empty:
        return None
    fac.to_csv(_cache_path(symbol, interval), index=False)
    return fac


def load_hyperliquid_cached(
    symbol: str,
    *,
    interval: Optional[str] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    加载 HL K 线：本地缓存 →（过期或 refresh）ccxt → 工厂 CSV 兜底。
    refresh=False 时若缓存过期仍会自动拉取，避免纸面一直用陈旧 mark。
    """
    interval = interval or cfg.MOSS_QUANT_KLINE_INTERVAL
    limit = cfg.MOSS_QUANT_KLINE_LIMIT
    sym = str(symbol).strip().upper()
    path = _cache_path(sym, interval)

    cached: Optional[pd.DataFrame] = None
    if path.is_file():
        cached = pd.read_csv(path, parse_dates=["timestamp"])
        if not cached.empty and not refresh and not _kline_stale(cached):
            return _trim_limit(cached, limit)

    need_live = refresh or cached is None or cached.empty or _kline_stale(cached)
    if need_live:
        try:
            return fetch_hyperliquid(sym, interval=interval, limit=limit)
        except Exception as e:
            logger.warning("[moss] HL live fetch failed %s: %s", sym, e)
            fac = _bootstrap_from_factory(sym, interval, limit)
            if fac is not None:
                return fac
            if cached is not None and not cached.empty:
                logger.warning("[moss] HL using stale cache for %s", sym)
                return _trim_limit(cached, limit)
            raise

    fac = _bootstrap_from_factory(sym, interval, limit)
    if fac is not None:
        return fac
    return fetch_hyperliquid(sym, interval=interval, limit=limit)


def catalog_entry(symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {"symbol": symbol, "bars": 0, "data_source": "hyperliquid"}
    factory = _factory_csv_path(symbol)
    return {
        "symbol": symbol,
        "bars": len(df),
        "start": _ts_iso(df["timestamp"].iloc[0]),
        "end": _ts_iso(df["timestamp"].iloc[-1]),
        "csv_path": str(_cache_path(symbol, cfg.MOSS_QUANT_KLINE_INTERVAL)),
        "data_source": "hyperliquid",
        "hl_ccxt_symbol": symbol_to_ccxt(symbol),
        "factory_csv": str(factory) if factory else None,
        "stale": _kline_stale(df),
    }
