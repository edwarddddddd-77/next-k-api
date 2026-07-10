#!/usr/bin/env python3
"""ORB K 线本地缓存：data/orb/kline/<SYMBOL>/（读写分离 legacy output 回退）。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from quant.common.paths import (
    OUTPUT_LEGACY_ROOT,
    PROJECT_ROOT,
    ensure_kline_dirs,
    resolve_kline_cache_root,
)

COLUMNS = ("open_time", "open", "high", "low", "close", "volume")


def default_cache_root() -> Path:
    return resolve_kline_cache_root()


def norm_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    return s if s.endswith("USDT") else s + "USDT"


def symbol_label(symbol: str) -> str:
    return norm_symbol(symbol).replace("USDT", "")


def symbol_cache_dir(symbol: str) -> Path:
    return default_cache_root() / symbol_label(symbol)


def legacy_symbol_cache_dir(symbol: str) -> Path:
    return OUTPUT_LEGACY_ROOT / symbol_label(symbol)


def kline_path(symbol: str, interval: str) -> Path:
    return symbol_cache_dir(symbol) / f"{interval.strip().lower()}.csv"


def legacy_kline_path(symbol: str, interval: str) -> Path:
    return legacy_symbol_cache_dir(symbol) / f"{interval.strip().lower()}.csv"


def meta_path(symbol: str) -> Path:
    return symbol_cache_dir(symbol) / "meta.json"


def resolve_kline_read_path(symbol: str, interval: str) -> Optional[Path]:
    primary = kline_path(symbol, interval)
    if primary.is_file():
        return primary
    legacy = legacy_kline_path(symbol, interval)
    if legacy.is_file():
        return legacy
    return None


def has_kline_cache(symbol: str, interval: str = "5m") -> bool:
    return resolve_kline_read_path(symbol, interval) is not None


def load_klines(
    symbol: str,
    interval: str,
    *,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
) -> pd.DataFrame:
    path = resolve_kline_read_path(symbol, interval)
    if path is None:
        return pd.DataFrame(columns=list(COLUMNS))
    if path.stat().st_size < 8:
        return pd.DataFrame(columns=list(COLUMNS))
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=list(COLUMNS))
    if df.empty:
        return df
    for c in COLUMNS:
        if c in df.columns:
            if c == "open_time":
                df[c] = df[c].astype("int64")
            else:
                df[c] = df[c].astype(float)
    df = df.drop_duplicates(subset=["open_time"], keep="last").sort_values("open_time").reset_index(drop=True)
    if start_ms is not None:
        df = df[df["open_time"] >= int(start_ms)]
    if end_ms is not None:
        df = df[df["open_time"] <= int(end_ms)]
    return df.reset_index(drop=True)


def save_klines(symbol: str, interval: str, df: pd.DataFrame) -> Path:
    ensure_kline_dirs()
    out = kline_path(symbol, interval)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in COLUMNS if c in df.columns]
    df[cols].to_csv(out, index=False)
    return out


def write_meta(symbol: str, *, days: float, intervals: list[str]) -> None:
    sym = norm_symbol(symbol)
    label = symbol_label(sym)
    ensure_kline_dirs()
    d = symbol_cache_dir(sym)
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "symbol": label,
        "binance_symbol": sym,
        "cache_root": str(default_cache_root()),
        "updated_at_ms": int(time.time() * 1000),
        "days": days,
        "intervals": intervals,
        "files": {
            iv: str(kline_path(sym, iv).relative_to(d)).replace("\\", "/") for iv in intervals
        },
    }
    meta_path(sym).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def session_dates_from_cache(symbol: str, cfg) -> list[str]:
    """K 线缓存中的 session_date（NYSE 交易日过滤）。"""
    from quant.common.config import OrbConfig
    from quant.common.session import session_day_str
    from quant.common.us_equity_calendar import is_us_equity_market, is_us_equity_trading_day

    if not isinstance(cfg, OrbConfig):
        cfg = OrbConfig.from_env()
    sym = norm_symbol(symbol)
    df = load_klines(sym, cfg.signal_interval)
    if df.empty:
        return []
    tz = cfg.session_tz
    open_time = cfg.session_open_time
    dates = {
        session_day_str(int(t), tz=tz, session_open_time=open_time)
        for t in df["open_time"].astype("int64")
    }
    out = sorted(d for d in dates if d)
    if is_us_equity_market(cfg.market):
        out = [d for d in out if is_us_equity_trading_day(d)]
    return out
