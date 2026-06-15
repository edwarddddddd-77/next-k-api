#!/usr/bin/env python3
"""ORB K 线本地缓存：data/orb/kline/<SYMBOL>/（读写分离 legacy output 回退）。"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from orb.data.paths import KLINE_ROOT, OUTPUT_LEGACY_ROOT, ensure_kline_dirs

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COLUMNS = ("open_time", "open", "high", "low", "close", "volume")


def default_cache_root() -> Path:
    raw = (os.getenv("ORB_KLINE_CACHE_ROOT") or "").strip()
    return Path(raw) if raw else KLINE_ROOT


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
    df = pd.read_csv(path)
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
