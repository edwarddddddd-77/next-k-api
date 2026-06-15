"""ORB universe K 线拉取（供月度训练 / 定时刷新）。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from orb.core.config import OrbConfig
from orb.core.kline_cache import has_kline_cache, norm_symbol, save_klines, symbol_cache_dir, symbol_label, write_meta
from orb.core.session import extended_fetch_anchor_ms
from orb.ml.samples import parse_symbol_list
from orb.core.backtest import _load_range


def fetch_symbol(
    sym: str,
    *,
    days: float,
    intervals: List[str],
    cfg: OrbConfig,
    end_ms: int,
    fetch_start: int,
) -> dict:
    label = symbol_label(sym)
    out_dir = symbol_cache_dir(sym)
    out_dir.mkdir(parents=True, exist_ok=True)
    sym_summary: dict = {"symbol": label, "dir": str(out_dir), "intervals": {}}

    for iv in intervals:
        iv_start = fetch_start
        if iv == "1d" and (cfg.sl_mode or "").strip().lower() == "atr_pct":
            iv_start = fetch_start - cfg.daily_atr_warmup_ms()
        t1 = time.time()
        df = _load_range(sym, iv, iv_start, end_ms)
        path = save_klines(sym, iv, df)
        sym_summary["intervals"][iv] = {
            "rows": len(df),
            "path": path.name,
            "elapsed_sec": round(time.time() - t1, 1),
        }

    write_meta(sym, days=days, intervals=intervals)
    return sym_summary


def fetch_universe_klines(
    *,
    symbols_file: Path,
    days: float = 180.0,
    intervals: Optional[List[str]] = None,
    skip_existing: bool = False,
    cfg: Optional[OrbConfig] = None,
) -> Dict[str, Any]:
    """拉取 universe 全部标的 K 线到 data/orb/kline/。"""
    syms = parse_symbol_list(symbols_file.read_text(encoding="utf-8"))
    if not syms:
        raise ValueError(f"empty symbols file: {symbols_file}")

    ivs = intervals or ["5m", "1m", "1d"]
    c = cfg or OrbConfig.from_env()
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(float(days) * 86_400_000)
    bar_step = c.bar_step_ms()
    fetch_start = extended_fetch_anchor_ms(start_ms, c) - bar_step * 96

    t0 = time.time()
    summary: Dict[str, Any] = {
        "symbols_file": str(symbols_file),
        "days": float(days),
        "intervals": ivs,
        "symbols": {},
        "skipped": [],
        "errors": [],
    }
    for sym in syms:
        sym = norm_symbol(sym)
        label = symbol_label(sym)
        if skip_existing and all(has_kline_cache(sym, iv) for iv in ivs):
            summary["skipped"].append(label)
            continue
        try:
            summary["symbols"][label] = fetch_symbol(
                sym,
                days=float(days),
                intervals=ivs,
                cfg=c,
                end_ms=end_ms,
                fetch_start=fetch_start,
            )
        except Exception as exc:
            summary["errors"].append({"symbol": label, "error": str(exc)})

    summary["elapsed_sec"] = round(time.time() - t0, 1)
    summary["fetched"] = len(summary["symbols"])
    summary["ok"] = not summary["errors"]
    return summary
