"""线上自动拉取 factory-en 风格 CSV 到 Moss2 自有 data_cache（不依赖 skills 目录）。"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from moss2 import config as cfg
from moss2.config import MOSS2_SEED_BASES, base_to_fetch_slash, en_data_cache_dir

logger = logging.getLogger(__name__)

BOOTSTRAP_MARKER_NAME = ".moss2_bootstrap_last.json"


def bootstrap_marker_path(cache_dir: Optional[Path] = None) -> Path:
    return (cache_dir or en_data_cache_dir()) / BOOTSTRAP_MARKER_NAME


def seed_csv_missing_count(cache_dir: Optional[Path] = None) -> int:
    """MOSS2_SEED_BASES 中尚无有效 CSV 的数量。"""
    root = cache_dir or en_data_cache_dir()
    n = 0
    for base in MOSS2_SEED_BASES:
        path = canonical_csv_path(root, base)
        if not path.is_file() or path.stat().st_size < 100:
            n += 1
    return n


def read_bootstrap_marker(cache_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    path = bootstrap_marker_path(cache_dir)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_bootstrap_marker(stats: Dict[str, Any], cache_dir: Optional[Path] = None) -> None:
    if not stats.get("ran_at_utc"):
        return
    if int(stats.get("saved") or 0) + int(stats.get("skipped") or 0) <= 0:
        return
    path = bootstrap_marker_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ran_at_utc": stats["ran_at_utc"],
        "ok": bool(stats.get("ok")),
        "saved": int(stats.get("saved") or 0),
        "skipped": int(stats.get("skipped") or 0),
        "failed": int(stats.get("failed") or 0),
        "bases": int(stats.get("bases") or 0),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def startup_bootstrap_needed(*, force: bool = False) -> Tuple[bool, str]:
    """
    启动时是否值得跑全量 seed bootstrap：
    - 有缺失 CSV，或从未成功写过 marker → 需要
    - 25 个 seed 均已就绪且已有 marker → 跳过（周任务负责刷新 stale）
    """
    if force:
        return True, "force"
    cache = en_data_cache_dir()
    missing = seed_csv_missing_count(cache)
    if missing > 0:
        return True, f"missing_csv:{missing}"
    if read_bootstrap_marker(cache) is None:
        return True, "no_marker"
    return False, "seed_cache_ready"


def canonical_csv_path(
    cache_dir: Path, base: str, *, since: Optional[str] = None, days: Optional[int] = None
) -> Path:
    b = str(base or "").strip().upper().replace("USDT", "")
    since = since or cfg.MOSS2_FETCH_SINCE
    days = int(days or cfg.MOSS2_FETCH_DAYS)
    return cache_dir / f"binanceusdm_{b}_USDT_USDT_15m_{since}_{days}d.csv"


def _csv_fresh(path: Path, *, max_age_hours: float) -> bool:
    if not path.is_file() or path.stat().st_size < 100:
        return False
    age_h = (time.time() - path.stat().st_mtime) / 3600.0
    return age_h <= max_age_hours


def fetch_base_to_cache(
    base: str,
    cache_dir: Optional[Path] = None,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    """从币安 U 本位拉 15m K 线并写入 Moss2 data_cache。"""
    cache_dir = cache_dir or en_data_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    b = str(base).strip().upper().replace("USDT", "")
    out = canonical_csv_path(cache_dir, b)
    fetch_sym = base_to_fetch_slash(b)
    row: Dict[str, Any] = {"base": b, "fetch_symbol": fetch_sym, "path": str(out)}

    if not force and _csv_fresh(out, max_age_hours=cfg.MOSS2_DATA_BOOTSTRAP_STALE_HOURS):
        row["status"] = "skipped"
        row["reason"] = "fresh"
        return row

    try:
        from moss2.variants.en.core.fetcher import fetch_ohlcv

        df = fetch_ohlcv(
            fetch_sym,
            timeframe=cfg.MOSS2_FETCH_TIMEFRAME,
            days=cfg.MOSS2_FETCH_DAYS,
            since_date=cfg.MOSS2_FETCH_SINCE,
            use_cache=False,
        )
        if df is None or df.empty:
            row["status"] = "failed"
            row["reason"] = "empty_df"
            return row
        df.to_csv(out, index=False)
        row["status"] = "saved"
        row["bars"] = int(len(df))
        row["start"] = str(df["timestamp"].iloc[0])
        row["end"] = str(df["timestamp"].iloc[-1])
        logger.info("[moss2] data_cache saved %s bars=%s", out.name, row["bars"])
        return row
    except Exception as e:
        logger.warning("[moss2] data_cache fetch %s failed: %s", b, e)
        row["status"] = "failed"
        row["reason"] = str(e)
        return row


def bootstrap_seed_data(
    *,
    bases: Optional[List[str]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """拉取 MOSS2_SEED_BASES（默认 25 核心）到 en_data_cache_dir。"""
    if not cfg.MOSS2_DATA_BOOTSTRAP_ENABLED:
        return {"ok": False, "reason": "bootstrap_disabled"}

    cache_dir = en_data_cache_dir()
    bases = list(bases or MOSS2_SEED_BASES)
    results: List[Dict[str, Any]] = []
    saved = skipped = failed = 0

    for i, base in enumerate(bases, 1):
        row = fetch_base_to_cache(base, cache_dir, force=force)
        results.append(row)
        st = row.get("status")
        if st == "saved":
            saved += 1
        elif st == "skipped":
            skipped += 1
        else:
            failed += 1
        if i < len(bases) and cfg.MOSS2_DATA_BOOTSTRAP_SLEEP_SEC > 0:
            time.sleep(cfg.MOSS2_DATA_BOOTSTRAP_SLEEP_SEC)

    stats = {
        "ok": failed == 0 and (saved > 0 or skipped > 0),
        "lane": "moss2",
        "cache_dir": str(cache_dir),
        "bases": len(bases),
        "saved": saved,
        "skipped": skipped,
        "failed": failed,
        "results": results,
        "ran_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    write_bootstrap_marker(stats, cache_dir)
    return stats
