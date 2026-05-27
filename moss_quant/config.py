"""Moss 量化 lane — env 配置。"""

from __future__ import annotations

import os
from pathlib import Path


def env_truthy(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


MOSS_QUANT_ENABLED = env_truthy("MOSS_QUANT_ENABLED", default=True)
MOSS_QUANT_PAPER_ENABLED = env_truthy("MOSS_QUANT_PAPER_ENABLED", default=True)
MOSS_QUANT_SCHEDULER_ENABLED = env_truthy("MOSS_QUANT_SCHEDULER_ENABLED", default=True)

MOSS_QUANT_SCAN_INTERVAL_MINUTES = max(
    1, int(os.getenv("MOSS_QUANT_SCAN_INTERVAL_MINUTES", "15") or 15)
)
MOSS_QUANT_DEFAULT_CAPITAL = max(
    100.0, float(os.getenv("MOSS_QUANT_DEFAULT_CAPITAL", "10000") or 10000)
)
MOSS_QUANT_SEGMENT_BARS = max(
    96, int(os.getenv("MOSS_QUANT_SEGMENT_BARS", "672") or 672)
)
MOSS_QUANT_KLINE_INTERVAL = (
    os.getenv("MOSS_QUANT_KLINE_INTERVAL", "15m") or "15m"
).strip()
MOSS_QUANT_KLINE_LIMIT = max(
    200, int(os.getenv("MOSS_QUANT_KLINE_LIMIT", "1500") or 1500)
)
MOSS_QUANT_MAX_ACTIVE_PROFILES = max(
    1, int(os.getenv("MOSS_QUANT_MAX_ACTIVE_PROFILES", "5") or 5)
)
MOSS_QUANT_REGIME_VERSION = (
    os.getenv("MOSS_QUANT_REGIME_VERSION", "v1") or "v1"
).strip()

def _default_cache_dir() -> Path:
    data_dir = os.getenv("DATA_DIR", "").strip()
    if data_dir:
        return Path(data_dir) / "moss_quant_cache"
    return Path(__file__).resolve().parent.parent / "data" / "moss_quant_cache"


_CACHE_ROOT = Path(os.getenv("MOSS_QUANT_CACHE_DIR", "") or _default_cache_dir())
MOSS_QUANT_CACHE_DIR = _CACHE_ROOT

# LLM 进化
MOSS_QUANT_LLM_ENABLED = env_truthy("MOSS_QUANT_LLM_ENABLED", default=True)
MOSS_QUANT_LLM_PROVIDER = (
    os.getenv("MOSS_QUANT_LLM_PROVIDER", "anthropic") or "anthropic"
).strip().lower()


def paper_scheduler_enabled() -> bool:
    return (
        MOSS_QUANT_ENABLED
        and MOSS_QUANT_PAPER_ENABLED
        and MOSS_QUANT_SCHEDULER_ENABLED
    )
