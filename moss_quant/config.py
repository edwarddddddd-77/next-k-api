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

# K 线数据源：hyperliquid（默认，与官方工厂一致）| binance（币安 U 本位）
MOSS_QUANT_DATA_SOURCE = (
    os.getenv("MOSS_QUANT_DATA_SOURCE", "hyperliquid") or "hyperliquid"
).strip().lower()
if MOSS_QUANT_DATA_SOURCE not in ("binance", "hyperliquid"):
    MOSS_QUANT_DATA_SOURCE = "hyperliquid"

# HL：可选官方工厂 data_cache 目录；空则自动探测 moss-trade-bot-skills-main
MOSS_QUANT_HL_FACTORY_CACHE = os.getenv("MOSS_QUANT_HL_FACTORY_CACHE", "").strip()
MOSS_QUANT_HL_FETCH_DAYS = max(
    7, int(os.getenv("MOSS_QUANT_HL_FETCH_DAYS", "45") or 45)
)
# 缓存最后一根 K 线超过该分钟数则纸面/扫描自动尝试 ccxt 更新（refresh=False 时也生效）
MOSS_QUANT_KLINE_STALE_MINUTES = max(
    5, int(os.getenv("MOSS_QUANT_KLINE_STALE_MINUTES", "20") or 20)
)


def data_source_label() -> str:
    return "Hyperliquid" if MOSS_QUANT_DATA_SOURCE == "hyperliquid" else "Binance USDT"


def _default_cache_dir() -> Path:
    data_dir = os.getenv("DATA_DIR", "").strip()
    if data_dir:
        return Path(data_dir) / "moss_quant_cache"
    return Path(__file__).resolve().parent.parent / "data" / "moss_quant_cache"


_CACHE_ROOT = Path(os.getenv("MOSS_QUANT_CACHE_DIR", "") or _default_cache_dir())
MOSS_QUANT_CACHE_DIR = _CACHE_ROOT

# LLM 进化
MOSS_QUANT_LLM_ENABLED = env_truthy("MOSS_QUANT_LLM_ENABLED", default=True)
def _resolve_llm_provider() -> str:
    explicit = (os.getenv("MOSS_QUANT_LLM_PROVIDER") or "").strip().lower()
    if explicit:
        return explicit
    if (os.getenv("GROQ_API_KEY") or "").strip():
        return "groq"
    if (os.getenv("ANTHROPIC_API_KEY") or "").strip():
        return "anthropic"
    return "groq"  # 未配 Key 时默认 groq，reflect 会提示 GROQ_API_KEY


MOSS_QUANT_LLM_PROVIDER = _resolve_llm_provider()

# 纸面扫描：每 profile 打印 composite / 阈值 / 持仓 SL·TP 距离（Railway 日志）
MOSS_QUANT_VERBOSE_LOG = env_truthy("MOSS_QUANT_VERBOSE_LOG", default=True)


def paper_scheduler_enabled() -> bool:
    return (
        MOSS_QUANT_ENABLED
        and MOSS_QUANT_PAPER_ENABLED
        and MOSS_QUANT_SCHEDULER_ENABLED
    )
