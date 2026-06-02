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
MOSS_QUANT_PROFILE_CAPITAL = max(
    100.0,
    float(
        os.getenv(
            "MOSS_QUANT_PROFILE_CAPITAL",
            os.getenv("MOSS_QUANT_DEFAULT_CAPITAL", "1000") or 1000,
        )
        or 1000
    ),
)
# 已废弃：全局纸面初始 = Profile 数 × MOSS_QUANT_PROFILE_CAPITAL（见 db.aggregate_moss_wallet_initial）
MOSS_QUANT_WALLET_INITIAL = MOSS_QUANT_PROFILE_CAPITAL
# 回测 / 寻优 / 单 bot 纸面 sizing 本金（与 MOSS_QUANT_PROFILE_CAPITAL 相同）
MOSS_QUANT_DEFAULT_CAPITAL = MOSS_QUANT_PROFILE_CAPITAL
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
MOSS_QUANT_EXTENDED_UNIVERSE = env_truthy(
    "MOSS_QUANT_EXTENDED_UNIVERSE", default=False
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
# 币安 fapi K 线最小请求间隔（秒）；limit=1500 单次权重约 10，43 币全刷约 430
MOSS_QUANT_BINANCE_KLINE_MIN_INTERVAL_SEC = max(
    0.0,
    float(os.getenv("MOSS_QUANT_BINANCE_KLINE_MIN_INTERVAL_SEC", "0.4") or 0.4),
)
# 每日寻优在 binance 源下是否每个标的都 refresh（默认仅第一个，省权重）
MOSS_QUANT_DAILY_OPTIMIZE_BINANCE_REFRESH_ALL = env_truthy(
    "MOSS_QUANT_DAILY_OPTIMIZE_BINANCE_REFRESH_ALL", default=False
)
# 回测/寻优：默认接受任意 XXXUSDT 格式；设为 0 则强制校验币安永续 TRADING 名单
MOSS_QUANT_RESEARCH_RELAX_SYMBOL_CHECK = env_truthy(
    "MOSS_QUANT_RESEARCH_RELAX_SYMBOL_CHECK", default=True
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
MOSS_QUANT_REAL_MODE = env_truthy("MOSS_QUANT_REAL_MODE", default=True)
# 纸面 moss_signals 为持仓真源；开/平仅通知 Protocol ingest
MOSS_QUANT_PAPER_SOURCE_OF_TRUTH = env_truthy(
    "MOSS_QUANT_PAPER_SOURCE_OF_TRUTH", default=True
)
MOSS_QUANT_VERBOSE_LOG = env_truthy("MOSS_QUANT_VERBOSE_LOG", default=True)


def paper_scheduler_enabled() -> bool:
    return (
        MOSS_QUANT_ENABLED
        and MOSS_QUANT_PAPER_ENABLED
        and MOSS_QUANT_SCHEDULER_ENABLED
    )


MOSS_QUANT_DAILY_OPTIMIZE_ENABLED = env_truthy(
    "MOSS_QUANT_DAILY_OPTIMIZE_ENABLED", default=True
)
# UTC 每日执行时刻，格式 HH:MM
MOSS_QUANT_DAILY_OPTIMIZE_UTC = (
    os.getenv("MOSS_QUANT_DAILY_OPTIMIZE_UTC", "06:30") or "06:30"
).strip()
MOSS_QUANT_DAILY_OPTIMIZE_REFRESH = env_truthy(
    "MOSS_QUANT_DAILY_OPTIMIZE_REFRESH", default=True
)
MOSS_QUANT_DAILY_OPTIMIZE_APPLY_PROFILES = env_truthy(
    "MOSS_QUANT_DAILY_OPTIMIZE_APPLY_PROFILES", default=True
)

# --- 寻优：训练/验证窗、评分、同步门控 ---
MOSS_QUANT_OPTIMIZE_TRAIN_RATIO = max(
    0.5, min(0.85, float(os.getenv("MOSS_QUANT_OPTIMIZE_TRAIN_RATIO", "0.7") or 0.7))
)
MOSS_QUANT_OPTIMIZE_REQUIRE_VALIDATION = env_truthy(
    "MOSS_QUANT_OPTIMIZE_REQUIRE_VALIDATION", default=True
)
MOSS_QUANT_OPTIMIZE_MIN_TRAIN_TRADES = max(
    1, int(os.getenv("MOSS_QUANT_OPTIMIZE_MIN_TRAIN_TRADES", "8") or 8)
)
MOSS_QUANT_OPTIMIZE_MIN_VAL_TRADES = max(
    1, int(os.getenv("MOSS_QUANT_OPTIMIZE_MIN_VAL_TRADES", "3") or 3)
)
MOSS_QUANT_OPTIMIZE_MAX_TRAIN_DRAWDOWN = max(
    0.05,
    min(0.95, float(os.getenv("MOSS_QUANT_OPTIMIZE_MAX_TRAIN_DRAWDOWN", "0.35") or 0.35)),
)
MOSS_QUANT_OPTIMIZE_MAX_VAL_DRAWDOWN = max(
    0.05,
    min(0.95, float(os.getenv("MOSS_QUANT_OPTIMIZE_MAX_VAL_DRAWDOWN", "0.40") or 0.40)),
)
MOSS_QUANT_OPTIMIZE_VALIDATION_TOP_K = max(
    1, int(os.getenv("MOSS_QUANT_OPTIMIZE_VALIDATION_TOP_K", "5") or 5)
)
MOSS_QUANT_OPTIMIZE_MIN_BARS = max(
    200, int(os.getenv("MOSS_QUANT_OPTIMIZE_MIN_BARS", "400") or 400)
)
MOSS_QUANT_OPTIMIZE_FULL_RISK_SLOTS = max(
    1, int(os.getenv("MOSS_QUANT_OPTIMIZE_FULL_RISK_SLOTS", "5") or 5)
)
MOSS_QUANT_OPTIMIZE_REDUCED_RISK_SCALE = max(
    0.1,
    min(1.0, float(os.getenv("MOSS_QUANT_OPTIMIZE_REDUCED_RISK_SCALE", "0.5") or 0.5)),
)
MOSS_QUANT_OPTIMIZE_TRAILING_FOR_TREND = env_truthy(
    "MOSS_QUANT_OPTIMIZE_TRAILING_FOR_TREND", default=False
)
MOSS_QUANT_MCAP_OBSERVATION_DAYS = max(
    1, int(os.getenv("MOSS_QUANT_MCAP_OBSERVATION_DAYS", "3") or 3)
)

# 市值扩展寻优：币安市值池大小、展示 Top N
MOSS_QUANT_MCAP_SCAN_POOL_LIMIT = max(
    10, int(os.getenv("MOSS_QUANT_MCAP_SCAN_POOL_LIMIT", "100") or 100)
)
MOSS_QUANT_MCAP_SCAN_REFRESH = env_truthy(
    "MOSS_QUANT_MCAP_SCAN_REFRESH", default=True
)
# 首次部署（尚无 daily_auto Profile）时自动跑一次全市场寻优
MOSS_QUANT_DAILY_OPTIMIZE_BOOTSTRAP = env_truthy(
    "MOSS_QUANT_DAILY_OPTIMIZE_BOOTSTRAP", default=False
)
# 启动后延迟再跑 bootstrap，避免与首屏 API 抢库（秒）
MOSS_QUANT_DAILY_OPTIMIZE_BOOTSTRAP_DELAY_SEC = max(
    60,
    int(os.getenv("MOSS_QUANT_DAILY_OPTIMIZE_BOOTSTRAP_DELAY_SEC", "1200") or 1200),
)

# --- 纸面池子治理：每日寻优后自动停/启 Profile（防抖 + TopN 补位）---
MOSS_QUANT_POOL_GOVERNANCE_ENABLED = env_truthy(
    "MOSS_QUANT_POOL_GOVERNANCE_ENABLED", default=True
)
MOSS_QUANT_POOL_AUTO_DISABLE = env_truthy("MOSS_QUANT_POOL_AUTO_DISABLE", default=True)
MOSS_QUANT_POOL_AUTO_ENABLE = env_truthy("MOSS_QUANT_POOL_AUTO_ENABLE", default=True)
MOSS_QUANT_POOL_DEGRADE_STREAK_B = max(
    1, int(os.getenv("MOSS_QUANT_POOL_DEGRADE_STREAK_B", "2") or 2)
)
MOSS_QUANT_POOL_DEGRADE_STREAK_C = max(
    1, int(os.getenv("MOSS_QUANT_POOL_DEGRADE_STREAK_C", "1") or 1)
)
MOSS_QUANT_POOL_UPGRADE_STREAK = max(
    1, int(os.getenv("MOSS_QUANT_POOL_UPGRADE_STREAK", "2") or 2)
)
MOSS_QUANT_POOL_AUTO_ADD_TOP_N = max(
    1, int(os.getenv("MOSS_QUANT_POOL_AUTO_ADD_TOP_N", "5") or 5)
)
MOSS_QUANT_POOL_MAX_AUTO_ENABLED = max(
    1,
    int(
        os.getenv(
            "MOSS_QUANT_POOL_MAX_AUTO_ENABLED",
            os.getenv("MOSS_QUANT_MAX_ACTIVE_PROFILES", "5") or 5,
        )
        or 5
    ),
)
MOSS_QUANT_POOL_RESPECT_MANUAL_DISABLE = env_truthy(
    "MOSS_QUANT_POOL_RESPECT_MANUAL_DISABLE", default=True
)


def pool_governance_enabled() -> bool:
    return MOSS_QUANT_POOL_GOVERNANCE_ENABLED and MOSS_QUANT_DAILY_OPTIMIZE_APPLY_PROFILES


def daily_optimize_scheduler_enabled() -> bool:
    return (
        MOSS_QUANT_ENABLED
        and MOSS_QUANT_DAILY_OPTIMIZE_ENABLED
        and MOSS_QUANT_SCHEDULER_ENABLED
    )


def daily_optimize_bootstrap_enabled() -> bool:
    return (
        daily_optimize_scheduler_enabled()
        and MOSS_QUANT_DAILY_OPTIMIZE_BOOTSTRAP
        and MOSS_QUANT_DAILY_OPTIMIZE_APPLY_PROFILES
    )


def parse_daily_optimize_utc() -> tuple[int, int]:
    raw = MOSS_QUANT_DAILY_OPTIMIZE_UTC.replace("：", ":")
    parts = raw.split(":")
    hour = int(parts[0]) if parts else 6
    minute = int(parts[1]) if len(parts) > 1 else 30
    return max(0, min(23, hour)), max(0, min(59, minute))
