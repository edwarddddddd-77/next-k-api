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
MOSS_QUANT_SEGMENT_BARS = 672
# --- K 线 / 数据源（固化默认，勿用 env 覆盖；改值请直接改本文件）---
MOSS_QUANT_KLINE_INTERVAL = "15m"
MOSS_QUANT_KLINE_LIMIT = 1500
MOSS_QUANT_RESEARCH_KLINE_BARS = 6720
MOSS_QUANT_DATA_SOURCE = "binance"
MOSS_QUANT_REGIME_VERSION = "v1"
MOSS_QUANT_KLINE_STALE_MINUTES = 20
MOSS_QUANT_BINANCE_KLINE_MIN_INTERVAL_SEC = 0.4
MOSS_QUANT_DAILY_OPTIMIZE_BINANCE_REFRESH_ALL = True
MOSS_QUANT_RESEARCH_RELAX_SYMBOL_CHECK = True
MOSS_QUANT_HL_FETCH_DAYS = 75

MOSS_QUANT_MAX_ACTIVE_PROFILES = max(
    1, int(os.getenv("MOSS_QUANT_MAX_ACTIVE_PROFILES", "5") or 5)
)
MOSS_QUANT_EXTENDED_UNIVERSE = env_truthy(
    "MOSS_QUANT_EXTENDED_UNIVERSE", default=False
)

# HL：可选官方工厂 data_cache 目录；空则自动探测 moss-trade-bot-skills-main
MOSS_QUANT_HL_FACTORY_CACHE = os.getenv("MOSS_QUANT_HL_FACTORY_CACHE", "").strip()


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

# --- 寻优 / 验证 / 网格（固化默认）---
MOSS_QUANT_OPTIMIZE_TRAIN_RATIO = 0.7
MOSS_QUANT_OPTIMIZE_REQUIRE_VALIDATION = True
MOSS_QUANT_OPTIMIZE_MIN_TRAIN_TRADES = 8
MOSS_QUANT_OPTIMIZE_MIN_VAL_TRADES = 3
MOSS_QUANT_OPTIMIZE_MAX_TRAIN_DRAWDOWN = 0.35
MOSS_QUANT_OPTIMIZE_MAX_VAL_DRAWDOWN = 0.40
MOSS_QUANT_OPTIMIZE_VALIDATION_TOP_K = 5
MOSS_QUANT_OPTIMIZE_MIN_BARS = 400
MOSS_QUANT_OPTIMIZE_VAL_WARMUP_BARS = 96
MOSS_QUANT_OPTIMIZE_GATE_PROXY_ENABLED = True
MOSS_QUANT_OPTIMIZE_GATE_PENALTY_SCALE = 0.20
MOSS_QUANT_OPTIMIZE_GATE_FAIL_RATIO = 0.0
MOSS_QUANT_OPTIMIZE_FULL_RISK_SLOTS = 5
MOSS_QUANT_OPTIMIZE_REDUCED_RISK_SCALE = 0.5
MOSS_QUANT_OPTIMIZE_TRAILING_FOR_TREND = False
MOSS_QUANT_OPTIMIZE_WF_FOLDS = 3
MOSS_QUANT_OPTIMIZE_WF_MIN_PASS_FOLDS = 2
MOSS_QUANT_OPTIMIZE_STABILITY_PENALTY = 0.4
MOSS_QUANT_OPTIMIZE_MAX_TRAIN_VAL_RATIO = 3.0
MOSS_QUANT_OPTIMIZE_MIN_TRAIN_VAL_RATIO = 0.3
MOSS_QUANT_OPTIMIZE_REGIME_FILTER_TEMPLATES = True
MOSS_QUANT_OPTIMIZE_MAX_COMBINATIONS = 72
MOSS_QUANT_OPTIMIZE_ENTRY_THRESHOLDS = (0.40, 0.44, 0.48)
MOSS_QUANT_OPTIMIZE_SL_ATR_MULTS = (2.0, 2.5)
MOSS_QUANT_OPTIMIZE_TP_RR_RATIOS = (2.0, 2.5, 3.0)
MOSS_QUANT_OPTIMIZE_API_TOP_N = 15

# --- 组合风控（纸面开仓前）---
MOSS_QUANT_PORTFOLIO_RISK_ENABLED = True
MOSS_QUANT_PORTFOLIO_MAX_SAME_SIDE_PCT = 0.8
MOSS_QUANT_PORTFOLIO_CORR_THRESHOLD = 0.7
MOSS_QUANT_PORTFOLIO_CORR_RISK_SCALE = 0.5
MOSS_QUANT_PORTFOLIO_CORR_LOOKBACK_BARS = 96

# --- 开仓 gate ---
MOSS_QUANT_GATE_FUNDING_EXTREME = True
MOSS_QUANT_GATE_FUNDING_ABS_MAX = 0.001
MOSS_QUANT_GATE_FUNDING_BUMP = 0.03
MOSS_QUANT_GATE_HARD_BLOCK = False
MOSS_QUANT_GATE_BLOCK_BUMP = 0.08
MOSS_QUANT_GATE_OI_SPIKE = True
MOSS_QUANT_GATE_OI_D6H_MIN = 2.0
MOSS_QUANT_GATE_OI_PX_FLAT_MAX = 5.0
MOSS_QUANT_GATE_OI_BUMP = 0.03

# --- 同步门控 ---
MOSS_QUANT_SYNC_BLOCK_RECENT_LOSS_ENABLED = True
MOSS_QUANT_SYNC_BLOCK_LOSS_DAYS = 7
MOSS_QUANT_SYNC_BLOCK_LOSS_PCT = -0.05

# --- 日内门槛微调 ---
MOSS_QUANT_INTRADAY_ADJUST_ENABLED = True
MOSS_QUANT_INTRADAY_DRAWDOWN_BUMP = 0.03
MOSS_QUANT_INTRADAY_DRAWDOWN_PCT = 0.05
MOSS_QUANT_INTRADAY_PROFIT_BUMP = 0.02
MOSS_QUANT_INTRADAY_PROFIT_PCT = 0.08

# --- 纸面扫描：训练窗 regime 与当前 regime 一致时，单侧放宽开仓阈值 ---
MOSS_QUANT_REGIME_ALIGN_ADJUST_ENABLED = True
# 一致时 favored 侧阈值下调比例（如 0.08 → 0.52 变为 0.478）
MOSS_QUANT_REGIME_ALIGN_RELAX_PCT = 0.08
# 背离时两侧阈值上调比例
MOSS_QUANT_REGIME_ALIGN_TIGHTEN_PCT = 0.04
# 震荡市 + mean_revert/balanced 时两侧对称小幅放宽
MOSS_QUANT_REGIME_ALIGN_SIDEWAYS_RELAX_PCT = 0.04

# 首次部署（尚无 daily_auto Profile）时自动跑一次全市场寻优
MOSS_QUANT_DAILY_OPTIMIZE_BOOTSTRAP = env_truthy(
    "MOSS_QUANT_DAILY_OPTIMIZE_BOOTSTRAP", default=False
)
# 启动后延迟再跑 bootstrap，避免与首屏 API 抢库（秒）
MOSS_QUANT_DAILY_OPTIMIZE_BOOTSTRAP_DELAY_SEC = max(
    60,
    int(os.getenv("MOSS_QUANT_DAILY_OPTIMIZE_BOOTSTRAP_DELAY_SEC", "1200") or 1200),
)

# --- 纸面池子治理（完全自动化；防抖：启用/停用均看连续 N 日寻优批次）---
MOSS_QUANT_POOL_GOVERNANCE_ENABLED = True
MOSS_QUANT_POOL_AUTO_DISABLE = True
MOSS_QUANT_POOL_AUTO_ENABLE = True
# B 池连续 N 个每日批次 → 停用（2=至少 2 天观察池不佳，避免单日抖动）
MOSS_QUANT_POOL_DEGRADE_STREAK_B = 2
# C 池连续 N 个每日批次 → 停用（剔除档 1 天即停）
MOSS_QUANT_POOL_DEGRADE_STREAK_C = 1
# A 池且可同步连续 N 个每日批次 → 才自动启用/补位（2=连续 2 天达标）
MOSS_QUANT_POOL_UPGRADE_STREAK = 2
MOSS_QUANT_POOL_AUTO_ADD_TOP_N = 5
MOSS_QUANT_POOL_MAX_AUTO_ENABLED = MOSS_QUANT_MAX_ACTIVE_PROFILES
MOSS_QUANT_POOL_RESPECT_MANUAL_DISABLE = True
# 已启用 Profile 近 N 日纸面亏损超阈值 → 立即停用（与 SYNC_BLOCK 同口径）
MOSS_QUANT_POOL_AUTO_DISABLE_ON_PAPER_LOSS = True


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
