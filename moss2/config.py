"""Moss2 lane 配置（与 MOSS_QUANT_* 完全独立）。

运行时开关与数值默认在本文件；未设 env 时均为下列默认值，仅显式 MOSS2_*=0 可关。
改行为请直接改本文件，勿在 .env 堆参数。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Literal

from moss_lane import lane_allows_moss2, moss_lane_snapshot

FactoryVariant = Literal["hl", "en"]


def env_truthy(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name, "")
    if not str(raw).strip():
        return default
    low = str(raw).strip().lower()
    if low in ("0", "false", "no", "off"):
        return False
    return low in ("1", "true", "yes", "on")


# --- 总开关（默认全开；仅 MOSS2_*=0 显式关闭）---
MOSS2_ENABLED = env_truthy("MOSS2_ENABLED", default=True)
MOSS2_PAPER_ENABLED = env_truthy("MOSS2_PAPER_ENABLED", default=True)
MOSS2_SCHEDULER_ENABLED = env_truthy("MOSS2_SCHEDULER_ENABLED", default=True)

# --- 固化默认（勿用 env 覆盖）---
MOSS2_SCAN_INTERVAL_MINUTES = 15
MOSS2_PROFILE_CAPITAL = 10_000.0
MOSS2_DEFAULT_CAPITAL = MOSS2_PROFILE_CAPITAL
MOSS2_KLINE_LIMIT = 1500
MOSS2_REGIME_VERSION = "v1"
# 运维单 lane：Protocol 接币安 U 本位 → factory-en（hl 代码保留，默认不开放）
MOSS2_OPS_VARIANT: FactoryVariant = "en"
MOSS2_PROTOCOL_VENUE = "binance"
MOSS2_HL_ENABLED = False
MOSS2_DEFAULT_VARIANT: FactoryVariant = MOSS2_OPS_VARIANT
MOSS2_DEFAULT_TEMPLATE = "balanced"
MOSS2_VERBOSE_LOG = True
# QuantStats HTML tearsheet（回测 / 纸面结算）
MOSS2_QUANTSTATS_ENABLED = env_truthy("MOSS2_QUANTSTATS_ENABLED", default=True)
MOSS2_QUANTSTATS_DEFAULT_BENCHMARK = "BTCUSDT"

# 实盘 / Protocol（lane=moss2 时发信号；无 PROTOCOL_API_URL 时 sender 自动跳过）
MOSS2_REAL_MODE = lane_allows_moss2()
# 纸面 moss2_signals 为持仓真源；开/平仅通知 Protocol（失败不阻断纸面）
MOSS2_PAPER_SOURCE_OF_TRUTH = env_truthy("MOSS2_PAPER_SOURCE_OF_TRUTH", default=True)
MOSS2_LIVE_KLINES_ENABLED = True
MOSS2_KLINE_STALE_MINUTES = 20

# 纪律实验室（L1）
MOSS2_DISCIPLINE_ENABLED = True
MOSS2_PAPER_LOG_MARGIN = True
MOSS2_REGIME_SNOW_ENABLED = False
MOSS2_REGIME_SNOW_NOTIONAL_SCALE = 0.5
MOSS2_REGIME_SNOW_REGIMES = ("BEAR", "CRISIS")
MOSS2_DISCIPLINE_BLOCK_EV = True
MOSS2_DISCIPLINE_MIN_SETTLED = 4
MOSS2_DISCIPLINE_MAX_CONSEC_LOSS = 4
MOSS2_HALF_KELLY_CAP = 0.15
# 扫描开仓质量（质量优先：对齐 Moss Quant，高阈值 + 余量 + 多根确认）
MOSS2_ENTRY_QUALITY_ENABLED = True
MOSS2_ENTRY_THRESHOLD_FLOOR = 0.40
MOSS2_ENTRY_MARGIN = 0.05
MOSS2_ENTRY_CONFIRM_BARS = 2
# 0 = 第 2 根不得比第 1 根软（防尖上）；>0 则允许小幅回吐
MOSS2_ENTRY_CONFIRM_RELAX = 0

# 慢进化（L2/L3）
MOSS2_EVOLVE_ENABLED = True
MOSS2_EVOLVE_INTERVAL_DAYS = 7
# 进化/回测主窗（15m）；CSV 拉取天数应覆盖此窗 + 指标预热（见 MOSS2_FETCH_DAYS）
MOSS2_EVOLVE_LIMIT_BARS = 4500
# 全自动运维：建 Profile / 进化发布 / 启用（与 Moss1 寻优无关）
MOSS2_AUTO_PROVISION_ENABLED = True
# 拉 CSV 成功后自动接 25 核建 Profile→进化→启用（默认开，无需维护面板手点）
MOSS2_CHAIN_PROVISION_AFTER_BOOTSTRAP = env_truthy(
    "MOSS2_CHAIN_PROVISION_AFTER_BOOTSTRAP", default=True
)
# 链式开启时默认不再单独 +12min 启动 provision / 周日 04:45 重复跑（可用 env 覆盖）
MOSS2_AUTO_PROVISION_ON_START = env_truthy(
    "MOSS2_AUTO_PROVISION_ON_START",
    default=not MOSS2_CHAIN_PROVISION_AFTER_BOOTSTRAP,
)
MOSS2_AUTO_PROVISION_WEEKLY = env_truthy(
    "MOSS2_AUTO_PROVISION_WEEKLY",
    default=not MOSS2_CHAIN_PROVISION_AFTER_BOOTSTRAP,
)
MOSS2_AUTO_PROVISION_BACKTEST_BARS = 4500
MOSS2_AUTO_REPROVISION_EXISTING = False
MOSS2_AUTO_ENABLE_PROFILES = True
# 全自动：evolve 已 auto-approve 即启用（不必再等 suggest reason 字符串）
MOSS2_AUTO_ENABLE_ON_APPROVED = True
MOSS2_EVOLVE_AUTO_APPROVE = True
MOSS2_DISCIPLINE_SNAPSHOT_WEEKLY = True

# 选优闸门（四模板 + 战术窄搜，创建/evolve 共用）— 质量优先：严回测、高 entry 网格
# 与 MOSS2_FETCH_DAYS 无关：闸门只看回测窗 limit_bars 内的成交/EV/Sharpe/MDD
MOSS2_SELECTION_MIN_TRADES = 12
MOSS2_AUTO_PROVISION_MIN_TRADES = MOSS2_SELECTION_MIN_TRADES
MOSS2_SELECTION_MIN_SHARPE = 0.55
MOSS2_SELECTION_MAX_MDD = 0.20
MOSS2_SELECTION_MIN_EV_PCT = 0.01
MOSS2_SELECTION_TACTICAL_NARROW = True
# 全自动最多启用 Profile 数（25 核心里只留头部）
MOSS2_MAX_AUTO_ENABLED_PROFILES = 8
# 全 lane 纸面/实盘最大同时持仓数（扫描时新开仓受该上限约束）
MOSS2_PORTFOLIO_MAX_OPEN_POSITIONS = 4

# 淘汰（启用 Profile 定期体检，不过关停用）
MOSS2_CULL_ENABLED = True
MOSS2_CULL_SCHEDULER_WEEKLY = True
MOSS2_CULL_AUTO_DISABLE = True
MOSS2_CULL_REBACKTEST_ENABLED = True
MOSS2_CULL_RECOMPETE_BEFORE_DISABLE = True
MOSS2_CULL_LIVE_MIN_TRADES = 12
MOSS2_CULL_LIVE_EV_FLOOR = 0.005
MOSS2_CULL_LIVE_MAX_CONSEC_LOSS = 6

# 线上 data_cache（不依赖 moss-trade-bot-skills-main；启动后自动拉取）
MOSS2_DATA_BOOTSTRAP_ENABLED = True
MOSS2_DATA_BOOTSTRAP_ON_START = True
MOSS2_DATA_BOOTSTRAP_WEEKLY = True
MOSS2_DATA_BOOTSTRAP_STALE_HOURS = 24
MOSS2_DATA_BOOTSTRAP_SLEEP_SEC = 1.5
# 每次 bootstrap 拉取前先清理：旧命名 CSV、同币重复文件；force 时清空 25 核心再拉
MOSS2_DATA_BOOTSTRAP_CLEAN_BEFORE_FETCH = True
# False：固定起点（复现用）；True：滚动最近 MOSS2_FETCH_DAYS 天，终点为当前 UTC（线上默认）
MOSS2_FETCH_SINCE_ROLLING = True
MOSS2_FETCH_SINCE = "2025-10-06"
# 线上 CSV 缓存：滚动最近 N 天。≈ ceil(EVOLVE_LIMIT_BARS/96)+预热，取 90 兼顾稳定与拉数耗时
MOSS2_FETCH_DAYS = 90
MOSS2_FETCH_TIMEFRAME = "15m"


def effective_fetch_since_date() -> Optional[str]:
    """传给 fetcher 的 since_date；滚动窗返回 None（由 fetcher 用 now-days）。"""
    if MOSS2_FETCH_SINCE_ROLLING:
        return None
    return MOSS2_FETCH_SINCE
# 仅本地开发且存在 skills 包时设为 1，可改读 moss-trade-bot-skills-main 下已有 CSV
MOSS2_PREFER_SKILLS_DATA_CACHE = env_truthy("MOSS2_PREFER_SKILLS_DATA_CACHE", default=False)

# 币安 U 本位合约名与 base 不一致（拉数用）
MOSS2_BINANCE_CONTRACT_BASE: Dict[str, str] = {
    "PEPE": "1000PEPE",
    "SHIB": "1000SHIB",
    "BONK": "1000BONK",
}


def base_to_fetch_slash(base: str) -> str:
    """ccxt 拉数 symbol，如 PEPE -> 1000PEPE/USDT。"""
    b = str(base or "").strip().upper().replace("USDT", "")
    contract = MOSS2_BINANCE_CONTRACT_BASE.get(b, b)
    return f"{contract}/USDT"

# Profile 建议 / 批量拉数：默认 25 核心 U 本位（HyperCore 主板 + ICP、TON）
MOSS2_SEED_BASES: tuple[str, ...] = (
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "DOGE",
    "APT",
    "ATOM",
    "AVAX",
    "BCH",
    "DOT",
    "FIL",
    "HBAR",
    "ICP",
    "LINK",
    "LTC",
    "NEAR",
    "OP",
    "SUI",
    "TON",
    "TRX",
    "UNI",
    "XRP",
    "ADA",
    "ARB",
    "HYPE",
)


def _skills_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "moss-trade-bot-skills-main"
        if candidate.is_dir():
            return candidate
    return None


def factory_hl_root() -> Path:
    skills = _skills_root()
    if skills:
        return skills / "moss-trade-bot-factory-1.0.27"
    return Path(__file__).resolve().parent / "vendor-hl-placeholder"


def factory_en_root() -> Path:
    skills = _skills_root()
    if skills:
        return skills / "moss-trade-bot-factory-en-1.0.3"
    return Path(__file__).resolve().parent / "vendor-en-placeholder"


def hl_data_cache_dir() -> Path:
    return factory_hl_root() / "scripts" / "data_cache"


def moss2_en_data_cache_default() -> Path:
    """线上默认目录：next-k-api/data/moss2_en_data_cache（或 DATA_DIR 下同名）。"""
    raw = os.getenv("MOSS2_EN_DATA_CACHE", "").strip()
    if raw:
        return Path(raw)
    data_dir = os.getenv("DATA_DIR", "").strip()
    root = Path(data_dir) if data_dir else Path(__file__).resolve().parent.parent / "data"
    return root / "moss2_en_data_cache"


def en_data_cache_dir() -> Path:
    """
    Moss2 回测/进化 CSV 目录。
    默认使用 moss2_en_data_cache；仅 MOSS2_PREFER_SKILLS_DATA_CACHE=1 且 skills 有文件时走旧路径。
    """
    if MOSS2_PREFER_SKILLS_DATA_CACHE:
        skills = _skills_root()
        if skills:
            legacy_root = skills / "moss-trade-bot-factory-en-1.0.3"
            for sub in (Path("data_cache"), Path("scripts") / "data_cache"):
                p = legacy_root / sub
                if p.is_dir() and any(p.glob("binanceusdm_*.csv")):
                    return p
    p = moss2_en_data_cache_default()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _default_cache_dir() -> Path:
    data_dir = os.getenv("DATA_DIR", "").strip()
    if data_dir:
        return Path(data_dir) / "moss2_cache"
    return Path(__file__).resolve().parent.parent / "data" / "moss2_cache"


MOSS2_CACHE_DIR = _default_cache_dir()


def effective_variant(variant: str | None = None) -> FactoryVariant:
    """解析请求/Profile 的 variant；运维模式下仅允许 MOSS2_OPS_VARIANT。"""
    v = (variant or MOSS2_OPS_VARIANT).strip().lower()
    if v not in ("hl", "en"):
        raise ValueError(f"invalid_variant:{variant}")
    if not MOSS2_HL_ENABLED and v != MOSS2_OPS_VARIANT:
        raise ValueError(
            f"variant_{v}_disabled: ops locked to {MOSS2_OPS_VARIANT} "
            f"(protocol={MOSS2_PROTOCOL_VENUE})"
        )
    return v  # type: ignore[return-value]


def profile_variant(profile: dict) -> FactoryVariant:
    return effective_variant(str(profile.get("variant") or MOSS2_OPS_VARIANT))


def is_ops_variant(variant: str | None) -> bool:
    try:
        return effective_variant(variant) == MOSS2_OPS_VARIANT
    except ValueError:
        return False


def paper_scheduler_enabled() -> bool:
    return MOSS2_ENABLED and MOSS2_PAPER_ENABLED and MOSS2_SCHEDULER_ENABLED


def real_mode_enabled() -> bool:
    return MOSS2_REAL_MODE and MOSS2_ENABLED


def evolve_scheduler_enabled() -> bool:
    return MOSS2_ENABLED and MOSS2_EVOLVE_ENABLED and MOSS2_SCHEDULER_ENABLED


def discipline_snapshot_scheduler_enabled() -> bool:
    return (
        MOSS2_ENABLED
        and MOSS2_DISCIPLINE_ENABLED
        and MOSS2_DISCIPLINE_SNAPSHOT_WEEKLY
        and MOSS2_SCHEDULER_ENABLED
    )


def data_bootstrap_scheduler_enabled() -> bool:
    return MOSS2_ENABLED and MOSS2_DATA_BOOTSTRAP_ENABLED and MOSS2_SCHEDULER_ENABLED


def cull_scheduler_enabled() -> bool:
    return (
        MOSS2_ENABLED
        and MOSS2_CULL_ENABLED
        and MOSS2_CULL_SCHEDULER_WEEKLY
        and MOSS2_SCHEDULER_ENABLED
    )


def auto_provision_scheduler_enabled() -> bool:
    return (
        MOSS2_ENABLED
        and MOSS2_AUTO_PROVISION_ENABLED
        and MOSS2_SCHEDULER_ENABLED
    )


def data_bootstrap_allowed(*, manual: bool = False) -> bool:
    """调度或维护面板手动拉 CSV。"""
    if not MOSS2_ENABLED or not MOSS2_DATA_BOOTSTRAP_ENABLED:
        return False
    if manual:
        return True
    return MOSS2_SCHEDULER_ENABLED


def auto_provision_allowed(*, manual: bool = False) -> bool:
    """调度、链式 bootstrap 后或维护面板手动全自动。"""
    if not MOSS2_ENABLED or not MOSS2_AUTO_PROVISION_ENABLED:
        return False
    if manual:
        return True
    return MOSS2_SCHEDULER_ENABLED


def moss2_runtime_snapshot() -> Dict[str, object]:
    snap = moss_lane_snapshot()
    return {
        "enabled": MOSS2_ENABLED,
        "paper_scheduler": paper_scheduler_enabled(),
        "real_mode": real_mode_enabled(),
        "moss_active_lane": snap["active_lane"],
        "protocol_moss_slot": snap["moss2_protocol"],
        "paper_source_of_truth": MOSS2_PAPER_SOURCE_OF_TRUTH,
        "live_klines": MOSS2_LIVE_KLINES_ENABLED,
        "discipline_enabled": MOSS2_DISCIPLINE_ENABLED,
        "regime_snow": MOSS2_REGIME_SNOW_ENABLED,
        "evolve_enabled": MOSS2_EVOLVE_ENABLED,
        "default_variant": MOSS2_DEFAULT_VARIANT,
        "ops_variant": MOSS2_OPS_VARIANT,
        "protocol_venue": MOSS2_PROTOCOL_VENUE,
        "hl_enabled": MOSS2_HL_ENABLED,
        "hl_factory_root": str(factory_hl_root()),
        "en_factory_root": str(factory_en_root()),
        "hl_data_cache": str(hl_data_cache_dir()),
        "en_data_cache": str(en_data_cache_dir()),
        "data_bootstrap_on_start": MOSS2_DATA_BOOTSTRAP_ON_START,
        "data_bootstrap_weekly": MOSS2_DATA_BOOTSTRAP_WEEKLY,
        "data_bootstrap_clean_before_fetch": MOSS2_DATA_BOOTSTRAP_CLEAN_BEFORE_FETCH,
        "auto_provision": MOSS2_AUTO_PROVISION_ENABLED,
        "chain_provision_after_bootstrap": MOSS2_CHAIN_PROVISION_AFTER_BOOTSTRAP,
        "auto_provision_on_start": MOSS2_AUTO_PROVISION_ON_START,
        "auto_provision_weekly": MOSS2_AUTO_PROVISION_WEEKLY,
        "full_auto_note": (
            "调度器：启动/每周拉 CSV 后链式全自动建 Profile；15m 纸面扫描；周日 evolve/cull"
            if MOSS2_CHAIN_PROVISION_AFTER_BOOTSTRAP
            else "调度器：拉 CSV 与建 Profile 分离（可手点维护面板）"
        ),
        "auto_enable_profiles": MOSS2_AUTO_ENABLE_PROFILES,
        "auto_enable_on_approved": MOSS2_AUTO_ENABLE_ON_APPROVED,
        "evolve_auto_approve": MOSS2_EVOLVE_AUTO_APPROVE,
        "selection_tactical_narrow": MOSS2_SELECTION_TACTICAL_NARROW,
        "selection_min_trades": MOSS2_SELECTION_MIN_TRADES,
        "selection_min_sharpe": MOSS2_SELECTION_MIN_SHARPE,
        "selection_min_ev_pct": MOSS2_SELECTION_MIN_EV_PCT,
        "selection_max_mdd": MOSS2_SELECTION_MAX_MDD,
        "entry_quality_enabled": MOSS2_ENTRY_QUALITY_ENABLED,
        "entry_threshold_floor": MOSS2_ENTRY_THRESHOLD_FLOOR,
        "entry_margin": MOSS2_ENTRY_MARGIN,
        "entry_confirm_bars": MOSS2_ENTRY_CONFIRM_BARS,
        "entry_confirm_relax": MOSS2_ENTRY_CONFIRM_RELAX,
        "max_auto_enabled_profiles": MOSS2_MAX_AUTO_ENABLED_PROFILES,
        "portfolio_max_open_positions": MOSS2_PORTFOLIO_MAX_OPEN_POSITIONS,
        "cull_enabled": MOSS2_CULL_ENABLED,
        "scan_interval_minutes": MOSS2_SCAN_INTERVAL_MINUTES,
        "quantstats_enabled": MOSS2_QUANTSTATS_ENABLED,
        "default_template": MOSS2_DEFAULT_TEMPLATE,
        "seed_bases_count": len(MOSS2_SEED_BASES),
        "seed_bases": list(MOSS2_SEED_BASES),
        "config_source": "moss2/config.py",
        "ops_note": "binance_en_only" if not MOSS2_HL_ENABLED else "multi_variant",
    }
