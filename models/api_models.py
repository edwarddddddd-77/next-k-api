from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SignalType(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class RadarItem(BaseModel):
    symbol: str
    name: str
    asset_type: str
    anomaly_score: float
    signal: SignalType
    signals: List[str]
    price: float
    price_change: float
    volatility_hint: str


class HealthResponse(BaseModel):
    status: str
    crypto_connected: bool
    stocks_available: bool
    forex_available: bool
    version: str
    uptime: float
    maintenance_auth_required: bool = False
    scheduler_embedded: bool = False
    scheduler_running: bool = False
    zct_vwap_scheduler_enabled: bool = False


class ZctVwapManualPatchBody(BaseModel):
    manual_entry_price: Optional[float] = None
    manual_exit_price: Optional[float] = None
    manual_notes: Optional[str] = None


class ZctTouchPoolScanBody(BaseModel):
    symbols: str = Field(
        default="ZECUSDT,ONDOUSDT,1000SHIBUSDT",
    )
    symbols_source: Literal[
        "request", "worth_watch_plus_default_22", "hot_oi_plus_default_22"
    ] = Field(default="worth_watch_plus_default_22")
    days: float = Field(
        default=1.0,
        ge=0.25,
        le=30.0,
        description="与 08:05 主筛一致：严格 24h",
    )
    min_touch_trades: int = Field(default=1, ge=0, le=200_000)
    min_touch_win_rate: float = Field(default=0.72, ge=0.0, le=1.0)
    strict_greater_touch: bool = Field(default=False)
    strict_greater_rate: bool = Field(default=False)
    min_total_trades: int = Field(default=20, ge=0, le=200_000)
    max_expired_ratio: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="1.0=关闭过期占比过滤；稳档可设 0.4",
    )
    min_win_loss_abs: int = Field(default=0, ge=0, le=200_000)
    min_touch_share: float = Field(default=0.0, ge=0.0, le=1.0)
    min_profit_factor: float = Field(default=1.25, gt=0.0)
    max_consecutive_losses_at_end: int = Field(
        default=2,
        ge=0,
        le=50,
        description="周期末连亏上限（2 即 <3）",
    )
    signal_interval: str = Field(default="1m")
    sleep_between_symbols: float = Field(default=0.25, ge=0.0, le=10.0)
    persist_db: bool = Field(default=True)


class ClearWatchTablesBody(BaseModel):
    tables: List[str] = Field(default_factory=lambda: ["ambush_watch"])


class TriggerCronBody(BaseModel):
    task: str = Field(...)


class VpRegimeScanBody(BaseModel):
    symbols: Optional[str] = Field(
        default=None,
        description="逗号分隔永续，如 BTCUSDT,ETHUSDT；与 watchlist 二选一",
    )
    watchlist: bool = Field(default=False, description="从收筹池 watchlist 取标的")
    persist: bool = Field(default=True, description="写入 vp_regime_snapshots")
    notify_tg: bool = Field(default=True, description="推送 Telegram")
