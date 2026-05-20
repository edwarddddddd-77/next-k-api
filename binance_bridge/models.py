"""Pydantic models for the Binance live-trading router."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ConfigUpdate(BaseModel):
    """Batch-update config key/value pairs."""

    pairs: Dict[str, str] = Field(..., description="键值对，如 {\"enabled\": \"true\"}")


class PositionOut(BaseModel):
    """Single position record returned by the API."""

    id: int
    signal_log_id: Optional[int] = None
    symbol: str
    side: str
    entry_order_id: Optional[str] = None
    sl_order_id: Optional[str] = None
    tp_order_id: Optional[str] = None
    entry_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    quantity: Optional[float] = None
    notional_usdt: Optional[float] = None
    leverage: Optional[int] = None
    opened_at: str
    expire_at: Optional[str] = None
    status: str
    close_reason: Optional[str] = None
    close_price: Optional[float] = None
    closed_at: Optional[str] = None
    pnl_usdt: Optional[float] = None
    pnl_pct: Optional[float] = None


class SignalLogOut(BaseModel):
    """Signal log entry."""

    id: int
    source: str
    api_signal_id: str
    symbol: str
    side: str
    entry_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    confidence: Optional[str] = None
    regime: Optional[str] = None
    notional_usdt: Optional[float] = None
    received_at: str
    status: str
    skip_reason: Optional[str] = None


class DailyPnl(BaseModel):
    day: str
    pnl: float


class PnlSummaryOut(BaseModel):
    """Aggregated P&L summary."""

    total: int
    wins: int
    losses: int
    total_pnl: float
    avg_pnl: float
    daily: List[DailyPnl]


class StatusOut(BaseModel):
    """Service status."""

    enabled: str
    testnet: str
    open_positions: int
    max_positions: str
    position_expire_hours: str
    api_key_set: bool


class SignalBridgeResult(BaseModel):
    """Result of one signal_bridge.on_scan_complete() run."""

    scanned: int = 0
    traded: int = 0
    skipped: int = 0
    errors: int = 0
    details: List[Dict[str, Any]] = Field(default_factory=list)
