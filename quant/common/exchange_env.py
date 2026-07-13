"""交易所 / 行情源环境变量 — 单一解析入口。"""

from __future__ import annotations

import os

SUPPORTED_EXCHANGES = ("binance", "bitget", "bitget_spot")
DEFAULT_EXCHANGE = "binance"


def _normalize_exchange(raw: str | None) -> str | None:
    if not raw or not str(raw).strip():
        return None
    value = str(raw).strip().lower()
    if value == "bybit":
        value = "bitget"
    if value not in SUPPORTED_EXCHANGES:
        return None
    return value


def _first_env(*keys: str, default: str = DEFAULT_EXCHANGE) -> str:
    for key in keys:
        hit = _normalize_exchange(os.getenv(key))
        if hit:
            return hit
    return default


def resolve_live_exchange_id(explicit: str | None = None) -> str:
    """实盘交易所：ORB_VNPY_LIVE_EXCHANGE > VNPY_LIVE_EXCHANGE > binance。"""
    hit = _normalize_exchange(explicit)
    if hit:
        return hit
    return _first_env("ORB_VNPY_LIVE_EXCHANGE", "VNPY_LIVE_EXCHANGE")


def resolve_market_data_exchange_id(explicit: str | None = None) -> str:
    """行情源：ORB_MARKET_DATA_EXCHANGE > MARKET_DATA_EXCHANGE > VNPY_MARKET_DATA_EXCHANGE > 实盘 env > binance。"""
    hit = _normalize_exchange(explicit)
    if hit:
        return hit
    return _first_env(
        "ORB_MARKET_DATA_EXCHANGE",
        "MARKET_DATA_EXCHANGE",
        "VNPY_MARKET_DATA_EXCHANGE",
        "ORB_VNPY_LIVE_EXCHANGE",
        "VNPY_LIVE_EXCHANGE",
    )


def resolve_lanes_market_data_exchange(lanes: list[tuple[str, object]]) -> str:
    """多 lane 须使用同一行情源。"""
    if not lanes:
        return resolve_market_data_exchange_id()
    ids = {
        resolve_market_data_exchange_id(getattr(cfg, "market_data_exchange", None))
        for _, cfg in lanes
    }
    if len(ids) > 1:
        raise ValueError(f"multiple market data exchanges in vnpy lanes not allowed: {sorted(ids)}")
    return next(iter(ids))


def resolve_lanes_live_exchange(lanes: list[tuple[str, object]]) -> str:
    """多 lane 须使用同一实盘交易所。"""
    if not lanes:
        return resolve_live_exchange_id()
    ids = {
        resolve_live_exchange_id(getattr(cfg, "live_exchange", None))
        for _, cfg in lanes
    }
    if len(ids) > 1:
        raise ValueError(f"multiple live exchanges in vnpy lanes not allowed: {sorted(ids)}")
    return next(iter(ids))
