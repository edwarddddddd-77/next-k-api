"""vnpy 运行时行情源上下文（bootstrap 时设置，可选）。"""

from __future__ import annotations

_runtime_market_data_exchange: str | None = None


def set_runtime_market_data_exchange(exchange_id: str | None) -> None:
    global _runtime_market_data_exchange
    _runtime_market_data_exchange = (exchange_id or "").strip().lower() or None


def get_runtime_market_data_exchange() -> str | None:
    return _runtime_market_data_exchange


def clear_runtime_market_data_exchange() -> None:
    set_runtime_market_data_exchange(None)
