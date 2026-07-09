"""vnpy 运行时实盘交易所上下文（bootstrap 时设置）。"""

from __future__ import annotations

_runtime_live_exchange: str | None = None


def set_runtime_live_exchange(exchange_id: str | None) -> None:
    global _runtime_live_exchange
    _runtime_live_exchange = (exchange_id or "").strip().lower() or None


def get_runtime_live_exchange() -> str | None:
    return _runtime_live_exchange


def clear_runtime_live_exchange() -> None:
    set_runtime_live_exchange(None)
