"""IB50 实盘守卫。"""

from __future__ import annotations

from quant.common.exchange_env import resolve_live_exchange_id
from quant.engine.exchanges.registry import get_adapter
from quant.ib50.config import Ib50Config


def exchange_credentials_configured(exchange_id: str | None = None) -> bool:
    return get_adapter(resolve_live_exchange_id(exchange_id)).credentials_configured()


def live_enabled(cfg: Ib50Config) -> bool:
    if not cfg.live_enabled:
        return False
    return exchange_credentials_configured(cfg.live_exchange)
