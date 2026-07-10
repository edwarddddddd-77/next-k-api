"""KAMA Trend 实盘守卫。"""

from __future__ import annotations

from quant.common.exchange_env import resolve_live_exchange_id
from quant.engine.exchanges.registry import get_adapter
from quant.kama_trend.config import KamaTrendConfig


def live_enabled(cfg: KamaTrendConfig) -> bool:
    if not cfg.live_enabled:
        return False
    return get_adapter(resolve_live_exchange_id(cfg.live_exchange)).credentials_configured()
