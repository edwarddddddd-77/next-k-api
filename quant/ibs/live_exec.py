"""IBS lane 实盘守卫。"""

from __future__ import annotations

from typing import Protocol

from quant.common.exchange_env import resolve_live_exchange_id
from quant.engine.exchanges.registry import get_adapter


class _LiveCfg(Protocol):
    live_enabled: bool
    live_exchange: str


def live_enabled(cfg: _LiveCfg) -> bool:
    if not cfg.live_enabled:
        return False
    return get_adapter(resolve_live_exchange_id(cfg.live_exchange)).credentials_configured()
