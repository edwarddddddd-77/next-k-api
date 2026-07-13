"""Anchor Drift 实盘守卫。"""

from __future__ import annotations

from quant.anchor_drift.config import AnchorDriftConfig
from quant.common.exchange_env import resolve_live_exchange_id
from quant.engine.exchanges.registry import get_adapter


def live_enabled(cfg: AnchorDriftConfig) -> bool:
    if not cfg.live_enabled:
        return False
    return get_adapter(resolve_live_exchange_id(cfg.live_exchange)).credentials_configured()
