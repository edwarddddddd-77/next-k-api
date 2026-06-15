"""ORB 是否通知币安 — 由 Railway 环境变量 ORB_LIVE_ENABLED 控制。"""

from __future__ import annotations

from typing import Any, Dict

from orb.core.config import OrbConfig
from orb.core.protocol_client import protocol_api_url, protocol_configured


def live_notify_status() -> Dict[str, Any]:
    cfg = OrbConfig.from_env()
    enabled = bool(cfg.live_enabled) and protocol_configured()
    return {
        "live_notify_binance": enabled,
        "orb_live_enabled": bool(cfg.live_enabled),
        "protocol_configured": protocol_configured(),
        "protocol_api_url": protocol_api_url(),
    }
