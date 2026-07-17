"""
CoinGecko 客户端（后台环境变量配置密钥）。

优先级：
1. COINGECKO_PRO_API_KEY  → pro-api.coingecko.com + x-cg-pro-api-key
2. COINGECKO_API_KEY / COINGECKO_DEMO_API_KEY → api.coingecko.com + x-cg-demo-api-key
3. 无 key → 公开端点（易 429，仅兜底）
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

import requests

logger = logging.getLogger(__name__)

PUBLIC_BASE = "https://api.coingecko.com/api/v3"
PRO_BASE = "https://pro-api.coingecko.com/api/v3"


def _strip(v: str | None) -> str:
    return (v or "").strip()


def coingecko_pro_key() -> str:
    return _strip(os.getenv("COINGECKO_PRO_API_KEY"))


def coingecko_demo_key() -> str:
    return _strip(os.getenv("COINGECKO_API_KEY")) or _strip(os.getenv("COINGECKO_DEMO_API_KEY"))


def coingecko_mode() -> str:
    if coingecko_pro_key():
        return "pro"
    if coingecko_demo_key():
        return "demo"
    return "public"


def coingecko_base_url() -> str:
    return PRO_BASE if coingecko_mode() == "pro" else PUBLIC_BASE


def coingecko_session() -> requests.Session:
    s = requests.Session()
    headers = {
        "User-Agent": "NextK-AlphaRadar/1.0",
        "Accept": "application/json",
    }
    mode = coingecko_mode()
    if mode == "pro":
        headers["x-cg-pro-api-key"] = coingecko_pro_key()
    elif mode == "demo":
        headers["x-cg-demo-api-key"] = coingecko_demo_key()
    s.headers.update(headers)
    return s


def coingecko_provider_status() -> Dict[str, Any]:
    mode = coingecko_mode()
    return {
        "provider": "coingecko",
        "mode": mode,
        "base": coingecko_base_url(),
        "keyed": mode in ("pro", "demo"),
    }


def ethplorer_key() -> str:
    return _strip(os.getenv("ETHPLORER_API_KEY")) or "freekey"


def binplorer_key() -> str:
    return _strip(os.getenv("BINPLORER_API_KEY")) or "freekey"


def holders_provider_status() -> Dict[str, Any]:
    eth = ethplorer_key()
    bnb = binplorer_key()
    return {
        "ethplorer": {
            "keyed": eth.lower() != "freekey",
            "mode": "keyed" if eth.lower() != "freekey" else "freekey",
        },
        "binplorer": {
            "keyed": bnb.lower() != "freekey",
            "mode": "keyed" if bnb.lower() != "freekey" else "freekey",
        },
    }


def alpha_providers_status() -> Dict[str, Any]:
    return {
        "coingecko": coingecko_provider_status(),
        **holders_provider_status(),
    }
