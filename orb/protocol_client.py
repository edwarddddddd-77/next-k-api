"""Next-k-protocol HTTP 客户端 — POST /api/binance/signals/ingest。"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SEC = 30.0
SOURCE_ORB = "orb"


def protocol_api_url() -> str:
    return (os.getenv("PROTOCOL_API_URL") or "http://localhost:8001").strip().rstrip("/")


def protocol_configured() -> bool:
    return bool(protocol_api_url())


def ingest_signals(signals: List[Dict[str, Any]], *, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> Dict[str, Any]:
    """推送一批信号到 Next-k-protocol。"""
    if not signals:
        return {"scanned": 0, "traded": 0, "skipped": 0, "errors": 0, "details": []}
    url = f"{protocol_api_url()}/api/binance/signals/ingest"
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, json={"signals": signals}, headers=headers, timeout=timeout_sec)
    if resp.status_code >= 400:
        body = resp.text[:500]
        raise RuntimeError(f"protocol ingest HTTP {resp.status_code}: {body}")
    data = resp.json()
    logger.info(
        "[orb] protocol ingest: scanned=%s traded=%s skipped=%s errors=%s",
        data.get("scanned"),
        data.get("traded"),
        data.get("skipped"),
        data.get("errors"),
    )
    return data
