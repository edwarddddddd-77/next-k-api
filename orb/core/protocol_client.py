"""Next-k-protocol HTTP 客户端 — POST /api/binance/signals/ingest。"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SEC = 30.0
SOURCE_ORB = "orb"
LIVE_PENDING_NOTE = "live_pending_entry"


def protocol_api_url() -> str:
    return (os.getenv("PROTOCOL_API_URL") or "http://localhost:8001").strip().rstrip("/")


def _live_enabled_env() -> bool:
    raw = (os.getenv("ORB_LIVE_ENABLED") or "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def protocol_configured() -> bool:
    """实盘开启时必须显式配置 PROTOCOL_API_URL；否则 localhost 默认可用。"""
    url = protocol_api_url()
    if not url:
        return False
    if _live_enabled_env():
        return bool((os.getenv("PROTOCOL_API_URL") or "").strip())
    return True


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


def reconcile_pending_entries(*, timeout_sec: float = DEFAULT_TIMEOUT_SEC) -> Dict[str, Any]:
    url = f"{protocol_api_url()}/api/binance/maintenance/reconcile-entries"
    resp = requests.post(url, timeout=timeout_sec)
    if resp.status_code >= 400:
        body = resp.text[:500]
        raise RuntimeError(f"protocol reconcile HTTP {resp.status_code}: {body}")
    return resp.json()


def lookup_signal(
    *,
    source: str,
    api_signal_id: str,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> Optional[Dict[str, Any]]:
    url = f"{protocol_api_url()}/api/binance/signals/lookup"
    resp = requests.get(
        url,
        params={"source": source, "api_signal_id": api_signal_id},
        timeout=timeout_sec,
    )
    if resp.status_code == 404:
        return None
    if resp.status_code >= 400:
        body = resp.text[:500]
        raise RuntimeError(f"protocol lookup HTTP {resp.status_code}: {body}")
    return resp.json()
