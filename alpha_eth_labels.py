#!/usr/bin/env python3
"""
免费地址身份：eth-labels.com（Etherscan 类公开标签镜像）

用于自动标 exchange / mm / burn / whale(fund)。
绝不自动标成 alpha / airdrop（原文这两类无可靠公开名单）。
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

CHAIN_TO_ID = {
    "ethereum": 1,
    "binance-smart-chain": 56,
    "base": 8453,
    "arbitrum": 42161,
}

_EXCHANGE_KEYS = (
    "exchange",
    "binance",
    "coinbase",
    "okx",
    "okex",
    "huobi",
    "htx",
    "kucoin",
    "bybit",
    "gate",
    "mexc",
    "bitfinex",
    "kraken",
    "crypto.com",
    "bitget",
    "gemini",
)
_MM_KEYS = (
    "wintermute",
    "jump trading",
    "jump-",
    "gsr",
    "market maker",
    "market-maker",
    "cumberland",
    "flow traders",
    "dwf",
)
_BURN_KEYS = ("burn", "null:", "dead", "black hole")
_FUND_KEYS = ("fund", "capital", "ventures", "a16z", "paradigm", "alameda", "three arrows")

_session: Optional[requests.Session] = None
_neg_cache: Dict[str, float] = {}
_NEG_TTL = 3600.0


def enabled() -> bool:
    raw = (os.getenv("ALPHA_ETH_LABELS") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _sess() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "NextK-AlphaEthLabels/1.0", "Accept": "application/json"})
        _session = s
    return _session


def _map_row(label: str, name_tag: str) -> Optional[Dict[str, str]]:
    blob = f"{label} {name_tag}".lower()
    if any(k in blob for k in _BURN_KEYS):
        return {"type": "burn", "label": name_tag or label or "Burn"}
    if any(k in blob for k in _EXCHANGE_KEYS):
        return {"type": "exchange", "label": name_tag or label or "Exchange"}
    if any(k.lower() in blob for k in _MM_KEYS):
        return {"type": "mm", "label": name_tag or label or "MM"}
    if any(k in blob for k in _FUND_KEYS):
        return {"type": "whale", "label": name_tag or label or "Fund"}
    # 其它 Etherscan 标签：有 nameTag 就当 whale，便于「同动」统计
    if name_tag and name_tag.strip() and name_tag.lower() not in ("", "unknown"):
        lab = (label or "").lower()
        if lab in ("blocked", "phish", "hack", "heist"):
            return {"type": "other", "label": name_tag}
        return {"type": "whale", "label": name_tag}
    return None


def lookup_address(address: str, chain_id: str = "ethereum") -> Optional[Dict[str, str]]:
    """查 eth-labels；命中返回 {type,label}，未命中/失败返回 None。"""
    if not enabled():
        return None
    addr = str(address or "").strip().lower()
    if not addr.startswith("0x") or len(addr) < 10:
        return None
    if chain_id not in CHAIN_TO_ID:
        return None

    now = time.time()
    neg_key = f"{chain_id}:{addr}"
    if neg_key in _neg_cache and now - _neg_cache[neg_key] < _NEG_TTL:
        return None

    url = f"https://eth-labels.com/labels/{addr}"
    try:
        r = _sess().get(url, timeout=6)
        if r.status_code != 200:
            _neg_cache[neg_key] = now
            return None
        rows = r.json()
        if not isinstance(rows, list) or not rows:
            _neg_cache[neg_key] = now
            return None
        want = CHAIN_TO_ID.get(chain_id)
        # 优先同链，其次任意链同地址
        ordered = sorted(
            rows,
            key=lambda x: 0 if int(x.get("chainId") or 0) == want else 1,
        )
        for row in ordered:
            if not isinstance(row, dict):
                continue
            mapped = _map_row(str(row.get("label") or ""), str(row.get("nameTag") or ""))
            if mapped:
                return mapped
        _neg_cache[neg_key] = now
        return None
    except Exception as e:
        logger.debug("eth-labels lookup failed %s: %s", addr[:12], e)
        _neg_cache[neg_key] = now
        return None


def enrich_and_persist(
    address: str,
    *,
    chain_id: str = "ethereum",
) -> Optional[Dict[str, str]]:
    """查找并写入全局标签库（下次本地命中）。"""
    hit = lookup_address(address, chain_id)
    if not hit:
        return None
    try:
        from alpha_labels import upsert_label

        upsert_label(
            address=address,
            type=hit["type"],
            label=hit["label"],
            coingecko_id="",
            chain=chain_id,
        )
    except Exception as e:
        logger.debug("persist eth-label failed: %s", e)
    return hit
