#!/usr/bin/env python3
"""
Alpha 地址标签（对齐原文：Alpha / 空投 / 交易所 / 做市 / 巨鲸）

持久化：DATA_DIR/alpha_address_labels.json
结构：
{
  "global": { "0xabc...": {"type":"exchange","label":"Binance"} },
  "by_coin": {
    "caldera": { "0x...": {"type":"alpha","label":"早期 Alpha"} }
  },
  "updated_at_cst": "..."
}

优先级：by_coin[cid][addr] > global[addr] > 内置 burn/交易所
未标注地址不得当作原文里的 alpha/airdrop。
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from quant.common.paths import resolve_data_dir

logger = logging.getLogger(__name__)

CST = timezone(timedelta(hours=8))
LABELS_NAME = "alpha_address_labels.json"
VALID_TYPES = ("alpha", "airdrop", "exchange", "mm", "whale", "burn", "other")

_lock = threading.Lock()


def _now_cst() -> datetime:
    return datetime.now(CST)


def _labels_path() -> Path:
    return resolve_data_dir() / LABELS_NAME


def _norm_addr(addr: str, chain_id: str = "ethereum") -> str:
    a = str(addr or "").strip()
    if chain_id == "solana":
        return a
    return a.lower()


def _empty_store() -> Dict[str, Any]:
    return {"ok": True, "global": {}, "by_coin": {}, "updated_at_cst": None}


def _load_store() -> Dict[str, Any]:
    path = _labels_path()
    store = _empty_store()
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                store["global"] = raw.get("global") if isinstance(raw.get("global"), dict) else {}
                store["by_coin"] = raw.get("by_coin") if isinstance(raw.get("by_coin"), dict) else {}
                store["updated_at_cst"] = raw.get("updated_at_cst")
        except Exception as e:
            logger.warning("alpha labels read failed: %s", e)

    # 合并环境变量（启动时注入；不覆盖磁盘已有同地址）
    raw_env = (os.getenv("ALPHA_ADDRESS_LABELS_JSON") or "").strip()
    if raw_env:
        try:
            data = json.loads(raw_env)
            _merge_env_into_store(store, data)
        except Exception as e:
            logger.warning("ALPHA_ADDRESS_LABELS_JSON parse failed: %s", e)
    return store


def _merge_env_into_store(store: Dict[str, Any], data: Any) -> None:
    g = store.setdefault("global", {})
    by_coin = store.setdefault("by_coin", {})

    def put_global(addr: str, typ: str, label: str, chain: str = "ethereum") -> None:
        key = _norm_addr(addr, chain)
        if key and key not in g:
            g[key] = {"type": typ, "label": label, "chain": chain}

    if isinstance(data, dict):
        # 兼容旧格式：直接 address -> type|obj
        # 或 {global:{}, by_coin:{}}
        if "global" in data or "by_coin" in data:
            for k, v in (data.get("global") or {}).items():
                if isinstance(v, dict):
                    put_global(str(k), str(v.get("type") or "whale"), str(v.get("label") or v.get("type") or "自定义"), str(v.get("chain") or "ethereum"))
                elif isinstance(v, str):
                    put_global(str(k), v, v)
            for cid, mapping in (data.get("by_coin") or {}).items():
                if not isinstance(mapping, dict):
                    continue
                bucket = by_coin.setdefault(str(cid).strip(), {})
                for addr, v in mapping.items():
                    key = _norm_addr(str(addr), "ethereum")
                    if not key or key in bucket:
                        continue
                    if isinstance(v, dict):
                        bucket[key] = {
                            "type": str(v.get("type") or "whale"),
                            "label": str(v.get("label") or v.get("type") or "自定义"),
                            "chain": str(v.get("chain") or "ethereum"),
                        }
                    elif isinstance(v, str):
                        bucket[key] = {"type": v, "label": v, "chain": "ethereum"}
            return
        for k, v in data.items():
            if isinstance(v, dict) and v.get("type"):
                put_global(str(k), str(v.get("type") or "whale"), str(v.get("label") or v.get("type") or "自定义"))
            elif isinstance(v, str):
                put_global(str(k), v, v)
    elif isinstance(data, list):
        for row in data:
            if not isinstance(row, dict) or not row.get("address"):
                continue
            cid = str(row.get("coingecko_id") or "").strip()
            typ = str(row.get("type") or "whale")
            label = str(row.get("label") or typ)
            chain = str(row.get("chain") or "ethereum")
            key = _norm_addr(str(row["address"]), chain)
            if cid:
                bucket = by_coin.setdefault(cid, {})
                if key not in bucket:
                    bucket[key] = {"type": typ, "label": label, "chain": chain}
            else:
                put_global(str(row["address"]), typ, label, chain)


def _save_store(store: Dict[str, Any]) -> None:
    path = _labels_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    store = dict(store)
    store["ok"] = True
    store["updated_at_cst"] = _now_cst().isoformat()
    path.write_text(json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def list_labels(coingecko_id: Optional[str] = None) -> Dict[str, Any]:
    with _lock:
        store = _load_store()
        cid = (coingecko_id or "").strip()
        coin = store.get("by_coin", {}).get(cid, {}) if cid else {}
        return {
            "ok": True,
            "valid_types": list(VALID_TYPES),
            "global": store.get("global") or {},
            "by_coin": {cid: coin} if cid else (store.get("by_coin") or {}),
            "coin_labels": coin,
            "updated_at_cst": store.get("updated_at_cst"),
            "hint": "原文对齐：请把已知 Alpha / 空投 / 做市地址标进 by_coin；未标注不会当成 alpha/airdrop。",
        }


def upsert_label(
    *,
    address: str,
    type: str,
    label: str = "",
    coingecko_id: str = "",
    chain: str = "ethereum",
) -> Dict[str, Any]:
    typ = str(type or "").strip().lower()
    if typ not in VALID_TYPES:
        raise ValueError(f"invalid_type: use one of {', '.join(VALID_TYPES)}")
    addr = str(address or "").strip()
    if not addr:
        raise ValueError("address_required")
    chain_id = str(chain or "ethereum").strip() or "ethereum"
    key = _norm_addr(addr, chain_id)
    entry = {
        "type": typ,
        "label": str(label or typ).strip() or typ,
        "chain": chain_id,
        "address": addr,
    }
    with _lock:
        store = _load_store()
        cid = str(coingecko_id or "").strip()
        if cid:
            bucket = store.setdefault("by_coin", {}).setdefault(cid, {})
            bucket[key] = entry
        else:
            store.setdefault("global", {})[key] = entry
        _save_store(store)
        return {"ok": True, "entry": entry, "coingecko_id": cid or None, "key": key}


def delete_label(*, address: str, coingecko_id: str = "", chain: str = "ethereum") -> Dict[str, Any]:
    key = _norm_addr(address, chain)
    with _lock:
        store = _load_store()
        cid = str(coingecko_id or "").strip()
        removed = False
        if cid:
            bucket = store.get("by_coin", {}).get(cid) or {}
            if key in bucket:
                del bucket[key]
                removed = True
        else:
            g = store.get("global") or {}
            if key in g:
                del g[key]
                removed = True
        if removed:
            _save_store(store)
        return {"ok": True, "removed": removed, "key": key}


def resolve_label(
    address: str,
    *,
    chain_id: str = "ethereum",
    coingecko_id: str = "",
    rank: int = 99,
    builtin: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, str]:
    """解析标签。未命中 → whale/other；可自动用 eth-labels 补 exchange/mm/burn。"""
    key = _norm_addr(address, chain_id)
    if builtin and key in builtin:
        return dict(builtin[key])

    with _lock:
        store = _load_store()
        cid = str(coingecko_id or "").strip()
        if cid:
            hit = (store.get("by_coin") or {}).get(cid, {}).get(key)
            if isinstance(hit, dict) and hit.get("type"):
                return {"type": str(hit["type"]), "label": str(hit.get("label") or hit["type"])}
        hit = (store.get("global") or {}).get(key)
        if isinstance(hit, dict) and hit.get("type"):
            return {"type": str(hit["type"]), "label": str(hit.get("label") or hit["type"])}

    # 免费公开标签：交易所 / 做市 / 烧币 / 基金
    try:
        from alpha_eth_labels import enrich_and_persist

        remote = enrich_and_persist(address, chain_id=chain_id)
        if remote:
            return remote
    except Exception:
        pass

    if rank <= 10:
        return {"type": "whale", "label": f"未标注巨鲸#{rank}"}
    return {"type": "other", "label": f"未标注#{rank}"}


def labeling_coverage(holders: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Top 持仓里有多少已被人工标成原文类别。"""
    total = len(holders or [])
    tagged = 0
    by_type: Dict[str, int] = {}
    for h in holders or []:
        t = str(h.get("type") or "other")
        by_type[t] = by_type.get(t, 0) + 1
        if t in ("alpha", "airdrop", "exchange", "mm", "burn"):
            tagged += 1
    return {
        "total": total,
        "tagged_playbook": tagged,
        "untagged": max(0, total - tagged),
        "by_type": by_type,
        "ready": tagged >= 3,
    }
