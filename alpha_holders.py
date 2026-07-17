#!/usr/bin/env python3
"""
Alpha 多链持仓监控：ETH / BSC / Base / Arbitrum / Solana

数据源（无需付费 key，可用环境变量覆盖）：
- ethereum     → Ethplorer getTopTokenHolders
- bsc          → Binplorer getTopTokenHolders
- base/arb     → Blockscout token holders
- solana       → RPC getTokenLargestAccounts + owner 解析

快照落盘后对比余额变动 → 多地址同流出 / 集中流出 信号。
地址标签：内置 burn/常见交易所 + ALPHA_ADDRESS_LABELS_JSON。
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from quant.common.paths import resolve_data_dir
from alpha_coingecko import binplorer_key, coingecko_base_url, coingecko_session, ethplorer_key

logger = logging.getLogger(__name__)

CST = timezone(timedelta(hours=8))
HOLDERS_DIR_NAME = "alpha_holders"
PLATFORMS_CACHE_DIR_NAME = "alpha_platforms_cache"
HOLDERS_TOP_N = 20
OUTFLOW_EPS_SHARE = 0.05  # 持仓占比下降超过 0.05pp 视为流出
DEFAULT_PLATFORMS_CACHE_HOURS = 168  # 7 天
_last_coingecko_platforms_at = 0.0
_COINGECKO_PLATFORMS_MIN_INTERVAL_SEC = 1.2

# 日历币写死合约：持仓监控禁止依赖 CoinGecko coins/{id}（免费端点易 429）
KNOWN_TOKEN_CONTRACTS: Dict[str, Dict[str, str]] = {
    "caldera": {
        "ethereum": "0xE2AD0BF751834f2fbdC62A41014f84d67cA1de2A",
        "binance-smart-chain": "0x00312400303d02c323295f6E8b7309bc30FB6BcE",
    },
    "block-street": {
        "ethereum": "0xdb6ba5d510f114f9b2ea08bea7d30e32eee33411",
    },
    "aspecta": {
        "binance-smart-chain": "0xad8c787992428cd158e451aab109f724b6bc36de",
    },
}

# CoinGecko platform_id → 内部链配置
CHAIN_CONFIG: Dict[str, Dict[str, Any]] = {
    "ethereum": {
        "id": "ethereum",
        "label": "Ethereum",
        "kind": "ethplorer",
        "base": "https://api.ethplorer.io",
        "explorer": "https://etherscan.io/token/{contract}?a={address}",
        "priority": 1,
    },
    "binance-smart-chain": {
        "id": "bsc",
        "label": "BNB Chain",
        "kind": "binplorer",
        "base": "https://api.binplorer.com",
        "explorer": "https://bscscan.com/token/{contract}?a={address}",
        "priority": 0,
    },
    "base": {
        "id": "base",
        "label": "Base",
        "kind": "blockscout",
        "base": "https://base.blockscout.com/api",
        "explorer": "https://basescan.org/token/{contract}?a={address}",
        "priority": 2,
    },
    "arbitrum-one": {
        "id": "arbitrum",
        "label": "Arbitrum",
        "kind": "blockscout",
        "base": "https://arbitrum.blockscout.com/api",
        "explorer": "https://arbiscan.io/token/{contract}?a={address}",
        "priority": 3,
    },
    "solana": {
        "id": "solana",
        "label": "Solana",
        "kind": "solana",
        "rpc": "https://solana-rpc.publicnode.com",
        "explorer": "https://solscan.io/account/{address}",
        "priority": 1,
    },
}

# 常见标签（小写地址）
_BUILTIN_LABELS: Dict[str, Dict[str, str]] = {
    "0x0000000000000000000000000000000000000000": {"type": "burn", "label": "零地址"},
    "0x000000000000000000000000000000000000dead": {"type": "burn", "label": "Burn"},
    "0xdead000000000000000000000000000000000000": {"type": "burn", "label": "Dead"},
    # Binance 常见热钱包（公开列表，不完全）
    "0x28c6c06298d514db089934071355e5743bf21d60": {"type": "exchange", "label": "Binance"},
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549": {"type": "exchange", "label": "Binance"},
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d": {"type": "exchange", "label": "Binance"},
    "0x56eddb7aa87536c19524ebeb1ff532bf2193b6c3": {"type": "exchange", "label": "Binance"},
    "0x9696f59e4d72e237be84ffd425dcad154bf96976": {"type": "exchange", "label": "Binance"},
    "0xf977814e90da44bfa03b6295a0616a897441acec": {"type": "exchange", "label": "Binance"},
    "0x001866ae5b3de6caa5a51543fd9fb64f524f5478": {"type": "exchange", "label": "Binance"},
    "0x5a52e96bacdabb82fd05763e253f78a7d0e8b8ba": {"type": "exchange", "label": "Binance"},
    "0xbe0eb53f46cd790cd13851f5ca7b6ba45fbc6e63": {"type": "exchange", "label": "Binance"},
    "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be": {"type": "exchange", "label": "Binance"},
}


def _now_cst() -> datetime:
    return datetime.now(CST)


def _holders_dir() -> Path:
    d = resolve_data_dir() / HOLDERS_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _session() -> requests.Session:
    """通用 HTTP；CoinGecko 请用 coingecko_session()。"""
    s = requests.Session()
    s.headers.update({"User-Agent": "NextK-AlphaHolders/1.0", "Accept": "application/json"})
    return s


def _norm_addr(addr: str, chain_id: str) -> str:
    a = str(addr or "").strip()
    if chain_id == "solana":
        return a
    return a.lower()


def _load_custom_labels() -> Dict[str, Dict[str, str]]:
    out = dict(_BUILTIN_LABELS)
    raw = (os.getenv("ALPHA_ADDRESS_LABELS_JSON") or "").strip()
    if not raw:
        return out
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    out[_norm_addr(str(k), "ethereum")] = {
                        "type": str(v.get("type") or "whale"),
                        "label": str(v.get("label") or v.get("type") or "自定义"),
                    }
                elif isinstance(v, str):
                    out[_norm_addr(str(k), "ethereum")] = {"type": v, "label": v}
        elif isinstance(data, list):
            for row in data:
                if not isinstance(row, dict):
                    continue
                addr = row.get("address")
                if not addr:
                    continue
                out[_norm_addr(str(addr), str(row.get("chain") or "ethereum"))] = {
                    "type": str(row.get("type") or "whale"),
                    "label": str(row.get("label") or row.get("type") or "自定义"),
                }
    except Exception as e:
        logger.warning("ALPHA_ADDRESS_LABELS_JSON parse failed: %s", e)
    return out


def label_address(address: str, chain_id: str, rank: int) -> Dict[str, str]:
    labels = _load_custom_labels()
    key = _norm_addr(address, chain_id)
    if key in labels:
        return dict(labels[key])
    # 对齐原文：Alpha / 交易所 / 做市 / 空投；未标注时头部先作 Alpha 候选
    if rank <= 3:
        return {"type": "alpha", "label": f"头部候选#{rank}"}
    if rank <= 10:
        return {"type": "whale", "label": f"巨鲸#{rank}"}
    return {"type": "other", "label": f"#{rank}"}


def _platforms_cache_hours() -> int:
    try:
        return max(1, int(os.getenv("ALPHA_PLATFORMS_CACHE_HOURS", str(DEFAULT_PLATFORMS_CACHE_HOURS))))
    except Exception:
        return DEFAULT_PLATFORMS_CACHE_HOURS


def _platforms_cache_path(coingecko_id: str) -> Path:
    safe = str(coingecko_id or "").strip().replace("/", "_")
    return resolve_data_dir() / PLATFORMS_CACHE_DIR_NAME / f"{safe}.json"


def _load_platforms_cache(coingecko_id: str, *, allow_stale: bool = False) -> Optional[List[Dict[str, Any]]]:
    path = _platforms_cache_path(coingecko_id)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        fetched = _parse_iso_cst(str(raw.get("fetched_at_cst") or ""))
        platforms = raw.get("platforms")
        if not isinstance(platforms, list):
            return None
        if fetched is None:
            return platforms if allow_stale else None
        age_h = (_now_cst() - fetched).total_seconds() / 3600.0
        if age_h <= _platforms_cache_hours() or allow_stale:
            return platforms
    except Exception as e:
        logger.warning("platforms cache read failed %s: %s", coingecko_id, e)
    return None


def _parse_iso_cst(s: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=CST)
        return dt.astimezone(CST)
    except Exception:
        return None


def _save_platforms_cache(coingecko_id: str, platforms: List[Dict[str, Any]]) -> None:
    path = _platforms_cache_path(coingecko_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "coingecko_id": coingecko_id,
                "fetched_at_cst": _now_cst().isoformat(),
                "platforms": platforms,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _normalize_platform_rows(
    rows: Any,
    *,
    primary: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """把 contracts dict / platforms list 统一成内部结构。"""
    out: List[Dict[str, Any]] = []
    if isinstance(rows, dict):
        items = list(rows.items())
    elif isinstance(rows, list):
        items = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            pid = str(row.get("platform_id") or row.get("chain") or "").strip()
            contract = str(row.get("contract") or row.get("address") or "").strip()
            if pid and contract:
                items.append((pid, contract))
    else:
        return []

    for platform_id, contract in items:
        pid = str(platform_id or "").strip()
        # 兼容日历里写 bsc / eth
        aliases = {
            "bsc": "binance-smart-chain",
            "bnb": "binance-smart-chain",
            "eth": "ethereum",
            "arb": "arbitrum-one",
            "arbitrum": "arbitrum-one",
        }
        pid = aliases.get(pid.lower(), pid)
        if pid not in CHAIN_CONFIG:
            continue
        c = str(contract or "").strip()
        if not c:
            continue
        cfg = CHAIN_CONFIG[pid]
        out.append(
            {
                "platform_id": pid,
                "chain": cfg["id"],
                "chain_label": cfg["label"],
                "contract": c,
                "is_primary": pid == primary if primary else False,
                "priority": int(cfg.get("priority", 9)),
            }
        )
    out.sort(key=lambda x: (0 if x["is_primary"] else 1, x["priority"]))
    return out


def platforms_from_known(coingecko_id: str) -> List[Dict[str, Any]]:
    cid = str(coingecko_id or "").strip().lower()
    raw = KNOWN_TOKEN_CONTRACTS.get(cid)
    if not raw:
        return []
    return _normalize_platform_rows(raw)


def platforms_from_calendar_item(item: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(item, dict):
        return []
    if item.get("platforms") is not None:
        return _normalize_platform_rows(item.get("platforms"), primary=item.get("primary_platform"))
    if item.get("contracts") is not None:
        return _normalize_platform_rows(item.get("contracts"), primary=item.get("primary_platform"))
    # 日历行未写 contracts 时，仍可用内置已知合约
    return platforms_from_known(str(item.get("coingecko_id") or ""))


def fetch_platforms(coingecko_id: str, *, force: bool = False) -> List[Dict[str, Any]]:
    """从 CoinGecko 解析该币在各链的合约；带磁盘缓存与限流，429 时回退旧缓存。

    注意：已知日历币应走 KNOWN_TOKEN_CONTRACTS，不要调用本函数。
    """
    cid = str(coingecko_id or "").strip()
    if not cid:
        return []

    known = platforms_from_known(cid)
    if known:
        return known

    if not force:
        cached = _load_platforms_cache(cid, allow_stale=False)
        if cached is not None:
            return _normalize_platform_rows(
                [{"platform_id": p.get("platform_id"), "contract": p.get("contract")} for p in cached],
                primary=next((p.get("platform_id") for p in cached if p.get("is_primary")), None),
            )

    # 无 Demo/Pro key 时跳过公开 coins/{id}，避免必然 429
    from alpha_coingecko import coingecko_mode

    if coingecko_mode() == "public":
        stale = _load_platforms_cache(cid, allow_stale=True)
        if stale is not None:
            logger.warning("skip CoinGecko public coins/%s; using stale platforms cache", cid)
            return _normalize_platform_rows(
                [{"platform_id": p.get("platform_id"), "contract": p.get("contract")} for p in stale],
                primary=next((p.get("platform_id") for p in stale if p.get("is_primary")), None),
            )
        raise RuntimeError(
            f"no_contracts:{cid} — 请在日历写 contracts，或配置 COINGECKO_API_KEY；"
            "公开 CoinGecko 易 429，持仓监控不再调用"
        )

    global _last_coingecko_platforms_at
    elapsed = time.time() - _last_coingecko_platforms_at
    if elapsed < _COINGECKO_PLATFORMS_MIN_INTERVAL_SEC:
        time.sleep(_COINGECKO_PLATFORMS_MIN_INTERVAL_SEC - elapsed)

    sess = coingecko_session()
    last_err: Optional[Exception] = None
    for attempt in range(3):
        try:
            _last_coingecko_platforms_at = time.time()
            r = sess.get(
                f"{coingecko_base_url()}/coins/{cid}",
                params={
                    "localization": "false",
                    "tickers": "false",
                    "market_data": "false",
                    "community_data": "false",
                    "developer_data": "false",
                },
                timeout=25,
            )
            if r.status_code == 429:
                raise RuntimeError(f"429 Too Many Requests for CoinGecko coins/{cid}")
            r.raise_for_status()
            data = r.json()
            platforms = data.get("platforms") or {}
            primary = data.get("asset_platform_id")
            out = _normalize_platform_rows(platforms, primary=primary)
            try:
                _save_platforms_cache(cid, out)
            except Exception as e:
                logger.warning("platforms cache write failed %s: %s", cid, e)
            return out
        except Exception as e:
            last_err = e
            is_429 = "429" in str(e) or (
                getattr(getattr(e, "response", None), "status_code", None) == 429
            )
            if is_429 and attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            break

    stale = _load_platforms_cache(cid, allow_stale=True)
    if stale is not None:
        logger.warning(
            "CoinGecko platforms %s failed (%s); using stale cache (%d chains)",
            cid,
            last_err,
            len(stale),
        )
        return _normalize_platform_rows(
            [{"platform_id": p.get("platform_id"), "contract": p.get("contract")} for p in stale],
            primary=next((p.get("platform_id") for p in stale if p.get("is_primary")), None),
        )
    if last_err:
        raise last_err
    return []


def resolve_platforms(
    coingecko_id: str,
    *,
    override: Optional[Any] = None,
    calendar_item: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """已知合约 → 日历 → override → 缓存/CoinGecko(仅有 key 时)。"""
    known = platforms_from_known(coingecko_id)
    if known:
        return known

    from_cal = platforms_from_calendar_item(calendar_item)
    if from_cal:
        return from_cal

    # 回查日历全部条目（同 id 可能有多条，取第一条带合约的）
    try:
        from alpha_radar import _load_calendar

        cid = str(coingecko_id or "").strip()
        for item in _load_calendar():
            if str(item.get("coingecko_id") or "").strip() != cid:
                continue
            rows = platforms_from_calendar_item(item)
            if rows:
                return rows
    except Exception as e:
        logger.warning("calendar contract lookup failed for %s: %s", coingecko_id, e)

    if override is not None:
        rows = _normalize_platform_rows(override)
        if rows:
            return rows
    return fetch_platforms(coingecko_id)


def _ethplorer_holders(base: str, contract: str, limit: int, api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    key = (api_key or "freekey").strip() or "freekey"
    sess = _session()
    r = sess.get(
        f"{base.rstrip('/')}/getTopTokenHolders/{contract}",
        params={"apiKey": key, "limit": limit},
        timeout=25,
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("error"):
        raise ValueError(f"ethplorer error: {data.get('error')}")
    holders = data.get("holders") if isinstance(data, dict) else None
    if not isinstance(holders, list):
        raise ValueError(f"ethplorer unexpected: {data}")
    out = []
    for i, h in enumerate(holders[:limit], start=1):
        share = float(h.get("share") or 0.0)
        raw = str(h.get("rawBalance") or h.get("balance") or "0")
        out.append(
            {
                "rank": i,
                "address": str(h.get("address") or ""),
                "balance_raw": raw,
                "share_pct": round(share, 4),
            }
        )
    return out


def _provider_api_key(kind: str) -> str:
    """各链独立 key；在 Railway / .env.oi 配置，未配则 freekey。"""
    if kind == "ethplorer":
        return ethplorer_key()
    if kind == "binplorer":
        return binplorer_key()
    return "freekey"


def _blockscout_holders(api_base: str, contract: str, limit: int) -> List[Dict[str, Any]]:
    sess = _session()
    r = sess.get(
        api_base,
        params={
            "module": "token",
            "action": "getTokenHolders",
            "contractaddress": contract,
            "page": 1,
            "offset": limit,
        },
        timeout=25,
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and str(data.get("status") or "") == "0":
        raise ValueError(
            f"blockscout holders failed: {data.get('message')} {data.get('result')}"
        )
    rows = data.get("result") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"blockscout holders unexpected: {data}")

    supply = 0.0
    try:
        rs = sess.get(
            api_base,
            params={"module": "stats", "action": "tokensupply", "contractaddress": contract},
            timeout=20,
        )
        if rs.ok:
            js = rs.json()
            supply = float(js.get("result") or 0)
    except Exception:
        supply = 0.0

    out = []
    for i, h in enumerate(rows[:limit], start=1):
        bal = float(h.get("value") or 0)
        share = (bal / supply * 100.0) if supply > 0 else 0.0
        out.append(
            {
                "rank": i,
                "address": str(h.get("address") or ""),
                "balance_raw": str(h.get("value") or "0"),
                "share_pct": round(share, 4),
            }
        )
    return out


def _solana_holders(rpc: str, mint: str, limit: int) -> List[Dict[str, Any]]:
    sess = _session()
    r = sess.post(
        rpc,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTokenLargestAccounts",
            "params": [mint],
        },
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    values = ((data.get("result") or {}).get("value")) if isinstance(data, dict) else None
    if not isinstance(values, list):
        raise ValueError(f"solana largest unexpected: {data}")

    # resolve owners (batch sequentially to keep simple)
    owners: List[Dict[str, Any]] = []
    total_ui = sum(float(v.get("uiAmount") or 0) for v in values[:limit]) or 0.0
    for i, v in enumerate(values[:limit], start=1):
        token_acc = str(v.get("address") or "")
        ui = float(v.get("uiAmount") or 0)
        owner = token_acc
        try:
            ri = sess.post(
                rpc,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getAccountInfo",
                    "params": [token_acc, {"encoding": "jsonParsed"}],
                },
                timeout=20,
            )
            if ri.ok:
                info = (
                    (((ri.json().get("result") or {}).get("value") or {}).get("data") or {})
                    .get("parsed", {})
                    .get("info", {})
                )
                owner = str(info.get("owner") or token_acc)
        except Exception as e:
            logger.debug("solana owner resolve failed %s: %s", token_acc, e)
        share = (ui / total_ui * 100.0) if total_ui > 0 else 0.0
        owners.append(
            {
                "rank": i,
                "address": owner,
                "token_account": token_acc,
                "balance_raw": str(v.get("amount") or "0"),
                "balance_ui": ui,
                "share_pct": round(share, 4),
            }
        )
    return owners


def fetch_top_holders(platform_id: str, contract: str, limit: int = HOLDERS_TOP_N) -> Dict[str, Any]:
    cfg = CHAIN_CONFIG.get(platform_id)
    if not cfg:
        raise ValueError(f"unsupported platform: {platform_id}")
    kind = cfg["kind"]
    chain_id = cfg["id"]
    if kind in ("ethplorer", "binplorer"):
        rows = _ethplorer_holders(cfg["base"], contract, limit, api_key=_provider_api_key(kind))
    elif kind == "blockscout":
        rows = _blockscout_holders(cfg["base"], contract, limit)
    elif kind == "solana":
        rpc = (os.getenv("SOLANA_RPC_URL") or cfg["rpc"]).strip()
        rows = _solana_holders(rpc, contract, limit)
    else:
        raise ValueError(f"unknown kind: {kind}")

    enriched = []
    for h in rows:
        addr = h.get("address") or ""
        lab = label_address(addr, chain_id, int(h.get("rank") or 99))
        explorer = cfg.get("explorer") or ""
        link = (
            explorer.replace("{contract}", contract).replace("{address}", addr)
            if explorer
            else ""
        )
        enriched.append({**h, "type": lab["type"], "label": lab["label"], "explorer_url": link})

    return {
        "platform_id": platform_id,
        "chain": chain_id,
        "chain_label": cfg["label"],
        "contract": contract,
        "holders": enriched,
        "fetched_at_cst": _now_cst().isoformat(),
        "source": kind,
    }


def _snap_key(coingecko_id: str, platform_id: str, contract: str) -> Path:
    safe = f"{coingecko_id}__{platform_id}__{contract}".replace("/", "_")
    return _holders_dir() / f"{safe}.json"


def load_holder_snapshot(coingecko_id: str, platform_id: str, contract: str) -> Optional[Dict[str, Any]]:
    path = _snap_key(coingecko_id, platform_id, contract)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def save_holder_snapshot(coingecko_id: str, platform_id: str, contract: str, payload: Dict[str, Any]) -> None:
    path = _snap_key(coingecko_id, platform_id, contract)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def diff_holders(
    prev: Optional[Dict[str, Any]],
    curr: Dict[str, Any],
    phase: Optional[str] = None,
) -> Dict[str, Any]:
    from alpha_playbook import evaluate_cj_playbook

    prev_map: Dict[str, Dict[str, Any]] = {}
    if prev and isinstance(prev.get("holders"), list):
        for h in prev["holders"]:
            prev_map[_norm_addr(str(h.get("address") or ""), str(curr.get("chain") or ""))] = h

    movers: List[Dict[str, Any]] = []
    outflow_share = 0.0
    inflow_share = 0.0
    types_out: Dict[str, int] = {}

    for h in curr.get("holders") or []:
        addr = _norm_addr(str(h.get("address") or ""), str(curr.get("chain") or ""))
        cur_share = float(h.get("share_pct") or 0)
        old = prev_map.get(addr)
        if not old:
            movers.append({**h, "delta_share_pct": None, "move": "new_in_top"})
            continue
        old_share = float(old.get("share_pct") or 0)
        delta = round(cur_share - old_share, 4)
        move = "flat"
        if delta <= -OUTFLOW_EPS_SHARE:
            move = "outflow"
            outflow_share += -delta
            t = str(h.get("type") or "other")
            types_out[t] = types_out.get(t, 0) + 1
        elif delta >= OUTFLOW_EPS_SHARE:
            move = "inflow"
            inflow_share += delta
        movers.append({**h, "delta_share_pct": delta, "move": move})

    outflow_addrs = [m for m in movers if m.get("move") == "outflow"]
    type_set = {str(m.get("type") or "other") for m in outflow_addrs}
    play = evaluate_cj_playbook(
        has_baseline=bool(prev),
        outflow_addrs=outflow_addrs,
        outflow_share=outflow_share,
        inflow_share=inflow_share,
        phase=phase,
    )

    return {
        "outflow_share_pct": round(outflow_share, 4),
        "inflow_share_pct": round(inflow_share, 4),
        "outflow_count": len(outflow_addrs),
        "outflow_types": sorted(type_set),
        "types_out_count": types_out,
        "signal": play["signal"],
        "signal_label": play["signal_label"],
        "bias": play["bias"],
        "action": play["action"],
        "quote": play.get("quote"),
        "pressure_pp": play.get("pressure_pp", 0.0),
        "simultaneous_movers": play.get("simultaneous_movers", len(outflow_addrs)),
        "playbook_steps": play.get("steps") or [],
        "movers": sorted(
            [m for m in movers if m.get("move") in ("outflow", "inflow", "new_in_top")],
            key=lambda x: abs(float(x.get("delta_share_pct") or 0)),
            reverse=True,
        )[:15],
        "has_baseline": bool(prev),
        "phase": phase,
    }



def watch_token_holders(
    coingecko_id: str,
    symbol: str = "",
    name: str = "",
    limit: int = HOLDERS_TOP_N,
    max_chains: int = 4,
    phase: Optional[str] = None,
    calendar_item: Optional[Dict[str, Any]] = None,
    platforms_override: Optional[Any] = None,
) -> Dict[str, Any]:
    """拉取该币所有支持链的 Top holders，并与上次快照对比。"""
    platforms = resolve_platforms(
        coingecko_id,
        override=platforms_override,
        calendar_item=calendar_item,
    )
    platforms = platforms[: max(1, max_chains)]
    chains_out: List[Dict[str, Any]] = []
    errors: List[str] = []

    if not platforms:
        return {
            "coingecko_id": coingecko_id,
            "symbol": symbol,
            "name": name,
            "chains": [],
            "aggregate": {
                "signal": "no_data",
                "signal_label": "无可用合约",
                "bias": "neutral",
                "action": "日历未写 contracts，且 CoinGecko 未返回链合约",
            },
            "errors": ["no_platforms"],
        }

    for p in platforms:
        platform_id = p["platform_id"]
        contract = p["contract"]
        try:
            curr = fetch_top_holders(platform_id, contract, limit=limit)
            prev = load_holder_snapshot(coingecko_id, platform_id, contract)
            analysis = diff_holders(prev, curr, phase=phase)
            # 把变动写回 holders 行，便于前端表格展示
            move_by_addr = {
                _norm_addr(str(m.get("address") or ""), curr.get("chain") or ""): m
                for m in (analysis.get("movers") or [])
            }
            for h in curr.get("holders") or []:
                key = _norm_addr(str(h.get("address") or ""), curr.get("chain") or "")
                m = move_by_addr.get(key)
                if m:
                    h["delta_share_pct"] = m.get("delta_share_pct")
                    h["move"] = m.get("move")
                elif prev:
                    h["delta_share_pct"] = 0.0
                    h["move"] = "flat"
            save_holder_snapshot(
                coingecko_id,
                platform_id,
                contract,
                {
                    **curr,
                    "coingecko_id": coingecko_id,
                    "symbol": symbol,
                    "analysis": {
                        "signal": analysis["signal"],
                        "outflow_share_pct": analysis["outflow_share_pct"],
                    },
                },
            )
            top_share = round(
                sum(float(h.get("share_pct") or 0) for h in (curr.get("holders") or [])[:10]),
                2,
            )
            chains_out.append(
                {
                    **p,
                    **curr,
                    "top10_share_pct": top_share,
                    "analysis": analysis,
                }
            )
            time.sleep(0.35)  # 轻限流
        except Exception as e:
            logger.warning("holders %s %s failed: %s", coingecko_id, platform_id, e)
            errors.append(f"{platform_id}:{e}")

    # 聚合：取最偏空的链信号优先
    bias_rank = {"bearish": 0, "volatile": 1, "neutral": 2, "bullish": 3}
    best = None
    for c in chains_out:
        a = c.get("analysis") or {}
        if best is None:
            best = a
            continue
        if bias_rank.get(a.get("bias"), 9) < bias_rank.get(best.get("bias"), 9):
            best = a
        elif a.get("bias") == best.get("bias") and float(a.get("outflow_share_pct") or 0) > float(
            best.get("outflow_share_pct") or 0
        ):
            best = a

    return {
        "coingecko_id": coingecko_id,
        "symbol": symbol,
        "name": name,
        "chains": chains_out,
        "aggregate": best
        or {
            "signal": "no_data",
            "signal_label": "暂无链上数据",
            "bias": "neutral",
            "action": "检查合约是否已部署或数据源限流",
            "outflow_share_pct": 0,
            "outflow_count": 0,
            "has_baseline": False,
        },
        "errors": errors,
        "generated_at_cst": _now_cst().isoformat(),
    }


WATCH_SNAPSHOT_NAME = "alpha_holders_watch.json"


def _watch_snapshot_path() -> Path:
    return resolve_data_dir() / WATCH_SNAPSHOT_NAME


def save_watch_snapshot(payload: Dict[str, Any]) -> None:
    path = _watch_snapshot_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_watch_snapshot() -> Dict[str, Any]:
    path = _watch_snapshot_path()
    if not path.is_file():
        return {
            "ok": False,
            "error": "no_snapshot",
            "message": "尚无链上持仓快照，请点击「刷新持仓」。",
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("bad snapshot")
        data["snapshot_source"] = "disk"
        return data
    except Exception as e:
        return {"ok": False, "error": "snapshot_corrupt", "message": str(e)}


def _phase_from_item(item: Dict[str, Any]) -> Optional[str]:
    if item.get("phase"):
        return str(item.get("phase"))
    start_s = str(item.get("start_at_cst") or "")
    if not start_s:
        return None
    try:
        start = datetime.fromisoformat(start_s.replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=CST)
        start = start.astimezone(CST)
        delta_min = (start - _now_cst()).total_seconds() / 60.0
        if -10 <= delta_min <= 0:
            return "early_window"
        if 0 < delta_min <= 24 * 60:
            return "upcoming"
        if delta_min < -10:
            return "live"
        return "scheduled"
    except Exception:
        return None


def watch_calendar_tokens(
    calendar_items: List[Dict[str, Any]],
    limit: int = HOLDERS_TOP_N,
) -> Dict[str, Any]:
    """对日历里带 coingecko_id 的标的做多链持仓监控。"""
    watches: List[Dict[str, Any]] = []
    for item in calendar_items:
        cid = str(item.get("coingecko_id") or "").strip()
        if not cid:
            continue
        try:
            w = watch_token_holders(
                cid,
                symbol=str(item.get("symbol") or ""),
                name=str(item.get("name") or ""),
                limit=limit,
                phase=_phase_from_item(item),
                calendar_item=item,
            )
            watches.append(w)
            time.sleep(0.4)
        except Exception as e:
            logger.warning("watch calendar token %s failed: %s", cid, e)
            watches.append(
                {
                    "coingecko_id": cid,
                    "symbol": item.get("symbol"),
                    "name": item.get("name"),
                    "chains": [],
                    "aggregate": {
                        "signal": "error",
                        "signal_label": "拉取失败",
                        "bias": "neutral",
                        "action": str(e),
                    },
                    "errors": [str(e)],
                }
            )
    payload = {
        "ok": True,
        "watches": watches,
        "generated_at_cst": _now_cst().isoformat(),
        "chains_supported": [
            {"platform_id": k, "chain": v["id"], "label": v["label"]}
            for k, v in CHAIN_CONFIG.items()
        ],
    }
    try:
        save_watch_snapshot(payload)
    except Exception as e:
        logger.warning("save holders watch snapshot failed: %s", e)

    # 每期总结写入历史（保留半年）
    try:
        from alpha_history import record_from_watch_payload

        record_from_watch_payload(watches, calendar_items)
    except Exception as e:
        logger.warning("alpha history record failed: %s", e)

    return payload


def refresh_calendar_holders_from_radar() -> Dict[str, Any]:
    """从 alpha_radar 日历拉持仓（供 API 调用）。"""
    from alpha_radar import _load_calendar

    return watch_calendar_tokens(_load_calendar())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    w = watch_token_holders("caldera", symbol="ERA", name="Caldera", limit=10)
    print(json.dumps({
        "symbol": w["symbol"],
        "n_chains": len(w["chains"]),
        "aggregate": w["aggregate"],
        "errors": w["errors"],
        "chains": [
            {
                "chain": c["chain"],
                "n": len(c.get("holders") or []),
                "signal": (c.get("analysis") or {}).get("signal"),
                "top3": [
                    {"addr": h["address"][:10], "share": h["share_pct"], "type": h["type"]}
                    for h in (c.get("holders") or [])[:3]
                ],
            }
            for c in w["chains"]
        ],
    }, ensure_ascii=False, indent=2))
