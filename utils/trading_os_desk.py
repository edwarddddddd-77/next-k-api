"""Trading OS desk extensions: wallets, alts, risk, alerts, strategy link."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from quant.common.paths import resolve_data_dir

logger = logging.getLogger(__name__)

UA = "NextK-TradingOS-Desk/1.0"
CST = timezone(timedelta(hours=8))
_lock = threading.Lock()

WALLETS_FILE = "trading_os_wallets.json"
WALLETS_SNAP = "trading_os_wallets_snap.json"
ALTS_FILE = "trading_os_alts.json"
ALTS_SNAP = "trading_os_alts_snap.json"
ALERT_STATE = "trading_os_alert_state.json"
DESK_BUNDLE = "trading_os_desk_bundle.json"

BINANCE = "https://api.binance.com"
MEMPOOL = "https://mempool.space/api"
SOL_RPC = (os.getenv("SOLANA_RPC_URL") or "https://api.mainnet-beta.solana.com").strip()



def _now_cst() -> datetime:
    return datetime.now(CST)


def _path(name: str) -> Path:
    return resolve_data_dir() / name


def _read_json(name: str, default: Any) -> Any:
    path = _path(name)
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(name: str, data: Any) -> None:
    path = _path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _http_json(url: str, *, timeout: float = 20.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        # mempool.space sometimes flakes; blockstream.info has compatible address txs
        if "mempool.space" in url and "/address/" in url and "/txs" in url:
            alt = url.replace("https://mempool.space/api", "https://blockstream.info/api")
            req2 = urllib.request.Request(alt, headers={"User-Agent": UA, "Accept": "application/json"})
            with urllib.request.urlopen(req2, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        raise


# ---------- wallets ----------

# CEX hot wallets we probe for large withdrawals → receiver = accumulation candidate.
# (Not the watch targets themselves — those are the receivers.)
CEX_PROBE_BTC: list[dict[str, str]] = [
    {
        "address": "1Kr6QSydW9bFQG1mXiPNNu6WpJGmUa9i1g",
        "label": "Bitfinex热钱包",
        "source": "bitfinex-por",
    },
    {
        # Official Bitfinex cold — rare but large outs are high-signal custody moves
        "address": "bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97",
        "label": "Bitfinex冷钱包",
        "source": "bitfinex-por",
    },
]

# Bitfinex official ETH hot (PoR) — probe for large ETH withdrawals
CEX_PROBE_ETH: list[dict[str, str]] = [
    {
        "address": "0x77134cbc06cb00b66f4c7e623d5fdbf6777635ec",
        "label": "Bitfinex-ETH热",
        "source": "bitfinex-por",
    },
]

# Known CEX / plumbing addresses — never treat as "smart money" watch targets
CEX_SKIP_BTC = {
    p["address"].lower() for p in CEX_PROBE_BTC
} | {
    "34xp4vRoCGJym3xR7yCVPFHoCNxv4Twseo".lower(),  # Binance cold (rich list)
    "3M219KR5vEneNb47ewrPfWyb5jQ2DjxRP6".lower(),
    "bc1ql49ydapnjafl5t2cp9zqpjwe6pdgmxy98859v2".lower(),  # Robinhood cold
    "bc1qazcm763858nkj2dj986etajv6wquslv8uxwczt".lower(),  # Bitfinex hack recovery
    "3JZq4atUahhuA9rLhXLMhhTo133J9rF97j".lower(),  # Bitfinex cold (legacy)
}

CEX_SKIP_ETH = {p["address"].lower() for p in CEX_PROBE_ETH} | {
    "0x742d35cc6634c0532925a3b844bc454e4438f44e",  # Bitfinex ETH cold
    "0xc61b9bb3a7a0767e3179713f3a5c7a9aedce193c",
    "0x876eabf441b2ee5b5b0554fd502a8e0600950cfa",
}

# Bitfinex official SOL hot/cold (PoR)
CEX_PROBE_SOL: list[dict[str, str]] = [
    {
        "address": "FxteHmLwG9nk1eL4pjNve3Eub2goGkkz6g6TbvdmW46a",
        "label": "Bitfinex-SOL热",
        "source": "bitfinex-por",
    },
]

CEX_SKIP_SOL = {p["address"] for p in CEX_PROBE_SOL} | {
    "FyJBKcfcEBzGN74uNxZ95GxnCxeuJJujQCELpPv14ZfN",  # Bitfinex SOL cold
}

LEGACY_BAD_SOURCES = {
    "bitinfocharts",
    "bitfinex-por",
    "ens/public",
    "lookonchain",
    "lookonchain/arkham",
    "default",
}

DISCOVER_SNAP = "trading_os_discover_snap.json"


def _detect_chain(address: str) -> str:
    a = address.strip()
    if re.fullmatch(r"0x[a-fA-F0-9]{40}", a):
        return "eth"
    if re.fullmatch(r"[13][a-km-zA-HJ-NP-Z1-9]{25,34}", a) or re.fullmatch(
        r"bc1[a-z0-9]{25,87}", a
    ):
        return "btc"
    # Solana base58 pubkeys (case-sensitive); avoid 0/O/I/l
    if re.fullmatch(r"[1-9A-HJ-NP-Za-km-z]{32,44}", a) and not a.startswith("bc1"):
        return "sol"
    return "unknown"


def _addr_key(address: str, chain: str | None = None) -> str:
    """Dedup key: SOL is case-sensitive; BTC/ETH are not."""
    a = (address or "").strip()
    ch = (chain or _detect_chain(a)).lower()
    if ch == "sol":
        return a
    return a.lower()


def _is_legacy_junk_watchlist(wallets: list[dict[str, Any]]) -> bool:
    """Old mistaken seed = CEX cold / celebrity wallets, not accumulation targets."""
    if not wallets:
        return True
    if not all(w.get("seeded") for w in wallets):
        return False
    sources = {str(w.get("source") or "") for w in wallets}
    return bool(sources & LEGACY_BAD_SOURCES) and not any(
        str(w.get("source") or "").startswith("cex_outflow") for w in wallets
    )


def discover_btc_accumulators(*, min_btc: float | None = None, per_probe: int = 40) -> dict[str, Any]:
    """Scan CEX hot/cold recent outs; receivers of large withdrawals = watch candidates.

    This matches「跟踪钱包」: follow where coins leave the exchange to, not the vault itself.
    """
    min_btc = float(min_btc if min_btc is not None else (os.getenv("TRADING_OS_DISCOVER_MIN_BTC", "10") or 10))
    min_sats = int(min_btc * 1e8)
    found: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    for probe in CEX_PROBE_BTC:
        probe_addr = probe["address"]
        try:
            txs = _http_json(
                f"{MEMPOOL}/address/{urllib.parse.quote(probe_addr)}/txs",
                timeout=45.0,
            )
            if not isinstance(txs, list):
                continue
            for tx in txs[:per_probe]:
                txid = str(tx.get("txid") or "")[:16]
                for vout in tx.get("vout") or []:
                    val = int(vout.get("value") or 0)
                    if val < min_sats:
                        continue
                    dest = str(vout.get("scriptpubkey_address") or "").strip()
                    if not dest:
                        continue
                    if dest.lower() == probe_addr.lower():
                        continue
                    if dest.lower() in CEX_SKIP_BTC:
                        continue
                    if _detect_chain(dest) != "btc":
                        continue
                    btc = val / 1e8
                    key = dest.lower()
                    prev = found.get(key)
                    if prev is None or btc > float(prev.get("withdraw_btc") or 0):
                        found[key] = {
                            "address": dest,
                            "chain": "btc",
                            "label": f"吸筹·{probe['label'][:6]}·{btc:.0f}BTC",
                            "source": "cex_outflow",
                            "probe": probe["label"],
                            "withdraw_btc": round(btc, 4),
                            "txid_prefix": txid,
                            "discovered_at_cst": _now_cst().isoformat(),
                        }
        except Exception as e:
            errors.append(f"{probe_addr}:{e}")
            logger.warning("discover probe failed %s: %s", probe_addr, e)

    rows = sorted(found.values(), key=lambda r: -float(r.get("withdraw_btc") or 0))
    max_n = int(os.getenv("TRADING_OS_DISCOVER_MAX_BTC", "12") or 12)
    rows = rows[: max(3, max_n)]
    snap = {
        "ok": True,
        "chain": "btc",
        "fetched_at_cst": _now_cst().isoformat(),
        "min_btc": min_btc,
        "candidates": rows,
        "count": len(rows),
        "errors": errors,
    }
    return snap


def discover_eth_accumulators(*, min_eth: float | None = None, per_probe: int = 50) -> dict[str, Any]:
    """Scan CEX ETH hot wallet outs; large withdrawals → accumulation candidates."""
    min_eth = float(min_eth if min_eth is not None else (os.getenv("TRADING_OS_DISCOVER_MIN_ETH", "50") or 50))
    key = (os.getenv("ETHPLORER_API_KEY") or "freekey").strip()
    found: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    for probe in CEX_PROBE_ETH:
        probe_addr = probe["address"]
        try:
            url = (
                f"https://api.ethplorer.io/getAddressTransactions/{urllib.parse.quote(probe_addr)}"
                f"?apiKey={urllib.parse.quote(key)}&limit={per_probe}"
            )
            txs = _http_json(url, timeout=45.0)
            if not isinstance(txs, list):
                continue
            for tx in txs:
                if tx.get("success") is False:
                    continue
                # Ethplorer: value is ETH amount
                try:
                    val = float(tx.get("value") or 0)
                except (TypeError, ValueError):
                    val = 0.0
                if val < min_eth:
                    continue
                dest = str(tx.get("to") or "").strip()
                src = str(tx.get("from") or "").strip()
                # Only count outs FROM the probe
                if src.lower() != probe_addr.lower():
                    continue
                if not dest or dest.lower() == probe_addr.lower():
                    continue
                if dest.lower() in CEX_SKIP_ETH:
                    continue
                if _detect_chain(dest) != "eth":
                    continue
                k = dest.lower()
                prev = found.get(k)
                if prev is None or val > float(prev.get("withdraw_eth") or 0):
                    found[k] = {
                        "address": dest,
                        "chain": "eth",
                        "label": f"吸筹·{probe['label'][:10]}·{val:.0f}ETH",
                        "source": "cex_outflow",
                        "probe": probe["label"],
                        "withdraw_eth": round(val, 4),
                        "txid_prefix": str(tx.get("hash") or "")[:16],
                        "discovered_at_cst": _now_cst().isoformat(),
                    }
        except Exception as e:
            errors.append(f"{probe_addr}:{e}")
            logger.warning("eth discover probe failed %s: %s", probe_addr, e)

    rows = sorted(found.values(), key=lambda r: -float(r.get("withdraw_eth") or 0))
    max_n = int(os.getenv("TRADING_OS_DISCOVER_MAX_ETH", "8") or 8)
    rows = rows[: max(2, max_n)]
    return {
        "ok": True,
        "chain": "eth",
        "fetched_at_cst": _now_cst().isoformat(),
        "min_eth": min_eth,
        "candidates": rows,
        "count": len(rows),
        "errors": errors,
    }


def _sol_rpc(method: str, params: list[Any], *, timeout: float = 40.0) -> Any:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = urllib.request.Request(
        SOL_RPC,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": UA},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = json.loads(resp.read().decode("utf-8", errors="replace"))
    if raw.get("error"):
        raise RuntimeError(str(raw["error"]))
    return raw.get("result")


def discover_sol_accumulators(*, min_sol: float | None = None, per_probe: int = 25) -> dict[str, Any]:
    """Scan CEX SOL hot wallet outs via balance deltas; large withdrawals → candidates."""
    min_sol = float(min_sol if min_sol is not None else (os.getenv("TRADING_OS_DISCOVER_MIN_SOL", "100") or 100))
    found: dict[str, dict[str, Any]] = {}
    errors: list[str] = []

    for probe in CEX_PROBE_SOL:
        probe_addr = probe["address"]
        try:
            sigs = _sol_rpc("getSignaturesForAddress", [probe_addr, {"limit": per_probe}]) or []
            for s in sigs:
                if s.get("err"):
                    continue
                sig = s.get("signature")
                if not sig:
                    continue
                try:
                    tx = _sol_rpc(
                        "getTransaction",
                        [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
                        timeout=40.0,
                    )
                except Exception:
                    continue
                if not tx:
                    continue
                meta = tx.get("meta") or {}
                msg = ((tx.get("transaction") or {}).get("message")) or {}
                keys = msg.get("accountKeys") or []
                accs: list[str] = []
                for k in keys:
                    if isinstance(k, dict):
                        accs.append(str(k.get("pubkey") or ""))
                    else:
                        accs.append(str(k))
                pre = meta.get("preBalances") or []
                post = meta.get("postBalances") or []
                if len(accs) != len(pre) or len(pre) != len(post):
                    continue
                # Probe must have sent SOL (balance down)
                try:
                    pi = accs.index(probe_addr)
                except ValueError:
                    continue
                probe_delta = (post[pi] - pre[pi]) / 1e9
                if probe_delta >= -min_sol:
                    continue
                for i, addr in enumerate(accs):
                    if not addr or addr == probe_addr:
                        continue
                    if addr in CEX_SKIP_SOL:
                        continue
                    if _detect_chain(addr) != "sol":
                        continue
                    delta = (post[i] - pre[i]) / 1e9
                    if delta < min_sol:
                        continue
                    prev = found.get(addr)
                    if prev is None or delta > float(prev.get("withdraw_sol") or 0):
                        found[addr] = {
                            "address": addr,
                            "chain": "sol",
                            "label": f"吸筹·{probe['label'][:10]}·{delta:.0f}SOL",
                            "source": "cex_outflow",
                            "probe": probe["label"],
                            "withdraw_sol": round(delta, 4),
                            "txid_prefix": str(sig)[:16],
                            "discovered_at_cst": _now_cst().isoformat(),
                        }
        except Exception as e:
            errors.append(f"{probe_addr}:{e}")
            logger.warning("sol discover probe failed %s: %s", probe_addr, e)

    rows = sorted(found.values(), key=lambda r: -float(r.get("withdraw_sol") or 0))
    max_n = int(os.getenv("TRADING_OS_DISCOVER_MAX_SOL", "8") or 8)
    rows = rows[: max(2, max_n)]
    return {
        "ok": True,
        "chain": "sol",
        "fetched_at_cst": _now_cst().isoformat(),
        "min_sol": min_sol,
        "candidates": rows,
        "count": len(rows),
        "errors": errors,
    }


def discover_accumulators() -> dict[str, Any]:
    """BTC + ETH + SOL CEX-outflow discovery in one pass."""
    btc = discover_btc_accumulators()
    eth = discover_eth_accumulators()
    sol = discover_sol_accumulators()
    candidates = (
        list(btc.get("candidates") or [])
        + list(eth.get("candidates") or [])
        + list(sol.get("candidates") or [])
    )
    snap = {
        "ok": True,
        "fetched_at_cst": _now_cst().isoformat(),
        "candidates": candidates,
        "count": len(candidates),
        "btc": {"count": btc.get("count"), "min_btc": btc.get("min_btc"), "errors": btc.get("errors")},
        "eth": {"count": eth.get("count"), "min_eth": eth.get("min_eth"), "errors": eth.get("errors")},
        "sol": {"count": sol.get("count"), "min_sol": sol.get("min_sol"), "errors": sol.get("errors")},
        "errors": list(btc.get("errors") or [])
        + list(eth.get("errors") or [])
        + list(sol.get("errors") or []),
    }
    _write_json(DISCOVER_SNAP, snap)
    return snap


def ensure_smart_watchlist(*, force: bool = False) -> dict[str, Any]:
    """Build/refresh watchlist from CEX-outflow discovery (BTC+ETH+SOL)."""
    data = _read_json(WALLETS_FILE, {"wallets": []})
    wallets = list(data.get("wallets") or [])
    junk = _is_legacy_junk_watchlist(wallets)

    disc = discover_accumulators()
    candidates = disc.get("candidates") or []

    with _lock:
        data = _read_json(WALLETS_FILE, {"wallets": []})
        wallets = list(data.get("wallets") or [])
        if force or junk:
            wallets = [
                w
                for w in wallets
                if not (
                    w.get("seeded")
                    and str(w.get("source") or "") in LEGACY_BAD_SOURCES
                )
            ]
        existing = {_addr_key(str(w.get("address") or ""), w.get("chain")): w for w in wallets}
        added: list[dict[str, Any]] = []
        for c in candidates:
            addr = c["address"]
            chain = str(c.get("chain") or "btc")
            key = _addr_key(addr, chain)
            if chain == "sol" and addr in CEX_SKIP_SOL:
                continue
            if chain == "eth" and addr.lower() in CEX_SKIP_ETH:
                continue
            if chain == "btc" and addr.lower() in CEX_SKIP_BTC:
                continue
            if key in existing:
                if existing[key].get("source") == "cex_outflow":
                    if chain == "eth":
                        existing[key]["withdraw_eth"] = c.get("withdraw_eth")
                    elif chain == "sol":
                        existing[key]["withdraw_sol"] = c.get("withdraw_sol")
                    else:
                        existing[key]["withdraw_btc"] = c.get("withdraw_btc")
                    existing[key]["label"] = c.get("label") or existing[key].get("label")
                continue
            item = {
                "address": addr,
                "label": c.get("label") or addr[:12],
                "chain": chain,
                "source": "cex_outflow",
                "seeded": True,
                "probe": c.get("probe"),
                "withdraw_btc": c.get("withdraw_btc"),
                "withdraw_eth": c.get("withdraw_eth"),
                "withdraw_sol": c.get("withdraw_sol"),
                "added_at_cst": _now_cst().isoformat(),
            }
            wallets.append(item)
            existing[key] = item
            added.append(item)

        auto = [w for w in wallets if w.get("source") == "cex_outflow"]
        manual = [w for w in wallets if w.get("source") != "cex_outflow"]
        max_btc = int(os.getenv("TRADING_OS_DISCOVER_MAX_BTC", "12") or 12)
        max_eth = int(os.getenv("TRADING_OS_DISCOVER_MAX_ETH", "8") or 8)
        max_sol = int(os.getenv("TRADING_OS_DISCOVER_MAX_SOL", "8") or 8)
        auto_btc = sorted(
            [w for w in auto if w.get("chain") == "btc"],
            key=lambda w: -float(w.get("withdraw_btc") or 0),
        )[:max_btc]
        auto_eth = sorted(
            [w for w in auto if w.get("chain") == "eth"],
            key=lambda w: -float(w.get("withdraw_eth") or 0),
        )[:max_eth]
        auto_sol = sorted(
            [w for w in auto if w.get("chain") == "sol"],
            key=lambda w: -float(w.get("withdraw_sol") or 0),
        )[:max_sol]
        wallets = manual + auto_btc + auto_eth + auto_sol
        _write_json(
            WALLETS_FILE,
            {
                "wallets": wallets,
                "mode": "cex_outflow_discovery",
                "last_discover_cst": _now_cst().isoformat(),
            },
        )
    return {
        "ok": True,
        "mode": "cex_outflow_discovery",
        "added": len(added),
        "count": len(wallets),
        "wallets": wallets,
        "discover": {
            "count": disc.get("count"),
            "btc": disc.get("btc"),
            "eth": disc.get("eth"),
            "sol": disc.get("sol"),
            "errors": disc.get("errors"),
        },
        "new": added,
    }


def seed_default_wallets(*, force: bool = False) -> dict[str, Any]:
    """Back-compat alias → smart discovery (not celebrity/CEX-cold list)."""
    return ensure_smart_watchlist(force=force)


def list_wallets() -> dict[str, Any]:
    data = _read_json(WALLETS_FILE, {"wallets": []})
    wallets = data.get("wallets") or []
    if not wallets or _is_legacy_junk_watchlist(wallets):
        out = ensure_smart_watchlist(force=True)
        return {
            "ok": True,
            "wallets": out.get("wallets") or [],
            "auto_seeded": True,
            "mode": "cex_outflow_discovery",
        }
    return {"ok": True, "wallets": wallets, "mode": data.get("mode") or "manual"}


def add_wallet(address: str, *, label: str = "", chain: str = "") -> dict[str, Any]:
    addr = address.strip()
    if len(addr) < 10:
        raise ValueError("address_invalid")
    ch = (chain or _detect_chain(addr)).lower()
    if ch not in ("btc", "eth", "sol"):
        raise ValueError("chain_unsupported")
    with _lock:
        data = _read_json(WALLETS_FILE, {"wallets": []})
        wallets = list(data.get("wallets") or [])
        key = _addr_key(addr, ch)
        for w in wallets:
            if _addr_key(str(w.get("address") or ""), w.get("chain")) == key:
                raise ValueError("address_exists")
        item = {
            "address": addr,
            "label": (label or "").strip() or addr[:10],
            "chain": ch,
            "added_at_cst": _now_cst().isoformat(),
        }
        wallets.append(item)
        _write_json(WALLETS_FILE, {"wallets": wallets})
    return {"ok": True, "wallet": item}


def remove_wallet(address: str) -> dict[str, Any]:
    raw = address.strip()
    key_l = raw.lower()
    with _lock:
        data = _read_json(WALLETS_FILE, {"wallets": []})
        wallets = []
        for w in data.get("wallets") or []:
            wa = str(w.get("address") or "")
            # SOL exact; others case-insensitive
            if w.get("chain") == "sol":
                if wa == raw:
                    continue
            elif wa.lower() == key_l:
                continue
            wallets.append(w)
        _write_json(WALLETS_FILE, {"wallets": wallets})
    return {"ok": True, "removed": address}


def _fetch_btc_balance(address: str) -> dict[str, Any]:
    raw = _http_json(f"{MEMPOOL}/address/{urllib.parse.quote(address)}", timeout=25.0)
    chain = raw.get("chain_stats") or {}
    mempool = raw.get("mempool_stats") or {}
    funded = float(chain.get("funded_txo_sum") or 0) + float(mempool.get("funded_txo_sum") or 0)
    spent = float(chain.get("spent_txo_sum") or 0) + float(mempool.get("spent_txo_sum") or 0)
    bal_sats = funded - spent
    return {
        "balance": bal_sats / 1e8,
        "unit": "BTC",
        "tx_count": int(chain.get("tx_count") or 0),
    }


def _fetch_eth_balance(address: str) -> dict[str, Any]:
    key = (os.getenv("ETHPLORER_API_KEY") or "freekey").strip()
    url = (
        f"https://api.ethplorer.io/getAddressInfo/{urllib.parse.quote(address)}"
        f"?apiKey={urllib.parse.quote(key)}"
    )
    raw = _http_json(url, timeout=25.0)
    eth = raw.get("ETH") or {}
    bal = float(eth.get("balance") or 0)
    tokens = raw.get("tokens") or []
    top = []
    for t in tokens[:8]:
        info = t.get("tokenInfo") or {}
        dec = int(info.get("decimals") or 0)
        raw_bal = float(t.get("balance") or 0)
        qty = raw_bal / (10**dec) if dec >= 0 else raw_bal
        top.append(
            {
                "symbol": info.get("symbol") or "?",
                "balance": round(qty, 6),
            }
        )
    return {"balance": bal, "unit": "ETH", "tokens": top, "tx_count": int(raw.get("countTxs") or 0)}


def _fetch_sol_balance(address: str) -> dict[str, Any]:
    raw = _sol_rpc("getBalance", [address], timeout=25.0) or {}
    lamports = float(raw.get("value") or 0)
    return {"balance": lamports / 1e9, "unit": "SOL", "tx_count": None}


def refresh_wallets(*, min_move_pct: float | None = None) -> dict[str, Any]:
    # Auto-discover CEX-outflow accumulators each cycle (no hand-fill)
    discover_meta = ensure_smart_watchlist(force=False)
    if min_move_pct is None:
        min_move_pct = float(os.getenv("TRADING_OS_WALLET_MOVE_PCT", "3") or 3)
    abs_btc = float(os.getenv("TRADING_OS_WALLET_ABS_BTC", "10") or 10)
    abs_eth = float(os.getenv("TRADING_OS_WALLET_ABS_ETH", "200") or 200)
    abs_sol = float(os.getenv("TRADING_OS_WALLET_ABS_SOL", "200") or 200)

    wallets = list_wallets()["wallets"]
    prev = _read_json(WALLETS_SNAP, {"balances": {}})
    prev_map = prev.get("balances") or {}
    balances: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    errors: list[str] = []

    # Surface newly discovered accumulators as alertable events
    for nw in discover_meta.get("new") or []:
        chain = nw.get("chain") or "btc"
        if chain == "eth":
            amt, unit = nw.get("withdraw_eth"), "ETH"
        elif chain == "sol":
            amt, unit = nw.get("withdraw_sol"), "SOL"
        else:
            amt, unit = nw.get("withdraw_btc"), "BTC"
        events.append(
            {
                "address": nw.get("address"),
                "label": nw.get("label"),
                "chain": chain,
                "old_balance": 0,
                "new_balance": amt,
                "change_pct": None,
                "unit": unit,
                "note": "discovered_cex_outflow",
                "withdraw_btc": nw.get("withdraw_btc"),
                "withdraw_eth": nw.get("withdraw_eth"),
                "withdraw_sol": nw.get("withdraw_sol"),
                "probe": nw.get("probe"),
            }
        )

    for w in wallets:
        addr = str(w.get("address") or "")
        chain = str(w.get("chain") or "")
        key = _addr_key(addr, chain)
        try:
            if chain == "btc":
                info = _fetch_btc_balance(addr)
            elif chain == "eth":
                info = _fetch_eth_balance(addr)
            elif chain == "sol":
                info = _fetch_sol_balance(addr)
            else:
                raise RuntimeError("unsupported_chain")
            balances[key] = {
                **info,
                "address": addr,
                "label": w.get("label"),
                "chain": chain,
                "checked_at_cst": _now_cst().isoformat(),
            }
            old = prev_map.get(key) or prev_map.get(addr.lower()) or {}
            old_bal = float(old.get("balance") or 0)
            new_bal = float(info.get("balance") or 0)
            delta = new_bal - old_bal
            if chain == "btc":
                abs_thr = abs_btc
            elif chain == "sol":
                abs_thr = abs_sol
            else:
                abs_thr = abs_eth
            pct_hit = False
            abs_hit = False
            chg = None
            if old_bal > 0:
                chg = (new_bal - old_bal) / old_bal * 100.0
                pct_hit = abs(chg) >= min_move_pct
            if old and abs(delta) >= abs_thr:
                abs_hit = True
            if pct_hit or abs_hit:
                events.append(
                    {
                        "address": addr,
                        "label": w.get("label"),
                        "chain": chain,
                        "old_balance": old_bal,
                        "new_balance": new_bal,
                        "delta": round(delta, 6),
                        "change_pct": round(chg, 2) if chg is not None else None,
                        "unit": info.get("unit"),
                        "trigger": "pct" if pct_hit else "abs",
                    }
                )
            elif new_bal > 0 and not old:
                events.append(
                    {
                        "address": addr,
                        "label": w.get("label"),
                        "chain": chain,
                        "old_balance": 0,
                        "new_balance": new_bal,
                        "change_pct": None,
                        "unit": info.get("unit"),
                        "note": "first_snapshot",
                    }
                )
        except Exception as e:
            errors.append(f"{addr}:{e}")
            logger.warning("wallet refresh failed %s: %s", addr, e)

    snap = {
        "ok": True,
        "fetched_at_cst": _now_cst().isoformat(),
        "balances": balances,
        "events": events,
        "errors": errors,
        "count": len(balances),
    }
    _write_json(WALLETS_SNAP, snap)
    return snap


def load_wallets_snap() -> dict[str, Any]:
    snap = _read_json(WALLETS_SNAP, {})
    if snap:
        return {**snap, "cached": True}
    return refresh_wallets()


# ---------- alts / orderbook radar ----------

# Majors / stables excluded from “山寨” auto universe
ALT_EXCLUDE_BASES = {
    "BTC",
    "ETH",
    "SOL",
    "BNB",
    "XRP",
    "USDC",
    "FDUSD",
    "TUSD",
    "DAI",
    "USDE",
    "EUR",
    "TRY",
    "BRL",
    "AEUR",
    "USDT",
    "WBTC",
    "STETH",
    "WETH",
    "TBTC",
}


def discover_alt_universe(*, limit: int | None = None) -> dict[str, Any]:
    """Pick high-liquidity Binance USDT alts by 24h quote volume (auto, no hand-fill)."""
    lim = int(limit if limit is not None else (os.getenv("TRADING_OS_ALTS_AUTO_N", "25") or 25))
    lim = max(5, min(lim, 40))
    min_vol = float(os.getenv("TRADING_OS_ALTS_MIN_QUOTE_VOL", "2000000") or 2_000_000)
    tickers = _http_json(f"{BINANCE}/api/v3/ticker/24hr", timeout=30.0)
    if not isinstance(tickers, list):
        raise RuntimeError("binance_ticker_failed")
    rows: list[dict[str, Any]] = []
    for t in tickers:
        sym = str(t.get("symbol") or "")
        if not sym.endswith("USDT"):
            continue
        base = sym[:-4]
        if base in ALT_EXCLUDE_BASES or base.endswith("UP") or base.endswith("DOWN"):
            continue
        # skip leveraged tokens like BTCUP already covered; also 3L/3S
        if base.endswith("3L") or base.endswith("3S") or base.endswith("2L") or base.endswith("2S"):
            continue
        try:
            qv = float(t.get("quoteVolume") or 0)
            chg = float(t.get("priceChangePercent") or 0)
        except (TypeError, ValueError):
            continue
        if qv < min_vol:
            continue
        rows.append(
            {
                "symbol": sym,
                "quote_volume_24h": qv,
                "change_24h_pct": chg,
                "last": float(t.get("lastPrice") or 0),
            }
        )
    rows.sort(key=lambda r: -float(r["quote_volume_24h"]))
    picked = rows[:lim]
    return {
        "ok": True,
        "mode": "auto",
        "symbols": [r["symbol"] for r in picked],
        "universe": picked,
        "count": len(picked),
        "min_quote_vol": min_vol,
    }


def list_alts() -> dict[str, Any]:
    data = _read_json(ALTS_FILE, {})
    mode = str(data.get("mode") or "auto")
    symbols = list(data.get("symbols") or [])
    if mode == "manual" and symbols:
        return {"ok": True, "mode": "manual", "symbols": symbols}
    # auto (default): refresh universe if empty or stale file
    if not symbols or mode != "manual":
        try:
            uni = discover_alt_universe()
            symbols = uni["symbols"]
            _write_json(
                ALTS_FILE,
                {
                    "mode": "auto",
                    "symbols": symbols,
                    "updated_at_cst": _now_cst().isoformat(),
                },
            )
            return {"ok": True, "mode": "auto", "symbols": symbols, "auto": True}
        except Exception as e:
            logger.warning("alt universe discover failed: %s", e)
            fallback = ["PEPEUSDT", "WIFUSDT", "BONKUSDT", "DOGEUSDT", "SHIBUSDT"]
            return {"ok": True, "mode": "auto", "symbols": symbols or fallback, "error": str(e)}
    return {"ok": True, "mode": mode, "symbols": symbols}


def set_alts(symbols: list[str], *, mode: str = "manual") -> dict[str, Any]:
    cleaned = []
    for s in symbols:
        u = str(s or "").strip().upper().replace("/", "").replace("-", "")
        if not u:
            continue
        if not u.endswith("USDT"):
            u = u + "USDT"
        if u not in cleaned:
            cleaned.append(u)
    m = "auto" if str(mode).lower() == "auto" else "manual"
    if m == "auto" or not cleaned:
        uni = discover_alt_universe()
        cleaned = uni["symbols"]
        m = "auto"
    _write_json(
        ALTS_FILE,
        {
            "mode": m,
            "symbols": cleaned[:40],
            "updated_at_cst": _now_cst().isoformat(),
        },
    )
    return {"ok": True, "mode": m, "symbols": cleaned[:40]}


def _scan_symbol(symbol: str, *, ticker: dict[str, Any] | None = None) -> dict[str, Any]:
    depth = _http_json(f"{BINANCE}/api/v3/depth?symbol={symbol}&limit=20", timeout=15.0)
    if ticker is None:
        ticker = _http_json(f"{BINANCE}/api/v3/ticker/24hr?symbol={symbol}", timeout=15.0)
    bids = depth.get("bids") or []
    asks = depth.get("asks") or []
    bid_qty = sum(float(x[1]) for x in bids)
    ask_qty = sum(float(x[1]) for x in asks)
    total = bid_qty + ask_qty
    imbalance = (bid_qty - ask_qty) / total if total > 0 else 0.0
    change = float(ticker.get("priceChangePercent") or 0)
    quote_vol = float(ticker.get("quoteVolume") or 0)
    last = float(ticker.get("lastPrice") or 0)
    # Heuristic: bid-heavy + not already pumped hard → possible stealth accumulation
    flag = imbalance >= 0.12 and change < 25 and quote_vol > 500_000
    return {
        "symbol": symbol,
        "last": last,
        "change_24h_pct": round(change, 2),
        "quote_volume_24h": round(quote_vol, 0),
        "imbalance": round(imbalance, 4),
        "bid_qty_top20": round(bid_qty, 2),
        "ask_qty_top20": round(ask_qty, 2),
        "accum_flag": flag,
    }


def refresh_alts() -> dict[str, Any]:
    data = _read_json(ALTS_FILE, {})
    mode = str(data.get("mode") or "auto")
    ticker_map: dict[str, dict[str, Any]] = {}
    if mode != "manual":
        try:
            uni = discover_alt_universe()
            symbols = uni["symbols"]
            for u in uni.get("universe") or []:
                ticker_map[u["symbol"]] = {
                    "priceChangePercent": u.get("change_24h_pct"),
                    "quoteVolume": u.get("quote_volume_24h"),
                    "lastPrice": u.get("last"),
                }
            _write_json(
                ALTS_FILE,
                {
                    "mode": "auto",
                    "symbols": symbols,
                    "updated_at_cst": _now_cst().isoformat(),
                },
            )
        except Exception as e:
            logger.warning("alt auto universe failed, using saved list: %s", e)
            symbols = list_alts()["symbols"]
    else:
        symbols = list(data.get("symbols") or []) or list_alts()["symbols"]

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for sym in symbols:
        try:
            rows.append(_scan_symbol(sym, ticker=ticker_map.get(sym)))
        except Exception as e:
            errors.append(f"{sym}:{e}")
    rows.sort(key=lambda r: (not r.get("accum_flag"), -float(r.get("imbalance") or 0)))
    snap = {
        "ok": True,
        "fetched_at_cst": _now_cst().isoformat(),
        "mode": mode if mode == "manual" else "auto",
        "symbols": symbols,
        "rows": rows,
        "flagged": [r for r in rows if r.get("accum_flag")],
        "errors": errors,
        "count": len(rows),
    }
    _write_json(ALTS_SNAP, snap)
    return snap


def load_alts_snap() -> dict[str, Any]:
    snap = _read_json(ALTS_SNAP, {})
    if snap:
        return {**snap, "cached": True}
    return refresh_alts()


# ---------- risk ----------

def compute_risk(equity_usd: float | None = None) -> dict[str, Any]:
    eq = equity_usd
    if eq is None:
        raw = (os.getenv("TRADING_OS_EQUITY_USD") or "").strip()
        eq = float(raw) if raw else 10_000.0
    eq = max(0.0, float(eq))
    satellite_pct = float(os.getenv("TRADING_OS_SATELLITE_PCT", "5") or 5)
    single_risk_pct = float(os.getenv("TRADING_OS_SINGLE_RISK_PCT", "1") or 1)
    lev_max = float(os.getenv("TRADING_OS_LEVERAGE_MAX", "3") or 3)
    satellite_cap = eq * satellite_pct / 100.0
    single_risk = eq * single_risk_pct / 100.0
    # Example notionals for stop distances
    examples = []
    for stop_pct in (2.0, 5.0, 10.0):
        notional = single_risk / (stop_pct / 100.0) if stop_pct else 0
        examples.append(
            {
                "stop_pct": stop_pct,
                "max_notional_usd": round(notional, 2),
                "note": "愿亏金额 ÷ 止损幅度",
            }
        )
    return {
        "ok": True,
        "equity_usd": eq,
        "satellite_pct_max": satellite_pct,
        "satellite_cap_usd": round(satellite_cap, 2),
        "single_risk_pct_max": single_risk_pct,
        "single_risk_usd": round(single_risk, 2),
        "leverage_max": lev_max,
        "position_examples": examples,
        "formula": "仓位名义 ≈ 愿亏金额 ÷ 止损%",
    }


# ---------- strategy link ----------

def load_strategy_link(limit: int = 20) -> dict[str, Any]:
    out: dict[str, Any] = {"ok": True, "lanes": {}}
    try:
        from quant.engine.strategy_signals import VALID_LANES, list_strategy_signals

        for lane in sorted(VALID_LANES):
            try:
                out["lanes"][lane] = list_strategy_signals(lane=lane, limit=limit)
            except Exception as e:
                out["lanes"][lane] = {"ok": False, "error": str(e)}
    except Exception as e:
        out["ok"] = False
        out["error"] = str(e)
    return out


# ---------- alerts / full auto monitor ----------

MONITOR_STATE = ALERT_STATE  # same file


def _alerts_enabled() -> bool:
    return (os.getenv("TRADING_OS_ALERTS", "1") or "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _send_tg(text: str) -> bool:
    try:
        from accumulation_radar import send_telegram

        send_telegram(text)
        return True
    except Exception as e:
        logger.warning("telegram send failed: %s", e)
        return False


def _cooldown_ok(state: dict[str, Any], key: str, hours: float) -> bool:
    raw = (state.get("cooldowns") or {}).get(key)
    if not raw:
        return True
    try:
        last = datetime.fromisoformat(str(raw))
        if last.tzinfo is None:
            last = last.replace(tzinfo=CST)
        return (_now_cst() - last).total_seconds() >= hours * 3600
    except Exception:
        return True


def _mark_cooldown(state: dict[str, Any], key: str) -> None:
    cds = dict(state.get("cooldowns") or {})
    cds[key] = _now_cst().isoformat()
    # trim
    if len(cds) > 200:
        items = sorted(cds.items(), key=lambda x: x[1])
        cds = dict(items[-150:])
    state["cooldowns"] = cds


def maybe_alert(
    *,
    phase: str | None,
    wallet_events: list[dict[str, Any]] | None = None,
    alt_flags: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Legacy entry; prefer run_auto_monitor for full coverage."""
    return run_auto_monitor(
        {
            "score": {"phase": phase} if phase else {},
            "desk": {
                "wallets": {"events": wallet_events or []},
                "alts": {"flagged": alt_flags or []},
            },
        }
    )


def run_auto_monitor(snap: dict[str, Any] | None = None) -> dict[str, Any]:
    """Full-auto: phase / score / wallets / alts / strategy / digest / boot.

    Called after each scheduled refresh. No page open required.
    """
    snap = snap or {}
    enabled = _alerts_enabled()
    state = _read_json(MONITOR_STATE, {})
    sent: list[str] = []
    score = snap.get("score") or {}
    phase = score.get("phase")
    desk = snap.get("desk") or {}
    wallets = desk.get("wallets") or {}
    alts = desk.get("alts") or {}
    strategy = desk.get("strategy") or {}

    alt_cd_h = float(os.getenv("TRADING_OS_ALT_ALERT_COOLDOWN_H", "6") or 6)
    score_cd_h = float(os.getenv("TRADING_OS_SCORE_ALERT_COOLDOWN_H", "4") or 4)
    digest_h = float(os.getenv("TRADING_OS_DIGEST_HOURS", "8") or 8)
    boot_once = (os.getenv("TRADING_OS_BOOT_PING", "1") or "1").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )

    if not enabled:
        state["last_run_cst"] = _now_cst().isoformat()
        state["mode"] = "auto_disabled"
        _write_json(MONITOR_STATE, state)
        return {
            "ok": True,
            "enabled": False,
            "mode": "auto",
            "sent": sent,
            "last_run_cst": state["last_run_cst"],
        }

    # --- boot ping (once) ---
    if boot_once and not state.get("booted"):
        price = snap.get("price")
        cvdd = (snap.get("cvdd") or {}).get("cvdd")
        msg = (
            "*Trading OS 全自动监控已启动*\n"
            f"阶段 `{phase or '?'}` · 分 {score.get('score')}/{score.get('score_max')}\n"
            f"BTC `${price}` · CVDD `${cvdd}`\n"
            f"{_now_cst().strftime('%Y-%m-%d %H:%M')} CST\n"
            "之后异动/阶段切换会自动推送，无需开网页。"
        )
        if _send_tg(msg):
            sent.append("boot")
        state["booted"] = True
        state["booted_at_cst"] = _now_cst().isoformat()

    # --- phase change ---
    if phase:
        last_phase = state.get("last_phase")
        if last_phase and last_phase != phase:
            msg = (
                f"*Trading OS 阶段切换*\n"
                f"`{last_phase}` → `{phase}`\n"
                f"价格 `${snap.get('price')}` · 距底区 {score.get('distance_pct')}%\n"
                f"{_now_cst().strftime('%Y-%m-%d %H:%M')} CST"
            )
            if _send_tg(msg):
                sent.append(f"phase:{last_phase}->{phase}")
        state["last_phase"] = phase

    # --- score / signal combo (approach zone) ---
    sigs = score.get("signals") or {}
    sc = int(score.get("score") or 0)
    if phase in ("approach", "confirmed") or sc >= 3:
        if _cooldown_ok(state, "score_zone", score_cd_h):
            bits = [k for k, v in sigs.items() if v]
            msg = (
                f"*周期关注区*\n"
                f"阶段 `{phase}` · 分 {sc}/{score.get('score_max')}\n"
                f"距底区 {score.get('distance_pct')}% · BTC `${snap.get('price')}`\n"
                f"信号: {', '.join(bits) or '—'}"
            )
            if _send_tg(msg):
                sent.append(f"score:{phase}:{sc}")
                _mark_cooldown(state, "score_zone")

    # --- wallets ---
    for ev in wallets.get("events") or []:
        if ev.get("note") == "first_snapshot":
            continue
        if ev.get("note") == "discovered_cex_outflow":
            amt = (
                ev.get("withdraw_sol")
                if ev.get("chain") == "sol"
                else ev.get("withdraw_eth")
                if ev.get("chain") == "eth"
                else ev.get("withdraw_btc")
            )
            unit = "SOL" if ev.get("chain") == "sol" else "ETH" if ev.get("chain") == "eth" else "BTC"
            key = f"disc:{ev.get('address')}:{amt}"
            if key in (state.get("wallet_dedup") or []):
                continue
            msg = (
                f"*新发现吸筹地址（CEX大额提币）*\n"
                f"{ev.get('label')}\n"
                f"`{ev.get('address')}`\n"
                f"约 {amt} {unit} · 来自 {ev.get('probe')}"
            )
            if _send_tg(msg):
                sent.append(f"discover:{ev.get('address')}")
            dedup = list(state.get("wallet_dedup") or [])
            dedup.append(key)
            state["wallet_dedup"] = dedup[-80:]
            continue
        key = f"w:{ev.get('address')}:{ev.get('change_pct')}:{ev.get('new_balance')}"
        if key in (state.get("wallet_dedup") or []):
            continue
        msg = (
            f"*钱包异动*\n"
            f"{ev.get('label')} (`{ev.get('chain')}`)\n"
            f"{ev.get('old_balance')} → {ev.get('new_balance')} {ev.get('unit')} "
            f"({ev.get('change_pct')}%)"
        )
        if _send_tg(msg):
            sent.append(f"wallet:{ev.get('address')}")
        dedup = list(state.get("wallet_dedup") or [])
        dedup.append(key)
        state["wallet_dedup"] = dedup[-50:]

    # --- alts (time cooldown per symbol) ---
    for row in alts.get("flagged") or []:
        sym = str(row.get("symbol") or "")
        cd_key = f"alt:{sym}"
        if not sym or not _cooldown_ok(state, cd_key, alt_cd_h):
            continue
        msg = (
            f"*山寨盘口吸筹标记*\n"
            f"`{sym}` imbalance={row.get('imbalance')} "
            f"24h={row.get('change_24h_pct')}% vol≈{row.get('quote_volume_24h')}"
        )
        if _send_tg(msg):
            sent.append(f"alt:{sym}")
            _mark_cooldown(state, cd_key)

    # --- new strategy opens ---
    seen = dict(state.get("strategy_seen") or {})
    for lane, pack in (strategy.get("lanes") or {}).items():
        sigs_list = pack.get("signals") or []
        ids = [int(s["id"]) for s in sigs_list if s.get("id") is not None]
        if not ids:
            continue
        newest = max(ids)
        prev = seen.get(lane)
        if prev is not None:
            for sg in sigs_list:
                sid = sg.get("id")
                if sid is None or int(sid) <= int(prev):
                    continue
                sym = sg.get("symbol") or sg.get("pair") or "?"
                side = sg.get("side") or ""
                msg = f"*策略新开仓信号*\n`{lane}` · {sym} {side} · id={sid}"
                if _send_tg(msg):
                    sent.append(f"strategy:{lane}:{sid}")
        seen[lane] = newest
    state["strategy_seen"] = seen

    # --- periodic digest ---
    if digest_h > 0 and _cooldown_ok(state, "digest", digest_h):
        flagged_n = len(alts.get("flagged") or [])
        w_n = len(wallets.get("balances") or {})
        msg = (
            f"*Trading OS 定时摘要*\n"
            f"阶段 `{phase}` · 分 {score.get('score')}/{score.get('score_max')}\n"
            f"BTC `${snap.get('price')}` · CVDD `${(snap.get('cvdd') or {}).get('cvdd')}`\n"
            f"距底区 {score.get('distance_pct')}% · 钱包 {w_n} · 吸筹标记 {flagged_n}\n"
            f"{_now_cst().strftime('%Y-%m-%d %H:%M')} CST"
        )
        if _send_tg(msg):
            sent.append("digest")
            _mark_cooldown(state, "digest")

    state["last_run_cst"] = _now_cst().isoformat()
    state["mode"] = "auto"
    state["last_sent"] = sent
    state["updated_at_cst"] = state["last_run_cst"]
    _write_json(MONITOR_STATE, state)
    return {
        "ok": True,
        "enabled": True,
        "mode": "auto",
        "sent": sent,
        "last_run_cst": state["last_run_cst"],
        "booted": bool(state.get("booted")),
        "digest_hours": digest_h,
    }


def monitor_status() -> dict[str, Any]:
    state = _read_json(MONITOR_STATE, {})
    interval = int(os.getenv("NEXT_K_TRADING_OS_INTERVAL_MIN", "15") or "15")
    return {
        "ok": True,
        "mode": "auto" if _alerts_enabled() else "auto_disabled",
        "alerts_enabled": _alerts_enabled(),
        "interval_min": max(5, interval) if interval > 0 else 0,
        "booted": bool(state.get("booted")),
        "booted_at_cst": state.get("booted_at_cst"),
        "last_run_cst": state.get("last_run_cst") or state.get("updated_at_cst"),
        "last_phase": state.get("last_phase"),
        "last_sent": state.get("last_sent") or [],
        "digest_hours": float(os.getenv("TRADING_OS_DIGEST_HOURS", "8") or 8),
        "note": "API 进程内调度器定时刷新；异动自动推 TG，无需开网页",
    }


# ---------- bundle used by scheduler / snapshot ----------

def refresh_desk_bundle(*, equity_usd: float | None = None) -> dict[str, Any]:
    wallets = refresh_wallets()
    alts = refresh_alts()
    risk = compute_risk(equity_usd)
    strategy = load_strategy_link(limit=15)
    bundle = {
        "ok": True,
        "fetched_at_cst": _now_cst().isoformat(),
        "wallets": wallets,
        "alts": alts,
        "risk": risk,
        "strategy": strategy,
    }
    _write_json(DESK_BUNDLE, bundle)
    return bundle


def load_desk_bundle() -> dict[str, Any]:
    b = _read_json(DESK_BUNDLE, {})
    if b.get("ok"):
        return {**b, "cached": True}
    return refresh_desk_bundle()


def attach_phase_alert(phase: str) -> dict[str, Any]:
    return run_auto_monitor({"score": {"phase": phase}})
