"""Hyperliquid short-term watchlist monitor (read-only, no private keys)."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

INFO_URL = "https://api.hyperliquid.xyz/info"
WATCHLIST_NAME = "hl_short_term_watchlist.json"
STATE_NAME = "hl_short_term_state.json"
EVENTS_NAME = "hl_short_term_events.jsonl"
BOARD_NAME = "hl_short_term_board.json"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_lock = threading.Lock()


def resolve_data_dir() -> Path:
    raw = (os.getenv("DATA_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser()
    for candidate in (Path("/app/data"), Path("/data")):
        if candidate.is_dir():
            return candidate
    return PROJECT_ROOT


def _watchlist_path() -> Path:
    """Watchlist is deploy config: prefer repo file when newer or when DATA_DIR has none.

    Runtime volumes often keep a stale ``hl_short_term_watchlist.json`` that would
    otherwise shadow a pushed A–J update forever.
    """
    root = PROJECT_ROOT / WATCHLIST_NAME
    data = resolve_data_dir() / WATCHLIST_NAME
    if root.is_file() and data.is_file():
        try:
            if root.stat().st_mtime >= data.stat().st_mtime:
                return root
        except OSError:
            return root
        return data
    if root.is_file():
        return root
    return data


def _state_path() -> Path:
    return resolve_data_dir() / STATE_NAME


def _events_path() -> Path:
    return resolve_data_dir() / EVENTS_NAME


def _board_path() -> Path:
    return resolve_data_dir() / BOARD_NAME


def http_json(body: dict) -> dict | list:
    req = urllib.request.Request(
        INFO_URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "next-k-hl-short/1.0"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def load_watchlist_doc() -> dict[str, Any]:
    path = _watchlist_path()
    if not path.exists():
        return {
            "updated": None,
            "venue": "hyperliquid",
            "wallets": [],
            "reject_for_now": [],
            "error": f"missing watchlist: {path}",
        }
    return json.loads(path.read_text(encoding="utf-8"))


def load_watchlist() -> list[dict]:
    doc = load_watchlist_doc()
    wallets = doc.get("wallets") or []
    return [w for w in wallets if isinstance(w, dict) and w.get("address")]


def snapshot_positions(address: str) -> dict:
    """Clearinghouse-only snapshot (no userFills) — used by paper mirror on each WS fill."""
    state = http_json({"type": "clearinghouseState", "user": address})
    av = float((state.get("marginSummary") or {}).get("accountValue") or 0)
    positions = []
    for item in state.get("assetPositions") or []:
        pos = item.get("position") or {}
        szi = float(pos.get("szi") or 0)
        if abs(szi) < 1e-12:
            continue
        positions.append(
            {
                "coin": pos.get("coin"),
                "szi": szi,
                "entry": pos.get("entryPx"),
                "uPnl": float(pos.get("unrealizedPnl") or 0),
                "lev": (pos.get("leverage") or {}).get("value"),
            }
        )
    return {
        "address": address,
        "account_value": av,
        "positions": positions,
        "recent_fills": [],
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def snapshot_spot_usdc(address: str) -> float:
    """Same wallet's Core spot USDC balance (not HyperEVM). Negative totals clamp to 0."""
    return float(snapshot_spot(address, fill_limit=0).get("usdc") or 0)


# HyperEVM native USDC (Circle) — monitor only; never used for copy sizing.
HYPEREVM_RPC = (os.getenv("HYPEREVM_RPC") or "https://rpc.hyperliquid.xyz/evm").strip()
HYPEREVM_USDC = (
    os.getenv("HYPEREVM_USDC") or "0xb88339CB7199b77E23DB6E890353E22632Ba630f"
).strip()


def snapshot_hyperevm_usdc(address: str) -> float | None:
    """USDC balance on HyperEVM for the same 0x address (bridge destination).

    Returns None on RPC failure so UI can distinguish unknown vs zero.
    """
    addr = str(address or "").strip().lower()
    if not addr.startswith("0x") or len(addr) != 42:
        return None
    if not HYPEREVM_RPC or not HYPEREVM_USDC:
        return None
    # balanceOf(address)
    data = "0x70a08231000000000000000000000000" + addr[2:]
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "eth_call",
            "params": [{"to": HYPEREVM_USDC, "data": data}, "latest"],
        }
    ).encode()
    req = urllib.request.Request(
        HYPEREVM_RPC,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            out = json.loads(resp.read().decode())
    except Exception as exc:
        logger.warning("HyperEVM USDC %s: %s", addr[:10], exc)
        return None
    if not isinstance(out, dict) or out.get("error"):
        logger.warning("HyperEVM USDC rpc error %s: %s", addr[:10], out)
        return None
    raw = out.get("result") or "0x0"
    try:
        return int(str(raw), 16) / 1e6
    except (TypeError, ValueError):
        return None


_spot_meta_at: float = 0.0
_spot_meta_pair: dict[str, str] = {}  # "@107" / "PURR/USDC" -> display name
_spot_meta_ttl_sec = float(os.getenv("HL_SPOT_META_TTL_SEC", "3600") or 3600)


def is_hl_spot_coin(coin: Any) -> bool:
    """Spot fills use PURR/USDC or @{universeIndex}; perps are bare tickers."""
    c = str(coin or "").strip()
    if not c:
        return False
    return c.startswith("@") or "/" in c


def _refresh_spot_meta(*, force: bool = False) -> dict[str, str]:
    global _spot_meta_at, _spot_meta_pair
    now = time.time()
    if (
        not force
        and _spot_meta_pair
        and (now - _spot_meta_at) < max(60.0, _spot_meta_ttl_sec)
    ):
        return _spot_meta_pair
    meta = http_json({"type": "spotMeta"})
    if not isinstance(meta, dict):
        return _spot_meta_pair
    tokens = {
        int(t["index"]): str(t.get("name") or "").strip()
        for t in (meta.get("tokens") or [])
        if isinstance(t, dict) and t.get("index") is not None
    }
    pair: dict[str, str] = {}
    for u in meta.get("universe") or []:
        if not isinstance(u, dict):
            continue
        name = str(u.get("name") or "").strip()
        try:
            idx = int(u.get("index"))
        except (TypeError, ValueError):
            idx = None
        toks = u.get("tokens") if isinstance(u.get("tokens"), list) else []
        base = tokens.get(int(toks[0])) if len(toks) >= 1 else None
        quote = tokens.get(int(toks[1])) if len(toks) >= 2 else None
        display = (
            f"{base}/{quote}"
            if base and quote
            else (name or (f"@{idx}" if idx is not None else ""))
        )
        if name:
            pair[name.upper()] = display
            pair[name] = display
        if idx is not None:
            pair[f"@{idx}"] = display
            pair[f"@{idx}".upper()] = display
        # Balance rows use token name (HYPE). Prefer */USDC; never overwrite that
        # with a non-USDC quote pair (e.g. HYPE/USDE).
        if base and base.upper() != "USDC":
            key_u = base.upper()
            quote_u = str(quote or "").upper()
            if quote_u == "USDC":
                pair[key_u] = display
                pair[base] = display
            elif key_u not in pair:
                pair[key_u] = display
                pair[base] = display
    if pair:
        _spot_meta_pair = pair
        _spot_meta_at = now
    return _spot_meta_pair


def resolve_spot_coin(coin: Any) -> str:
    raw = str(coin or "").strip()
    if not raw:
        return ""
    maps = _refresh_spot_meta()
    return maps.get(raw) or maps.get(raw.upper()) or (
        raw.upper() if not is_hl_spot_coin(raw) else raw
    )


def snapshot_spot(address: str, *, fill_limit: int = 20) -> dict:
    """Core spot balances + recent spot fills for the same wallet (monitor only)."""
    spot = http_json({"type": "spotClearinghouseState", "user": address})
    usdc = 0.0
    balances: list[dict] = []
    if isinstance(spot, dict):
        for b in spot.get("balances") or []:
            if not isinstance(b, dict):
                continue
            coin = str(b.get("coin") or "").strip()
            if not coin:
                continue
            try:
                total = float(b.get("total") or 0)
            except (TypeError, ValueError):
                continue
            try:
                hold = float(b.get("hold") or 0)
            except (TypeError, ValueError):
                hold = 0.0
            try:
                entry_ntl = float(b.get("entryNtl") or 0)
            except (TypeError, ValueError):
                entry_ntl = 0.0
            if coin.upper() == "USDC":
                usdc = max(0.0, total)
            # Skip dust / accounting noise
            if abs(total) < 1e-8 and abs(hold) < 1e-8:
                continue
            if coin.upper() == "USDC" and usdc < 0.01:
                continue
            balances.append(
                {
                    "coin": resolve_spot_coin(coin),
                    "coin_raw": coin,
                    "total": round(total, 8),
                    "hold": round(hold, 8),
                    "entry_ntl": round(entry_ntl, 4),
                }
            )
    balances.sort(
        key=lambda x: (
            0 if str(x.get("coin_raw") or x.get("coin") or "").upper() == "USDC" else 1,
            -abs(float(x.get("entry_ntl") or 0)),
            -abs(float(x.get("total") or 0)),
        )
    )

    recent: list[dict] = []
    if fill_limit > 0:
        fills = http_json({"type": "userFills", "user": address})
        if isinstance(fills, list):
            for f in fills:
                if not isinstance(f, dict):
                    continue
                if not is_hl_spot_coin(f.get("coin")):
                    continue
                side = str(f.get("side") or "").strip().upper()
                if side in ("B", "BUY"):
                    side_l = "buy"
                elif side in ("A", "SELL"):
                    side_l = "sell"
                else:
                    side_l = side.lower() or None
                raw_coin = str(f.get("coin") or "")
                try:
                    px = float(f.get("px") or 0)
                    sz = float(f.get("sz") or 0)
                except (TypeError, ValueError):
                    px, sz = 0.0, 0.0
                if px <= 0 or abs(sz) <= 0:
                    continue
                ft = f.get("time")
                try:
                    ft_f = float(ft) if ft is not None else None
                except (TypeError, ValueError):
                    ft_f = None
                if ft_f is not None and ft_f > 1e12:
                    ft_sec = ft_f / 1000.0
                elif ft_f is not None and ft_f > 1e9:
                    ft_sec = ft_f
                else:
                    ft_sec = None
                recent.append(
                    {
                        "coin": resolve_spot_coin(raw_coin),
                        "coin_raw": raw_coin,
                        "side": side_l,
                        "dir": f.get("dir"),
                        "sz": sz,
                        "px": px,
                        "notional": round(abs(px * sz), 4),
                        "time": f.get("time"),
                        "tid": f.get("tid") or f.get("hash"),
                        "ts": (
                            datetime.fromtimestamp(ft_sec, timezone.utc).isoformat()
                            if ft_sec is not None
                            else None
                        ),
                    }
                )
            recent.sort(
                key=lambda r: float(r.get("time") or 0),
                reverse=True,
            )
            recent = recent[:fill_limit]

    return {
        "address": address,
        "usdc": round(usdc, 4),
        "balances": balances[:40],
        "recent_fills": recent,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def snapshot(address: str) -> dict:
    out = snapshot_positions(address)
    fills = http_json({"type": "userFills", "user": address})
    if not isinstance(fills, list):
        fills = []
    recent = []
    for f in fills[:15]:
        recent.append(
            {
                "coin": f.get("coin"),
                "side": f.get("side"),
                "sz": f.get("sz"),
                "px": f.get("px"),
                "closedPnl": f.get("closedPnl"),
                "time": f.get("time"),
                "tid": f.get("tid") or f.get("hash"),
            }
        )
    out["recent_fills"] = recent
    return out


def diff_snapshot(prev: dict | None, cur: dict) -> list[dict]:
    events: list[dict] = []
    if not prev:
        events.append(
            {
                "type": "baseline",
                "address": cur["address"],
                "account_value": cur["account_value"],
                "positions": cur["positions"],
            }
        )
        return events

    if abs(cur["account_value"] - prev.get("account_value", 0)) >= 1.0:
        events.append(
            {
                "type": "account_value_change",
                "address": cur["address"],
                "from": prev.get("account_value"),
                "to": cur["account_value"],
            }
        )

    prev_map = {p["coin"]: p for p in prev.get("positions") or []}
    cur_map = {p["coin"]: p for p in cur["positions"]}

    for coin, p in cur_map.items():
        if coin not in prev_map:
            events.append({"type": "position_open", "address": cur["address"], "position": p})
        else:
            old = prev_map[coin]
            if abs(float(old["szi"]) - float(p["szi"])) > 1e-9:
                events.append(
                    {
                        "type": "position_size_change",
                        "address": cur["address"],
                        "coin": coin,
                        "from_szi": old["szi"],
                        "to_szi": p["szi"],
                        "uPnl": p["uPnl"],
                    }
                )

    for coin, p in prev_map.items():
        if coin not in cur_map:
            events.append({"type": "position_close", "address": cur["address"], "position": p})

    prev_ids = {f.get("tid") for f in prev.get("recent_fills") or []}
    for f in cur["recent_fills"]:
        tid = f.get("tid")
        if tid and tid not in prev_ids:
            events.append({"type": "new_fill", "address": cur["address"], "fill": f})

    return events


def _load_state() -> dict:
    path = _state_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(state: dict) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def append_events(events: list[dict]) -> None:
    if not events:
        return
    path = _events_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for e in events:
            row = dict(e)
            row["logged_at"] = datetime.now(timezone.utc).isoformat()
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_events(*, limit: int = 50) -> list[dict]:
    path = _events_path()
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[dict] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(out) >= limit:
            break
    return out


def load_board() -> dict[str, Any] | None:
    path = _board_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _save_board(board: dict[str, Any]) -> None:
    path = _board_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(board, indent=2, ensure_ascii=False), encoding="utf-8")


def refresh_board() -> dict[str, Any]:
    """Poll HL for all watchlist wallets; update state, events, board snapshot."""
    with _lock:
        wallets = load_watchlist()
        state = _load_state()
        now_state = dict(state)
        rows: list[dict] = []
        all_events: list[dict] = []
        errors: list[dict] = []

        for w in sorted(wallets, key=lambda x: x.get("priority", 99)):
            addr = str(w["address"])
            label = w.get("id") or addr[:10]
            try:
                cur = snapshot(addr)
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
                logger.warning("hl_short snapshot failed %s: %s", label, exc)
                errors.append({"id": label, "address": addr, "error": str(exc)})
                prev = state.get(addr)
                rows.append(
                    {
                        "meta": w,
                        "snapshot": prev,
                        "error": str(exc),
                    }
                )
                continue

            prev = state.get(addr)
            events = diff_snapshot(prev, cur)
            append_events(events)
            all_events.extend(events)
            now_state[addr] = cur
            rows.append({"meta": w, "snapshot": cur, "events_this_poll": events, "error": None})

        _save_state(now_state)
        board = {
            "ok": True,
            "venue": "hyperliquid",
            "mode": "read_only",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "wallet_count": len(wallets),
            "wallets": rows,
            "errors": errors,
            "events_logged": len(all_events),
        }
        _save_board(board)
        return board


def build_board(*, refresh: bool = False) -> dict[str, Any]:
    if refresh:
        return refresh_board()
    snap = load_board()
    if snap:
        snap = dict(snap)
        snap["snapshot_source"] = "cache"
        return snap
    return refresh_board()
