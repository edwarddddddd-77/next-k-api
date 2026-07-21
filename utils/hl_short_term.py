"""Hyperliquid short-term watchlist monitor (read-only, no private keys)."""

from __future__ import annotations

import json
import logging
import os
import threading
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
    """Prefer DATA_DIR override; fall back to committed file under project root."""
    data = resolve_data_dir() / WATCHLIST_NAME
    if data.exists():
        return data
    return PROJECT_ROOT / WATCHLIST_NAME


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


def snapshot(address: str) -> dict:
    state = http_json({"type": "clearinghouseState", "user": address})
    fills = http_json({"type": "userFills", "user": address})
    if not isinstance(fills, list):
        fills = []

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

    return {
        "address": address,
        "account_value": av,
        "positions": positions,
        "recent_fills": recent,
        "ts": datetime.now(timezone.utc).isoformat(),
    }


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
